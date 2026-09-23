package inference

import (
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
)

// ModelWeight is one member of a blended generation: a domain model and its
// ensemble weight (typically the meta-router's probability for that domain).
type ModelWeight struct {
	Model  *model.Transformer
	Weight float32
	Domain string
}

// Ensemble runs blended autoregressive generation over several domain models.
// Every selected model keeps its own KV cache; the next-token logits are the
// weighted sum of the members' logits, so a token is sampled once from the
// combined distribution.
//
// This is an ensemble over whole models, not a per-token mixture: all member
// models run on every step, which is the cost of blending independent domain
// models whose weights cannot be merged.
type Ensemble struct {
	Models    []ModelWeight
	Tokenizer *tokenizer.Tokenizer
	// Tools is the registry used to execute tool calls; nil disables them.
	Tools *Registry
}

// NewEnsemble builds a blended engine over the given model weights.
func NewEnsemble(models []ModelWeight, tokenizerImpl *tokenizer.Tokenizer) *Ensemble {
	return &Ensemble{Models: models, Tokenizer: tokenizerImpl, Tools: NewCalculator()}
}

// normalizedWeights returns member weights scaled to sum to one (uniformly when
// they carry no mass).
func (ensemble *Ensemble) normalizedWeights() []float32 {
	weights := make([]float32, len(ensemble.Models))
	var sum float32
	for index, member := range ensemble.Models {
		weight := member.Weight
		if weight < 0 {
			weight = 0
		}
		weights[index] = weight
		sum += weight
	}
	if sum <= 0 {
		for index := range weights {
			weights[index] = 1 / float32(len(weights))
		}
		return weights
	}
	for index := range weights {
		weights[index] /= sum
	}
	return weights
}

// newCacheForModel allocates a KV buffer sized for a model's config, including
// the MLA and compressed/indexer tiers that config needs.
func newCacheForModel(config model.Config, batch, capacity int) *model.KVBuffer {
	cache := model.NewKVBuffer(batch, capacity, config.NumLayer, config.NumKVHead, config.HeadDim())
	if config.MLAEnabled() {
		cache.EnableMLA(config.MLALatent, config.NumKVHead*config.MLARotaryDimension())
	}
	if ratio := config.Compression(); ratio > 1 {
		kvWidth := config.NumKVHead * config.HeadDim()
		cache.EnableCompressionLayers(ratio, config.EmbedDim, kvWidth, capacity/ratio+1, compressionAllocMask(config))
		if config.SparseTopK > 0 {
			cache.EnableIndexerKeysLayers(indexerKeyWidth(config), indexerAllocMask(config))
		}
	}
	return cache
}

// prefillModel runs one member over the prompt and returns its decode cache and
// the final prompt position's logits.
func (ensemble *Ensemble) prefillModel(member ModelWeight, tokens []int, numSamples, maxTokens int) (*model.KVBuffer, []float32) {
	config := member.Model.Config
	inputs := tensors.NewInt32sWithData([]int{1, len(tokens)}, toI32(tokens))
	prefillCache := newCacheForModel(config, 1, len(tokens))
	var logits *tensors.Tensor
	if config.CEDEnabled() {
		logits = member.Model.PrefillCED(inputs, prefillCache)
	} else {
		logits = member.Model.Forward(inputs, prefillCache)
	}
	rows := logits.Shape[1]
	vocab := config.VocabSize
	last := logits.Data[(rows-1)*vocab : rows*vocab]
	combined := append([]float32(nil), last...)

	decodeCache := newCacheForModel(config, numSamples, len(tokens)+maxTokens)
	model.PrefillFrom(decodeCache, prefillCache)
	return decodeCache, combined
}

// GenerateWith runs blended generation, yielding (tokenColumn, tokenMask) per
// step exactly like Engine.GenerateWith. Thinking-budget handling matches the
// single-model engine.
func (ensemble *Ensemble) GenerateWith(tokens []int, numSamples, maxTokens int, temperature float32, topK int, seed uint64, options GenerateOptions) func(yield func([]int, []int) bool) {
	return func(yield func([]int, []int) bool) {
		if len(ensemble.Models) == 0 {
			panic("inference: ensemble has no models")
		}
		weights := ensemble.normalizedWeights()
		vocab := ensemble.Models[0].Model.Config.VocabSize

		caches := make([]*model.KVBuffer, len(ensemble.Models))
		combined := make([]float32, numSamples*vocab)
		for index, member := range ensemble.Models {
			cache, last := ensemble.prefillModel(member, tokens, numSamples, maxTokens)
			caches[index] = cache
			for row := 0; row < numSamples; row++ {
				addScaledSlice(combined[row*vocab:(row+1)*vocab], last, weights[index])
			}
		}

		logitsTensor := tensors.NewWithData([]int{numSamples, vocab}, combined)

		states := make([]*RowState, numSamples)
		for index := range states {
			state := &RowState{}
			state.currentTokens = append([]int(nil), tokens...)
			states[index] = state
		}

		specialToken := func(name string) int { return ensemble.Tokenizer.EncodeSpecial(name) }
		toolStart := specialToken("<|tool_start|>")
		toolEnd := specialToken("<|tool_end|>")
		toolOutputStart := specialToken("<|tool_output_start|>")
		toolOutputEnd := specialToken("<|tool_output_end|>")
		assistantEnd := specialToken("<|assistant_end|>")
		thinkStart := specialToken("<|think_start|>")
		thinkEnd := specialToken("<|think_end|>")
		bosToken := ensemble.Tokenizer.BOSTokenID()

		if (options.Thinking || options.ThinkingBudget > 0) && len(tokens) > 0 && tokens[len(tokens)-1] == thinkStart {
			for _, state := range states {
				state.inThinking = true
			}
		}

		randomGenerator := tensors.NewRNG(seed)
		generated := 0
		for {
			if maxTokens > 0 && generated >= maxTokens {
				return
			}
			allDone := true
			for _, state := range states {
				if !state.completed {
					allDone = false
					break
				}
			}
			if allDone {
				return
			}

			nextIDs := SampleNextToken(logitsTensor, randomGenerator, temperature, topK)
			tokenColumn := make([]int, numSamples)
			tokenMask := make([]int, numSamples)
			for index, state := range states {
				isForced := len(state.forcedTokens) > 0
				if isForced {
					tokenMask[index] = 0
				} else {
					tokenMask[index] = 1
				}
				var nextToken int
				if isForced {
					nextToken = state.forcedTokens[0]
					state.forcedTokens = state.forcedTokens[1:]
				} else {
					nextToken = int(nextIDs.Data[index])
				}
				tokenColumn[index] = nextToken
				state.currentTokens = append(state.currentTokens, nextToken)
				if nextToken == assistantEnd || nextToken == bosToken {
					state.completed = true
				}
				switch {
				case nextToken == toolStart:
					state.inToolCall = true
					state.toolCallTokens = nil
				case nextToken == toolEnd && state.inToolCall:
					state.inToolCall = false
					if len(state.toolCallTokens) > 0 && ensemble.Tools != nil {
						expression := ensemble.Tokenizer.Decode(state.toolCallTokens)
						if result, ok := ensemble.Tools.Execute(expression); ok {
							resultTokens := ensemble.Tokenizer.Encode(result)
							state.forcedTokens = append(state.forcedTokens, toolOutputStart)
							state.forcedTokens = append(state.forcedTokens, resultTokens...)
							state.forcedTokens = append(state.forcedTokens, toolOutputEnd)
						}
					}
					state.toolCallTokens = nil
				case state.inToolCall:
					state.toolCallTokens = append(state.toolCallTokens, nextToken)
				}
				if options.Thinking || options.ThinkingBudget > 0 {
					switch {
					case nextToken == thinkStart:
						state.inThinking = true
						state.thinkingTokens = 0
					case nextToken == thinkEnd && state.inThinking:
						state.inThinking = false
					case state.inThinking:
						state.thinkingTokens++
						if options.ThinkingBudget > 0 && state.thinkingTokens >= options.ThinkingBudget {
							state.forcedTokens = append(state.forcedTokens, thinkEnd)
						}
					}
				}
			}

			if !yield(tokenColumn, tokenMask) {
				return
			}
			generated++

			// Run every member on the shared token column and combine logits.
			nextInputIDs := tensors.NewInt32sWithData([]int{numSamples, 1}, toI32(tokenColumn))
			for row := range combined {
				combined[row] = 0
			}
			for index, member := range ensemble.Models {
				stepLogits := member.Model.Forward(nextInputIDs, caches[index]) // [B,1,vocab]
				flat := stepLogits.Data
				for row := 0; row < numSamples; row++ {
					base := row * vocab
					addScaledSlice(combined[base:base+vocab], flat[base:base+vocab], weights[index])
				}
			}
			logitsTensor = tensors.NewWithData([]int{numSamples, vocab}, combined)
		}
	}
}

// GenerateBatch runs blended generation and returns the final sequences.
func (ensemble *Ensemble) GenerateBatch(tokens []int, numSamples, maxTokens int, temperature float32, topK int, seed uint64) ([][]int, [][]int) {
	assistantEnd := ensemble.Tokenizer.EncodeSpecial("<|assistant_end|>")
	bosToken := ensemble.Tokenizer.BOSTokenID()

	results := make([][]int, numSamples)
	masks := make([][]int, numSamples)
	for index := 0; index < numSamples; index++ {
		results[index] = append([]int(nil), tokens...)
		masks[index] = make([]int, len(tokens))
	}
	completed := make([]bool, numSamples)

	generate := ensemble.GenerateWith(tokens, numSamples, maxTokens, temperature, topK, seed, GenerateOptions{})
	generate(func(column, mask []int) bool {
		for index := range column {
			if completed[index] {
				continue
			}
			if column[index] == assistantEnd || column[index] == bosToken {
				completed[index] = true
			} else {
				results[index] = append(results[index], column[index])
				masks[index] = append(masks[index], mask[index])
			}
		}
		for _, done := range completed {
			if !done {
				return true
			}
		}
		return false
	})
	return results, masks
}

// addScaledSlice computes destination += weight*source.
func addScaledSlice(destination, source []float32, weight float32) {
	for index := range destination {
		destination[index] += weight * source[index]
	}
}
