package inference

import (
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
)

// RowState tracks the state of one generation row, including forced tokens
// (tool use) and completion status.
type RowState struct {
	currentTokens  []int
	forcedTokens   []int
	inToolCall     bool
	toolCallTokens []int
	completed      bool
}

// Engine performs batched autoregressive generation with a KV cache and a
// tool-call state machine.
type Engine struct {
	Model     *model.Transformer
	Tokenizer *tokenizer.Tokenizer
	// Tools is the registry used to execute tool calls. It defaults to the
	// built-in calculator; register additional Go tools to extend it.
	Tools *Registry
	// Prefix, when non-nil, enables in-memory prefix caching: a request whose
	// prompt strictly extends a previously prefilled prompt reuses that KV
	// state and only prefills the new suffix. It is ignored for CED models.
	Prefix *PrefixCache
	// Cache, when non-nil, enables the multi-entry persistent KV cache tier
	// (DeepSeek-V4.1 §3.2.1). It takes precedence over Prefix. It is ignored
	// for CED models.
	Cache *CacheManager
	// SWACache, when non-nil, is a small short-TTL cache of full KV snapshots
	// (including sliding-window state) checked before Cache. It is the paper's
	// host-DRAM SWA pool: a hit avoids the bounded replay a stripped Cache entry
	// would need.
	SWACache *CacheManager
	// Drafter, when set alongside Speculative, enables exact greedy speculative
	// decoding (DeepSeek-V4.1 §2.4.3) with the given draft model.
	Drafter *model.Transformer
	// DSpark, when set alongside Speculative, enables speculative decoding with
	// the DSpark heads: the confidence head schedules how many drafted tokens
	// are verified. It takes precedence over Drafter.
	DSpark *model.DSpark
	// ConfidenceThreshold is the minimum confidence for a drafted token to be
	// verified. Zero uses the default of 0.5.
	ConfidenceThreshold float32
	// Speculative enables speculative decoding when Drafter is set. It only
	// applies to single-row greedy requests on uncompressed models and does not
	// run the tool-call state machine.
	Speculative bool
	// DraftLength is the number of tokens the drafter proposes per round.
	// Values below 2 use the default of 5.
	DraftLength int
}

// prefixCacheStore is the common interface of the single-entry PrefixCache and
// the multi-entry CacheManager.
type prefixCacheStore interface {
	Lookup(tokens []int) (*model.KVBuffer, int)
	Store(tokens []int, state *model.KVBuffer)
}

// prefixStore returns the active prefix cache, preferring the multi-entry
// CacheManager when both are configured.
func (engine *Engine) prefixStore() prefixCacheStore {
	if engine.Cache != nil {
		return engine.Cache
	}
	if engine.Prefix != nil {
		return engine.Prefix
	}
	return nil
}

// NewEngine builds an inference engine over the given model and tokenizer. It
// installs the built-in calculator tool.
func NewEngine(transformer *model.Transformer, tokenizerImpl *tokenizer.Tokenizer) *Engine {
	return &Engine{Model: transformer, Tokenizer: tokenizerImpl, Tools: NewCalculator()}
}

// Generate produces token columns for num_samples rows, seeded from the given
// prompt tokens. It yields (tokenColumn, tokenMask) for each decode step,
// where tokenMask is 1 for sampled tokens and 0 for forced (tool) tokens.
func (engine *Engine) Generate(tokens []int, numSamples, maxTokens int, temperature float32, topK int, seed uint64) func(yield func([]int, []int) bool) {
	return func(yield func([]int, []int) bool) {
		if engine.speculativeEligible(numSamples, temperature) {
			engine.speculativeGenerate(tokens, maxTokens, yield)
			return
		}
		config := engine.Model.Config
		headDim := config.HeadDim()
		store := engine.prefixStore()
		swaStore := engine.SWACache

		// 1) Batch-1 prefill of the prompt, reusing a cached prefix when the
		// cached prompt strictly precedes the new one. The full-state SWA pool
		// is checked first; a stripped persistent entry triggers bounded replay.
		var prefillCache *model.KVBuffer
		matched := 0
		if !config.CEDEnabled() && swaStore != nil {
			if cached, hit := swaStore.Lookup(tokens); cached != nil {
				prefillCache = cached
				matched = hit
			}
		}
		if prefillCache == nil && !config.CEDEnabled() && store != nil {
			if cached, hit := store.Lookup(tokens); cached != nil {
				engine.replayLocalState(cached, tokens, hit)
				prefillCache = cached
				matched = hit
			}
		}
		if prefillCache == nil {
			// When prefix caching is enabled the prefill buffer is sized to the
			// full context so a later request can extend it in place.
			capacity := len(tokens)
			if (store != nil || swaStore != nil) && !config.CEDEnabled() {
				capacity = config.SequenceLen
			}
			prefillCache = model.NewKVBuffer(1, capacity, config.NumLayer, config.NumKVHead, headDim)
			if config.MLAEnabled() {
				prefillCache.EnableMLA(config.MLALatent, config.NumKVHead*config.MLARotaryDimension())
			}
			if ratio := config.Compression(); ratio > 1 {
				kvWidth := config.NumKVHead * headDim
				prefillCache.EnableCompressionLayers(ratio, config.EmbedDim, kvWidth, capacity/ratio+1, compressionAllocMask(config))
				if config.SparseTopK > 0 {
					prefillCache.EnableIndexerKeysLayers(indexerKeyWidth(config), indexerAllocMask(config))
				}
			}
		}
		inputTokens := tokens
		if matched > 0 {
			inputTokens = tokens[matched:]
		}
		inputIDs := tensors.NewInt32sWithData([]int{1, len(inputTokens)}, toI32(inputTokens))
		var logits *tensors.Tensor
		if config.CEDEnabled() {
			// The causal encoder-decoder prefill runs the encoder over the full
			// prompt and replays only the last window tokens through the
			// decoder; the returned logits cover that replay.
			logits = engine.Model.PrefillCED(inputIDs, prefillCache)
		} else {
			logits = engine.Model.Forward(inputIDs, prefillCache) // [1, T, vocab]
		}
		if !config.CEDEnabled() {
			if store != nil {
				store.Store(tokens, prefillCache)
			}
			if swaStore != nil {
				swaStore.Store(tokens, prefillCache)
			}
		}
		vocab := config.VocabSize
		rows := logits.Shape[1]
		lastLogits := logits.Reshape(rows, vocab)
		// Expand the last position's logits to numSamples rows.
		base := lastLogits.Data[(rows-1)*vocab : rows*vocab]
		expanded := make([]float32, numSamples*vocab)
		for index := 0; index < numSamples; index++ {
			copy(expanded[index*vocab:], base)
		}
		logitsTensor := tensors.NewWithData([]int{numSamples, vocab}, expanded)

		// 2) Replicate the KV cache across rows.
		cacheLen := len(tokens)
		if maxTokens > 0 {
			cacheLen += maxTokens
		}
		decodeCache := model.NewKVBuffer(numSamples, cacheLen, config.NumLayer, config.NumKVHead, headDim)
		if config.MLAEnabled() {
			decodeCache.EnableMLA(config.MLALatent, config.NumKVHead*config.MLARotaryDimension())
		}
		if ratio := config.Compression(); ratio > 1 {
			kvWidth := config.NumKVHead * headDim
			decodeCache.EnableCompressionLayers(ratio, config.EmbedDim, kvWidth, cacheLen/ratio+1, compressionAllocMask(config))
			if config.SparseTopK > 0 {
				decodeCache.EnableIndexerKeysLayers(indexerKeyWidth(config), indexerAllocMask(config))
			}
		}
		model.PrefillFrom(decodeCache, prefillCache)

		// 3) Row states.
		states := make([]*RowState, numSamples)
		for index := range states {
			state := &RowState{}
			state.currentTokens = append([]int(nil), tokens...)
			states[index] = state
		}

		specialToken := func(name string) int { return engine.Tokenizer.EncodeSpecial(name) }
		toolStart := specialToken("<|tool_start|>")
		toolEnd := specialToken("<|tool_end|>")
		toolOutputStart := specialToken("<|tool_output_start|>")
		toolOutputEnd := specialToken("<|tool_output_end|>")
		assistantEnd := specialToken("<|assistant_end|>")
		bosToken := engine.Tokenizer.BOSTokenID()

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

			nextIDs := SampleNextToken(logitsTensor, randomGenerator, temperature, topK) // [numSamples]

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
					if len(state.toolCallTokens) > 0 {
						expression := engine.Tokenizer.Decode(state.toolCallTokens)
						if result, ok := engine.Tools.Execute(expression); ok {
							resultTokens := engine.Tokenizer.Encode(result)
							state.forcedTokens = append(state.forcedTokens, toolOutputStart)
							state.forcedTokens = append(state.forcedTokens, resultTokens...)
							state.forcedTokens = append(state.forcedTokens, toolOutputEnd)
						}
					}
					state.toolCallTokens = nil
				case state.inToolCall:
					state.toolCallTokens = append(state.toolCallTokens, nextToken)
				}
			}

			if !yield(tokenColumn, tokenMask) {
				return
			}
			generated++

			// Forward the next token column.
			nextInputIDs := tensors.NewInt32sWithData([]int{numSamples, 1}, toI32(tokenColumn))
			logits = engine.Model.Forward(nextInputIDs, decodeCache) // [B, 1, vocab]
			logitsTensor = logits.Reshape(numSamples, vocab)
		}
	}
}

// GenerateBatch runs non-streaming generation and returns the final token
// sequences (excluding terminal tokens) and their masks.
func (engine *Engine) GenerateBatch(tokens []int, numSamples, maxTokens int, temperature float32, topK int, seed uint64) ([][]int, [][]int) {
	assistantEnd := engine.Tokenizer.EncodeSpecial("<|assistant_end|>")
	bosToken := engine.Tokenizer.BOSTokenID()

	results := make([][]int, numSamples)
	masks := make([][]int, numSamples)
	for index := 0; index < numSamples; index++ {
		results[index] = append([]int(nil), tokens...)
		masks[index] = make([]int, len(tokens))
	}
	completed := make([]bool, numSamples)

	generate := engine.Generate(tokens, numSamples, maxTokens, temperature, topK, seed)
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

// indexerKeyWidth returns the cached indexer key width for a config.
func indexerKeyWidth(config model.Config) int {
	dim := config.IndexerDim
	if dim <= 0 {
		dim = 64
	}
	heads := config.IndexerHeads
	if heads <= 0 {
		heads = 1
	}
	return dim * heads
}

// compressionAllocMask marks the layers that own compressed KV buffers: only
// full layers produce compressed state, while reindex/reuse layers borrow it.
func compressionAllocMask(config model.Config) []bool {
	mask := make([]bool, config.NumLayer)
	for layer := 0; layer < config.NumLayer; layer++ {
		mask[layer] = config.OwnsCompressed(layer)
	}
	return mask
}

// indexerAllocMask marks the layers that cache their own indexer keys: full and
// reindex layers. Pure reuse layers inherit the producer's selection.
func indexerAllocMask(config model.Config) []bool {
	if config.SparseTopK <= 0 {
		return nil
	}
	mask := make([]bool, config.NumLayer)
	for layer := 0; layer < config.NumLayer; layer++ {
		mask[layer] = config.OwnsIndexer(layer)
	}
	return mask
}

func toI32(ids []int) []int32 {
	converted := make([]int32, len(ids))
	for index, value := range ids {
		converted[index] = int32(value)
	}
	return converted
}
