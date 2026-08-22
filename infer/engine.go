package infer

import (
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensor"
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
}

// NewEngine builds an inference engine over the given model and tokenizer. It
// installs the built-in calculator tool.
func NewEngine(m *model.Transformer, tok *tokenizer.Tokenizer) *Engine {
	return &Engine{Model: m, Tokenizer: tok, Tools: NewCalculator()}
}

// Generate produces token columns for num_samples rows, seeded from the given
// prompt tokens. It yields (tokenColumn, tokenMask) for each decode step,
// where tokenMask is 1 for sampled tokens and 0 for forced (tool) tokens.
func (e *Engine) Generate(tokens []int, numSamples, maxTokens int, temperature float32, topK int, seed uint64) func(yield func([]int, []int) bool) {
	return func(yield func([]int, []int) bool) {
		cfg := e.Model.Config
		headDim := cfg.HeadDim()

		// 1) Batch-1 prefill of the prompt.
		prefillCache := model.NewKVBuffer(1, len(tokens), cfg.NumLayer, cfg.NumKVHead, headDim)
		idx := tensor.NewInt32sWithData([]int{1, len(tokens)}, toI32(tokens))
		logits := e.Model.Forward(idx, prefillCache) // [1, T, vocab]
		vocab := cfg.VocabSize
		lastLogits := logits.Reshape(len(tokens), vocab)
		// Expand the last position's logits to numSamples rows.
		base := lastLogits.Data[(len(tokens)-1)*vocab : len(tokens)*vocab]
		expanded := make([]float32, numSamples*vocab)
		for i := 0; i < numSamples; i++ {
			copy(expanded[i*vocab:], base)
		}
		logitsTensor := tensor.NewWithData([]int{numSamples, vocab}, expanded)

		// 2) Replicate the KV cache across rows.
		kvLen := len(tokens)
		if maxTokens > 0 {
			kvLen += maxTokens
		}
		decodeCache := model.NewKVBuffer(numSamples, kvLen, cfg.NumLayer, cfg.NumKVHead, headDim)
		model.PrefillFrom(decodeCache, prefillCache)

		// 3) Row states.
		states := make([]*RowState, numSamples)
		for i := range states {
			s := &RowState{}
			s.currentTokens = append([]int(nil), tokens...)
			states[i] = s
		}

		special := func(name string) int { return e.Tokenizer.EncodeSpecial(name) }
		toolStart := special("<|tool_start|>")
		toolEnd := special("<|tool_end|>")
		toolOutputStart := special("<|tool_output_start|>")
		toolOutputEnd := special("<|tool_output_end|>")
		assistantEnd := special("<|assistant_end|>")
		bos := e.Tokenizer.BOSTokenID()

		rng := tensor.NewRNG(seed)

		generated := 0
		for {
			if maxTokens > 0 && generated >= maxTokens {
				return
			}
			allDone := true
			for _, s := range states {
				if !s.completed {
					allDone = false
					break
				}
			}
			if allDone {
				return
			}

			nextIDs := SampleNextToken(logitsTensor, rng, temperature, topK) // [numSamples]

			tokenColumn := make([]int, numSamples)
			tokenMask := make([]int, numSamples)
			for i, s := range states {
				isForced := len(s.forcedTokens) > 0
				if isForced {
					tokenMask[i] = 0
				} else {
					tokenMask[i] = 1
				}
				var next int
				if isForced {
					next = s.forcedTokens[0]
					s.forcedTokens = s.forcedTokens[1:]
				} else {
					next = int(nextIDs.Data[i])
				}
				tokenColumn[i] = next
				s.currentTokens = append(s.currentTokens, next)
				if next == assistantEnd || next == bos {
					s.completed = true
				}
			switch {
			case next == toolStart:
				s.inToolCall = true
				s.toolCallTokens = nil
			case next == toolEnd && s.inToolCall:
				s.inToolCall = false
				if len(s.toolCallTokens) > 0 {
					expr := e.Tokenizer.Decode(s.toolCallTokens)
					if result, ok := e.Tools.Execute(expr); ok {
						resultTokens := e.Tokenizer.Encode(result)
						s.forcedTokens = append(s.forcedTokens, toolOutputStart)
						s.forcedTokens = append(s.forcedTokens, resultTokens...)
						s.forcedTokens = append(s.forcedTokens, toolOutputEnd)
					}
				}
				s.toolCallTokens = nil
			case s.inToolCall:
				s.toolCallTokens = append(s.toolCallTokens, next)
			}
			}

			if !yield(tokenColumn, tokenMask) {
				return
			}
			generated++

			// Forward the next token column.
			nextIdx := tensor.NewInt32sWithData([]int{numSamples, 1}, toI32(tokenColumn))
			logits = e.Model.Forward(nextIdx, decodeCache) // [B, 1, vocab]
			logitsTensor = logits.Reshape(numSamples, vocab)
		}
	}
}

// GenerateBatch runs non-streaming generation and returns the final token
// sequences (excluding terminal tokens) and their masks.
func (e *Engine) GenerateBatch(tokens []int, numSamples, maxTokens int, temperature float32, topK int, seed uint64) ([][]int, [][]int) {
	assistantEnd := e.Tokenizer.EncodeSpecial("<|assistant_end|>")
	bos := e.Tokenizer.BOSTokenID()

	results := make([][]int, numSamples)
	masks := make([][]int, numSamples)
	for i := 0; i < numSamples; i++ {
		results[i] = append([]int(nil), tokens...)
		masks[i] = make([]int, len(tokens))
	}
	completed := make([]bool, numSamples)

	gen := e.Generate(tokens, numSamples, maxTokens, temperature, topK, seed)
	gen(func(column, mask []int) bool {
		for i := range column {
			if completed[i] {
				continue
			}
			if column[i] == assistantEnd || column[i] == bos {
				completed[i] = true
			} else {
				results[i] = append(results[i], column[i])
				masks[i] = append(masks[i], mask[i])
			}
		}
		for _, c := range completed {
			if !c {
				return true
			}
		}
		return false
	})
	return results, masks
}

func toI32(ids []int) []int32 {
	out := make([]int32, len(ids))
	for i, v := range ids {
		out[i] = int32(v)
	}
	return out
}
