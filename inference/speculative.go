package inference

import (
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
)

// defaultDraftLength is the number of tokens the drafter proposes per round.
const defaultDraftLength = 5

// speculativeEligible reports whether a request can use exact greedy
// speculative decoding. Speculation is limited to single-row, greedy,
// uncompressed models and must be enabled explicitly, because it does not run
// the tool-call state machine.
func (engine *Engine) speculativeEligible(numSamples int, temperature float32) bool {
	return engine.Speculative && engine.Drafter != nil &&
		engine.Model.Config.Compression() == 1 &&
		numSamples == 1 && temperature <= 0
}

// speculativeGenerate runs exact greedy speculative decoding for one row. It
// produces exactly the same tokens as greedy decoding with the target model,
// but verifies up to DraftLength drafted tokens per target-model forward.
func (engine *Engine) speculativeGenerate(tokens []int, maxTokens int, yield func([]int, []int) bool) {
	config := engine.Model.Config
	vocabSize := config.VocabSize
	if len(tokens) == 0 {
		return
	}
	draftLength := engine.DraftLength
	if draftLength < 2 {
		draftLength = defaultDraftLength
	}

	cache := model.NewKVBuffer(1, config.SequenceLen, config.NumLayer, config.NumKVHead, config.HeadDim())
	prefill := tensors.NewInt32sWithData([]int{1, len(tokens)}, toI32(tokens))
	logits := engine.Model.Forward(prefill, cache)
	lastLogits := logits.Data[(len(tokens)-1)*vocabSize : len(tokens)*vocabSize]

	assistantEnd := engine.Tokenizer.EncodeSpecial("<|assistant_end|>")
	bosToken := engine.Tokenizer.BOSTokenID()

	sequence := append([]int(nil), tokens...)
	emitted := 0
	pending := 0
	hasPending := false

	emit := func(token int) bool {
		if token == assistantEnd || token == bosToken {
			return false
		}
		if !yield([]int{token}, []int{1}) {
			return false
		}
		sequence = append(sequence, token)
		emitted++
		return emitted < maxTokens
	}

	for emitted < maxTokens {
		var first int
		if hasPending {
			first = pending
			hasPending = false
		} else {
			first = argmaxSlice(lastLogits)
		}

		// Cap the candidate block so it never exceeds the context window.
		blockSize := draftLength
		if room := config.SequenceLen - cache.Position(); blockSize > room {
			blockSize = room
		}
		if blockSize < 1 {
			return
		}
		draftCount := blockSize - 1
		candidates := make([]int, 0, blockSize)
		candidates = append(candidates, first)
		if draftCount > 0 {
			draftContext := append(append([]int(nil), sequence...), first)
			candidates = append(candidates, engine.Drafter.DraftTokens(draftContext, draftCount)...)
		}
		blockSize = len(candidates)

		basePosition := cache.Position()
		candidateIDs := tensors.NewInt32sWithData([]int{1, blockSize}, toI32(candidates))
		verified := engine.Model.Forward(candidateIDs, cache)

		// Accept the longest prefix of drafted tokens that matches the target
		// model's greedy tokens. The first candidate came from the target model
		// and is always accepted.
		matched := 0
		for matched < blockSize-1 && argmaxRow(verified, matched) == candidates[matched+1] {
			matched++
		}
		accepted := 1 + matched
		cache.SetPosition(basePosition + accepted)
		engine.Model.SetPreviousEmbeddingForToken(cache, candidates[accepted-1])

		// The candidate that follows the accepted prefix (the target's own
		// token on a mismatch, or the bonus token after a full match) is carried
		// as the next round's first candidate.
		var next int
		if matched < blockSize-1 {
			next = argmaxRow(verified, matched)
		} else {
			next = argmaxRow(verified, blockSize-1)
		}

		for index := 0; index < accepted; index++ {
			if !emit(candidates[index]) {
				return
			}
		}
		pending = next
		hasPending = true
	}
}

// argmaxRow returns the argmax of row `row` of a [1, rows, vocab] tensor.
func argmaxRow(logits *tensors.Tensor, row int) int {
	vocabSize := logits.Shape[len(logits.Shape)-1]
	base := row * vocabSize
	return argmaxSlice(logits.Data[base : base+vocabSize])
}

// argmaxSlice returns the index of the largest value in a slice.
func argmaxSlice(values []float32) int {
	best := 0
	for index := 1; index < len(values); index++ {
		if values[index] > values[best] {
			best = index
		}
	}
	return best
}
