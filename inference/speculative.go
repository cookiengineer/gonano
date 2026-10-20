package inference

import (
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
)

// defaultDraftLength is the number of tokens the drafter proposes per round.
const defaultDraftLength = 5

// defaultConfidenceThreshold is the minimum confidence for a drafted token to
// be included in the verification block.
const defaultConfidenceThreshold = 0.5

// scheduledLength returns how many candidates to verify: the first candidate
// plus the leading drafts while the estimated prefix survival probability (the
// product of the per-position conditional acceptance confidences) stays at or
// above the threshold, at least one. Combining the confidences multiplicatively
// follows the paper's prefix-survival estimate for the DSpark scheduler
// (DeepSeek-V4.1 §2.4.3), rather than testing each confidence in isolation.
func scheduledLength(confidences []float32, threshold float32) int {
	limit := 1
	survival := float32(1)
	for _, confidence := range confidences {
		survival *= confidence
		if survival < threshold {
			break
		}
		limit++
	}
	if limit < 1 {
		limit = 1
	}
	return limit
}

// speculativeEligible reports whether a request can use exact greedy
// speculative decoding. Speculation is limited to single-row, greedy requests
// and must be enabled explicitly, because it does not run the tool-call state
// machine. Compressed and MLA targets are supported through cache rollback.
func (engine *Engine) speculativeEligible(numSamples int, temperature float32) bool {
	return engine.Speculative && (engine.DFlash != nil || engine.DSpark != nil || engine.Drafter != nil) &&
		numSamples == 1 && temperature <= 0
}

// speculativeGenerate runs exact greedy speculative decoding for one row. It
// produces exactly the same tokens as greedy decoding with the target model,
// but verifies up to DraftLength drafted tokens per target-model forward.
func (engine *Engine) speculativeGenerate(tokens []int, maxTokens int, yield func([]int, []int) bool) {
	config := engine.Model.Config
	cache := newCacheForModel(config, 1, config.SequenceLen)
	defer cache.Release()
	engine.speculativeGenerateWithCache(tokens, maxTokens, cache, yield)
}

// speculativeGenerateWithCache runs the speculative loop against an existing
// empty cache. The scheduler's solo path uses it so the request stays inside the
// shared block budget.
func (engine *Engine) speculativeGenerateWithCache(tokens []int, maxTokens int, cache *model.KVBuffer, yield func([]int, []int) bool) {
	config := engine.Model.Config
	vocabSize := config.VocabSize
	if len(tokens) == 0 {
		return
	}
	draftLength := engine.DraftLength
	if draftLength < 2 {
		draftLength = defaultDraftLength
	}

	prefill := tensors.NewInt32sWithData([]int{1, len(tokens)}, toI32(tokens))
	var logits *tensors.Tensor
	var lastHidden *tensors.Tensor
	if engine.DFlash != nil {
		var hidden *tensors.Tensor
		logits, hidden = engine.Model.ForwardHiddenAtLayers(prefill, cache, engine.DFlash.TargetIDs)
		lastHidden = lastHiddenRow(hidden)
	} else {
		logits = engine.Model.Forward(prefill, cache)
	}
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
		candidates := engine.draftCandidates(sequence, first, lastHidden, draftCount)
		blockSize = len(candidates)

		basePosition := cache.Position()
		mark := cache.Mark()
		candidateIDs := tensors.NewInt32sWithData([]int{1, blockSize}, toI32(candidates))
		var verified *tensors.Tensor
		var verifiedHidden *tensors.Tensor
		if engine.DFlash != nil {
			verified, verifiedHidden = engine.Model.ForwardHiddenAtLayers(candidateIDs, cache, engine.DFlash.TargetIDs)
		} else {
			verified = engine.Model.Forward(candidateIDs, cache)
		}

		// Accept the longest prefix of drafted tokens that matches the target
		// model's greedy tokens. The first candidate came from the target model
		// and is always accepted.
		matched := 0
		for matched < blockSize-1 && argmaxRow(verified, matched) == candidates[matched+1] {
			matched++
		}
		accepted := 1 + matched
		if accepted < blockSize {
			// Rejected drafts may have completed compression blocks or advanced
			// the MLA latent state, so restore the pre-verification snapshot and
			// replay only the accepted prefix. This keeps the cache exactly
			// consistent with a non-speculative decode.
			cache.Rollback(mark)
			acceptedIDs := tensors.NewInt32sWithData([]int{1, accepted}, toI32(candidates[:accepted]))
			var replayHidden *tensors.Tensor
			if engine.DFlash != nil {
				_, replayHidden = engine.Model.ForwardHiddenAtLayers(acceptedIDs, cache, engine.DFlash.TargetIDs)
				lastHidden = lastHiddenRow(replayHidden)
			} else {
				engine.Model.Forward(acceptedIDs, cache)
			}
		} else if engine.DFlash != nil && verifiedHidden != nil {
			lastHidden = hiddenRow(verifiedHidden, accepted-1)
		}
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

// draftCandidates builds the verification block for one row: the target's own
// next token followed by up to draftCount drafted tokens. Confidence scheduling
// may shorten the block for DFlash and DSpark.
func (engine *Engine) draftCandidates(context []int, first int, lastHidden *tensors.Tensor, draftCount int) []int {
	candidates := make([]int, 0, draftCount+1)
	candidates = append(candidates, first)
	if draftCount <= 0 {
		return candidates
	}
	draftContext := append(append([]int(nil), context...), first)
	threshold := engine.ConfidenceThreshold
	if threshold <= 0 {
		threshold = defaultConfidenceThreshold
	}
	var confidences []float32
	switch {
	case engine.DFlash != nil:
		drafted, draftedConfidences := engine.DFlash.Draft(draftContext, lastHidden, draftCount)
		candidates = append(candidates, drafted...)
		confidences = draftedConfidences
	case engine.DSpark != nil:
		drafted, draftedConfidences := engine.DSpark.Draft(draftContext, draftCount)
		candidates = append(candidates, drafted...)
		confidences = draftedConfidences
	default:
		candidates = append(candidates, engine.Drafter.DraftTokens(draftContext, draftCount)...)
		return candidates
	}
	if limit := scheduledLength(confidences, threshold); limit < len(candidates) {
		candidates = candidates[:limit]
	}
	return candidates
}

// lastHiddenRow returns the last sequence row of a [1, T, D] hidden tensor.
func lastHiddenRow(hidden *tensors.Tensor) *tensors.Tensor {
	if hidden == nil {
		return nil
	}
	dim := hidden.Shape[len(hidden.Shape)-1]
	rows := hidden.Numel() / dim
	return hiddenRow(hidden, rows-1)
}

// hiddenRow returns row `row` of a [1, T, D] hidden tensor as [1, 1, D].
func hiddenRow(hidden *tensors.Tensor, row int) *tensors.Tensor {
	if hidden == nil {
		return nil
	}
	dim := hidden.Shape[len(hidden.Shape)-1]
	rows := hidden.Numel() / dim
	if row < 0 {
		row = 0
	}
	if row >= rows {
		row = rows - 1
	}
	output := tensors.New(1, 1, dim)
	copy(output.Data, hidden.Data[row*dim:(row+1)*dim])
	return output
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
