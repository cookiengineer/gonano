// Package infer provides the autoregressive inference engine: KV-cache-driven
// batched generation, token sampling, and the calculator tool-use state
// machine.
package infer

import (
	"math"
	"sort"

	"github.com/cookiengineer/gonano/tensor"
)

// SampleNextToken samples one next token per row from the given logits of
// shape [B, vocab]. temperature <= 0 selects the argmax deterministically.
// topK > 0 restricts sampling to the k highest logits.
func SampleNextToken(logits *tensor.Tensor, rng *tensor.RNG, temperature float32, topK int) *tensor.Int32s {
	b, vocab := logits.Shape[0], logits.Shape[1]
	out := tensor.NewInt32s(b)
	for i := 0; i < b; i++ {
		row := logits.Data[i*vocab : (i+1)*vocab]
		out.Data[i] = int32(sampleRow(row, rng, temperature, topK))
	}
	return out
}

func sampleRow(logits []float32, rng *tensor.RNG, temperature float32, topK int) int {
	if temperature <= 0 {
		return argMax(logits)
	}
	row := logits
	if topK > 0 && topK < len(logits) {
		row = maskTopK(logits, topK)
	}
	return tensor.SampleMultinomial(rng, row, temperature)
}

func argMax(s []float32) int {
	best := 0
	for i := 1; i < len(s); i++ {
		if s[i] > s[best] {
			best = i
		}
	}
	return best
}

// maskTopK returns a copy of logits with all but the top-k entries set to
// -inf (so they cannot be sampled).
func maskTopK(logits []float32, k int) []float32 {
	threshold := kthLargest(logits, k)
	out := make([]float32, len(logits))
	for i, v := range logits {
		if v >= threshold {
			out[i] = v
		} else {
			out[i] = float32(math.Inf(-1))
		}
	}
	return out
}

// kthLargest returns the k-th largest value (1-indexed).
func kthLargest(vals []float32, k int) float32 {
	top := make([]float32, k)
	copy(top, vals[:k])
	sort.Slice(top, func(i, j int) bool { return top[i] < top[j] }) // ascending
	for _, v := range vals[k:] {
		if v > top[0] {
			top[0] = v
			// bubble the new minimum into place
			for i := 0; i+1 < k && top[i] > top[i+1]; i++ {
				top[i], top[i+1] = top[i+1], top[i]
			}
		}
	}
	return top[0]
}
