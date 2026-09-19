// Package inference provides the autoregressive inference engine: KV-cache-driven
// batched generation, token sampling, and the calculator tool-use state
// machine.
package inference

import (
	"math"
	"sort"

	"github.com/cookiengineer/gonano/tensors"
)

// SampleNextToken samples one next token per row from the given logits of
// shape [B, vocab]. temperature <= 0 selects the argmax deterministically.
// topK > 0 restricts sampling to the k highest logits.
func SampleNextToken(logits *tensors.Tensor, randomGenerator *tensors.RNG, temperature float32, topK int) *tensors.Int32s {
	batchSize, vocab := logits.Shape[0], logits.Shape[1]
	out := tensors.NewInt32s(batchSize)
	for index := 0; index < batchSize; index++ {
		row := logits.Data[index*vocab : (index+1)*vocab]
		out.Data[index] = int32(sampleRow(row, randomGenerator, temperature, topK))
	}
	return out
}

func sampleRow(logits []float32, randomGenerator *tensors.RNG, temperature float32, topK int) int {
	if temperature <= 0 {
		return argMax(logits)
	}
	row := logits
	if topK > 0 && topK < len(logits) {
		row = maskTopK(logits, topK)
	}
	return tensors.SampleMultinomial(randomGenerator, row, temperature)
}

func argMax(values []float32) int {
	bestIndex := 0
	for index := 1; index < len(values); index++ {
		if values[index] > values[bestIndex] {
			bestIndex = index
		}
	}
	return bestIndex
}

// maskTopK returns a copy of logits with all but the top-k entries set to
// -inf (so they cannot be sampled).
func maskTopK(logits []float32, topK int) []float32 {
	threshold := kthLargest(logits, topK)
	masked := make([]float32, len(logits))
	for index, value := range logits {
		if value >= threshold {
			masked[index] = value
		} else {
			masked[index] = float32(math.Inf(-1))
		}
	}
	return masked
}

// kthLargest returns the k-th largest value (1-indexed).
func kthLargest(values []float32, rank int) float32 {
	top := make([]float32, rank)
	copy(top, values[:rank])
	sort.Slice(top, func(left, right int) bool { return top[left] < top[right] }) // ascending
	for _, value := range values[rank:] {
		if value > top[0] {
			top[0] = value
			// bubble the new minimum into place
			for index := 0; index+1 < rank && top[index] > top[index+1]; index++ {
				top[index], top[index+1] = top[index+1], top[index]
			}
		}
	}
	return top[0]
}
