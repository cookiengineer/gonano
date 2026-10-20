// Package inference provides the autoregressive inference engine: KV-cache-driven
// batched generation, token sampling, and the calculator tool-use state
// machine.
package inference

import (
	"math"
	"sort"

	"github.com/cookiengineer/gonano/tensors"
)

// SamplingOptions controls token filtering.
type SamplingOptions struct {
	Temperature float32
	TopK        int
	TopP        float32
	MinP        float32
	// Allowed masks token ids for one row. Nil disables the mask.
	Allowed []bool
	// AllowedRows provides a per-row mask; it takes precedence over Allowed.
	AllowedRows [][]bool
}

// SampleNextToken samples one next token per row from the given logits of
// shape [B, vocab]. temperature <= 0 selects the argmax deterministically.
// topK > 0 restricts sampling to the k highest logits.
func SampleNextToken(logits *tensors.Tensor, randomGenerator *tensors.RNG, temperature float32, topK int) *tensors.Int32s {
	return SampleNextTokenWithOptions(logits, randomGenerator, SamplingOptions{Temperature: temperature, TopK: topK})
}

// SampleNextTokenWithOptions applies temperature, top-k, top-p and min-p
// filtering before sampling one token per row.
func SampleNextTokenWithOptions(logits *tensors.Tensor, randomGenerator *tensors.RNG, options SamplingOptions) *tensors.Int32s {
	batchSize, vocab := logits.Shape[0], logits.Shape[1]
	out := tensors.NewInt32s(batchSize)
	for index := 0; index < batchSize; index++ {
		row := logits.Data[index*vocab : (index+1)*vocab]
		rowOptions := options
		if options.AllowedRows != nil && index < len(options.AllowedRows) {
			rowOptions.Allowed = options.AllowedRows[index]
			rowOptions.AllowedRows = nil
		}
		out.Data[index] = int32(sampleRowWithOptions(row, randomGenerator, rowOptions))
	}
	return out
}

func sampleRow(logits []float32, randomGenerator *tensors.RNG, temperature float32, topK int) int {
	return sampleRowWithOptions(logits, randomGenerator, SamplingOptions{Temperature: temperature, TopK: topK})
}

func sampleRowWithOptions(logits []float32, randomGenerator *tensors.RNG, options SamplingOptions) int {
	if options.Allowed != nil {
		masked := make([]float32, len(logits))
		for index, value := range logits {
			if index < len(options.Allowed) && options.Allowed[index] {
				masked[index] = value
			} else {
				masked[index] = float32(math.Inf(-1))
			}
		}
		logits = masked
	}
	if options.Temperature <= 0 {
		return argMax(logits)
	}
	if options.TopP <= 0 || options.TopP >= 1 {
		if options.MinP <= 0 {
			row := logits
			if options.TopK > 0 && options.TopK < len(logits) {
				row = maskTopK(logits, options.TopK)
			}
			return tensors.SampleMultinomial(randomGenerator, row, options.Temperature)
		}
	}
	// Compute the temperature-scaled distribution once, then apply the
	// probability filters and sample from the log-probabilities with
	// temperature already folded in.
	scaled := make([]float32, len(logits))
	maximum := float32(math.Inf(-1))
	for index, value := range logits {
		scaled[index] = value / options.Temperature
		if scaled[index] > maximum {
			maximum = scaled[index]
		}
	}
	var total float64
	for index := range scaled {
		scaled[index] = float32(math.Exp(float64(scaled[index] - maximum)))
		total += float64(scaled[index])
	}
	if total <= 0 {
		return argMax(logits)
	}
	probabilities := make([]float32, len(scaled))
	for index := range scaled {
		probabilities[index] = float32(float64(scaled[index]) / total)
	}
	keep := make([]bool, len(logits))
	for index := range keep {
		keep[index] = true
	}
	if options.TopK > 0 && options.TopK < len(logits) {
		threshold := kthLargest(probabilities, options.TopK)
		for index := range probabilities {
			if probabilities[index] < threshold {
				keep[index] = false
			}
		}
	}
	if options.MinP > 0 {
		maxProbability := probabilities[argMax(probabilities)]
		threshold := options.MinP * maxProbability
		for index, probability := range probabilities {
			if probability < threshold {
				keep[index] = false
			}
		}
	}
	if options.TopP > 0 && options.TopP < 1 {
		order := make([]int, 0, len(probabilities))
		for index := range probabilities {
			if keep[index] {
				order = append(order, index)
			}
		}
		sort.SliceStable(order, func(left, right int) bool { return probabilities[order[left]] > probabilities[order[right]] })
		var cumulative float64
		cutoff := len(order)
		for position, index := range order {
			cumulative += float64(probabilities[index])
			if cumulative >= float64(options.TopP) {
				cutoff = position + 1
				break
			}
		}
		if cutoff < len(order) {
			for position := cutoff; position < len(order); position++ {
				keep[order[position]] = false
			}
		}
	}
	filtered := make([]float32, len(logits))
	kept := 0
	for index := range filtered {
		if keep[index] {
			filtered[index] = float32(math.Log(float64(probabilities[index]) + 1e-30))
			kept++
		} else {
			filtered[index] = float32(math.Inf(-1))
		}
	}
	if kept == 0 {
		return argMax(logits)
	}
	return tensors.SampleMultinomial(randomGenerator, filtered, 1.0)
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
