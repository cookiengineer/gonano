package inference

import (
	"math"
	"sort"
)

// TokenAlternative is one top-logprob entry.
type TokenAlternative struct {
	Token   int
	Logprob float32
}

// TokenLogprob is the sampled token's log-probability and optional top-k
// alternatives.
type TokenLogprob struct {
	Token   int
	Logprob float32
	Top     []TokenAlternative
}

// computeTokenLogprob returns the log-probability of token under the
// temperature-scaled logits and the top-k alternatives. Temperature <= 0
// returns a neutral logprob because argmax has no probabilistic interpretation.
func computeTokenLogprob(logits []float32, token int, temperature float32, topK int) TokenLogprob {
	result := TokenLogprob{Token: token}
	if temperature <= 0 || token < 0 || token >= len(logits) {
		return result
	}
	scaled := make([]float32, len(logits))
	maximum := float32(math.Inf(-1))
	for index, value := range logits {
		scaled[index] = value / temperature
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
		return result
	}
	logDenominator := maximum + float32(math.Log(total))
	result.Logprob = scaled[token] - logDenominator
	if topK > 0 {
		if topK > len(scaled) {
			topK = len(scaled)
		}
		order := make([]int, len(scaled))
		for index := range order {
			order[index] = index
		}
		sort.SliceStable(order, func(left, right int) bool { return scaled[order[left]] > scaled[order[right]] })
		result.Top = make([]TokenAlternative, 0, topK)
		for index := 0; index < topK; index++ {
			id := order[index]
			result.Top = append(result.Top, TokenAlternative{
				Token:   id,
				Logprob: scaled[id] - logDenominator,
			})
		}
	}
	return result
}
