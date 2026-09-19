package tensors

import (
	"math"

	"github.com/cookiengineer/gonano/internal/parallel"
)

// CrossEntropyPerPosition computes the per-position cross-entropy loss between
// logits (shape [rowCount, vocabularySize]) and targets (shape [rowCount]).
// Positions where target == ignoreIndex are skipped and receive a loss of zero.
// It returns the per-position losses and the number of valid positions.
//
// The computation uses the numerically stable log-sum-exp formulation:
// loss = logsumexp(row) - row[target].
func CrossEntropyPerPosition(logits *Tensor, targets *Int32s, ignoreIndex int32) ([]float32, int) {
	rowCount, vocabularySize := logits.Shape[0], logits.Shape[1]
	if targets.Numel() != rowCount {
		panic("tensors: CrossEntropy logits/targets row mismatch")
	}
	losses := make([]float32, rowCount)
	validCount := 0
	for _, target := range targets.Data {
		if target != ignoreIndex {
			validCount++
		}
	}

	backend := KernelBackend()
	parallel.Default().Chunks(0, rowCount, func(rowStart, rowEnd int) {
		for row := rowStart; row < rowEnd; row++ {
			target := targets.Data[row]
			if target == ignoreIndex {
				continue
			}
			rowLogits := logits.Data[row*vocabularySize : (row+1)*vocabularySize]
			maximum := backend.Max(rowLogits)
			var sum float64
			for _, value := range rowLogits {
				sum += math.Exp(float64(value - maximum))
			}
			losses[row] = maximum + float32(math.Log(sum)) - rowLogits[int(target)]
		}
	})
	return losses, validCount
}

// CrossEntropy returns the mean cross-entropy loss over valid positions. It
// returns zero when there are no valid positions.
func CrossEntropy(logits *Tensor, targets *Int32s, ignoreIndex int32) float32 {
	losses, validCount := CrossEntropyPerPosition(logits, targets, ignoreIndex)
	if validCount == 0 {
		return 0
	}
	var sum float64
	for _, loss := range losses {
		sum += float64(loss)
	}
	return float32(sum / float64(validCount))
}

// CrossEntropySum returns the summed cross-entropy loss over valid positions.
func CrossEntropySum(logits *Tensor, targets *Int32s, ignoreIndex int32) float32 {
	losses, _ := CrossEntropyPerPosition(logits, targets, ignoreIndex)
	var sum float64
	for _, loss := range losses {
		sum += float64(loss)
	}
	return float32(sum)
}
