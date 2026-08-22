package tensor

import "math"

// CrossEntropyPerPosition computes the per-position cross-entropy loss between
// logits (shape [rows, vocab]) and targets (shape [rows]). Positions where
// target == ignore are skipped and receive a loss of zero. It returns the
// per-position losses and the number of valid (non-ignored) positions.
//
// The computation uses the numerically stable log-sum-exp formulation:
// loss = logsumexp(row) - row[target].
func CrossEntropyPerPosition(logits *Tensor, targets *Int32s, ignore int32) ([]float32, int) {
	rows, vocab := logits.Shape[0], logits.Shape[1]
	if targets.Numel() != rows {
		panic("tensor: CrossEntropy logits/targets row mismatch")
	}
	losses := make([]float32, rows)
	valid := 0
	for _, y := range targets.Data {
		if y != ignore {
			valid++
		}
	}
	ld := logits.Data
	td := targets.Data
	parallelChunks(rows, func(r0, r1 int) {
		for i := r0; i < r1; i++ {
			y := td[i]
			if y == ignore {
				continue
			}
			row := ld[i*vocab : (i+1)*vocab]
			m := sliceMax(row)
			var sum float64
			for _, v := range row {
				sum += math.Exp(float64(v - m))
			}
			losses[i] = m + float32(math.Log(sum)) - row[int(y)]
		}
	})
	return losses, valid
}

// CrossEntropy returns the mean cross-entropy loss over valid positions.
// It returns 0 when there are no valid positions.
func CrossEntropy(logits *Tensor, targets *Int32s, ignore int32) float32 {
	losses, valid := CrossEntropyPerPosition(logits, targets, ignore)
	if valid == 0 {
		return 0
	}
	var sum float64
	for _, l := range losses {
		sum += float64(l)
	}
	return float32(sum / float64(valid))
}

// CrossEntropySum returns the summed cross-entropy loss over valid positions.
func CrossEntropySum(logits *Tensor, targets *Int32s, ignore int32) float32 {
	losses, _ := CrossEntropyPerPosition(logits, targets, ignore)
	var sum float64
	for _, l := range losses {
		sum += float64(l)
	}
	return float32(sum)
}
