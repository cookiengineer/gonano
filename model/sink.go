package model

import (
	"math"

	"github.com/cookiengineer/gonano/tensors"
)

// logAddExp returns log(exp(a)+exp(b)) without overflow.
func logAddExp(a, b float32) float32 {
	if a == float32(math.Inf(-1)) {
		return b
	}
	if b == float32(math.Inf(-1)) {
		return a
	}
	maximum := a
	if b > maximum {
		maximum = b
	}
	return maximum + float32(math.Log(float64(math.Exp(float64(a-maximum))+math.Exp(float64(b-maximum)))))
}

// applyAttentionSink folds the per-head sink logit into the attention
// denominator. output is [batch, head, sequence, headDim]; logSumExp is
// [batch, head, sequence]. The numerator is unchanged, so the output rows are
// scaled by exp(lse - lseWithSink).
func applyAttentionSink(output, logSumExp, sink *tensors.Tensor, batchSize, headCount, sequenceLength, headDimension int) {
	if sink == nil {
		return
	}
	for batch := 0; batch < batchSize; batch++ {
		for head := 0; head < headCount; head++ {
			sinkLogit := sink.Data[head]
			rowBase := (batch*headCount + head) * sequenceLength
			for position := 0; position < sequenceLength; position++ {
				index := rowBase + position
				lse := logSumExp.Data[index]
				withSink := logAddExp(lse, sinkLogit)
				scale := float32(0)
				if withSink != float32(math.Inf(-1)) {
					scale = float32(math.Exp(float64(lse - withSink)))
				}
				base := index * headDimension
				for dimension := 0; dimension < headDimension; dimension++ {
					output.Data[base+dimension] *= scale
				}
				logSumExp.Data[index] = withSink
			}
		}
	}
}

// accumulateAttentionSinkGradient adds the sink gradient for every query row.
// rowCorrection holds dO . O per row, so dL/dsink =
// -exp(sink - lseWithSink) * (dO . O).
func accumulateAttentionSinkGradient(sink *tensors.Tensor, rowCorrection, logSumExp *tensors.Tensor, batchSize, headCount, sequenceLength int) {
	if sink == nil || rowCorrection == nil {
		return
	}
	sink.EnsureGrad()
	for batch := 0; batch < batchSize; batch++ {
		for head := 0; head < headCount; head++ {
			sinkLogit := sink.Data[head]
			rowBase := (batch*headCount + head) * sequenceLength
			for position := 0; position < sequenceLength; position++ {
				index := rowBase + position
				lse := logSumExp.Data[index]
				if lse == float32(math.Inf(-1)) {
					continue
				}
				probability := float32(math.Exp(float64(sinkLogit - lse)))
				sink.Grad[head] += -probability * rowCorrection.Data[index]
			}
		}
	}
}
