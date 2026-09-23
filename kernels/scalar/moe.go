package scalar

import "math"

// TopKIndices fills destination[row*k+slot] with the index of the slot-th
// largest score of row `row`, descending, ties toward the smaller index. See
// kernels.MoE.
func (backend *Backend) TopKIndices(destination []int32, scores []float32, rowCount, count, k int) {
	if count == 0 || k <= 0 {
		return
	}
	limit := k
	if limit > count {
		limit = count
	}
	working := make([]float32, count)
	for row := 0; row < rowCount; row++ {
		copy(working, scores[row*count:(row+1)*count])
		for slot := 0; slot < limit; slot++ {
			best := 0
			for expert := 1; expert < count; expert++ {
				if working[expert] > working[best] {
					best = expert
				}
			}
			destination[row*k+slot] = int32(best)
			working[best] = float32(math.Inf(-1))
		}
	}
}

// MoEGateTopK computes the router softmax and the top-k expert selection for
// every token row. See kernels.MoE.
func (backend *Backend) MoEGateTopK(logits, bias, probs []float32, selected []int32, selectedWeights []float32, rowCount, expertCount, k int) {
	backend.SoftmaxLastDim(probs, logits, rowCount, expertCount)
	scores := make([]float32, rowCount*expertCount)
	for index := range scores {
		scores[index] = logits[index] + bias[index%expertCount]
	}
	backend.TopKIndices(selected, scores, rowCount, expertCount, k)
	limit := k
	if limit > expertCount {
		limit = expertCount
	}
	for row := 0; row < rowCount; row++ {
		for slot := 0; slot < limit; slot++ {
			selectedWeights[row*k+slot] = probs[row*expertCount+int(selected[row*k+slot])]
		}
	}
}

// GroupedMatMulTransposed computes one matmul per expert over rows grouped by
// expert. See kernels.MoE.
func (backend *Backend) GroupedMatMulTransposed(destination, input, weight []float32, tokenCounts []int32, expertCount, columnCount, innerCount int) {
	rowOffset := 0
	for expert := 0; expert < expertCount; expert++ {
		count := int(tokenCounts[expert])
		weightBase := expert * columnCount * innerCount
		for row := 0; row < count; row++ {
			inputRow := input[(rowOffset+row)*innerCount:]
			destinationRow := destination[(rowOffset+row)*columnCount:]
			for column := 0; column < columnCount; column++ {
				weightRow := weight[weightBase+column*innerCount:]
				var total float32
				for inner := 0; inner < innerCount; inner++ {
					total += inputRow[inner] * weightRow[inner]
				}
				destinationRow[column] = total
			}
		}
		rowOffset += count
	}
}
