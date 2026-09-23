package simdbackend

import (
	"math"

	"simd"

	"github.com/cookiengineer/gonano/internal/parallel"
)

// TopKIndices fills destination[row*k+slot] with the index of the slot-th
// largest score of row `row`, descending, ties toward the smaller index. The
// maximum is found with a vectorized reduction and its first index with a
// vectorized equality mask, so the selection is O(k * count / lanes) rather
// than a scalar scan. See kernels.MoE.
func (backend *Backend) TopKIndices(destination []int32, scores []float32, rowCount, count, k int) {
	if count == 0 || k <= 0 {
		return
	}
	limit := k
	if limit > count {
		limit = count
	}
	parallel.KernelPool().Chunks(0, rowCount, func(rowStart, rowEnd int) {
		working := make([]float32, count)
		for row := rowStart; row < rowEnd; row++ {
			copy(working, scores[row*count:(row+1)*count])
			for slot := 0; slot < limit; slot++ {
				index := maxIndexFloat32s(working)
				destination[row*k+slot] = int32(index)
				working[index] = float32(math.Inf(-1))
			}
		}
	})
}

// maxIndexFloat32s returns the index of the first largest value, or -1 for an
// empty slice. The first pass computes the maximum with a vectorized reduction;
// the second finds the earliest lane equal to it, preserving the
// larger-score/smaller-index tie-break.
func maxIndexFloat32s(values []float32) int {
	length := len(values)
	if length == 0 {
		return -1
	}
	maximumVector := simd.BroadcastFloat32s(float32(math.Inf(-1)))
	index := 0
	for ; index+float32LaneCount <= length; index += float32LaneCount {
		maximumVector = maximumVector.Max(simd.LoadFloat32s(values[index:]))
	}
	maximum := horizontalMax(maximumVector)
	for ; index < length; index++ {
		if values[index] > maximum {
			maximum = values[index]
		}
	}
	index = 0
	for ; index+float32LaneCount <= length; index += float32LaneCount {
		mask := simd.LoadFloat32s(values[index:]).Equal(simd.BroadcastFloat32s(maximum)).ToInt32s()
		var buffer [maximumFloat32Lanes]int32
		mask.Store(buffer[:float32LaneCount])
		for lane := 0; lane < float32LaneCount; lane++ {
			if buffer[lane] != 0 {
				return index + lane
			}
		}
	}
	for ; index < length; index++ {
		if values[index] == maximum {
			return index
		}
	}
	return length - 1
}

// MoEGateTopK computes the router softmax (vectorized and parallelized across
// rows) and the vectorized top-k expert selection for every token row. See
// kernels.MoE.
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
// expert, parallelized across the gathered rows. See kernels.MoE.
func (backend *Backend) GroupedMatMulTransposed(destination, input, weight []float32, tokenCounts []int32, expertCount, columnCount, innerCount int) {
	totalRows := 0
	for expert := 0; expert < expertCount; expert++ {
		totalRows += int(tokenCounts[expert])
	}
	if totalRows == 0 {
		return
	}
	// Map every gathered row to its expert so the parallel row loop can find the
	// packed weight slice without re-walking the counts.
	rowExpert := make([]int, totalRows)
	rowOffset := 0
	for expert := 0; expert < expertCount; expert++ {
		count := int(tokenCounts[expert])
		for row := 0; row < count; row++ {
			rowExpert[rowOffset+row] = expert
		}
		rowOffset += count
	}
	parallel.KernelPool().Chunks(0, totalRows, func(rowStart, rowEnd int) {
		for row := rowStart; row < rowEnd; row++ {
			inputRow := input[row*innerCount:]
			destinationRow := destination[row*columnCount:]
			weightBase := rowExpert[row] * columnCount * innerCount
			for column := 0; column < columnCount; column++ {
				destinationRow[column] = dotProduct(inputRow, weight[weightBase+column*innerCount:], innerCount)
			}
		}
	})
}
