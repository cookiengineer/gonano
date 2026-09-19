package simdbackend

import (
	"simd"

	"github.com/cookiengineer/gonano/internal/parallel"
)

// columnBlockSize controls how many output columns each worker computes when
// MatMulTransposed partitions by column (small row counts).
const columnBlockSize = 64

// MatMul computes destination = left @ right, where left is
// [rowCount, innerCount] and right is [innerCount, columnCount]. It uses rank-1
// updates vectorized over columns and parallelized over row chunks.
func (backend *Backend) MatMul(destination, left, right []float32, rowCount, columnCount, innerCount int) {
	matMulCore(destination, left, right, rowCount, columnCount, innerCount)
}

// matMulCore holds the vectorized body of MatMul, for the same compiler reason
// as addCore in elementwise.go.
func matMulCore(destination, left, right []float32, rowCount, columnCount, innerCount int) {
	parallel.KernelPool().Chunks(0, rowCount, func(rowStart, rowEnd int) {
		for row := rowStart; row < rowEnd; row++ {
			clear(destination[row*columnCount : (row+1)*columnCount])
		}
		for inner := 0; inner < innerCount; inner++ {
			rightRow := right[inner*columnCount:]
			for row := rowStart; row < rowEnd; row++ {
				element := left[row*innerCount+inner]
				elementVector := simd.BroadcastFloat32s(element)
				destinationRow := destination[row*columnCount:]
				column := 0
				for ; column+float32LaneCount <= columnCount; column += float32LaneCount {
					accumulator := simd.LoadFloat32s(destinationRow[column:])
					rightVector := simd.LoadFloat32s(rightRow[column:])
					rightVector.MulAdd(elementVector, accumulator).Store(destinationRow[column:])
				}
				for ; column < columnCount; column++ {
					destinationRow[column] += element * rightRow[column]
				}
			}
		}
	})
}

// MatMulTransposed computes destination = left @ right^T, where left is
// [rowCount, innerCount] and right is [columnCount, innerCount]. It is
// vectorized over the inner dimension.
//
// Parallelization adapts to the shape. With enough rows (the training/prefill
// case) it partitions rows, so each row reads the full weight matrix once and
// the left operand is streamed once. With few rows (small decode batches) it
// partitions output columns instead, so batched decode still fans out across
// cores instead of leaving the whole matmul on one.
func (backend *Backend) MatMulTransposed(destination, left, right []float32, rowCount, columnCount, innerCount int) {
	pool := parallel.KernelPool()
	if rowCount >= pool.Workers() {
		pool.Chunks(0, rowCount, func(rowStart, rowEnd int) {
			for row := rowStart; row < rowEnd; row++ {
				leftRow := left[row*innerCount:]
				destinationRow := destination[row*columnCount:]
				for column := 0; column < columnCount; column++ {
					destinationRow[column] = dotProduct(leftRow, right[column*innerCount:], innerCount)
				}
			}
		})
		return
	}
	numberColumnBlocks := (columnCount + columnBlockSize - 1) / columnBlockSize
	pool.For(0, numberColumnBlocks, func(blockIndex int) {
		columnStart := blockIndex * columnBlockSize
		columnEnd := min(columnStart+columnBlockSize, columnCount)
		for row := 0; row < rowCount; row++ {
			leftRow := left[row*innerCount:]
			destinationRow := destination[row*columnCount:]
			for column := columnStart; column < columnEnd; column++ {
				destinationRow[column] = dotProduct(leftRow, right[column*innerCount:], innerCount)
			}
		}
	})
}

// DotProduct returns the sum of left[i]*right[i] over the shorter input.
func (backend *Backend) DotProduct(left, right []float32) float32 {
	return dotProduct(left, right, min(len(left), len(right)))
}

// dotProduct computes the dot product of two vectors over length elements,
// vectorized with a single horizontal fold at the end.
func dotProduct(left, right []float32, length int) float32 {
	accumulator := simd.BroadcastFloat32s(0)
	index := 0
	for ; index+float32LaneCount <= length; index += float32LaneCount {
		leftVector := simd.LoadFloat32s(left[index:])
		rightVector := simd.LoadFloat32s(right[index:])
		// MulAdd(x, y, z) = x*y + z, so this accumulates left*right.
		accumulator = leftVector.MulAdd(rightVector, accumulator)
	}
	total := horizontalSum(accumulator)
	for ; index < length; index++ {
		total += left[index] * right[index]
	}
	return total
}
