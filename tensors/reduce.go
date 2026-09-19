package tensors

import "github.com/cookiengineer/gonano/internal/parallel"

// SumAll returns the sum of every element.
func SumAll(source *Tensor) float32 { return KernelBackend().Sum(source.Data) }

// MaxAll returns the largest element.
func MaxAll(source *Tensor) float32 { return KernelBackend().Max(source.Data) }

// ArgMax returns the index of the largest element of a rank-1 tensor.
func ArgMax(source *Tensor) int { return KernelBackend().ArgMax(source.Data) }

// SumLastDim reduces a rank-2 tensor over its last (column) dimension,
// returning a [rowCount] tensor of row sums. Rows are processed in parallel.
func SumLastDim(source *Tensor) *Tensor {
	rowCount, columnCount := source.Shape[0], source.Shape[1]
	result := New(rowCount)
	backend := KernelBackend()
	parallel.Default().Chunks(0, rowCount, func(rowStart, rowEnd int) {
		for row := rowStart; row < rowEnd; row++ {
			result.Data[row] = backend.Sum(source.Data[row*columnCount : (row+1)*columnCount])
		}
	})
	return result
}

// MaxLastDim reduces a rank-2 tensor over its last dimension, returning a
// [rowCount] tensor of row maxima. Rows are processed in parallel.
func MaxLastDim(source *Tensor) *Tensor {
	rowCount, columnCount := source.Shape[0], source.Shape[1]
	result := New(rowCount)
	backend := KernelBackend()
	parallel.Default().Chunks(0, rowCount, func(rowStart, rowEnd int) {
		for row := rowStart; row < rowEnd; row++ {
			result.Data[row] = backend.Max(source.Data[row*columnCount : (row+1)*columnCount])
		}
	})
	return result
}

// ArgMaxLastDim returns the column index of the maximum in each row of a rank-2
// tensor, as an int32 tensor of shape [rowCount].
func ArgMaxLastDim(source *Tensor) *Int32s {
	rowCount, columnCount := source.Shape[0], source.Shape[1]
	result := NewInt32s(rowCount)
	backend := KernelBackend()
	parallel.Default().Chunks(0, rowCount, func(rowStart, rowEnd int) {
		for row := rowStart; row < rowEnd; row++ {
			result.Data[row] = int32(backend.ArgMax(source.Data[row*columnCount : (row+1)*columnCount]))
		}
	})
	return result
}
