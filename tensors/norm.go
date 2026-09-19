package tensors

// SoftmaxLastDim applies the numerically stable softmax over the last (column)
// dimension of a rank-2 tensor. Rows are independent.
func SoftmaxLastDim(source *Tensor) *Tensor {
	rowCount, columnCount := source.Shape[0], source.Shape[1]
	result := New(rowCount, columnCount)
	KernelBackend().SoftmaxLastDim(result.Data, source.Data, rowCount, columnCount)
	return result
}

// Softmax is an alias of SoftmaxLastDim for a rank-2 tensor.
func Softmax(source *Tensor) *Tensor { return SoftmaxLastDim(source) }

// RMSNormLastDim applies RMSNorm (normalize each row to unit root-mean-square,
// without affine parameters) over the last dimension of a rank-2 tensor. This
// matches nanochat's stateless norm(x) = rms_norm(x, -1).
func RMSNormLastDim(source *Tensor, epsilon float32) *Tensor {
	rowCount, columnCount := source.Shape[0], source.Shape[1]
	result := New(rowCount, columnCount)
	KernelBackend().RMSNormLastDim(result.Data, source.Data, rowCount, columnCount, epsilon)
	return result
}
