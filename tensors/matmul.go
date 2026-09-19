package tensors

// MatMul computes result = left @ right, where left is [rowCount, innerCount]
// and right is [innerCount, columnCount]. The result is [rowCount,
// columnCount]. It is used by the attention scores@values product.
func MatMul(left, right *Tensor) *Tensor {
	if left.Rank() != 2 || right.Rank() != 2 {
		panic("tensors: MatMul requires rank-2 operands")
	}
	rowCount, innerCount := left.Shape[0], left.Shape[1]
	if right.Shape[0] != innerCount {
		panic("tensors: MatMul inner dimension mismatch")
	}
	columnCount := right.Shape[1]
	result := New(rowCount, columnCount)
	KernelBackend().MatMul(result.Data, left.Data, right.Data, rowCount, columnCount, innerCount)
	return result
}

// MatMulTransposed computes result = left @ right^T, where left is
// [rowCount, innerCount] and right is [columnCount, innerCount] (a weight
// matrix stored as [out, in]). The result is [rowCount, columnCount]. This is
// the primitive used by every linear layer and by the query@key^T product.
func MatMulTransposed(left, right *Tensor) *Tensor {
	if left.Rank() != 2 || right.Rank() != 2 {
		panic("tensors: MatMulTransposed requires rank-2 operands")
	}
	rowCount, innerCount := left.Shape[0], left.Shape[1]
	if right.Shape[1] != innerCount {
		panic("tensors: MatMulTransposed inner dimension mismatch")
	}
	columnCount := right.Shape[0]
	result := New(rowCount, columnCount)
	KernelBackend().MatMulTransposed(result.Data, left.Data, right.Data, rowCount, columnCount, innerCount)
	return result
}
