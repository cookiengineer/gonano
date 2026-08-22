package tensor

import "math"

// SoftmaxLastDim applies the numerically stable softmax over the last (column)
// dimension of a 2D tensor. Rows are independent and processed in parallel.
func SoftmaxLastDim(x *Tensor) *Tensor {
	rows, cols := x.Shape[0], x.Shape[1]
	out := New(rows, cols)
	parallelChunks(rows, func(r0, r1 int) {
		for r := r0; r < r1; r++ {
			row := x.Data[r*cols : (r+1)*cols]
			dst := out.Data[r*cols : (r+1)*cols]
			m := sliceMax(row)
			var sum float64
			for c := 0; c < cols; c++ {
				e := float32(math.Exp(float64(row[c] - m)))
				dst[c] = e
				sum += float64(e)
			}
			inv := float32(1.0 / sum)
			scaleKernel(dst, dst, inv)
		}
	})
	return out
}

// Softmax is an alias of SoftmaxLastDim for a 2D tensor.
func Softmax(x *Tensor) *Tensor { return SoftmaxLastDim(x) }

// RMSNormLastDim applies RMSNorm (normalize each row to unit RMS, no affine
// parameters) over the last dimension of a 2D tensor. This matches nanochat's
// stateless norm(x) = rms_norm(x, -1).
func RMSNormLastDim(x *Tensor, eps float32) *Tensor {
	rows, cols := x.Shape[0], x.Shape[1]
	out := New(rows, cols)
	parallelChunks(rows, func(r0, r1 int) {
		for r := r0; r < r1; r++ {
			row := x.Data[r*cols : (r+1)*cols]
			dst := out.Data[r*cols : (r+1)*cols]
			var sumSq float64
			for _, v := range row {
				sumSq += float64(v) * float64(v)
			}
			meanSq := float32(sumSq / float64(cols))
			rstd := float32(1.0 / math.Sqrt(float64(meanSq)+float64(eps)))
			scaleKernel(dst, row, rstd)
		}
	})
	return out
}
