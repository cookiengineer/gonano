package tensor

import "simd"

// sliceSum returns the sum of all elements using SIMD accumulation with a
// single horizontal fold at the end.
func sliceSum(s []float32) float32 {
	n := len(s)
	acc := simd.BroadcastFloat32s(0)
	i := 0
	for ; i+float32Lanes <= n; i += float32Lanes {
		acc = acc.Add(simd.LoadFloat32s(s[i:]))
	}
	sum := horizontalSum(acc)
	for ; i < n; i++ {
		sum += s[i]
	}
	return sum
}

// sliceMax returns the maximum element using SIMD accumulation. Small slices
// shorter than one vector fall back to a scalar scan.
func sliceMax(s []float32) float32 {
	n := len(s)
	if n == 0 {
		return 0
	}
	if n < float32Lanes {
		return maxScalar(s)
	}
	acc := simd.LoadFloat32s(s)
	i := float32Lanes
	for ; i+float32Lanes <= n; i += float32Lanes {
		acc = acc.Max(simd.LoadFloat32s(s[i:]))
	}
	m := horizontalMax(acc)
	for ; i < n; i++ {
		if s[i] > m {
			m = s[i]
		}
	}
	return m
}

// sliceMaxIndex returns the index of the maximum element. Ties resolve to the
// lowest index.
func sliceMaxIndex(s []float32) int {
	best := 0
	for i := 1; i < len(s); i++ {
		if s[i] > s[best] {
			best = i
		}
	}
	return best
}

// SumAll returns the sum of all elements.
func SumAll(t *Tensor) float32 { return sliceSum(t.Data) }

// MaxAll returns the maximum element.
func MaxAll(t *Tensor) float32 { return sliceMax(t.Data) }

// ArgMax returns the index of the maximum element of a 1D tensor.
func ArgMax(t *Tensor) int { return sliceMaxIndex(t.Data) }

// SumLastDim reduces a 2D tensor over its last (column) dimension, returning a
// [rows] tensor of row sums.
func SumLastDim(x *Tensor) *Tensor {
	rows, cols := x.Shape[0], x.Shape[1]
	out := New(rows)
	parallelChunks(rows, func(r0, r1 int) {
		for r := r0; r < r1; r++ {
			out.Data[r] = sliceSum(x.Data[r*cols : (r+1)*cols])
		}
	})
	return out
}

// MaxLastDim reduces a 2D tensor over its last dimension, returning a [rows]
// tensor of row maxima.
func MaxLastDim(x *Tensor) *Tensor {
	rows, cols := x.Shape[0], x.Shape[1]
	out := New(rows)
	parallelChunks(rows, func(r0, r1 int) {
		for r := r0; r < r1; r++ {
			out.Data[r] = sliceMax(x.Data[r*cols : (r+1)*cols])
		}
	})
	return out
}

// ArgMaxLastDim returns the column index of the maximum in each row of a 2D
// tensor, as an integer tensor of shape [rows].
func ArgMaxLastDim(x *Tensor) *Int32s {
	rows, cols := x.Shape[0], x.Shape[1]
	out := NewInt32s(rows)
	parallelChunks(rows, func(r0, r1 int) {
		for r := r0; r < r1; r++ {
			out.Data[r] = int32(sliceMaxIndex(x.Data[r*cols : (r+1)*cols]))
		}
	})
	return out
}
