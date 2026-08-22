package tensor

import "simd"

// MatMulTransB computes C = A @ B^T where A is [M,K] and B is [N,K] (i.e. B
// is the transposed right operand, laid out as a weight matrix [out,in]).
// The result is [M,N]. This is the primitive used by every linear layer and
// by the query@key^T attention product. It is SIMD-accelerated (dot products
// vectorized over K) and parallelized over output rows.
func MatMulTransB(a, b *Tensor) *Tensor {
	if a.Rank() != 2 || b.Rank() != 2 {
		panic("tensor: MatMulTransB requires rank-2 operands")
	}
	m, k := a.Shape[0], a.Shape[1]
	n := b.Shape[0]
	if b.Shape[1] != k {
		panic("tensor: MatMulTransB inner dimension mismatch")
	}
	out := New(m, n)
	gemmTransB(m, n, k, a.Data, b.Data, out.Data)
	return out
}

// MatMul computes C = A @ B where A is [M,K] and B is [K,N]. The result is
// [M,N]. It is used by the attention scores@values product. SIMD-accelerated
// (rank-1 updates vectorized over N) and parallelized over output rows.
func MatMul(a, b *Tensor) *Tensor {
	if a.Rank() != 2 || b.Rank() != 2 {
		panic("tensor: MatMul requires rank-2 operands")
	}
	m, k := a.Shape[0], a.Shape[1]
	if b.Shape[0] != k {
		panic("tensor: MatMul inner dimension mismatch")
	}
	n := b.Shape[1]
	out := New(m, n)
	gemmNN(m, n, k, a.Data, b.Data, out.Data)
	return out
}

// gemmTransB computes out[M,N] = a[M,K] @ bt[N,K]^T. The row-block size mc
// controls cache reuse of bt across rows of a.
func gemmTransB(m, n, k int, a, bt, out []float32) {
	const mc = 64
	const nc = 64
	numRowBlocks := (m + mc - 1) / mc
	parallelFor(numRowBlocks, func(block int) {
		m0 := block * mc
		m1 := min(m0+mc, m)
		for n0 := 0; n0 < n; n0 += nc {
			n1 := min(n0+nc, n)
			for mm := m0; mm < m1; mm++ {
				aRow := a[mm*k:]
				outRow := out[mm*n:]
				for nn := n0; nn < n1; nn++ {
					outRow[nn] = dotKernel(aRow, bt[nn*k:], k)
				}
			}
		}
	})
}

// gemmNN computes out[M,N] = a[M,K] @ b[K,N] via rank-1 updates. b is read
// once per row block, giving good cache reuse of the right operand.
func gemmNN(m, n, k int, a, b, out []float32) {
	const mc = 64
	numRowBlocks := (m + mc - 1) / mc
	parallelFor(numRowBlocks, func(block int) {
		m0 := block * mc
		m1 := min(m0+mc, m)
		for mm := m0; mm < m1; mm++ {
			clear(out[mm*n : (mm+1)*n])
		}
		for kk := 0; kk < k; kk++ {
			bRow := b[kk*n:]
			for mm := m0; mm < m1; mm++ {
				av := a[mm*k+kk]
				sv := simd.BroadcastFloat32s(av)
				outRow := out[mm*n:]
				nn := 0
				for ; nn+float32Lanes <= n; nn += float32Lanes {
					acc := simd.LoadFloat32s(outRow[nn:])
					bv := simd.LoadFloat32s(bRow[nn:])
					bv.MulAdd(sv, acc).Store(outRow[nn:])
				}
				for ; nn < n; nn++ {
					outRow[nn] += av * bRow[nn]
				}
			}
		}
	})
}

// dotKernel computes the dot product of a and b over k elements, vectorized
// with a single horizontal fold at the end.
func dotKernel(a, b []float32, k int) float32 {
	acc := simd.BroadcastFloat32s(0)
	i := 0
	for ; i+float32Lanes <= k; i += float32Lanes {
		av := simd.LoadFloat32s(a[i:])
		bv := simd.LoadFloat32s(b[i:])
		// MulAdd(x, y, z) = x*y + z, so av.MulAdd(bv, acc) = av*bv + acc.
		acc = av.MulAdd(bv, acc)
	}
	sum := horizontalSum(acc)
	for ; i < k; i++ {
		sum += a[i] * b[i]
	}
	return sum
}
