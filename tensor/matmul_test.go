package tensor

import "testing"

// scalarMatMulTransB is a float64-accumulated reference for C = A @ B^T.
func scalarMatMulTransB(m, n, k int, a, bt []float32) []float32 {
	out := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var acc float64
			for p := 0; p < k; p++ {
				acc += float64(a[i*k+p]) * float64(bt[j*k+p])
			}
			out[i*n+j] = float32(acc)
		}
	}
	return out
}

// scalarMatMul is a float64-accumulated reference for C = A @ B.
func scalarMatMul(m, n, k int, a, b []float32) []float32 {
	out := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var acc float64
			for p := 0; p < k; p++ {
				acc += float64(a[i*k+p]) * float64(b[p*n+j])
			}
			out[i*n+j] = float32(acc)
		}
	}
	return out
}

func TestMatMulTransBParity(t *testing.T) {
	m, n, k := 97, 131, 256
	a := NewWithData([]int{m, k}, testData(m*k))
	bt := NewWithData([]int{n, k}, testDataB(n*k))
	got := MatMulTransB(a, bt)
	want := scalarMatMulTransB(m, n, k, a.Data, bt.Data)
	assertTensorClose(t, got, NewWithData([]int{m, n}, want), 2e-3, 1e-5)
}

func TestMatMulTransBParitySmall(t *testing.T) {
	m, n, k := 3, 5, 17
	a := NewWithData([]int{m, k}, testData(m*k))
	bt := NewWithData([]int{n, k}, testDataB(n*k))
	got := MatMulTransB(a, bt)
	want := scalarMatMulTransB(m, n, k, a.Data, bt.Data)
	assertTensorClose(t, got, NewWithData([]int{m, n}, want), 1e-5, 1e-6)
}

func TestMatMulParity(t *testing.T) {
	m, n, k := 64, 129, 200
	a := NewWithData([]int{m, k}, testData(m*k))
	b := NewWithData([]int{k, n}, testDataB(k*n))
	got := MatMul(a, b)
	want := scalarMatMul(m, n, k, a.Data, b.Data)
	assertTensorClose(t, got, NewWithData([]int{m, n}, want), 2e-3, 1e-5)
}

func TestMatMulParitySmall(t *testing.T) {
	m, n, k := 4, 6, 13
	a := NewWithData([]int{m, k}, testData(m*k))
	b := NewWithData([]int{k, n}, testDataB(k*n))
	got := MatMul(a, b)
	want := scalarMatMul(m, n, k, a.Data, b.Data)
	assertTensorClose(t, got, NewWithData([]int{m, n}, want), 1e-5, 1e-6)
}

func TestMatMulShapeMismatch(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on inner-dim mismatch")
		}
	}()
	MatMul(New(3, 4), New(5, 6))
}

func TestMatMulTransBShapeMismatch(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on inner-dim mismatch")
		}
	}()
	MatMulTransB(New(3, 4), New(5, 6))
}

// TestMatMulTransBLarge exercises the parallel tiling path (M > one block).
func TestMatMulTransBLarge(t *testing.T) {
	m, n, k := 400, 128, 512
	a := NewWithData([]int{m, k}, testData(m*k))
	bt := NewWithData([]int{n, k}, testDataB(n*k))
	got := MatMulTransB(a, bt)
	want := scalarMatMulTransB(m, n, k, a.Data, bt.Data)
	assertTensorClose(t, got, NewWithData([]int{m, n}, want), 5e-3, 1e-4)
}

// TestDotKernelHandlesExactMultiple verifies the no-tail fast path.
func TestDotKernelExactMultiple(t *testing.T) {
	k := float32Lanes * 4
	a := make([]float32, k)
	b := make([]float32, k)
	for i := 0; i < k; i++ {
		a[i] = float32(i+1) / 10
		b[i] = float32(k-i) / 10
	}
	got := dotKernel(a, b, k)
	var want float64
	for i := 0; i < k; i++ {
		want += float64(a[i]) * float64(b[i])
	}
	if !close(got, float32(want), 1e-5, 1e-6) {
		t.Fatalf("dot = %v want %v", got, want)
	}
}
