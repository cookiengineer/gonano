package tensors

import (
	"testing"

	"github.com/cookiengineer/gonano/kernels/scalar"
)

func TestMatMulTransposedSmall(t *testing.T) {
	left := NewWithData([]int{2, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
	})
	right := NewWithData([]int{3, 3}, []float32{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	})
	got := MatMulTransposed(left, right)
	want := []float32{1, 2, 3, 4, 5, 6}
	for index := range want {
		if !close(got.Data[index], want[index], 1e-5, 1e-6) {
			t.Fatalf("MatMulTransposed[%d] = %v, want %v", index, got.Data[index], want[index])
		}
	}
}

func TestMatMulSmall(t *testing.T) {
	left := NewWithData([]int{2, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
	})
	right := NewWithData([]int{3, 2}, []float32{
		1, 2,
		3, 4,
		5, 6,
	})
	got := MatMul(left, right)
	want := []float32{22, 28, 49, 64}
	for index := range want {
		if !close(got.Data[index], want[index], 1e-5, 1e-6) {
			t.Fatalf("MatMul[%d] = %v, want %v", index, got.Data[index], want[index])
		}
	}
}

func TestMatMulMatchesScalarBackend(t *testing.T) {
	left := NewWithData([]int{40, 33}, testData(40*33))
	right := NewWithData([]int{33, 29}, testDataB(33*29))
	got := MatMul(left, right)

	previous := KernelBackend()
	UseKernelBackend(scalar.New())
	want := MatMul(left, right)
	UseKernelBackend(previous)

	assertTensorClose(t, got, want, 5e-3, 1e-4)
}

func TestMatMulShapeMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on inner-dim mismatch")
		}
	}()
	MatMul(New(3, 4), New(5, 6))
}

func TestMatMulTransposedShapeMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on inner-dim mismatch")
		}
	}()
	MatMulTransposed(New(3, 4), New(5, 6))
}
