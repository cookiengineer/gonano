package simdbackend_test

import (
	"testing"

	"github.com/cookiengineer/gonano/internal/parallel"
	"github.com/cookiengineer/gonano/kernels/kerneltest"
	"github.com/cookiengineer/gonano/kernels/scalar"
	simdbackend "github.com/cookiengineer/gonano/kernels/simd"
)

// TestMatMulTransposedPartitioning forces a small worker count so both
// partitioning strategies are exercised deterministically: row counts below
// the worker count take the column-partition path (batched decode), and larger
// row counts take the row-partition path (training/prefill).
func TestMatMulTransposedPartitioning(t *testing.T) {
	previous := parallel.Default()
	parallel.SetDefault(parallel.NewPool(8))
	defer parallel.SetDefault(previous)

	simdBackend := simdbackend.New()
	scalarBackend := scalar.New()

	shapes := []struct{ rows, columns, inner int }{
		{1, 129, 17},  // column path
		{2, 130, 64},  // column path
		{64, 130, 17}, // row path
		{400, 64, 128},
	}
	for _, shape := range shapes {
		left := kerneltest.Data(shape.rows * shape.inner)
		right := kerneltest.DataB(shape.columns * shape.inner)
		got := make([]float32, shape.rows*shape.columns)
		want := make([]float32, shape.rows*shape.columns)
		simdBackend.MatMulTransposed(got, left, right, shape.rows, shape.columns, shape.inner)
		scalarBackend.MatMulTransposed(want, left, right, shape.rows, shape.columns, shape.inner)
		kerneltest.AssertSlicesClose(t, got, want, 2e-3, 1e-4)
	}
}
