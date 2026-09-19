package simdbackend_test

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/internal/parallel"
	"github.com/cookiengineer/gonano/kernels/kerneltest"
	"github.com/cookiengineer/gonano/kernels/scalar"
	simdbackend "github.com/cookiengineer/gonano/kernels/simd"
)

// quantizeRowsPerChannel quantizes each row of a [rows, columns] matrix to
// symmetric int8 and returns the packed data, per-row scales, and the
// dequantized float32 matrix.
func quantizeRowsPerChannel(values []float32, rows, columns int) ([]int8, []float32, []float32) {
	packed := make([]int8, rows*columns)
	scales := make([]float32, rows)
	dequantized := make([]float32, rows*columns)
	for row := 0; row < rows; row++ {
		maximum := float32(0)
		for column := 0; column < columns; column++ {
			if magnitude := float32(math.Abs(float64(values[row*columns+column]))); magnitude > maximum {
				maximum = magnitude
			}
		}
		scale := float32(1)
		if maximum > 0 {
			scale = maximum / 127
		}
		scales[row] = scale
		for column := 0; column < columns; column++ {
			quantized := math.Round(float64(values[row*columns+column] / scale))
			if quantized > 127 {
				quantized = 127
			}
			if quantized < -127 {
				quantized = -127
			}
			packed[row*columns+column] = int8(quantized)
			dequantized[row*columns+column] = float32(quantized) * scale
		}
	}
	return packed, scales, dequantized
}

func TestMatMulTransposedInt8Parity(t *testing.T) {
	simdBackend := simdbackend.New()
	scalarBackend := scalar.New()
	shapes := []struct{ rows, columns, inner int }{
		{1, 130, 17},
		{3, 129, 200},
		{64, 131, 256},
		{400, 128, 512},
	}
	for _, shape := range shapes {
		left := kerneltest.Data(shape.rows * shape.inner)
		weights := kerneltest.DataB(shape.columns * shape.inner)
		packed, scales, dequantized := quantizeRowsPerChannel(weights, shape.columns, shape.inner)

		got := make([]float32, shape.rows*shape.columns)
		want := make([]float32, shape.rows*shape.columns)
		reference := make([]float32, shape.rows*shape.columns)
		simdBackend.MatMulTransposedInt8(got, left, packed, scales, shape.rows, shape.columns, shape.inner)
		scalarBackend.MatMulTransposedInt8(want, left, packed, scales, shape.rows, shape.columns, shape.inner)
		kerneltest.AssertSlicesClose(t, got, want, 2e-3, 1e-4)

		// The int8 path must equal the fp32 path over the dequantized weights.
		simdBackend.MatMulTransposed(reference, left, dequantized, shape.rows, shape.columns, shape.inner)
		kerneltest.AssertSlicesClose(t, got, reference, 2e-3, 1e-4)
	}
}

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
