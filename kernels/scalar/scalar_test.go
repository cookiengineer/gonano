package scalar_test

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/kernels"
	"github.com/cookiengineer/gonano/kernels/kerneltest"
	"github.com/cookiengineer/gonano/kernels/scalar"
)

func TestElementwise(t *testing.T) {
	backend := scalar.New()
	left := []float32{1, 2, 3}
	right := []float32{4, 5, 6}
	destination := make([]float32, 3)

	backend.Add(destination, left, right)
	kerneltest.AssertSlicesClose(t, destination, []float32{5, 7, 9}, 1e-6, 1e-6)
	backend.Subtract(destination, right, left)
	kerneltest.AssertSlicesClose(t, destination, []float32{3, 3, 3}, 1e-6, 1e-6)
	backend.Multiply(destination, left, right)
	kerneltest.AssertSlicesClose(t, destination, []float32{4, 10, 18}, 1e-6, 1e-6)
	backend.Divide(destination, right, left)
	kerneltest.AssertSlicesClose(t, destination, []float32{4, 2.5, 2}, 1e-6, 1e-6)
	backend.Scale(destination, left, 2)
	kerneltest.AssertSlicesClose(t, destination, []float32{2, 4, 6}, 1e-6, 1e-6)
	backend.AddScaled(destination, left, right, 0.5)
	kerneltest.AssertSlicesClose(t, destination, []float32{3, 4.5, 6}, 1e-6, 1e-6)
	backend.Negate(destination, left)
	kerneltest.AssertSlicesClose(t, destination, []float32{-1, -2, -3}, 1e-6, 1e-6)
	backend.Abs(destination, []float32{-1, 2, -3})
	kerneltest.AssertSlicesClose(t, destination, []float32{1, 2, 3}, 1e-6, 1e-6)
	backend.Square(destination, left)
	kerneltest.AssertSlicesClose(t, destination, []float32{1, 4, 9}, 1e-6, 1e-6)
	backend.ReluSquared(destination, []float32{-1, 2, -3})
	kerneltest.AssertSlicesClose(t, destination, []float32{0, 4, 0}, 1e-6, 1e-6)
}

func TestTranscendentals(t *testing.T) {
	backend := scalar.New()
	destination := make([]float32, 1)

	backend.Exp(destination, []float32{1})
	kerneltest.AssertSlicesClose(t, destination, []float32{float32(math.E)}, 1e-6, 1e-6)
	backend.Sigmoid(destination, []float32{0})
	kerneltest.AssertSlicesClose(t, destination, []float32{0.5}, 1e-6, 1e-6)
	backend.Tanh(destination, []float32{0})
	kerneltest.AssertSlicesClose(t, destination, []float32{0}, 1e-6, 1e-6)
	backend.Rsqrt(destination, []float32{4})
	kerneltest.AssertSlicesClose(t, destination, []float32{0.5}, 1e-6, 1e-6)
}

func TestReductionsAndLinearAlgebra(t *testing.T) {
	backend := scalar.New()
	if backend.Sum([]float32{1, 2, 3, 4}) != 10 {
		t.Fatal("Sum")
	}
	if backend.Max([]float32{1, 9, 3}) != 9 {
		t.Fatal("Max")
	}
	if backend.Max(nil) != 0 {
		t.Fatal("Max of empty slice should be zero")
	}
	if backend.ArgMax([]float32{1, 9, 3}) != 1 {
		t.Fatal("ArgMax")
	}

	left := []float32{1, 2, 3, 4, 5, 6}     // [2,3]
	right := []float32{7, 8, 9, 10, 11, 12} // [3,2]
	destination := make([]float32, 4)
	backend.MatMul(destination, left, right, 2, 2, 3)
	kerneltest.AssertSlicesClose(t, destination, []float32{58, 64, 139, 154}, 1e-5, 1e-5)

	backend.MatMulTransposed(destination, left, right, 2, 2, 3)
	kerneltest.AssertSlicesClose(t, destination, []float32{50, 68, 122, 167}, 1e-5, 1e-5)

	if !kerneltest.Close(backend.DotProduct([]float32{1, 2, 3}, []float32{4, 5, 6}), 32, 1e-6, 1e-6) {
		t.Fatal("DotProduct")
	}
}

func TestRows(t *testing.T) {
	backend := scalar.New()
	source := []float32{1, 2, 3, 4}
	destination := make([]float32, 4)
	backend.SoftmaxLastDim(destination, source, 2, 2)
	var total float64
	for _, value := range destination {
		total += float64(value)
	}
	if math.Abs(total-2) > 1e-6 {
		t.Fatalf("softmax rows should each sum to one, total = %v", total)
	}
	backend.RMSNormLastDim(destination, []float32{1, 1, 1, 1}, 2, 2, 1e-6)
	kerneltest.AssertSlicesClose(t, destination, []float32{1, 1, 1, 1}, 1e-5, 1e-5)
}

func TestAttentionReference(t *testing.T) {
	backend := scalar.New()
	query := []float32{1, 0, 0, 1}
	key := []float32{1, 0, 0, 1}
	value := []float32{1, 0, 0, 1}

	output := make([]float32, 4)
	logSumExp := make([]float32, 2)
	backend.AttentionForward(
		kernels.AttentionForwardParameters{Query: query, Key: key, Value: value, QueryLength: 2, KeyLength: 2, HeadDim: 2, PositionOffset: 0, Window: -1},
		kernels.AttentionForwardResult{Output: output, LogSumExp: logSumExp},
	)
	// Row 0 attends only to key 0, so its output equals value row 0.
	kerneltest.AssertSlicesClose(t, output[0:2], []float32{1, 0}, 1e-5, 1e-6)

	queryGradient := make([]float32, 4)
	keyGradient := make([]float32, 4)
	valueGradient := make([]float32, 4)
	backend.AttentionBackward(
		kernels.AttentionBackwardParameters{
			Query: query, Key: key, Value: value, Output: output,
			OutputGradient: []float32{1, 1, 1, 1}, LogSumExp: logSumExp,
			QueryLength: 2, KeyLength: 2, HeadDim: 2, PositionOffset: 0, Window: -1,
		},
		kernels.AttentionBackwardResult{QueryGradient: queryGradient, KeyGradient: keyGradient, ValueGradient: valueGradient},
	)
	for _, gradient := range [][]float32{queryGradient, keyGradient, valueGradient} {
		for _, element := range gradient {
			if element != element {
				t.Fatal("attention backward produced NaN")
			}
		}
	}
}
