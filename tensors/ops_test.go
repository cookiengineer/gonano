package tensors

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/kernels/scalar"
)

// useScalarBackend swaps the active kernel backend for the portable scalar
// reference so wrapper tests assert on exact, backend-independent values.
func useScalarBackend(t *testing.T) {
	t.Helper()
	previous := KernelBackend()
	UseKernelBackend(scalar.New())
	t.Cleanup(func() { UseKernelBackend(previous) })
}

func TestElementwiseOperations(t *testing.T) {
	useScalarBackend(t)
	left := NewWithData([]int{4}, []float32{1, 2, 3, 4})
	right := NewWithData([]int{4}, []float32{5, 6, 7, 8})

	assertTensorClose(t, Add(left, right), NewWithData([]int{4}, []float32{6, 8, 10, 12}), 1e-6, 1e-6)
	assertTensorClose(t, Subtract(right, left), NewWithData([]int{4}, []float32{4, 4, 4, 4}), 1e-6, 1e-6)
	assertTensorClose(t, Multiply(left, right), NewWithData([]int{4}, []float32{5, 12, 21, 32}), 1e-6, 1e-6)
	assertTensorClose(t, Divide(right, left), NewWithData([]int{4}, []float32{5, 3, 7.0 / 3.0, 2}), 1e-6, 1e-6)
	assertTensorClose(t, Scale(left, 2), NewWithData([]int{4}, []float32{2, 4, 6, 8}), 1e-6, 1e-6)
	assertTensorClose(t, AddScaled(left, right, 0.5), NewWithData([]int{4}, []float32{3.5, 5, 6.5, 8}), 1e-6, 1e-6)
	assertTensorClose(t, Negate(left), NewWithData([]int{4}, []float32{-1, -2, -3, -4}), 1e-6, 1e-6)
	assertTensorClose(t, Abs(Negate(left)), left, 1e-6, 1e-6)
	assertTensorClose(t, Square(right), NewWithData([]int{4}, []float32{25, 36, 49, 64}), 1e-6, 1e-6)
	assertTensorClose(t, ReluSquared(NewWithData([]int{4}, []float32{-1, 2, -3, 4})), NewWithData([]int{4}, []float32{0, 4, 0, 16}), 1e-6, 1e-6)
}

func TestElementwiseShapeMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on shape mismatch")
		}
	}()
	Add(New(2, 3), New(3, 2))
}

func TestTranscendentalOperations(t *testing.T) {
	useScalarBackend(t)
	source := NewWithData([]int{3}, []float32{0, 1, -1})

	sigmoid := Sigmoid(source)
	if !close(sigmoid.Data[0], 0.5, 1e-5, 1e-6) || !close(sigmoid.Data[1], 0.731058, 1e-5, 1e-6) {
		t.Fatalf("sigmoid = %v", sigmoid.Data)
	}
	hyperbolic := Tanh(source)
	if !close(hyperbolic.Data[0], 0, 1e-5, 1e-6) || !close(hyperbolic.Data[1], 0.761594, 1e-5, 1e-6) {
		t.Fatalf("tanh = %v", hyperbolic.Data)
	}
	exponential := Exp(NewWithData([]int{1}, []float32{1}))
	if !close(exponential.Data[0], float32(math.E), 1e-5, 1e-6) {
		t.Fatalf("exp = %v", exponential.Data[0])
	}
	reciprocal := Rsqrt(NewWithData([]int{1}, []float32{4}))
	if !close(reciprocal.Data[0], 0.5, 1e-5, 1e-6) {
		t.Fatalf("rsqrt = %v", reciprocal.Data[0])
	}
	softcapped := Softcap(NewWithData([]int{1}, []float32{7.5}), 15)
	if !close(softcapped.Data[0], 15*float32(math.Tanh(0.5)), 1e-4, 1e-5) {
		t.Fatalf("softcap = %v", softcapped.Data[0])
	}
}
