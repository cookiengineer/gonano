package layers

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/tensors"
)

func TestLinearInt8ForwardMatchesFakeQuant(t *testing.T) {
	linear := NewLinear(4, 3)
	rng := tensors.NewRNG(5)
	for index := range linear.Weight.Data {
		linear.Weight.Data[index] = rng.NormFloat32()
	}
	input := tensors.New(2, 4)
	for index := range input.Data {
		input.Data[index] = rng.NormFloat32()
	}

	linear.QuantMode = QuantInt8
	got := linear.Forward(input)

	reference := &Linear{InFeatures: 4, OutFeatures: 3, Weight: tensors.FakeQuantizeInt8Rows(linear.Weight)}
	want := reference.Forward(input)

	for index := range got.Data {
		if math.Abs(float64(got.Data[index]-want.Data[index])) > 1e-5 {
			t.Fatalf("quantized forward mismatch at %d: got %v want %v", index, got.Data[index], want.Data[index])
		}
	}
}
