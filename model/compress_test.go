package model

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/tensors"
)

func TestChannelCompressorForwardMeansBlocks(t *testing.T) {
	// Zero logits => uniform softmax => each complete block is a mean; a
	// trailing partial block keeps its single row.
	compressor := NewChannelCompressor(1, 1, 2)
	hidden := tensors.New(1, 3, 1)
	value := tensors.NewWithData([]int{1, 3, 1}, []float32{2, 4, 6})
	compressed, _ := compressor.Forward(hidden, value)
	if compressed.Shape[0] != 1 || compressed.Shape[1] != 2 || compressed.Shape[2] != 1 {
		t.Fatalf("compressed shape = %v, want [1 2 1]", compressed.Shape)
	}
	if compressed.Data[0] != 3 { // (2+4)/2
		t.Fatalf("block 0 = %v, want 3", compressed.Data[0])
	}
	if compressed.Data[1] != 6 { // single remaining row
		t.Fatalf("block 1 = %v, want 6", compressed.Data[1])
	}
}

func TestChannelCompressorGradient(t *testing.T) {
	compressor := NewChannelCompressor(4, 3, 2)
	rng := tensors.NewRNG(7)
	for _, parameter := range []*tensors.Tensor{compressor.logitWeight.Weight, compressor.bias} {
		for index := range parameter.Data {
			parameter.Data[index] = rng.NormFloat32() * 0.5
		}
	}

	hidden := tensors.New(2, 5, 4)
	tensors.FillNormal(hidden, rng, 1)
	value := tensors.New(2, 5, 3)
	tensors.FillNormal(value, rng, 1)

	// A fixed random cotangent for the compressed output.
	compressedShapeLen := 2 * 3 * 3 // B=2, blocks=3, C=3
	gradCompressed := tensors.NewWithData([]int{2, 3, 3}, make([]float32, compressedShapeLen))
	for index := range gradCompressed.Data {
		gradCompressed.Data[index] = rng.NormFloat32()
	}

	compressor.ZeroGrad()
	_, context := compressor.Forward(hidden, value)
	gradHidden, gradValue := compressor.Backward(gradCompressed, context)

	// Random directions for every input and parameter.
	dirHidden := tensors.New(hidden.Shape...)
	dirValue := tensors.New(value.Shape...)
	tensors.FillNormal(dirHidden, rng, 1)
	tensors.FillNormal(dirValue, rng, 1)
	dirWeight := tensors.New(compressor.logitWeight.Weight.Shape...)
	dirBias := tensors.New(compressor.bias.Shape...)
	tensors.FillNormal(dirWeight, rng, 1)
	tensors.FillNormal(dirBias, rng, 1)

	analytic := float64(0)
	for index, value := range gradHidden.Data {
		analytic += float64(value) * float64(dirHidden.Data[index])
	}
	for index, value := range gradValue.Data {
		analytic += float64(value) * float64(dirValue.Data[index])
	}
	for index, value := range compressor.logitWeight.Weight.Grad {
		analytic += float64(value) * float64(dirWeight.Data[index])
	}
	for index, value := range compressor.bias.Grad {
		analytic += float64(value) * float64(dirBias.Data[index])
	}

	loss := func() float32 {
		compressed, _ := compressor.Forward(hidden, value)
		var total float32
		for index, value := range compressed.Data {
			total += value * gradCompressed.Data[index]
		}
		return total
	}

	epsilon := float32(1e-3)
	perturb := func(sign float32) {
		for index := range hidden.Data {
			hidden.Data[index] += sign * epsilon * dirHidden.Data[index]
		}
		for index := range value.Data {
			value.Data[index] += sign * epsilon * dirValue.Data[index]
		}
		for index := range compressor.logitWeight.Weight.Data {
			compressor.logitWeight.Weight.Data[index] += sign * epsilon * dirWeight.Data[index]
		}
		for index := range compressor.bias.Data {
			compressor.bias.Data[index] += sign * epsilon * dirBias.Data[index]
		}
	}
	perturb(1)
	lossPlus := loss()
	perturb(-2)
	lossMinus := loss()
	perturb(1)

	numeric := float64(lossPlus-lossMinus) / (2 * float64(epsilon))
	scale := math.Abs(analytic) + 1e-4
	if math.Abs(numeric-analytic) > 5e-2*scale {
		t.Fatalf("compressor directional gradient mismatch: analytic=%v numeric=%v", analytic, numeric)
	}
}
