// Package layers provides the neural-network building blocks used by the
// gonano transformer: linear layers, embeddings, and weight initializers.
//
// All layers store their parameters as float32 tensors and compute through the
// tensors package, which delegates to the active kernels.Backend.
package layers

import (
	"github.com/cookiengineer/gonano/internal/parallel"
	"github.com/cookiengineer/gonano/tensors"
)

// Quantization modes for a Linear layer's weight.
const (
	QuantNone = 0
	QuantInt8 = 1
)

// Linear is a bias-free linear layer: out = input @ W^T, with W of shape
// [out_features, in_features]. nanochat uses no biases anywhere.
type Linear struct {
	InFeatures  int
	OutFeatures int
	Weight      *tensors.Tensor // [out, in]
	// QuantMode enables quantization-aware training for the forward pass. The
	// master weight stays float32; the forward use a fake-quantized copy so the
	// network adapts to quantization (straight-through on the backward).
	QuantMode int
	// Packed, PackedWeight, and PackedScales hold an int8 inference weight
	// representation produced by PackInt8. When Packed is set the forward uses
	// the quantized GEMM, which reads packed weights once per forward.
	Packed       bool
	PackedWeight []int8
	PackedScales []float32
}

// PackInt8 quantizes the float32 master weight into a per-row symmetric int8
// representation for the quantized inference GEMM.
func (layer *Linear) PackInt8() {
	quantized := tensors.QuantizeInt8Rows(layer.Weight)
	layer.PackedWeight = quantized.Packed
	layer.PackedScales = quantized.Scales
	layer.Packed = true
}

// NewLinear allocates a zero-initialized Linear layer.
func NewLinear(in, out int) *Linear {
	return &Linear{
		InFeatures:  in,
		OutFeatures: out,
		Weight:      tensors.New(out, in),
	}
}

// Forward computes out = input @ W^T for input of shape [..., in]. The
// trailing in_features dimension is contracted; leading dims are preserved.
func (layer *Linear) Forward(input *tensors.Tensor) *tensors.Tensor {
	if input.Shape[len(input.Shape)-1] != layer.InFeatures {
		panic("nn: input feature dimension mismatch")
	}
	rows := input.Numel() / layer.InFeatures
	inputMatrix := input.Reshape(rows, layer.InFeatures)
	var outputMatrix *tensors.Tensor
	switch {
	case layer.Packed:
		// Int8 inference: read packed weights once per forward.
		outputMatrix = tensors.MatMulTransposedInt8(inputMatrix, layer.PackedWeight, layer.PackedScales, layer.OutFeatures)
	case layer.QuantMode == QuantInt8:
		// Quantization-aware training: use a fake-quantized weight in the
		// forward pass. Backward keeps the straight-through estimator.
		outputMatrix = tensors.MatMulTransposed(inputMatrix, tensors.FakeQuantizeInt8Rows(layer.Weight))
	default:
		outputMatrix = tensors.MatMulTransposed(inputMatrix, layer.Weight)
	}
	return outputMatrix.Reshape(append(append([]int(nil), input.Shape[:len(input.Shape)-1]...), layer.OutFeatures)...)
}

// Backward computes gradients given the forward input and the gradient of the
// output. It accumulates the weight gradient into layer.Weight.Grad and
// returns the gradient with respect to the input.
func (layer *Linear) Backward(input, gradOutput *tensors.Tensor) *tensors.Tensor {
	rows := input.Numel() / layer.InFeatures
	inputMatrix := input.Reshape(rows, layer.InFeatures)
	gradOutputMatrix := gradOutput.Reshape(rows, layer.OutFeatures)
	gradWeight := tensors.MatMul(tensors.Transpose(gradOutputMatrix), inputMatrix) // [out, in]
	layer.Weight.AddGrad(gradWeight.Data)
	gradInput := tensors.MatMul(gradOutputMatrix, layer.Weight) // [rows, out] @ [out, in] = [rows, in]
	return gradInput.Reshape(input.Shape...)
}

// Embedding maps integer token ids to dense vectors via a lookup table of
// shape [num_embeddings, dim].
type Embedding struct {
	NumEmbeddings int
	Dim           int
	Weight        *tensors.Tensor // [num, dim]
}

// NewEmbedding allocates a zero-initialized embedding table.
func NewEmbedding(num, dim int) *Embedding {
	return &Embedding{NumEmbeddings: num, Dim: dim, Weight: tensors.New(num, dim)}
}

// Forward gathers rows of the weight table for each token id. indices has
// shape [d0, d1, ...]; the result has shape [d0, d1, ..., dim]. Out-of-range
// ids panic (they indicate a bug in the caller).
func (embedding *Embedding) Forward(indices *tensors.Int32s) *tensors.Tensor {
	count := indices.Numel()
	dim := embedding.Dim
	output := tensors.New(append(append([]int(nil), indices.Shape...), dim)...)
	weight := embedding.Weight.Data
	outputData := output.Data
	parallel.Default().Chunks(0, count, func(lo, hi int) {
		for index := lo; index < hi; index++ {
			tokenID := indices.Data[index]
			source := weight[int(tokenID)*dim : (int(tokenID)+1)*dim]
			copy(outputData[index*dim:(index+1)*dim], source)
		}
	})
	return output
}

// Backward accumulates the gradient of the output into the embedding table
// rows indexed by indices. It has no return value because an embedding has no
// trainable input.
func (embedding *Embedding) Backward(indices *tensors.Int32s, gradOutput *tensors.Tensor) {
	count := indices.Numel()
	dim := embedding.Dim
	embedding.Weight.EnsureGrad()
	weightGrad := embedding.Weight.Grad
	gradData := gradOutput.Data
	for index := 0; index < count; index++ {
		tokenID := indices.Data[index]
		base := index * dim
		weightBase := int(tokenID) * dim
		for column := 0; column < dim; column++ {
			weightGrad[weightBase+column] += gradData[base+column]
		}
	}
}
