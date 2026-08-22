// Package nn provides the neural-network building blocks used by the gonano
// transformer: linear layers, embeddings, and weight initializers.
//
// All layers store their parameters as float32 tensors and use the SIMD
// kernels in the tensor package for their forward passes.
package nn

import (
	"github.com/cookiengineer/gonano/parallel"
	"github.com/cookiengineer/gonano/tensor"
)

// Linear is a bias-free linear layer: out = x @ W^T, with W of shape
// [out_features, in_features]. nanochat uses no biases anywhere.
type Linear struct {
	InFeatures  int
	OutFeatures int
	Weight      *tensor.Tensor // [out, in]
}

// NewLinear allocates a zero-initialized Linear layer.
func NewLinear(in, out int) *Linear {
	return &Linear{
		InFeatures:  in,
		OutFeatures: out,
		Weight:      tensor.New(out, in),
	}
}

// Forward computes out = x @ W^T for x of shape [..., in]. The trailing
// in_features dimension is contracted; leading dims are preserved.
func (l *Linear) Forward(x *tensor.Tensor) *tensor.Tensor {
	if x.Shape[len(x.Shape)-1] != l.InFeatures {
		panic("nn: input feature dimension mismatch")
	}
	m := x.Numel() / l.InFeatures
	x2d := x.Reshape(m, l.InFeatures)
	y2d := tensor.MatMulTransB(x2d, l.Weight) // [m, out]
	return y2d.Reshape(append(append([]int(nil), x.Shape[:len(x.Shape)-1]...), l.OutFeatures)...)
}

// Backward computes gradients given the forward input x and the gradient of
// the output. It accumulates the weight gradient into l.Weight.Grad and
// returns the gradient with respect to the input.
func (l *Linear) Backward(x, gradOut *tensor.Tensor) *tensor.Tensor {
	m := x.Numel() / l.InFeatures
	x2d := x.Reshape(m, l.InFeatures)
	go2d := gradOut.Reshape(m, l.OutFeatures)
	gradW := tensor.MatMul(tensor.Transpose(go2d), x2d) // [out, in]
	l.Weight.AddGrad(gradW.Data)
	gradIn := tensor.MatMul(go2d, l.Weight) // [m, out] @ [out, in] = [m, in]
	return gradIn.Reshape(x.Shape...)
}

// Embedding maps integer token ids to dense vectors via a lookup table of
// shape [num_embeddings, dim].
type Embedding struct {
	NumEmbeddings int
	Dim           int
	Weight        *tensor.Tensor // [num, dim]
}

// NewEmbedding allocates a zero-initialized embedding table.
func NewEmbedding(num, dim int) *Embedding {
	return &Embedding{NumEmbeddings: num, Dim: dim, Weight: tensor.New(num, dim)}
}

// Forward gathers rows of the weight table for each token id. idx has shape
// [d0, d1, ...]; the result has shape [d0, d1, ..., dim]. Out-of-range ids
// panic (they indicate a bug in the caller).
func (e *Embedding) Forward(idx *tensor.Int32s) *tensor.Tensor {
	n := idx.Numel()
	dim := e.Dim
	out := tensor.New(append(append([]int(nil), idx.Shape...), dim)...)
	weight := e.Weight.Data
	outData := out.Data
	parallel.Default().Chunks(0, n, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			id := idx.Data[i]
			src := weight[int(id)*dim : (int(id)+1)*dim]
			copy(outData[i*dim:(i+1)*dim], src)
		}
	})
	return out
}

// Backward accumulates the gradient of the output into the embedding table
// rows indexed by idx. It has no return value because an embedding has no
// trainable input.
func (e *Embedding) Backward(idx *tensor.Int32s, gradOut *tensor.Tensor) {
	n := idx.Numel()
	dim := e.Dim
	e.Weight.EnsureGrad()
	wg := e.Weight.Grad
	gd := gradOut.Data
	for i := 0; i < n; i++ {
		id := idx.Data[i]
		base := i * dim
		wbase := int(id) * dim
		for j := 0; j < dim; j++ {
			wg[wbase+j] += gd[base+j]
		}
	}
}
