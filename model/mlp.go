package model

import (
	"github.com/cookiengineer/gonano/model/layers"
	"github.com/cookiengineer/gonano/tensors"
)

// MLP is the feed-forward sublayer: input projection (4x expansion) ->
// ReLU-squared -> output projection.
type MLP struct {
	inputProjection  *layers.Linear
	outputProjection *layers.Linear
}

// NewMLP builds an MLP with the standard 4x expansion ratio.
func NewMLP(embeddingDimension int) *MLP {
	return &MLP{
		inputProjection:  layers.NewLinear(embeddingDimension, 4*embeddingDimension),
		outputProjection: layers.NewLinear(4*embeddingDimension, embeddingDimension),
	}
}

// Forward computes mlp(input) for input of shape [batch, sequence, embedding].
func (mlp *MLP) Forward(input *tensors.Tensor) *tensors.Tensor {
	hidden := tensors.ReluSquared(mlp.inputProjection.Forward(input))
	return mlp.outputProjection.Forward(hidden)
}
