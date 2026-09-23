package model

import (
	"github.com/cookiengineer/gonano/model/layers"
	"github.com/cookiengineer/gonano/tensors"
)

// MLPActivation selects the feed-forward nonlinearity.
type MLPActivation uint8

const (
	// ActivationReluSquared is the historical nanochat MLP activation:
	// max(x, 0)^2.
	ActivationReluSquared MLPActivation = iota
	// ActivationSwiGLU is the DeepSeekMoE activation: silu(clamp(gate)) *
	// clamp(up). It uses a separate gate projection.
	ActivationSwiGLU
)

// MLP is the feed-forward sublayer. With the ReLU² activation it is
// input-projection -> ReLU² -> output-projection (the historical nanochat
// MLP). With SwiGLU it is a gated input/up projection -> clamped SwiGLU ->
// output projection, used by the shared expert when MoE is enabled.
type MLP struct {
	HiddenDim        int
	Activation       MLPActivation
	Clamp            float32 // SwiGLU clamp threshold
	inputProjection  *layers.Linear
	gateProjection   *layers.Linear // SwiGLU gate; nil for ReLU²
	outputProjection *layers.Linear
}

// NewMLP builds an MLP with the standard 4x expansion ratio and the historical
// ReLU² activation.
func NewMLP(embeddingDimension int) *MLP {
	return NewMLPWithHidden(embeddingDimension, 4*embeddingDimension)
}

// NewMLPWithHidden builds a ReLU² MLP with an explicit intermediate width.
func NewMLPWithHidden(embeddingDimension, hiddenDim int) *MLP {
	return &MLP{
		HiddenDim:        hiddenDim,
		Activation:       ActivationReluSquared,
		inputProjection:  layers.NewLinear(embeddingDimension, hiddenDim),
		outputProjection: layers.NewLinear(hiddenDim, embeddingDimension),
	}
}

// NewSwiGLUMLP builds a clamped SwiGLU MLP with an explicit intermediate width.
func NewSwiGLUMLP(embeddingDimension, hiddenDim int, clamp float32) *MLP {
	return &MLP{
		HiddenDim:        hiddenDim,
		Activation:       ActivationSwiGLU,
		Clamp:            clamp,
		inputProjection:  layers.NewLinear(embeddingDimension, hiddenDim),
		gateProjection:   layers.NewLinear(embeddingDimension, hiddenDim),
		outputProjection: layers.NewLinear(hiddenDim, embeddingDimension),
	}
}

// Forward computes mlp(input) for input of shape [batch, sequence, embedding].
func (mlp *MLP) Forward(input *tensors.Tensor) *tensors.Tensor {
	up := mlp.inputProjection.Forward(input)
	hidden := mlp.activate(up, input)
	return mlp.outputProjection.Forward(hidden)
}

// activate applies the configured nonlinearity. For SwiGLU the gate projection
// is computed from the same input.
func (mlp *MLP) activate(up, input *tensors.Tensor) *tensors.Tensor {
	if mlp.gateProjection != nil {
		gate := mlp.gateProjection.Forward(input)
		return tensors.SwiGLU(gate, up, mlp.Clamp)
	}
	return tensors.ReluSquared(up)
}

// parameters returns every weight of the MLP.
func (mlp *MLP) parameters() []*tensors.Tensor {
	weights := []*tensors.Tensor{mlp.inputProjection.Weight}
	if mlp.gateProjection != nil {
		weights = append(weights, mlp.gateProjection.Weight)
	}
	weights = append(weights, mlp.outputProjection.Weight)
	return weights
}
