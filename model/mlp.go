package model

import (
	"github.com/cookiengineer/gonano/nn"
	"github.com/cookiengineer/gonano/tensor"
)

// MLP is the feed-forward sublayer: c_fc (4x expansion) -> ReLU² -> c_proj.
type MLP struct {
	CFc   *nn.Linear
	CProj *nn.Linear
}

// NewMLP builds an MLP with the standard 4x expansion ratio.
func NewMLP(embedDim int) *MLP {
	return &MLP{
		CFc:   nn.NewLinear(embedDim, 4*embedDim),
		CProj: nn.NewLinear(4*embedDim, embedDim),
	}
}

// Forward computes mlp(x) for x of shape [B,T,C].
func (m *MLP) Forward(x *tensor.Tensor) *tensor.Tensor {
	h := tensor.Relu2(m.CFc.Forward(x))
	return m.CProj.Forward(h)
}
