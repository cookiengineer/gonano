package model

import (
	"github.com/cookiengineer/gonano/tensor"
)

// Block is a single transformer block: pre-norm attention and MLP with
// residual connections, scaled by the per-layer resid and x0 lambdas.
type Block struct {
	Attn *CausalSelfAttention
	MLP  *MLP
}

// NewBlock builds a transformer block.
func NewBlock(cfg Config, hasVE bool) *Block {
	return &Block{
		Attn: NewCausalSelfAttention(cfg, hasVE),
		MLP:  NewMLP(cfg.EmbedDim),
	}
}

// Forward applies attention and MLP with residual connections:
//
//	x = x + attn(norm(x))
//	x = x + mlp(norm(x))
func (b *Block) Forward(x, ve, cos, sin *tensor.Tensor, t0 int, window [2]int, cache *KVBuffer, layer int) *tensor.Tensor {
	x = tensor.Add(x, b.Attn.Forward(normLastDim(x), ve, cos, sin, t0, window, cache, layer))
	x = tensor.Add(x, b.MLP.Forward(normLastDim(x)))
	return x
}
