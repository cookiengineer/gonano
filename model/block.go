package model

import (
	"github.com/cookiengineer/gonano/tensors"
)

// Block is a single transformer block: pre-norm attention and MLP with
// residual connections, scaled by the per-layer resid and x0 lambdas.
type Block struct {
	attention *CausalSelfAttention
	mlp       *MLP
}

// NewBlock builds a transformer block.
func NewBlock(configuration Config, hasValueEmbedding bool) *Block {
	return &Block{
		attention: NewCausalSelfAttention(configuration, hasValueEmbedding),
		mlp:       NewMLP(configuration.EmbedDim),
	}
}

// Forward applies attention and MLP with residual connections:
//
//	input = input + attention(norm(input))
//	input = input + mlp(norm(input))
func (block *Block) Forward(input, valueEmbedding, cosine, sine *tensors.Tensor, positionOffset int, window [2]int, cache *KVBuffer, layer int) *tensors.Tensor {
	input = tensors.Add(input, block.attention.Forward(normalizeLastDim(input), valueEmbedding, cosine, sine, positionOffset, window, cache, layer))
	input = tensors.Add(input, block.mlp.Forward(normalizeLastDim(input)))
	return input
}
