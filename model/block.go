package model

import (
	"github.com/cookiengineer/gonano/tensors"
)

// Block is a single transformer block: pre-norm attention and feed-forward with
// residual connections, scaled by the per-layer resid and x0 lambdas. The
// feed-forward is either a dense MLP or a shared expert plus a routed MoE
// (DeepSeekMoE).
type Block struct {
	attention *CausalSelfAttention
	mlp       *MLP // dense MLP, or the always-active shared expert when MoE is on
	moe       *MoE // routed experts; nil for the dense MLP
}

// NewBlock builds a transformer block. layer is the block index, used to
// resolve the cross-layer reuse role.
func NewBlock(configuration Config, hasValueEmbedding bool, layer int) *Block {
	block := &Block{
		attention: NewCausalSelfAttention(configuration, hasValueEmbedding, layer),
	}
	if configuration.MoEEnabled() {
		block.mlp = NewSwiGLUMLP(configuration.EmbedDim, configuration.MoESharedHidden(), configuration.MoEEffectiveClamp())
		block.moe = NewMoE(configuration)
	} else {
		block.mlp = NewMLP(configuration.EmbedDim)
	}
	return block
}

// Forward applies attention and feed-forward with residual connections:
//
//	input = input + attention(norm(input))
//	input = input + ffn(norm(input))
//
// where ffn is the dense MLP or the shared expert plus routed MoE output.
func (block *Block) Forward(input, valueEmbedding, cosine, sine *tensors.Tensor, positionOffset int, window [2]int, cache *KVBuffer, layer int, share *compressionShare) *tensors.Tensor {
	input = tensors.Add(input, block.attention.Forward(normalizeLastDim(input), valueEmbedding, cosine, sine, positionOffset, window, cache, layer, share))
	ffnInput := normalizeLastDim(input)
	ffnOutput := block.mlp.Forward(ffnInput)
	if block.moe != nil {
		ffnOutput = tensors.Add(ffnOutput, block.moe.Forward(ffnInput))
	}
	input = tensors.Add(input, ffnOutput)
	return input
}
