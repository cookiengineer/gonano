package model

import (
	"github.com/cookiengineer/gonano/model/layers"
	"github.com/cookiengineer/gonano/tensors"
)

// MLAttention holds the absorbed Multi-head Latent Attention parameters
// (DeepSeek-V4.1 §2.3/§4.2.1). Content keys and values are up-projected from a
// shared latent c_kv; a decoupled rotary key is stored alongside it. At
// inference the content up-projections are absorbed into the query and output
// so attention reads the cached latent directly, which is what makes the KV
// cache hold a single latent row per token instead of a key and a value per
// head.
type MLAttention struct {
	// QueryDown projects the hidden state to the query latent.
	QueryDown *layers.Linear // [latentDim, EmbedDim]
	// QueryUp expands the query latent to the per-head content query.
	QueryUp *layers.Linear // [NumHead*contentDim, latentDim]
	// QueryRope projects the decoupled rotary query.
	QueryRope *layers.Linear // [NumHead*ropeDim, EmbedDim]
	// KVDown projects the hidden state to the shared key/value latent.
	KVDown *layers.Linear // [latentDim, EmbedDim]
	// KeyUp expands the latent to the per-head content key.
	KeyUp *layers.Linear // [NumKVHead*contentDim, latentDim]
	// ValueUp expands the latent to the per-head value.
	ValueUp *layers.Linear // [NumKVHead*valueDim, latentDim]
	// KeyRope projects the decoupled rotary key.
	KeyRope *layers.Linear // [NumKVHead*ropeDim, EmbedDim]

	queryHeadCount int
	kvHeadCount    int
	contentDim     int
	valueDim       int
	ropeDim        int
	latentDim      int
}

// newMLAttention builds the MLA parameters for a config. The caller guarantees
// MLAEnabled.
func newMLAttention(config Config) *MLAttention {
	headDimension := config.HeadDim()
	ropeDim := config.MLARotaryDimension()
	return &MLAttention{
		QueryDown:      layers.NewLinear(config.EmbedDim, config.MLARank()),
		QueryUp:        layers.NewLinear(config.MLARank(), config.NumHead*(headDimension-ropeDim)),
		QueryRope:      layers.NewLinear(config.EmbedDim, config.NumHead*ropeDim),
		KVDown:         layers.NewLinear(config.EmbedDim, config.MLARank()),
		KeyUp:          layers.NewLinear(config.MLARank(), config.NumKVHead*(headDimension-ropeDim)),
		ValueUp:        layers.NewLinear(config.MLARank(), config.NumKVHead*headDimension),
		KeyRope:        layers.NewLinear(config.EmbedDim, config.NumKVHead*ropeDim),
		queryHeadCount: config.NumHead,
		kvHeadCount:    config.NumKVHead,
		contentDim:     headDimension - ropeDim,
		valueDim:       headDimension,
		ropeDim:        ropeDim,
		latentDim:      config.MLARank(),
	}
}

// parameters returns the MLA weight tensors in a deterministic order.
func (mla *MLAttention) parameters() []*tensors.Tensor {
	return []*tensors.Tensor{
		mla.QueryDown.Weight,
		mla.QueryUp.Weight,
		mla.QueryRope.Weight,
		mla.KVDown.Weight,
		mla.KeyUp.Weight,
		mla.ValueUp.Weight,
		mla.KeyRope.Weight,
	}
}

// numParameters returns the total number of MLA weights.
func (mla *MLAttention) numParameters() int {
	total := 0
	for _, parameter := range mla.parameters() {
		total += parameter.Numel()
	}
	return total
}

// mlaParamsForConfig returns the per-layer MLA matrix-parameter count without
// building a model, for the scaling-law and FLOP formulas.
func mlaParamsForConfig(config Config) int {
	embeddingDimension := config.EmbedDim
	headDimension := config.HeadDim()
	ropeDim := config.MLARotaryDimension()
	contentDim := headDimension - ropeDim
	latent := config.MLARank()
	queryHeads := config.NumHead
	kvHeads := config.NumKVHead
	return latent*embeddingDimension +
		queryHeads*contentDim*latent +
		queryHeads*ropeDim*embeddingDimension +
		latent*embeddingDimension +
		kvHeads*contentDim*latent +
		kvHeads*headDimension*latent +
		kvHeads*ropeDim*embeddingDimension
}
