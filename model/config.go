// Package model implements the nanochat GPT transformer: rotary embeddings,
// QK-norm, untied token embedding/lm_head, ReLU² MLP, Group-Query Attention,
// value embeddings (ResFormer), and sliding-window attention.
package model

import "fmt"

// Config describes a GPT transformer. It mirrors nanochat's GPTConfig.
type Config struct {
	SequenceLen   int    `json:"sequence_len"`   // maximum context length T
	VocabSize     int    `json:"vocab_size"`     // size of the token vocabulary
	NumLayer      int    `json:"n_layer"`        // number of transformer blocks (depth)
	NumHead       int    `json:"n_head"`         // number of query heads
	NumKVHead     int    `json:"n_kv_head"`      // number of key/value heads (GQA)
	EmbedDim      int    `json:"n_embd"`         // transformer width (n_embd)
	WindowPattern string `json:"window_pattern"` // sliding-window pattern (L=long, S=short)
	// RopeTheta is the RoPE frequency base (default 100000).
	RopeTheta float64 `json:"rope_theta,omitempty"`
	// CompressRopeTheta is the frequency base for compressed layers; zero uses
	// RopeTheta.
	CompressRopeTheta float64 `json:"compress_rope_theta,omitempty"`
	// RopeScaling selects the rotary scaling mode: empty/"none" (base RoPE) or
	// "yarn".
	RopeScaling string `json:"rope_scaling,omitempty"`
	// RopeScalingFactor is the YaRN context-extension factor.
	RopeScalingFactor float64 `json:"rope_scaling_factor,omitempty"`
	// RopeOriginalMaxPosition is the original trained context length for YaRN.
	RopeOriginalMaxPosition int `json:"rope_original_max_position,omitempty"`
	// RopeBetaFast and RopeBetaSlow are the YaRN correction bounds.
	RopeBetaFast float64 `json:"rope_beta_fast,omitempty"`
	RopeBetaSlow float64 `json:"rope_beta_slow,omitempty"`
	// RopeMScale applies the YaRN magnitude scale to the rotary tables.
	RopeMScale bool `json:"rope_mscale,omitempty"`
	// RotaryDims is the number of trailing head dimensions that receive RoPE
	// (partial rotary embedding, as used by DeepSeek-V4). Zero means the full
	// head dimension, preserving the historical behavior.
	RotaryDims int `json:"rotary_dims,omitempty"`
	// CompressionRatio enables HCA-style dense KV compression on every layer
	// when > 1: each ratio consecutive key/value rows are merged into one
	// compressed entry, and queries attend to strictly preceding compressed
	// blocks. Zero or one disables compression. CompressionRatios, when set,
	// overrides this field with a per-layer layout.
	CompressionRatio int `json:"compression_ratio,omitempty"`
	// CompressionRatios is the optional per-layer compression layout used by
	// DeepSeek-V4.1-style models. Each entry is one layer's ratio:
	//   0 - sliding-window-only layer (no packed global KV)
	//   1 - C1 packed KV layer (identity compression, no row merging)
	//  >=2 - compressed layer (each ratio rows merge into one entry)
	// When empty, CompressionRatio applies uniformly to every layer, preserving
	// the legacy checkpoint behavior.
	CompressionRatios []int `json:"compression_ratios,omitempty"`
	// PrefillLayer is the first decoder/consumer layer of the causal
	// encoder-decoder split (DeepSeek-V4.1 YOCO). It is only meaningful when
	// CED is enabled; zero derives the historical NumLayer/2 split.
	PrefillLayer int `json:"prefill_layer,omitempty"`
	// SparseTopK enables CSA-style sparse attention on compressed layers: each
	// query attends only to the top-k compressed blocks selected by the
	// lightning indexer. Zero disables sparsity (dense compressed attention).
	SparseTopK int `json:"sparse_top_k,omitempty"`
	// IndexerDim is the per-head dimension of the lightning indexer.
	IndexerDim int `json:"indexer_dim,omitempty"`
	// IndexerHeads is the number of indexer query heads.
	IndexerHeads int `json:"indexer_heads,omitempty"`
	// IndexerLossWeight scales the indexer distillation loss used to train the
	// sparse selection. Zero uses the default of 1.
	IndexerLossWeight float32 `json:"indexer_loss_weight,omitempty"`
	// IndexerPool enables the hierarchical (coarse-to-fine) indexer: compressed
	// entries are pooled into super-blocks of this size before fine scoring.
	// Zero or one disables it.
	IndexerPool int `json:"indexer_pool,omitempty"`
	// IndexerCandidates bounds the number of entries fully scored per token by
	// the hierarchical indexer. Zero uses 8*SparseTopK (minimum 64).
	IndexerCandidates int `json:"indexer_candidates,omitempty"`
	// ReusePattern enables cross-layer compressed KV/index reuse
	// (DeepSeek-V4.1 §2.3.1). It is cycled per layer: 'F' (full) owns a
	// compressor and produces compressed KV and, when sparse, the top-k
	// selection; 'R' (reindex) reuses the group's compressed KV but runs its
	// own indexer for fresh top-k; 'U' (reuse) reuses both the compressed KV
	// and the most recently published selection. The pattern must start with
	// 'F'. Empty or all-'F' keeps the historical all-layers-full behaviour.
	ReusePattern string `json:"reuse_pattern,omitempty"`
	// SWAWindow enables the local sliding-window attention branch on
	// compressed layers (DeepSeek-V4.1 §2.2). Each compressed query attends to
	// the global compressed blocks together with the raw keys/values in the
	// preceding window. Zero disables the local branch, preserving the
	// historical global-only compressed attention. It is ignored unless
	// CompressionRatio > 1.
	SWAWindow int `json:"swa_window,omitempty"`
	// CED enables the Causal Encoder-Decoder split (DeepSeek-V4.1 §2.2). The
	// bottom half of the layers is a causal encoder; the top half is a decoder
	// whose global (compressed) keys/values are projected from the encoder's
	// final hidden state rather than from the decoder layer's own hidden state.
	// The decoder's local sliding-window branch still reads its own hidden
	// state. CED requires compression and a local window.
	CED bool `json:"ced,omitempty"`
	// AttentionSink enables a learnable per-query-head sink logit in the
	// attention softmax (DeepSeek-V4.1 §2.3). The sink participates in the
	// denominator but has a zero value vector, so it absorbs probability mass
	// without contributing to the output.
	AttentionSink bool `json:"attention_sink,omitempty"`
	// AttentionSinkInit is the initial sink logit value (default 0).
	AttentionSinkInit float32 `json:"attention_sink_init,omitempty"`
	// HeadWiseMuon splits the query and key projection weights by attention
	// head and orthogonalizes each head independently with Muon (DeepSeek-V4.1
	// §2.5). It is a training-only setting; it adds no inference cost.
	HeadWiseMuon bool `json:"head_wise_muon,omitempty"`
	// QueryCompressionDim enables a low-rank (MLA-style) query projection: the
	// query is down-projected to this width and then up-projected per head
	// (DeepSeek-V4.1 §2.3/§4.2.1). Zero keeps the full-rank projection.
	QueryCompressionDim int `json:"query_compression_dim,omitempty"`
	// KVLatentDim enables a shared low-rank KV latent: key and value are
	// projected from a single latent of this width instead of from the hidden
	// state (MLA). Zero keeps the full-rank projections. The KV cache still
	// stores the full per-head keys/values, so this reduces projection compute
	// and parameters, not cache bytes.
	KVLatentDim int `json:"kv_latent_dim,omitempty"`
	// MLALatent enables absorbed Multi-head Latent Attention: content keys and
	// values are up-projected from a shared latent of this width, a decoupled
	// rotary key is stored alongside it, and the up-projections are absorbed
	// into the query and output so attention reads the latent directly
	// (DeepSeek-V4.1 §2.3/§4.2.1). It reduces both parameters and KV-cache
	// bytes. It is a genuine architectural change and must be trained from
	// scratch; the full-rank, low-rank, compressed, and CED paths are separate
	// modes. Zero disables it.
	MLALatent int `json:"mla_latent,omitempty"`
	// MLARotaryDims is the decoupled rotary width of the MLA key/query. Zero
	// uses RotaryDimension.
	MLARotaryDims int `json:"mla_rotary_dims,omitempty"`
	// NumExperts > 0 enables the DeepSeekMoE feed-forward in every block
	// (DeepSeek-V4.1 §2.1, §4.2.1): one shared expert plus NumExperts routed
	// experts, of which NumExpertsPerToken are active per token. Zero keeps the
	// historical dense ReLU² MLP. It is orthogonal to the attention design and
	// composes with GQA, compression, SWA, CED, and MLA.
	NumExperts int `json:"num_experts,omitempty"`
	// NumExpertsPerToken is the number of routed experts activated per token
	// (top-k). It must be in [1, NumExperts].
	NumExpertsPerToken int `json:"num_experts_per_token,omitempty"`
	// ExpertHiddenDim is the intermediate width of every routed expert
	// (DeepSeek's fine-grained experts are much narrower than the 4x dense
	// MLP). Zero defaults to one attention width (Config.EmbedDim).
	ExpertHiddenDim int `json:"expert_hidden_dim,omitempty"`
	// SharedExpertHiddenDim is the intermediate width of the always-active
	// shared expert. Zero defaults to ExpertHiddenDim.
	SharedExpertHiddenDim int `json:"shared_expert_hidden_dim,omitempty"`
	// MoEScoring selects the router scoring function: empty/"softmax" (legacy)
	// or "sqrtsoftplus" (DeepSeek-V4.1).
	MoEScoring string `json:"moe_scoring,omitempty"`
	// MoETopKMethod selects the selection method: empty/"noaux_tc" keeps the
	// DeepSeek-V4.1 correction-bias selection.
	MoETopKMethod string `json:"moe_topk_method,omitempty"`
	// MoENormTopKProb normalizes the selected routed weights to sum to one.
	MoENormTopKProb bool `json:"moe_norm_topk_prob,omitempty"`
	// MoEClamp is the SwiGLU clamp threshold; gate is clamped above it and up
	// into [-MoEClamp, MoEClamp]. Zero uses the paper's default of 10; a
	// negative value disables clamping.
	MoEClamp float32 `json:"moe_clamp,omitempty"`
	// MoEScale multiplies the routed expert output before the shared expert is
	// added. Zero defaults to 1.
	MoEScale float32 `json:"moe_scale,omitempty"`
	// RouterBiasUpdate is the auxiliary-loss-free load-balancing bias update
	// speed (DeepSeek-V4.1 §2.1.1, §4.2.2). Zero defaults to 0.001.
	RouterBiasUpdate float32 `json:"router_bias_update,omitempty"`
	// NumNextNPredictLayers is the number of DFlash/MTP draft stages attached to
	// the backbone. Zero disables DFlash.
	NumNextNPredictLayers int `json:"num_nextn_predict_layers,omitempty"`
	// DFlashTargetLayerIDs lists the backbone layers whose hidden states feed
	// the DFlash projection.
	DFlashTargetLayerIDs []int `json:"dflash_target_layer_ids,omitempty"`
	// DFlashBlockSize is the number of draft tokens proposed per step.
	DFlashBlockSize int `json:"dflash_block_size,omitempty"`
	// DFlashNoiseToken is the placeholder token id used for draft positions.
	DFlashNoiseToken int `json:"dflash_noise_token,omitempty"`
	// DFlashMarkovRank is the low-rank width of the Markov correction head.
	DFlashMarkovRank int `json:"dflash_markov_rank,omitempty"`
	// MoEBalanceWeight is the weight of the small sequence-level load-balancing
	// auxiliary loss (DeepSeek-V4.1 §4.2.2, weight 1e-4) that prevents extreme
	// expert imbalance within a single training sequence. Zero uses the paper's
	// default of 1e-4; a negative value disables the loss, leaving only the
	// auxiliary-loss-free bias update.
	MoEBalanceWeight float32 `json:"moe_balance_weight,omitempty"`
}

// defaultMoEClamp is the SwiGLU clamp threshold from DeepSeek-V4.1 §4.2.1.
const defaultMoEClamp = 10.0

// defaultRouterBiasUpdate is the auxiliary-loss-free bias update speed from
// DeepSeek-V4.1 §4.2.2.
const defaultRouterBiasUpdate = 0.001

// defaultMoEBalanceWeight is the sequence-level load-balancing auxiliary loss
// weight from DeepSeek-V4.1 §4.2.2.
const defaultMoEBalanceWeight = 1e-4

// MoEEnabled reports whether the mixture-of-experts feed-forward is active.
func (config Config) MoEEnabled() bool { return config.NumExperts > 0 }

// MoEActiveExperts returns the number of routed experts selected per token.
func (config Config) MoEActiveExperts() int {
	if config.NumExpertsPerToken <= 0 {
		return 0
	}
	if config.NumExpertsPerToken > config.NumExperts {
		return config.NumExperts
	}
	return config.NumExpertsPerToken
}

// MoEEffectiveClamp returns the SwiGLU clamp threshold (default 10).
func (config Config) MoEEffectiveClamp() float32 {
	if config.MoEClamp == 0 {
		return defaultMoEClamp
	}
	return config.MoEClamp
}

// MoEEffectiveScale returns the routed output scale (default 1).
func (config Config) MoEEffectiveScale() float32 {
	if config.MoEScale == 0 {
		return 1
	}
	return config.MoEScale
}

// MoERouterBiasUpdate returns the bias update speed (default 0.001).
func (config Config) MoERouterBiasUpdate() float32 {
	if config.RouterBiasUpdate <= 0 {
		return defaultRouterBiasUpdate
	}
	return config.RouterBiasUpdate
}

// MoEBalanceLossWeight returns the effective sequence-level balance loss weight:
// negative disables it, zero selects the paper's default (1e-4), and a positive
// value overrides it.
func (config Config) MoEBalanceLossWeight() float32 {
	if config.MoEBalanceWeight < 0 {
		return 0
	}
	if config.MoEBalanceWeight == 0 {
		return defaultMoEBalanceWeight
	}
	return config.MoEBalanceWeight
}

// MoEExpertHidden returns the routed expert intermediate width, applying the
// default.
func (config Config) MoEExpertHidden() int {
	if config.ExpertHiddenDim > 0 {
		return config.ExpertHiddenDim
	}
	return config.EmbedDim
}

// MoESharedHidden returns the shared expert intermediate width, applying the
// default.
func (config Config) MoESharedHidden() int {
	if config.SharedExpertHiddenDim > 0 {
		return config.SharedExpertHiddenDim
	}
	return config.MoEExpertHidden()
}

// ApplyMoEDefaults fills the derived MoE hyperparameters from the depth
// (Config.NumLayer) and width when they are left unset. It mutates the config;
// callers that persist a config should call it before NewTransformer so the
// derived values are checkpointed.
func (config *Config) ApplyMoEDefaults() {
	if !config.MoEEnabled() {
		return
	}
	if config.ExpertHiddenDim <= 0 {
		config.ExpertHiddenDim = config.EmbedDim
	}
	if config.SharedExpertHiddenDim <= 0 {
		config.SharedExpertHiddenDim = config.ExpertHiddenDim
	}
	if config.NumExpertsPerToken <= 0 {
		config.NumExpertsPerToken = 2
	}
	if config.NumExpertsPerToken > config.NumExperts {
		config.NumExpertsPerToken = config.NumExperts
	}
}

// CEDEnabled reports whether the causal encoder-decoder split is active.
func (config Config) CEDEnabled() bool { return config.CED }

// QueryRank returns the low-rank query bottleneck width, or 0 when the query
// projection is full-rank.
func (config Config) QueryRank() int {
	if config.QueryCompressionDim <= 0 {
		return 0
	}
	return config.QueryCompressionDim
}

// KVRank returns the shared low-rank KV latent width, or 0 when the key/value
// projections are full-rank.
func (config Config) KVRank() int {
	if config.KVLatentDim <= 0 {
		return 0
	}
	return config.KVLatentDim
}

// MLAEnabled reports whether absorbed Multi-head Latent Attention is active.
func (config Config) MLAEnabled() bool { return config.MLALatent > 0 }

// MLARank returns the MLA shared latent width.
func (config Config) MLARank() int { return config.MLALatent }

// MLAContentDim returns the non-rotary content width, the part of the head that
// is up-projected from the shared latent.
func (config Config) MLAContentDim() int { return config.HeadDim() - config.MLARotaryDimension() }

// MLARotaryDimension returns the decoupled rotary width of the MLA key/query.
// Zero defaults to half the head dimension (capped at the standard 64), so the
// content width is always positive.
func (config Config) MLARotaryDimension() int {
	if config.MLARotaryDims <= 0 {
		width := config.HeadDim() / 2
		if width > defaultRotaryDims {
			width = defaultRotaryDims
		}
		if width%2 != 0 {
			width--
		}
		return width
	}
	if config.MLARotaryDims > config.HeadDim() {
		panic(fmt.Sprintf("model: MLARotaryDims %d exceeds head dim %d", config.MLARotaryDims, config.HeadDim()))
	}
	if config.MLARotaryDims%2 != 0 {
		panic(fmt.Sprintf("model: MLARotaryDims %d must be even", config.MLARotaryDims))
	}
	return config.MLARotaryDims
}

// CEDSplit returns the first decoder layer index, or 0 when CED is disabled.
// The encoder is layers [0, CEDSplit) and the decoder is [CEDSplit, NumLayer).
// An explicit PrefillLayer overrides the historical half-depth split.
func (config Config) CEDSplit() int {
	if !config.CED {
		return 0
	}
	if config.PrefillLayer > 0 && config.PrefillLayer < config.NumLayer {
		return config.PrefillLayer
	}
	return config.NumLayer / 2
}

// IsEncoderLayer reports whether a layer belongs to the CED causal encoder.
func (config Config) IsEncoderLayer(layer int) bool {
	split := config.CEDSplit()
	return split > 0 && layer < split
}

// IsDecoderLayer reports whether a layer belongs to the CED decoder.
func (config Config) IsDecoderLayer(layer int) bool {
	split := config.CEDSplit()
	return split > 0 && layer >= split
}

// ReuseMode is the per-layer compressed-attention reuse role.
type ReuseMode uint8

const (
	// ReuseFull owns a compressor/indexer and produces compressed KV/selection.
	ReuseFull ReuseMode = iota
	// ReuseReindex reuses the group's compressed KV, with its own selection.
	ReuseReindex
	// ReuseReuse reuses the group's compressed KV and selection.
	ReuseReuse
)

// reuseHalfStart returns the first layer of the reuse cycle that contains the
// given layer. Under CED the pattern restarts at the decoder split so that a
// decoder group never borrows an encoder layer's compressed KV. Without CED the
// cycle always starts at layer zero.
func (config Config) reuseHalfStart(layer int) int {
	if config.CED && layer >= config.CEDSplit() && config.CEDSplit() > 0 {
		return config.CEDSplit()
	}
	return 0
}

// LayeredCompression reports whether the config carries an explicit per-layer
// compression layout rather than the legacy uniform CompressionRatio.
func (config Config) LayeredCompression() bool {
	return len(config.CompressionRatios) > 0
}

// LayerRatioAt returns the effective compression ratio of one layer. A ratio
// of zero is a sliding-window-only layer, one is a C1 packed layer (no row
// merging), and values above one merge that many rows into one entry.
func (config Config) LayerRatioAt(layer int) int {
	if config.LayeredCompression() {
		if layer < 0 || layer >= len(config.CompressionRatios) {
			return 0
		}
		ratio := config.CompressionRatios[layer]
		if ratio < 0 {
			return 0
		}
		return ratio
	}
	if layer < 0 || layer >= config.NumLayer {
		return config.Compression()
	}
	return config.Compression()
}

// HasCompressionAt reports whether a layer uses the packed/global KV path. C1
// layers (ratio 1) have packed KV but do not merge rows.
func (config Config) HasCompressionAt(layer int) bool {
	if config.LayeredCompression() {
		return config.LayerRatioAt(layer) >= 1
	}
	return config.Compression() > 1
}

// CompressesAt reports whether a layer actually merges rows.
func (config Config) CompressesAt(layer int) bool {
	return config.LayerRatioAt(layer) > 1
}

// SparseAt reports whether a layer owns the lightning indexer.
func (config Config) SparseAt(layer int) bool {
	return config.SparseTopK > 0 && config.HasCompressionAt(layer)
}

// AnyCompression reports whether at least one layer uses the packed global KV
// path.
func (config Config) AnyCompression() bool {
	if !config.LayeredCompression() {
		return config.Compression() > 1
	}
	for layer := range config.CompressionRatios {
		if config.CompressionRatios[layer] >= 1 {
			return true
		}
	}
	return false
}

// AnySparse reports whether at least one layer owns a lightning indexer.
func (config Config) AnySparse() bool {
	if config.SparseTopK <= 0 {
		return false
	}
	if !config.LayeredCompression() {
		return config.Compression() > 1
	}
	for layer := range config.CompressionRatios {
		if config.CompressionRatios[layer] >= 1 {
			return true
		}
	}
	return false
}

// LayerRatios returns the effective per-layer compression layout. Legacy
// configs are expanded to a uniform list so cache allocation can use one path.
func (config Config) LayerRatios() []int {
	ratios := make([]int, config.NumLayer)
	for layer := 0; layer < config.NumLayer; layer++ {
		ratios[layer] = config.LayerRatioAt(layer)
	}
	return ratios
}

// CompressedMaxBlocks returns the maximum number of compressed entries each
// layer can hold for a cache capacity of capacity tokens.
func (config Config) CompressedMaxBlocks(capacity int) []int {
	maxBlocks := make([]int, config.NumLayer)
	for layer := 0; layer < config.NumLayer; layer++ {
		ratio := config.LayerRatioAt(layer)
		if ratio < 1 {
			continue
		}
		maxBlocks[layer] = capacity/ratio + 1
	}
	// A reindex/reuse layer shares its producer's compressed KV and, for
	// reindex layers, caches indexer keys for every producer entry. Its
	// buffers must therefore cover the producer's capacity even when its own
	// compression ratio is higher.
	for layer := 0; layer < config.NumLayer; layer++ {
		if !config.HasCompressionAt(layer) || config.OwnsCompressed(layer) {
			continue
		}
		producer := config.ReuseProducer(layer)
		if producer >= 0 && producer < len(maxBlocks) && maxBlocks[producer] > maxBlocks[layer] {
			maxBlocks[layer] = maxBlocks[producer]
		}
	}
	return maxBlocks
}

// ReuseModeAt returns the reuse mode of a compressed layer, cycling the
// configured pattern over the compressed layers of its encoder/decoder segment.
// An empty pattern yields ReuseFull for every layer. Sliding-window-only layers
// are never reuse layers and always return ReuseFull.
func (config Config) ReuseModeAt(layer int) ReuseMode {
	if !config.HasCompressionAt(layer) {
		return ReuseFull
	}
	pattern := config.ReusePattern
	if len(pattern) == 0 {
		return ReuseFull
	}
	start := config.reuseHalfStart(layer)
	position := 0
	for index := start; index < layer; index++ {
		if config.HasCompressionAt(index) {
			position++
		}
	}
	switch pattern[position%len(pattern)] {
	case 'R', 'r':
		return ReuseReindex
	case 'U', 'u':
		return ReuseReuse
	default:
		return ReuseFull
	}
}

// OwnsCompressed reports whether a layer produces its own compressed KV rather
// than borrowing the producing layer's. Sliding-window-only layers own no
// compressed KV.
func (config Config) OwnsCompressed(layer int) bool {
	return config.HasCompressionAt(layer) && config.ReuseModeAt(layer) == ReuseFull
}

// OwnsIndexer reports whether a layer owns a lightning indexer. Full and
// reindex compressed layers do when sparsity is enabled; pure reuse layers do
// not.
func (config Config) OwnsIndexer(layer int) bool {
	return config.SparseAt(layer) && config.ReuseModeAt(layer) != ReuseReuse
}

// ReuseProducer returns the index of the producing full compressed layer for a
// reuse layer, or the layer itself when it is full. It assumes Validate has
// passed, which guarantees a preceding full compressed layer in the segment.
func (config Config) ReuseProducer(layer int) int {
	if config.ReuseModeAt(layer) == ReuseFull {
		return layer
	}
	for index := layer - 1; index >= config.reuseHalfStart(layer); index-- {
		if config.HasCompressionAt(index) && config.ReuseModeAt(index) == ReuseFull {
			return index
		}
	}
	return layer
}

// IndexerCandidateBudget returns the hierarchical indexer candidate budget.
func (config Config) IndexerCandidateBudget() int {
	if config.IndexerCandidates > 0 {
		return config.IndexerCandidates
	}
	budget := 8 * config.SparseTopK
	if budget < 64 {
		budget = 64
	}
	return budget
}

// IndexerWeight returns the effective indexer distillation loss weight.
func (config Config) IndexerWeight() float32 {
	if config.IndexerLossWeight <= 0 {
		return 1.0
	}
	return config.IndexerLossWeight
}

// indexerDefaults returns the indexer dimension and head count, applying
// defaults when unset.
func (config Config) indexerDefaults() (dim, heads int) {
	dim = config.IndexerDim
	if dim <= 0 {
		dim = 64
	}
	heads = config.IndexerHeads
	if heads <= 0 {
		heads = 1
	}
	return dim, heads
}

// Compression returns the maximum effective KV compression ratio (>= 1).
// Legacy configs use the uniform CompressionRatio; layered configs report the
// widest layer ratio for accounting and legacy callers.
func (config Config) Compression() int {
	maxRatio := 1
	if config.LayeredCompression() {
		for _, ratio := range config.CompressionRatios {
			if ratio > maxRatio {
				maxRatio = ratio
			}
		}
		return maxRatio
	}
	if config.CompressionRatio > maxRatio {
		return config.CompressionRatio
	}
	return maxRatio
}

// SWAWindowSize returns the local sliding-window width used by compressed
// layers. Zero disables the local branch.
func (config Config) SWAWindowSize() int {
	if config.SWAWindow < 0 {
		return 0
	}
	return config.SWAWindow
}

// defaultRotaryDims is the partial-RoPE width used by ConfigForDepthRatio when
// the head dimension allows it, matching DeepSeek-V4's 64-dim rotary slice.
const defaultRotaryDims = 64

// RotaryDimension returns the number of trailing head dimensions that receive
// RoPE. It validates that the value is even and does not exceed the head
// dimension.
func (config Config) RotaryDimension() int {
	if config.MLAEnabled() {
		return config.MLARotaryDimension()
	}
	if config.RotaryDims <= 0 {
		return config.HeadDim()
	}
	if config.RotaryDims > config.HeadDim() {
		panic(fmt.Sprintf("model: RotaryDims %d exceeds head dim %d", config.RotaryDims, config.HeadDim()))
	}
	if config.RotaryDims%2 != 0 {
		panic(fmt.Sprintf("model: RotaryDims %d must be even", config.RotaryDims))
	}
	return config.RotaryDims
}

// EffectiveRopeTheta returns the base rotary frequency, defaulting to 100000.
func (config Config) EffectiveRopeTheta() float64 {
	if config.RopeTheta > 0 {
		return config.RopeTheta
	}
	return 100000
}

// EffectiveCompressRopeTheta returns the compressed-layer rotary frequency.
func (config Config) EffectiveCompressRopeTheta() float64 {
	if config.CompressRopeTheta > 0 {
		return config.CompressRopeTheta
	}
	return config.EffectiveRopeTheta()
}

// ropeScalingConfig returns the YaRN configuration used for the rotary tables.
func (config Config) ropeScalingConfig() RopeScalingConfig {
	mode := config.RopeScaling
	if mode == "none" {
		mode = ""
	}
	return RopeScalingConfig{
		Mode:                mode,
		Factor:              config.RopeScalingFactor,
		OriginalMaxPosition: config.RopeOriginalMaxPosition,
		BetaFast:            config.RopeBetaFast,
		BetaSlow:            config.RopeBetaSlow,
		MScale:              config.RopeMScale,
	}
}

// RopeThetaForLayer returns the rotary frequency of one layer.
func (config Config) RopeThetaForLayer(layer int) float64 {
	if config.CompressesAt(layer) {
		return config.EffectiveCompressRopeTheta()
	}
	return config.EffectiveRopeTheta()
}

// HeadDim returns the per-head dimension d = n_embd / n_head.
func (config Config) HeadDim() int { return config.EmbedDim / config.NumHead }

// KVHeadDim returns the KV per-head dimension, equal to HeadDim by construction.
func (config Config) KVHeadDim() int { return config.EmbedDim / config.NumHead }

// Validate panics if the config is inconsistent.
func (config Config) Validate() {
	if config.EmbedDim%config.NumHead != 0 {
		panic(fmt.Sprintf("model: EmbedDim %d not divisible by NumHead %d", config.EmbedDim, config.NumHead))
	}
	if config.NumKVHead > config.NumHead || config.NumHead%config.NumKVHead != 0 {
		panic(fmt.Sprintf("model: NumKVHead %d must divide NumHead %d", config.NumKVHead, config.NumHead))
	}
	for _, patternChar := range config.WindowPattern {
		if patternChar != 'L' && patternChar != 'S' && patternChar != 'l' && patternChar != 's' {
			panic(fmt.Sprintf("model: invalid window pattern %q", config.WindowPattern))
		}
	}
	if config.LayeredCompression() {
		if len(config.CompressionRatios) != config.NumLayer {
			panic(fmt.Sprintf("model: CompressionRatios length %d != NumLayer %d", len(config.CompressionRatios), config.NumLayer))
		}
		for layer, ratio := range config.CompressionRatios {
			if ratio < 0 {
				panic(fmt.Sprintf("model: layer %d compression ratio %d must be >= 0", layer, ratio))
			}
			if ratio == 1 && config.SWAWindowSize() <= 0 {
				panic(fmt.Sprintf("model: C1 layer %d requires SWAWindow > 0", layer))
			}
		}
	}
	if config.PrefillLayer < 0 || config.PrefillLayer > config.NumLayer {
		panic(fmt.Sprintf("model: PrefillLayer %d must be in [0, NumLayer=%d]", config.PrefillLayer, config.NumLayer))
	}
	if config.PrefillLayer > 0 && !config.CED {
		panic("model: PrefillLayer requires CED")
	}
	if config.SparseTopK > 0 && !config.AnyCompression() {
		panic("model: SparseTopK requires at least one packed/compressed layer")
	}
	if config.SWAWindow < 0 {
		panic(fmt.Sprintf("model: SWAWindow %d must be >= 0", config.SWAWindow))
	}
	if config.QueryCompressionDim < 0 {
		panic(fmt.Sprintf("model: QueryCompressionDim %d must be >= 0", config.QueryCompressionDim))
	}
	if config.KVLatentDim < 0 {
		panic(fmt.Sprintf("model: KVLatentDim %d must be >= 0", config.KVLatentDim))
	}
	if config.MLALatent < 0 {
		panic(fmt.Sprintf("model: MLALatent %d must be >= 0", config.MLALatent))
	}
	if config.MLARotaryDims < 0 {
		panic(fmt.Sprintf("model: MLARotaryDims %d must be >= 0", config.MLARotaryDims))
	}
	if config.MLAEnabled() {
		if config.QueryCompressionDim > 0 || config.KVLatentDim > 0 {
			panic("model: MLA is incompatible with QueryCompressionDim/KVLatentDim")
		}
		if config.HeadWiseMuon {
			panic("model: MLA is not supported with head-wise Muon")
		}
		if config.MLAContentDim() <= 0 {
			panic("model: MLA content dimension must be positive")
		}
	}
	if config.ReusePattern != "" {
		if !config.AnyCompression() {
			panic("model: ReusePattern requires at least one packed/compressed layer")
		}
		for _, patternChar := range config.ReusePattern {
			switch patternChar {
			case 'F', 'R', 'U', 'f', 'r', 'u':
			default:
				panic(fmt.Sprintf("model: invalid reuse pattern %q", config.ReusePattern))
			}
		}
		starts := []int{0}
		if config.CED && config.CEDSplit() > 0 {
			starts = append(starts, config.CEDSplit())
		}
		for _, start := range starts {
			for layer := start; layer < config.NumLayer; layer++ {
				if config.HasCompressionAt(layer) {
					if config.ReuseModeAt(layer) != ReuseFull {
						panic(fmt.Sprintf("model: reuse pattern %q must start each segment with F (layer %d)", config.ReusePattern, layer))
					}
					break
				}
			}
		}
	}
	if config.CED {
		for layer := config.CEDSplit(); layer < config.NumLayer; layer++ {
			if !config.HasCompressionAt(layer) {
				panic(fmt.Sprintf("model: CED decoder layer %d requires packed/compressed KV", layer))
			}
		}
		if config.SWAWindowSize() <= 0 {
			panic("model: CED requires SWAWindow > 0")
		}
		if config.NumLayer < 2 {
			panic(fmt.Sprintf("model: CED requires NumLayer >= 2, got %d", config.NumLayer))
		}
	}
	if config.NumExperts < 0 {
		panic(fmt.Sprintf("model: NumExperts %d must be >= 0", config.NumExperts))
	}
	if config.NumExperts > 0 {
		if config.NumExpertsPerToken < 0 || config.NumExpertsPerToken > config.NumExperts {
			panic(fmt.Sprintf("model: NumExpertsPerToken %d must be in [0, NumExperts=%d]", config.NumExpertsPerToken, config.NumExperts))
		}
		if config.ExpertHiddenDim < 0 {
			panic(fmt.Sprintf("model: ExpertHiddenDim %d must be >= 0", config.ExpertHiddenDim))
		}
		if config.SharedExpertHiddenDim < 0 {
			panic(fmt.Sprintf("model: SharedExpertHiddenDim %d must be >= 0", config.SharedExpertHiddenDim))
		}
		if config.MoEScale < 0 {
			panic(fmt.Sprintf("model: MoEScale %v must be >= 0", config.MoEScale))
		}
		if config.RouterBiasUpdate < 0 {
			panic(fmt.Sprintf("model: RouterBiasUpdate %v must be >= 0", config.RouterBiasUpdate))
		}
	} else if config.NumExpertsPerToken != 0 || config.ExpertHiddenDim != 0 || config.SharedExpertHiddenDim != 0 {
		panic("model: MoE hyperparameters set without NumExperts > 0")
	}
}

// ConfigForDepth derives a compute-optimal config from a single dial: the
// depth. model_dim = ceil(depth*aspectRatio / headDim) * headDim, and heads =
// model_dim / headDim. All other hyperparameters (batch size, learning rates,
// horizons) are derived later from the scaling laws in package trainer.
//
// It uses full multi-head attention (one key/value head per query head). Use
// ConfigForDepthRatio to enable grouped-query attention.
func ConfigForDepth(depth, vocabSize, aspectRatio, headDim, seqLen int, windowPattern string) Config {
	return ConfigForDepthRatio(depth, vocabSize, aspectRatio, headDim, seqLen, windowPattern, 1)
}

// ConfigForDepthRatio is ConfigForDepth with an explicit grouped-query
// attention ratio: there is exactly one key/value head per kvHeadRatio query
// heads. A ratio of 1 is full multi-head attention; a ratio equal to the query
// head count is single-head multi-query attention. The query head count must
// be divisible by the ratio.
func ConfigForDepthRatio(depth, vocabSize, aspectRatio, headDim, seqLen int, windowPattern string, kvHeadRatio int) Config {
	if kvHeadRatio < 1 {
		panic(fmt.Sprintf("model: kvHeadRatio %d must be >= 1", kvHeadRatio))
	}
	baseDimension := depth * aspectRatio
	modelDimension := ((baseDimension + headDim - 1) / headDim) * headDim
	numHeads := modelDimension / headDim
	if numHeads%kvHeadRatio != 0 {
		panic(fmt.Sprintf("model: numHeads %d not divisible by kvHeadRatio %d", numHeads, kvHeadRatio))
	}
	rotaryDims := defaultRotaryDims
	if headDim < rotaryDims {
		rotaryDims = headDim
	}
	if rotaryDims%2 != 0 {
		rotaryDims--
	}
	config := Config{
		SequenceLen:   seqLen,
		VocabSize:     vocabSize,
		NumLayer:      depth,
		NumHead:       numHeads,
		NumKVHead:     numHeads / kvHeadRatio,
		EmbedDim:      modelDimension,
		WindowPattern: windowPattern,
		RotaryDims:    rotaryDims,
	}
	config.Validate()
	return config
}

// MoEExpertsForDepth derives the routed expert count from the depth dial. Two
// experts per layer (floored at 8) keeps the routed-parameter growth
// proportional to depth, in the spirit of nanochat's single-dial scaling.
func MoEExpertsForDepth(depth int) int {
	experts := 2 * depth
	if experts < 8 {
		experts = 8
	}
	return experts
}

// vocabPaddingTo is the multiple to which the vocabulary is padded for
// efficient embedding/matmul alignment (nanochat uses 64).
const vocabPaddingTo = 64

// PaddedVocab returns vocab size rounded up to the nearest multiple of 64.
func (config Config) PaddedVocab() int {
	return ((config.VocabSize + vocabPaddingTo - 1) / vocabPaddingTo) * vocabPaddingTo
}

// WindowSizes computes the per-layer (left, right) sliding-window sizes. The
// final layer always gets full context. S = quarter context, rounded up to a
// multiple of 128. Returns one [2]int per layer, {left, 0}.
func (config Config) WindowSizes() [][2]int {
	pattern := config.WindowPattern
	longWindow := config.SequenceLen
	shortWindow := -((-config.SequenceLen / 4) / 128) * 128 // ceil(T/4/128)*128
	sizes := make([][2]int, config.NumLayer)
	for layerIndex := 0; layerIndex < config.NumLayer; layerIndex++ {
		// A layered V4.1 encoder uses sliding-window-only layers. Compressed
		// layers use the global compressed branch plus the local SWAWindow, so
		// this per-layer window is only consumed by non-compressed layers.
		if config.LayeredCompression() && !config.HasCompressionAt(layerIndex) && config.SWAWindowSize() > 0 {
			sizes[layerIndex] = [2]int{config.SWAWindowSize(), 0}
			continue
		}
		if len(pattern) == 0 {
			sizes[layerIndex] = [2]int{longWindow, 0}
			continue
		}
		patternChar := pattern[layerIndex%len(pattern)]
		switch patternChar {
		case 'L', 'l':
			sizes[layerIndex] = [2]int{longWindow, 0}
		case 'S', 's':
			sizes[layerIndex] = [2]int{shortWindow, 0}
		}
	}
	if !config.LayeredCompression() || config.HasCompressionAt(config.NumLayer-1) {
		sizes[config.NumLayer-1] = [2]int{longWindow, 0}
	}
	return sizes
}

// RingWindows returns the per-layer local sliding-window width that can be
// served from a bounded ring instead of storing every raw K/V row. Zero means
// the layer needs the full-context raw cache (or has no local branch).
func (config Config) RingWindows() []int {
	windows := config.WindowSizes()
	result := make([]int, config.NumLayer)
	for layer := 0; layer < config.NumLayer; layer++ {
		if config.HasCompressionAt(layer) {
			// Compressed layers read raw K/V only for the local branch.
			if config.SWAWindowSize() > 0 && config.SWAWindowSize() < config.SequenceLen {
				result[layer] = config.SWAWindowSize()
			}
			continue
		}
		if layer < len(windows) {
			width := windows[layer][0]
			if width > 0 && width < config.SequenceLen {
				result[layer] = width
			}
		}
	}
	return result
}
