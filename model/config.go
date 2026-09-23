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
	// RotaryDims is the number of trailing head dimensions that receive RoPE
	// (partial rotary embedding, as used by DeepSeek-V4). Zero means the full
	// head dimension, preserving the historical behavior.
	RotaryDims int `json:"rotary_dims,omitempty"`
	// CompressionRatio enables HCA-style dense KV compression on every layer
	// when > 1: each ratio consecutive key/value rows are merged into one
	// compressed entry, and queries attend to strictly preceding compressed
	// blocks. Zero or one disables compression.
	CompressionRatio int `json:"compression_ratio,omitempty"`
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

// CEDSplit returns the first decoder layer index, or 0 when CED is disabled.
// The encoder is layers [0, CEDSplit) and the decoder is [CEDSplit, NumLayer).
func (config Config) CEDSplit() int {
	if !config.CED {
		return 0
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

// ReuseModeAt returns the reuse mode of a layer, cycling the configured
// pattern. An empty pattern yields ReuseFull for every layer. Under CED the
// pattern is applied independently to the encoder and decoder halves.
func (config Config) ReuseModeAt(layer int) ReuseMode {
	pattern := config.ReusePattern
	if len(pattern) == 0 {
		return ReuseFull
	}
	start := config.reuseHalfStart(layer)
	switch pattern[(layer-start)%len(pattern)] {
	case 'R', 'r':
		return ReuseReindex
	case 'U', 'u':
		return ReuseReuse
	default:
		return ReuseFull
	}
}

// OwnsCompressed reports whether a layer produces its own compressed KV rather
// than borrowing the producing layer's.
func (config Config) OwnsCompressed(layer int) bool {
	return config.ReuseModeAt(layer) == ReuseFull
}

// OwnsIndexer reports whether a layer owns a lightning indexer. Full and
// reindex layers do when sparsity is enabled; pure reuse layers do not.
func (config Config) OwnsIndexer(layer int) bool {
	return config.SparseTopK > 0 && config.ReuseModeAt(layer) != ReuseReuse
}

// ReuseProducer returns the index of the producing full layer for a reuse
// layer, or the layer itself when it is full. It assumes Validate has passed,
// which guarantees at least one preceding full layer.
func (config Config) ReuseProducer(layer int) int {
	if config.ReuseModeAt(layer) == ReuseFull {
		return layer
	}
	for index := layer - 1; index >= config.reuseHalfStart(layer); index-- {
		if config.ReuseModeAt(index) == ReuseFull {
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

// Compression returns the effective KV compression ratio (>= 1).
func (config Config) Compression() int {
	if config.CompressionRatio <= 1 {
		return 1
	}
	return config.CompressionRatio
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
	if config.SparseTopK > 0 && config.Compression() <= 1 {
		panic("model: SparseTopK requires CompressionRatio > 1")
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
	if config.ReusePattern != "" {
		if config.Compression() <= 1 {
			panic("model: ReusePattern requires CompressionRatio > 1")
		}
		for _, patternChar := range config.ReusePattern {
			switch patternChar {
			case 'F', 'R', 'U', 'f', 'r', 'u':
			default:
				panic(fmt.Sprintf("model: invalid reuse pattern %q", config.ReusePattern))
			}
		}
		if config.ReuseModeAt(0) != ReuseFull {
			panic(fmt.Sprintf("model: reuse pattern %q must start with F", config.ReusePattern))
		}
	}
	if config.CED {
		if config.Compression() <= 1 {
			panic("model: CED requires CompressionRatio > 1")
		}
		if config.SWAWindowSize() <= 0 {
			panic("model: CED requires SWAWindow > 0")
		}
		if config.NumLayer < 2 {
			panic(fmt.Sprintf("model: CED requires NumLayer >= 2, got %d", config.NumLayer))
		}
		if config.ReusePattern != "" && config.ReuseModeAt(config.CEDSplit()) != ReuseFull {
			panic(fmt.Sprintf("model: CED reuse pattern %q must start each half with F", config.ReusePattern))
		}
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
		patternChar := pattern[layerIndex%len(pattern)]
		switch patternChar {
		case 'L', 'l':
			sizes[layerIndex] = [2]int{longWindow, 0}
		case 'S', 's':
			sizes[layerIndex] = [2]int{shortWindow, 0}
		}
	}
	sizes[config.NumLayer-1] = [2]int{longWindow, 0}
	return sizes
}
