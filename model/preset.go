package model

import (
	"fmt"
	"strings"
)

// Preset is a named set of architecture defaults. The recommended preset,
// PresetFlash, turns on the DeepSeek-V4.1-Flash long-context stack; the other
// presets exist as escape hatches and alternate architectures.
type Preset string

const (
	// PresetFlash is the recommended DeepSeek-V4.1-Flash long-context stack:
	// HCA KV compression, CSA sparse attention with the hierarchical indexer,
	// cross-layer KV/index reuse, the local sliding window, the causal
	// encoder-decoder split, SwiGLU DeepSeekMoE, grouped-query attention,
	// head-wise Muon, and partial RoPE.
	PresetFlash Preset = "flash"
	// PresetLatent uses absorbed Multi-head Latent Attention instead of the
	// compression stack (the two are mutually exclusive), together with SwiGLU
	// DeepSeekMoE and grouped-query attention.
	PresetLatent Preset = "latent"
	// PresetDense is the historical dense nanochat decoder: full attention, a
	// ReLU² MLP, and no KV compression.
	PresetDense Preset = "dense"
)

// DefaultPreset is the preset applied when none is specified.
const DefaultPreset = PresetFlash

// ParsePreset resolves a preset name. The empty string selects DefaultPreset.
func ParsePreset(name string) (Preset, error) {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "", string(PresetFlash):
		return PresetFlash, nil
	case string(PresetLatent):
		return PresetLatent, nil
	case string(PresetDense):
		return PresetDense, nil
	default:
		return "", fmt.Errorf("model: unknown preset %q (want flash, latent, or dense)", name)
	}
}

// UsesSinkhorn reports whether the preset routes the embedding table, the
// language-model head, and the value embeddings to the Sinkhorn-balanced
// momentum update (DeepSeek-V4.1 §2.5). It is true for every preset except the
// dense baseline.
func (preset Preset) UsesSinkhorn() bool { return preset != PresetDense }

// ApplyPreset fills the config with the preset's architecture defaults. Options
// the caller already enabled are respected; only disabled options are turned
// on, so an explicit configuration is never silently broadened.
func (configuration *Config) ApplyPreset(preset Preset) {
	switch preset {
	case PresetDense:
		return
	case PresetLatent:
		configuration.applyLatentPreset()
	default:
		configuration.applyFlashPreset()
	}
}

// ConfigForPreset derives a config from the depth dial and applies a preset.
// It is the single entry point training commands use, so the recommended
// architecture needs no per-feature flags.
func ConfigForPreset(preset Preset, depth, vocabSize, aspectRatio, headDim, seqLen int, windowPattern string) Config {
	config := ConfigForDepth(depth, vocabSize, aspectRatio, headDim, seqLen, windowPattern)
	config.ApplyPreset(preset)
	config.Validate()
	return config
}

// applyFlashPreset turns on the DeepSeek-V4.1-Flash long-context stack.
func (configuration *Config) applyFlashPreset() {
	configuration.applyDefaultGroupedQuery()
	configuration.HeadWiseMuon = true
	if configuration.Compression() <= 1 {
		configuration.CompressionRatio = 4
	}
	if configuration.SparseTopK <= 0 {
		configuration.SparseTopK = 8
	}
	if configuration.IndexerPool <= 1 {
		configuration.IndexerPool = 8
	}
	if configuration.ReusePattern == "" {
		configuration.ReusePattern = "FRU"
	}
	if configuration.SWAWindow <= 0 {
		configuration.SWAWindow = 128
	}
	if configuration.NumLayer >= 2 {
		// CED splices the backbone into equal encoder/decoder halves and needs
		// at least two layers; a single-layer model keeps global attention.
		configuration.CED = true
	}
	configuration.enableMoE()
}

// applyLatentPreset turns on absorbed MLA plus MoE. MLA is incompatible with
// the compression stack and head-wise Muon, so those stay off.
func (configuration *Config) applyLatentPreset() {
	configuration.applyDefaultGroupedQuery()
	configuration.HeadWiseMuon = false
	if configuration.MLALatent <= 0 {
		configuration.MLALatent = configuration.EmbedDim / 2
	}
	configuration.enableMoE()
}

// applyDefaultGroupedQuery halves the key/value heads when the caller left
// full multi-head attention and the query head count is even.
func (configuration *Config) applyDefaultGroupedQuery() {
	if configuration.NumKVHead != configuration.NumHead || configuration.NumHead == 0 {
		return
	}
	if kvHeads := defaultKVHeadCount(configuration.NumHead); kvHeads < configuration.NumKVHead {
		configuration.NumKVHead = kvHeads
	}
}

// enableMoE turns on the DeepSeekMoE feed-forward, deriving its shape from the
// depth dial if the caller did not set it.
func (configuration *Config) enableMoE() {
	if configuration.NumExperts <= 0 {
		configuration.NumExperts = MoEExpertsForDepth(configuration.NumLayer)
	}
	if configuration.NumExpertsPerToken <= 0 {
		configuration.NumExpertsPerToken = 2
	}
	configuration.ApplyMoEDefaults()
}

// defaultKVHeadCount returns the grouped-query key/value head count: half the
// query heads when even, otherwise full multi-head attention.
func defaultKVHeadCount(numHeads int) int {
	if numHeads%2 == 0 {
		return numHeads / 2
	}
	return numHeads
}
