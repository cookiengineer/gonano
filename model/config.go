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
}

// ConfigForDepth derives a compute-optimal config from a single dial: the
// depth. model_dim = ceil(depth*aspectRatio / headDim) * headDim, and heads =
// model_dim / headDim. All other hyperparameters (batch size, learning rates,
// horizons) are derived later from the scaling laws in package trainer.
func ConfigForDepth(depth, vocabSize, aspectRatio, headDim, seqLen int, windowPattern string) Config {
	baseDimension := depth * aspectRatio
	modelDimension := ((baseDimension + headDim - 1) / headDim) * headDim
	numHeads := modelDimension / headDim
	config := Config{
		SequenceLen:   seqLen,
		VocabSize:     vocabSize,
		NumLayer:      depth,
		NumHead:       numHeads,
		NumKVHead:     numHeads,
		EmbedDim:      modelDimension,
		WindowPattern: windowPattern,
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
