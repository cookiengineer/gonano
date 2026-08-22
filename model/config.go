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
func (c Config) HeadDim() int { return c.EmbedDim / c.NumHead }

// KVHeadDim returns the KV per-head dimension, equal to HeadDim by construction.
func (c Config) KVHeadDim() int { return c.EmbedDim / c.NumHead }

// Validate panics if the config is inconsistent.
func (c Config) Validate() {
	if c.EmbedDim%c.NumHead != 0 {
		panic(fmt.Sprintf("model: EmbedDim %d not divisible by NumHead %d", c.EmbedDim, c.NumHead))
	}
	if c.NumKVHead > c.NumHead || c.NumHead%c.NumKVHead != 0 {
		panic(fmt.Sprintf("model: NumKVHead %d must divide NumHead %d", c.NumKVHead, c.NumHead))
	}
	for _, ch := range c.WindowPattern {
		if ch != 'L' && ch != 'S' && ch != 'l' && ch != 's' {
			panic(fmt.Sprintf("model: invalid window pattern %q", c.WindowPattern))
		}
	}
}

// ConfigForDepth derives a compute-optimal config from a single dial: the
// depth. model_dim = ceil(depth*aspectRatio / headDim) * headDim, and heads =
// model_dim / headDim. All other hyperparameters (batch size, learning rates,
// horizons) are derived later from the scaling laws in package train.
func ConfigForDepth(depth, vocabSize, aspectRatio, headDim, seqLen int, windowPattern string) Config {
	baseDim := depth * aspectRatio
	modelDim := ((baseDim + headDim - 1) / headDim) * headDim
	numHeads := modelDim / headDim
	cfg := Config{
		SequenceLen:   seqLen,
		VocabSize:     vocabSize,
		NumLayer:      depth,
		NumHead:       numHeads,
		NumKVHead:     numHeads,
		EmbedDim:      modelDim,
		WindowPattern: windowPattern,
	}
	cfg.Validate()
	return cfg
}

// vocabPaddingTo is the multiple to which the vocabulary is padded for
// efficient embedding/matmul alignment (nanochat uses 64).
const vocabPaddingTo = 64

// PaddedVocab returns vocab size rounded up to the nearest multiple of 64.
func (c Config) PaddedVocab() int {
	return ((c.VocabSize + vocabPaddingTo - 1) / vocabPaddingTo) * vocabPaddingTo
}

// WindowSizes computes the per-layer (left, right) sliding-window sizes. The
// final layer always gets full context. S = quarter context, rounded up to a
// multiple of 128. Returns one [2]int per layer, {left, 0}.
func (c Config) WindowSizes() [][2]int {
	pattern := c.WindowPattern
	long := c.SequenceLen
	short := -((-c.SequenceLen / 4) / 128) * 128 // ceil(T/4/128)*128
	sizes := make([][2]int, c.NumLayer)
	for i := 0; i < c.NumLayer; i++ {
		ch := pattern[i%len(pattern)]
		switch ch {
		case 'L', 'l':
			sizes[i] = [2]int{long, 0}
		case 'S', 's':
			sizes[i] = [2]int{short, 0}
		}
	}
	sizes[c.NumLayer-1] = [2]int{long, 0}
	return sizes
}
