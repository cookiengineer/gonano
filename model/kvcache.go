package model

import "github.com/cookiengineer/gonano/tensor"

// KVBuffer is the key/value cache used during autoregressive inference. It
// stores, for every layer and every batch row, a contiguous [maxSeqLen, D]
// buffer per KV head. New keys/values are written in place; attention reads a
// contiguous prefix as the available context.
type KVBuffer struct {
	batchSize int
	maxSeqLen int
	numLayers int
	numKVHead int
	headDim   int

	// k[layer][row*numKVHead+head] is a [maxSeqLen, headDim] tensor.
	k [][]*tensor.Tensor
	v [][]*tensor.Tensor

	// seqLen is the current sequence length per batch row (all rows advance
	// together, as in nanochat's engine).
	seqLen int32

	// prevEmbedding caches the previous token's post-norm embedding for the
	// smear mechanism during single-token decode.
	prevEmbedding *tensor.Tensor
}

// NewKVBuffer allocates a zeroed KV cache for the given model geometry.
func NewKVBuffer(batchSize, maxSeqLen, numLayers, numKVHead, headDim int) *KVBuffer {
	cache := &KVBuffer{
		batchSize: batchSize,
		maxSeqLen: maxSeqLen,
		numLayers: numLayers,
		numKVHead: numKVHead,
		headDim:   headDim,
		k:         make([][]*tensor.Tensor, numLayers),
		v:         make([][]*tensor.Tensor, numLayers),
	}
	for l := 0; l < numLayers; l++ {
		cache.k[l] = make([]*tensor.Tensor, batchSize*numKVHead)
		cache.v[l] = make([]*tensor.Tensor, batchSize*numKVHead)
		for i := 0; i < batchSize*numKVHead; i++ {
			cache.k[l][i] = tensor.New(maxSeqLen, headDim)
			cache.v[l][i] = tensor.New(maxSeqLen, headDim)
		}
	}
	return cache
}

// Position returns the current sequence length (assumed uniform across rows).
func (c *KVBuffer) Position() int { return int(c.seqLen) }

// Advance moves the cache position forward by n tokens.
func (c *KVBuffer) Advance(n int) { c.seqLen += int32(n) }

// NumLayers returns the number of layers.
func (c *KVBuffer) NumLayers() int { return c.numLayers }

// BatchSize returns the number of batch rows.
func (c *KVBuffer) BatchSize() int { return c.batchSize }

// Reset clears the cache and the smear state.
func (c *KVBuffer) Reset() {
	c.seqLen = 0
	c.prevEmbedding = nil
}

// PrevEmbedding returns the cached previous-token embedding (may be nil).
func (c *KVBuffer) PrevEmbedding() *tensor.Tensor { return c.prevEmbedding }

// SetPrevEmbedding stores the previous-token embedding for the smear step.
func (c *KVBuffer) SetPrevEmbedding(x *tensor.Tensor) { c.prevEmbedding = x }

// writeKeyValue copies the new k/v for layer l, row b, head h into the cache
// at the current position, and returns the contiguous key/value prefix
// [pos+T, D] available for attention.
func (c *KVBuffer) writeKeyValue(layer, row, head int, kNew, vNew []float32) (kFull, vFull []float32) {
	pos := int(c.seqLen)
	t := len(kNew) / c.headDim
	kBuf := c.k[layer][row*c.numKVHead+head].Data
	vBuf := c.v[layer][row*c.numKVHead+head].Data
	copy(kBuf[pos*c.headDim:], kNew)
	copy(vBuf[pos*c.headDim:], vNew)
	upto := (pos + t) * c.headDim
	return kBuf[:upto], vBuf[:upto]
}

// PrefillFrom copies the cache contents (and smear state) of src into dst,
// expanding a batch-1 cache into a larger batch. Used by the engine to
// replicate a single-row prefill across many decode rows.
func PrefillFrom(dst, src *KVBuffer) {
	if dst.Position() != 0 {
		panic("model: cannot prefill a non-empty KV cache")
	}
	pos := src.Position()
	for l := 0; l < dst.numLayers; l++ {
		for b := 0; b < dst.batchSize; b++ {
			for h := 0; h < dst.numKVHead; h++ {
				// src has batch 1, so its head index is just h.
				copy(dst.k[l][b*dst.numKVHead+h].Data[:pos*dst.headDim], src.k[l][h].Data[:pos*src.headDim])
				copy(dst.v[l][b*dst.numKVHead+h].Data[:pos*dst.headDim], src.v[l][h].Data[:pos*src.headDim])
			}
		}
	}
	dst.seqLen = src.seqLen
	if src.prevEmbedding != nil {
		// Expand the batch-1 previous embedding across all decode rows.
		c := src.prevEmbedding.Shape[2]
		dst.prevEmbedding = tensor.New(dst.batchSize, 1, c)
		for b := 0; b < dst.batchSize; b++ {
			copy(dst.prevEmbedding.Data[b*c:(b+1)*c], src.prevEmbedding.Data[:c])
		}
	}
}
