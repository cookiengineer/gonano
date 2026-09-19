package model

import "github.com/cookiengineer/gonano/tensors"

// KVBuffer is the key/value cache used during autoregressive inference. It
// stores, for every layer and every batch row, a contiguous
// [maximumSequenceLength, headDimension] buffer per key/value head. New
// keys/values are written in place; attention reads a contiguous prefix as the
// available context.
type KVBuffer struct {
	batchSize             int
	maximumSequenceLength int
	layerCount            int
	keyValueHeadCount     int
	headDimension         int

	// keyCache[layer][row*keyValueHeadCount+head] is a
	// [maximumSequenceLength, headDimension] tensor.
	keyCache   [][]*tensors.Tensor
	valueCache [][]*tensors.Tensor

	// sequenceLength is the current sequence length per batch row (all rows
	// advance together, as in nanochat's engine).
	sequenceLength int32

	// previousEmbedding caches the previous token's post-norm embedding for the
	// smear mechanism during single-token decode.
	previousEmbedding *tensors.Tensor
}

// NewKVBuffer allocates a zeroed KV cache for the given model geometry.
func NewKVBuffer(batchSize, maximumSequenceLength, layerCount, keyValueHeadCount, headDimension int) *KVBuffer {
	cache := &KVBuffer{
		batchSize:             batchSize,
		maximumSequenceLength: maximumSequenceLength,
		layerCount:            layerCount,
		keyValueHeadCount:     keyValueHeadCount,
		headDimension:         headDimension,
		keyCache:              make([][]*tensors.Tensor, layerCount),
		valueCache:            make([][]*tensors.Tensor, layerCount),
	}
	for layer := 0; layer < layerCount; layer++ {
		cache.keyCache[layer] = make([]*tensors.Tensor, batchSize*keyValueHeadCount)
		cache.valueCache[layer] = make([]*tensors.Tensor, batchSize*keyValueHeadCount)
		for index := 0; index < batchSize*keyValueHeadCount; index++ {
			cache.keyCache[layer][index] = tensors.New(maximumSequenceLength, headDimension)
			cache.valueCache[layer][index] = tensors.New(maximumSequenceLength, headDimension)
		}
	}
	return cache
}

// Position returns the current sequence length (assumed uniform across rows).
func (cache *KVBuffer) Position() int { return int(cache.sequenceLength) }

// Advance moves the cache position forward by tokenCount tokens.
func (cache *KVBuffer) Advance(tokenCount int) { cache.sequenceLength += int32(tokenCount) }

// NumLayers returns the number of layers.
func (cache *KVBuffer) NumLayers() int { return cache.layerCount }

// BatchSize returns the number of batch rows.
func (cache *KVBuffer) BatchSize() int { return cache.batchSize }

// Reset clears the cache and the smear state.
func (cache *KVBuffer) Reset() {
	cache.sequenceLength = 0
	cache.previousEmbedding = nil
}

// PrevEmbedding returns the cached previous-token embedding (may be nil).
func (cache *KVBuffer) PrevEmbedding() *tensors.Tensor { return cache.previousEmbedding }

// SetPrevEmbedding stores the previous-token embedding for the smear step.
func (cache *KVBuffer) SetPrevEmbedding(embedding *tensors.Tensor) {
	cache.previousEmbedding = embedding
}

// writeKeyValue copies the new key/value for the given layer, row, and head
// into the cache at the current position, and returns the contiguous
// key/value prefix [position+tokenCount, headDimension] available for
// attention.
func (cache *KVBuffer) writeKeyValue(layer, row, head int, newKey, newValue []float32) (fullKey, fullValue []float32) {
	position := int(cache.sequenceLength)
	tokenCount := len(newKey) / cache.headDimension
	keyBuffer := cache.keyCache[layer][row*cache.keyValueHeadCount+head].Data
	valueBuffer := cache.valueCache[layer][row*cache.keyValueHeadCount+head].Data
	copy(keyBuffer[position*cache.headDimension:], newKey)
	copy(valueBuffer[position*cache.headDimension:], newValue)
	end := (position + tokenCount) * cache.headDimension
	return keyBuffer[:end], valueBuffer[:end]
}

// PrefillFrom copies the cache contents (and smear state) of source into
// destination, expanding a batch-1 cache into a larger batch. It is used by the
// engine to replicate a single-row prefill across many decode rows.
func PrefillFrom(destination, source *KVBuffer) {
	if destination.Position() != 0 {
		panic("model: cannot prefill a non-empty KV cache")
	}
	position := source.Position()
	for layer := 0; layer < destination.layerCount; layer++ {
		for row := 0; row < destination.batchSize; row++ {
			for head := 0; head < destination.keyValueHeadCount; head++ {
				// source has batch 1, so its head index is just head.
				copy(destination.keyCache[layer][row*destination.keyValueHeadCount+head].Data[:position*destination.headDimension], source.keyCache[layer][head].Data[:position*source.headDimension])
				copy(destination.valueCache[layer][row*destination.keyValueHeadCount+head].Data[:position*destination.headDimension], source.valueCache[layer][head].Data[:position*source.headDimension])
			}
		}
	}
	destination.sequenceLength = source.sequenceLength
	if source.previousEmbedding != nil {
		// Expand the batch-1 previous embedding across all decode rows.
		channels := source.previousEmbedding.Shape[2]
		destination.previousEmbedding = tensors.New(destination.batchSize, 1, channels)
		for row := 0; row < destination.batchSize; row++ {
			copy(destination.previousEmbedding.Data[row*channels:(row+1)*channels], source.previousEmbedding.Data[:channels])
		}
	}
}
