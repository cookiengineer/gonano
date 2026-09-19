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

	// HCA-style dense compression state, allocated by EnableCompression.
	compressionRatio       int
	compressedMaxBlocks    int
	compressedEmbeddingDim int
	compressedKVWidth      int
	// tailHidden[layer][batch] is [ratio, d] flattened.
	tailHidden [][][]float32
	// tailKey/tailValue[layer][batch] are [ratio, kvWidth] flattened.
	tailKey   [][][]float32
	tailValue [][][]float32
	// tailLength[layer][batch] is the number of buffered rows.
	tailLength [][]int
	// compressedKey/compressedValue[layer][batch*kvHead] are
	// [maxBlocks, headDim].
	compressedKey   [][]*tensors.Tensor
	compressedValue [][]*tensors.Tensor
	// compressedCount[layer][batch] is the number of complete blocks.
	compressedCount [][]int
	// indexerKey[layer][batch*kvHead] caches the indexer key projections of the
	// compressed entries as [maxBlocks, indexerKeyWidth].
	indexerKeyWidth int
	indexerKey      [][]*tensors.Tensor
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

// EnableCompression allocates the HCA-style dense compression state. ratio is
// the number of tokens merged into one compressed entry, embeddingDimension is
// the compressor hidden width, kvWidth is keyValueHeadCount*headDim, and
// maxBlocks is the maximum number of compressed entries per (layer, row, head).
func (cache *KVBuffer) EnableCompression(ratio, embeddingDimension, kvWidth, maxBlocks int) {
	cache.compressionRatio = ratio
	cache.compressedMaxBlocks = maxBlocks
	cache.compressedEmbeddingDim = embeddingDimension
	cache.compressedKVWidth = kvWidth
	headDimension := kvWidth / cache.keyValueHeadCount

	cache.tailHidden = make([][][]float32, cache.layerCount)
	cache.tailKey = make([][][]float32, cache.layerCount)
	cache.tailValue = make([][][]float32, cache.layerCount)
	cache.tailLength = make([][]int, cache.layerCount)
	cache.compressedKey = make([][]*tensors.Tensor, cache.layerCount)
	cache.compressedValue = make([][]*tensors.Tensor, cache.layerCount)
	cache.compressedCount = make([][]int, cache.layerCount)

	for layer := 0; layer < cache.layerCount; layer++ {
		cache.tailHidden[layer] = make([][]float32, cache.batchSize)
		cache.tailKey[layer] = make([][]float32, cache.batchSize*cache.keyValueHeadCount)
		cache.tailValue[layer] = make([][]float32, cache.batchSize*cache.keyValueHeadCount)
		cache.tailLength[layer] = make([]int, cache.batchSize)
		cache.compressedCount[layer] = make([]int, cache.batchSize)
		cache.compressedKey[layer] = make([]*tensors.Tensor, cache.batchSize*cache.keyValueHeadCount)
		cache.compressedValue[layer] = make([]*tensors.Tensor, cache.batchSize*cache.keyValueHeadCount)
		for batch := 0; batch < cache.batchSize; batch++ {
			cache.tailHidden[layer][batch] = make([]float32, ratio*embeddingDimension)
		}
		for index := range cache.tailKey[layer] {
			cache.tailKey[layer][index] = make([]float32, ratio*headDimension)
			cache.tailValue[layer][index] = make([]float32, ratio*headDimension)
		}
		for index := range cache.compressedKey[layer] {
			cache.compressedKey[layer][index] = tensors.New(maxBlocks, headDimension)
			cache.compressedValue[layer][index] = tensors.New(maxBlocks, headDimension)
		}
	}
}

// EnableIndexerKeys allocates the cached indexer key projections, one
// [maxBlocks, width] buffer per (layer, row, kv-head). EnableCompression must
// be called first.
func (cache *KVBuffer) EnableIndexerKeys(width int) {
	cache.indexerKeyWidth = width
	cache.indexerKey = make([][]*tensors.Tensor, cache.layerCount)
	for layer := 0; layer < cache.layerCount; layer++ {
		cache.indexerKey[layer] = make([]*tensors.Tensor, cache.batchSize*cache.keyValueHeadCount)
		for index := range cache.indexerKey[layer] {
			cache.indexerKey[layer][index] = tensors.New(cache.compressedMaxBlocks, width)
		}
	}
}

// AppendIndexerKey stores the indexer key projection of one compressed entry.
func (cache *KVBuffer) AppendIndexerKey(layer, batch, head int, key []float32) {
	if cache.indexerKey == nil {
		return
	}
	count := cache.compressedCount[layer][batch]
	index := batch*cache.keyValueHeadCount + head
	copy(cache.indexerKey[layer][index].Data[count*cache.indexerKeyWidth:], key)
}

// IndexerKey returns the first count cached indexer key projections for a
// (row, kv-head).
func (cache *KVBuffer) IndexerKey(layer, batch, head, count int) []float32 {
	if cache.indexerKey == nil {
		return nil
	}
	index := batch*cache.keyValueHeadCount + head
	return cache.indexerKey[layer][index].Data[:count*cache.indexerKeyWidth]
}

// CompressionEnabled reports whether compression state was allocated.
func (cache *KVBuffer) CompressionEnabled() bool { return cache.compressionRatio > 1 }

// CompressionRatio returns the configured compression ratio.
func (cache *KVBuffer) CompressionRatio() int { return cache.compressionRatio }

// TailLength returns the number of buffered (uncompressed) rows for a row.
func (cache *KVBuffer) TailLength(layer, batch int) int { return cache.tailLength[layer][batch] }

// TailHidden returns the [ratio*d] buffered hidden rows for a row.
func (cache *KVBuffer) TailHidden(layer, batch int) []float32 { return cache.tailHidden[layer][batch] }

// TailKey returns the [ratio*headDim] buffered key rows for a (row, kv-head).
func (cache *KVBuffer) TailKey(layer, index int) []float32 { return cache.tailKey[layer][index] }

// TailValue returns the [ratio*headDim] buffered value rows for a (row, kv-head).
func (cache *KVBuffer) TailValue(layer, index int) []float32 { return cache.tailValue[layer][index] }

// AppendTailRow appends one hidden/key/value row to the buffered tail. keyNew
// and valueNew are flattened [keyValueHeadCount*headDim] rows.
func (cache *KVBuffer) AppendTailRow(layer, batch int, hiddenNew, keyNew, valueNew []float32) {
	position := cache.tailLength[layer][batch]
	embeddingDimension := cache.compressedEmbeddingDim
	headDimension := cache.compressedKVWidth / cache.keyValueHeadCount
	copy(cache.tailHidden[layer][batch][position*embeddingDimension:], hiddenNew)
	for head := 0; head < cache.keyValueHeadCount; head++ {
		index := batch*cache.keyValueHeadCount + head
		copy(cache.tailKey[layer][index][position*headDimension:], keyNew[head*headDimension:(head+1)*headDimension])
		copy(cache.tailValue[layer][index][position*headDimension:], valueNew[head*headDimension:(head+1)*headDimension])
	}
	cache.tailLength[layer][batch]++
}

// ResetTail clears the buffered tail for a row.
func (cache *KVBuffer) ResetTail(layer, batch int) { cache.tailLength[layer][batch] = 0 }

// AppendCompressed writes one compressed entry for a (row, kv-head).
func (cache *KVBuffer) AppendCompressed(layer, batch, head int, key, value []float32) {
	count := cache.compressedCount[layer][batch]
	index := batch*cache.keyValueHeadCount + head
	headDimension := cache.compressedKey[layer][index].Shape[1]
	copy(cache.compressedKey[layer][index].Data[count*headDimension:], key)
	copy(cache.compressedValue[layer][index].Data[count*headDimension:], value)
}

// AdvanceCompressedCount marks that every (batch, head) gained one entry.
func (cache *KVBuffer) AdvanceCompressedCount(layer int) {
	for batch := 0; batch < cache.batchSize; batch++ {
		cache.compressedCount[layer][batch]++
	}
}

// CompressedCount returns the number of complete blocks for a row.
func (cache *KVBuffer) CompressedCount(layer, batch int) int {
	return cache.compressedCount[layer][batch]
}

// CompressedKey returns the first count compressed key rows for a (row, head).
func (cache *KVBuffer) CompressedKey(layer, batch, head, count int) []float32 {
	index := batch*cache.keyValueHeadCount + head
	headDimension := cache.compressedKey[layer][index].Shape[1]
	return cache.compressedKey[layer][index].Data[:count*headDimension]
}

// CompressedValue returns the first count compressed value rows for a (row, head).
func (cache *KVBuffer) CompressedValue(layer, batch, head, count int) []float32 {
	index := batch*cache.keyValueHeadCount + head
	headDimension := cache.compressedValue[layer][index].Shape[1]
	return cache.compressedValue[layer][index].Data[:count*headDimension]
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
	if destination.CompressionEnabled() {
		for layer := 0; layer < destination.layerCount; layer++ {
			sourceCount := source.compressedCount[layer][0]
			for batch := 0; batch < destination.batchSize; batch++ {
				destination.compressedCount[layer][batch] = sourceCount
				copy(destination.tailHidden[layer][batch], source.tailHidden[layer][0])
				copy(destination.tailKey[layer][batch], source.tailKey[layer][0])
				copy(destination.tailValue[layer][batch], source.tailValue[layer][0])
				destination.tailLength[layer][batch] = source.tailLength[layer][0]
			}
			for head := 0; head < destination.keyValueHeadCount; head++ {
				headDimension := source.compressedKey[layer][head].Shape[1]
				for batch := 0; batch < destination.batchSize; batch++ {
					destinationIndex := batch*destination.keyValueHeadCount + head
					copy(destination.compressedKey[layer][destinationIndex].Data[:sourceCount*headDimension], source.compressedKey[layer][head].Data[:sourceCount*headDimension])
					copy(destination.compressedValue[layer][destinationIndex].Data[:sourceCount*headDimension], source.compressedValue[layer][head].Data[:sourceCount*headDimension])
					if destination.indexerKey != nil && source.indexerKey != nil {
						indexerWidth := source.indexerKey[layer][head].Shape[1]
						copy(destination.indexerKey[layer][destinationIndex].Data[:sourceCount*indexerWidth], source.indexerKey[layer][head].Data[:sourceCount*indexerWidth])
					}
				}
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
