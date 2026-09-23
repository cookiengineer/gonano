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

	// MLA latent cache (DeepSeek-V4.1 §2.3/§4.2.1): the shared key/value latent
	// c_kv is one row per token (not per head), and the decoupled rotary key is
	// per key/value head. Allocated by EnableMLA.
	mlaLatentWidth  int
	mlaRopeKeyWidth int
	mlaLatent       [][]*tensors.Tensor // [layer][batch] [maxSeq, mlaLatentWidth]
	mlaRopeKey      [][]*tensors.Tensor // [layer][batch] [maxSeq, mlaRopeKeyWidth]

	// rawStripped marks a snapshot whose raw key/value buffers were dropped
	// (DeepSeek-V4.1 §3.2.1): only the global compressed/indexer state and the
	// buffered tail survive. Bounded replay rebuilds the local sliding-window
	// rows from the cached tokens.
	rawStripped bool
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

// SetPosition rewinds or advances the cache position to position. Speculative
// verification uses it to discard rejected draft tokens: entries beyond the
// position are ignored and overwritten by the next forward. It is only safe for
// uncompressed caches, which carry no per-position compression bookkeeping.
func (cache *KVBuffer) SetPosition(position int) {
	if position < 0 || position > cache.maximumSequenceLength {
		panic("model: KV cache position out of range")
	}
	cache.sequenceLength = int32(position)
}

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

// EnableCompression allocates the HCA-style dense compression state on every
// layer. See EnableCompressionLayers.
func (cache *KVBuffer) EnableCompression(ratio, embeddingDimension, kvWidth, maxBlocks int) {
	cache.EnableCompressionLayers(ratio, embeddingDimension, kvWidth, maxBlocks, nil)
}

// EnableCompressionLayers allocates the HCA-style dense compression state. ratio
// is the number of tokens merged into one compressed entry, embeddingDimension
// is the compressor hidden width, kvWidth is keyValueHeadCount*headDim, and
// maxBlocks is the maximum number of compressed entries per (layer, row, head).
//
// allocate selects which layers own compression buffers; layers that reuse
// another layer's compressed cache (reindex/reuse) pass false and allocate only
// the small bookkeeping counters. A nil allocate allocates every layer.
func (cache *KVBuffer) EnableCompressionLayers(ratio, embeddingDimension, kvWidth, maxBlocks int, allocate []bool) {
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
		// Bookkeeping counters are tiny and needed by reindex layers to track
		// how many indexer keys they have cached.
		cache.tailLength[layer] = make([]int, cache.batchSize)
		cache.compressedCount[layer] = make([]int, cache.batchSize)
		if allocate != nil && !allocate[layer] {
			continue
		}
		cache.tailHidden[layer] = make([][]float32, cache.batchSize)
		cache.tailKey[layer] = make([][]float32, cache.batchSize*cache.keyValueHeadCount)
		cache.tailValue[layer] = make([][]float32, cache.batchSize*cache.keyValueHeadCount)
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

// EnableIndexerKeys allocates the cached indexer key projections on every
// layer. See EnableIndexerKeysLayers.
func (cache *KVBuffer) EnableIndexerKeys(width int) {
	cache.EnableIndexerKeysLayers(width, nil)
}

// EnableIndexerKeysLayers allocates the cached indexer key projections, one
// [maxBlocks, width] buffer per (layer, row, kv-head) whose allocate flag is
// true. EnableCompression must be called first. A nil allocate allocates every
// layer.
func (cache *KVBuffer) EnableIndexerKeysLayers(width int, allocate []bool) {
	cache.indexerKeyWidth = width
	cache.indexerKey = make([][]*tensors.Tensor, cache.layerCount)
	for layer := 0; layer < cache.layerCount; layer++ {
		if allocate != nil && !allocate[layer] {
			continue
		}
		cache.indexerKey[layer] = make([]*tensors.Tensor, cache.batchSize*cache.keyValueHeadCount)
		for index := range cache.indexerKey[layer] {
			cache.indexerKey[layer][index] = tensors.New(cache.compressedMaxBlocks, width)
		}
	}
}

// EnableMLA allocates the MLA latent and rope-key buffers: one shared latent row
// and one rope-key row per token per (layer, batch).
func (cache *KVBuffer) EnableMLA(latentWidth, ropeKeyWidth int) {
	cache.mlaLatentWidth = latentWidth
	cache.mlaRopeKeyWidth = ropeKeyWidth
	cache.mlaLatent = make([][]*tensors.Tensor, cache.layerCount)
	cache.mlaRopeKey = make([][]*tensors.Tensor, cache.layerCount)
	for layer := 0; layer < cache.layerCount; layer++ {
		cache.mlaLatent[layer] = make([]*tensors.Tensor, cache.batchSize)
		cache.mlaRopeKey[layer] = make([]*tensors.Tensor, cache.batchSize)
		for batch := 0; batch < cache.batchSize; batch++ {
			cache.mlaLatent[layer][batch] = tensors.New(cache.maximumSequenceLength, latentWidth)
			cache.mlaRopeKey[layer][batch] = tensors.New(cache.maximumSequenceLength, ropeKeyWidth)
		}
	}
}

// MLAEnabled reports whether the MLA latent buffers were allocated.
func (cache *KVBuffer) MLAEnabled() bool { return cache.mlaLatent != nil }

// MLALatentWidth returns the shared key/value latent width.
func (cache *KVBuffer) MLALatentWidth() int { return cache.mlaLatentWidth }

// MLARopeKeyWidth returns the decoupled rotary key width (NumKVHead*ropeDim).
func (cache *KVBuffer) MLARopeKeyWidth() int { return cache.mlaRopeKeyWidth }

// AppendMLALatentAt writes one token's shared latent at position+row.
func (cache *KVBuffer) AppendMLALatentAt(layer, batch, row int, latent []float32) {
	position := int(cache.sequenceLength) + row
	copy(cache.mlaLatent[layer][batch].Data[position*cache.mlaLatentWidth:], latent)
}

// AppendMLARopeKeyAt writes one token's decoupled rotary key at position+row.
func (cache *KVBuffer) AppendMLARopeKeyAt(layer, batch, row int, key []float32) {
	position := int(cache.sequenceLength) + row
	copy(cache.mlaRopeKey[layer][batch].Data[position*cache.mlaRopeKeyWidth:], key)
}

// MLALatent returns the first count shared latent rows.
func (cache *KVBuffer) MLALatent(layer, batch, count int) []float32 {
	if cache.mlaLatent == nil {
		return nil
	}
	return cache.mlaLatent[layer][batch].Data[:count*cache.mlaLatentWidth]
}

// MLARopeKey returns the first count decoupled rotary key rows.
func (cache *KVBuffer) MLARopeKey(layer, batch, count int) []float32 {
	if cache.mlaRopeKey == nil {
		return nil
	}
	return cache.mlaRopeKey[layer][batch].Data[:count*cache.mlaRopeKeyWidth]
}

// AppendIndexerKey stores the indexer key projection of one compressed entry.
func (cache *KVBuffer) AppendIndexerKey(layer, batch, head int, key []float32) {
	if cache.indexerKey == nil || cache.indexerKey[layer] == nil {
		return
	}
	count := cache.compressedCount[layer][batch]
	index := batch*cache.keyValueHeadCount + head
	copy(cache.indexerKey[layer][index].Data[count*cache.indexerKeyWidth:], key)
}

// IndexerKey returns the first count cached indexer key projections for a
// (row, kv-head).
func (cache *KVBuffer) IndexerKey(layer, batch, head, count int) []float32 {
	if cache.indexerKey == nil || cache.indexerKey[layer] == nil {
		return nil
	}
	index := batch*cache.keyValueHeadCount + head
	return cache.indexerKey[layer][index].Data[:count*cache.indexerKeyWidth]
}

// CompressedBlock returns one compressed key entry [headDim] for a (row, head).
func (cache *KVBuffer) CompressedBlock(layer, batch, head, block int) []float32 {
	index := batch*cache.keyValueHeadCount + head
	headDimension := cache.compressedKey[layer][index].Shape[1]
	return cache.compressedKey[layer][index].Data[block*headDimension : (block+1)*headDimension]
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
	if cache.compressedKey[layer] == nil {
		return nil
	}
	index := batch*cache.keyValueHeadCount + head
	headDimension := cache.compressedKey[layer][index].Shape[1]
	return cache.compressedKey[layer][index].Data[:count*headDimension]
}

// CompressedValue returns the first count compressed value rows for a (row, head).
func (cache *KVBuffer) CompressedValue(layer, batch, head, count int) []float32 {
	if cache.compressedValue[layer] == nil {
		return nil
	}
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
			}
			if destination.compressedKey[layer] == nil {
				// Reuse layer: it owns no compressed KV, but a reindex layer
				// still caches its own indexer-key projections.
				copyIndexerKeys(destination, source, layer, sourceCount)
				continue
			}
			for batch := 0; batch < destination.batchSize; batch++ {
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
				}
			}
			copyIndexerKeys(destination, source, layer, sourceCount)
		}
	}

	if destination.mlaLatent != nil {
		position := source.Position()
		latentWidth := destination.mlaLatentWidth
		ropeWidth := destination.mlaRopeKeyWidth
		for layer := 0; layer < destination.layerCount; layer++ {
			for batch := 0; batch < destination.batchSize; batch++ {
				copy(destination.mlaLatent[layer][batch].Data[:position*latentWidth], source.mlaLatent[layer][0].Data[:position*latentWidth])
				copy(destination.mlaRopeKey[layer][batch].Data[:position*ropeWidth], source.mlaRopeKey[layer][0].Data[:position*ropeWidth])
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

// Clone returns a deep copy of the cache, including compression and indexer
// state. It is used by the inference prefix cache to snapshot a prefill and by
// callers that need to branch from an existing state.
func (cache *KVBuffer) Clone() *KVBuffer {
	clone := &KVBuffer{
		batchSize:             cache.batchSize,
		maximumSequenceLength: cache.maximumSequenceLength,
		layerCount:            cache.layerCount,
		keyValueHeadCount:     cache.keyValueHeadCount,
		headDimension:         cache.headDimension,
		sequenceLength:        cache.sequenceLength,
		rawStripped:           cache.rawStripped,
		keyCache:              make([][]*tensors.Tensor, cache.layerCount),
		valueCache:            make([][]*tensors.Tensor, cache.layerCount),
	}
	for layer := 0; layer < cache.layerCount; layer++ {
		clone.keyCache[layer] = make([]*tensors.Tensor, len(cache.keyCache[layer]))
		clone.valueCache[layer] = make([]*tensors.Tensor, len(cache.valueCache[layer]))
		for index := range cache.keyCache[layer] {
			clone.keyCache[layer][index] = cache.keyCache[layer][index].Clone()
			clone.valueCache[layer][index] = cache.valueCache[layer][index].Clone()
		}
	}

	if cache.compressionRatio > 1 {
		clone.compressionRatio = cache.compressionRatio
		clone.compressedMaxBlocks = cache.compressedMaxBlocks
		clone.compressedEmbeddingDim = cache.compressedEmbeddingDim
		clone.compressedKVWidth = cache.compressedKVWidth
		clone.tailHidden = make([][][]float32, cache.layerCount)
		clone.tailKey = make([][][]float32, cache.layerCount)
		clone.tailValue = make([][][]float32, cache.layerCount)
		clone.tailLength = make([][]int, cache.layerCount)
		clone.compressedKey = make([][]*tensors.Tensor, cache.layerCount)
		clone.compressedValue = make([][]*tensors.Tensor, cache.layerCount)
		clone.compressedCount = make([][]int, cache.layerCount)
		for layer := 0; layer < cache.layerCount; layer++ {
			clone.tailLength[layer] = append([]int(nil), cache.tailLength[layer]...)
			clone.compressedCount[layer] = append([]int(nil), cache.compressedCount[layer]...)
			if cache.tailHidden[layer] == nil {
				continue
			}
			clone.tailHidden[layer] = make([][]float32, len(cache.tailHidden[layer]))
			for batch := range cache.tailHidden[layer] {
				clone.tailHidden[layer][batch] = append([]float32(nil), cache.tailHidden[layer][batch]...)
			}
			clone.tailKey[layer] = make([][]float32, len(cache.tailKey[layer]))
			clone.tailValue[layer] = make([][]float32, len(cache.tailValue[layer]))
			for index := range cache.tailKey[layer] {
				clone.tailKey[layer][index] = append([]float32(nil), cache.tailKey[layer][index]...)
				clone.tailValue[layer][index] = append([]float32(nil), cache.tailValue[layer][index]...)
			}
			clone.compressedKey[layer] = make([]*tensors.Tensor, len(cache.compressedKey[layer]))
			clone.compressedValue[layer] = make([]*tensors.Tensor, len(cache.compressedValue[layer]))
			for index := range cache.compressedKey[layer] {
				clone.compressedKey[layer][index] = cache.compressedKey[layer][index].Clone()
				clone.compressedValue[layer][index] = cache.compressedValue[layer][index].Clone()
			}
		}
	}

	if cache.indexerKey != nil {
		clone.indexerKeyWidth = cache.indexerKeyWidth
		clone.indexerKey = make([][]*tensors.Tensor, cache.layerCount)
		for layer := 0; layer < cache.layerCount; layer++ {
			if cache.indexerKey[layer] == nil {
				continue
			}
			clone.indexerKey[layer] = make([]*tensors.Tensor, len(cache.indexerKey[layer]))
			for index := range cache.indexerKey[layer] {
				clone.indexerKey[layer][index] = cache.indexerKey[layer][index].Clone()
			}
		}
	}

	if cache.mlaLatent != nil {
		clone.mlaLatentWidth = cache.mlaLatentWidth
		clone.mlaRopeKeyWidth = cache.mlaRopeKeyWidth
		clone.mlaLatent = make([][]*tensors.Tensor, cache.layerCount)
		clone.mlaRopeKey = make([][]*tensors.Tensor, cache.layerCount)
		for layer := 0; layer < cache.layerCount; layer++ {
			clone.mlaLatent[layer] = make([]*tensors.Tensor, len(cache.mlaLatent[layer]))
			clone.mlaRopeKey[layer] = make([]*tensors.Tensor, len(cache.mlaRopeKey[layer]))
			for batch := range cache.mlaLatent[layer] {
				clone.mlaLatent[layer][batch] = cache.mlaLatent[layer][batch].Clone()
				clone.mlaRopeKey[layer][batch] = cache.mlaRopeKey[layer][batch].Clone()
			}
		}
	}

	if cache.previousEmbedding != nil {
		clone.previousEmbedding = cache.previousEmbedding.Clone()
	}
	return clone
}

// CompressedBytesAllocated returns the bytes allocated for compressed key/value
// storage and cached indexer keys across every layer. Reuse/reindex layers own
// no compressed KV, so only full layers contribute key/value bytes; reindex
// layers still contribute their cached indexer keys.
func (cache *KVBuffer) CompressedBytesAllocated() int {
	total := 0
	for layer := 0; layer < cache.layerCount; layer++ {
		if cache.compressedKey != nil {
			for _, tensor := range cache.compressedKey[layer] {
				if tensor != nil {
					total += tensor.Numel() * 4
				}
			}
		}
		if cache.compressedValue != nil {
			for _, tensor := range cache.compressedValue[layer] {
				if tensor != nil {
					total += tensor.Numel() * 4
				}
			}
		}
		if cache.indexerKey != nil {
			for _, tensor := range cache.indexerKey[layer] {
				if tensor != nil {
					total += tensor.Numel() * 4
				}
			}
		}
	}
	return total
}

// BytesAllocated returns the approximate bytes allocated for every key/value,
// compression, indexer, tail, and previous-embedding buffer. It is used by the
// inference persistent cache to enforce a memory budget.
func (cache *KVBuffer) BytesAllocated() int {
	total := 0
	for layer := 0; layer < cache.layerCount; layer++ {
		for _, tensor := range cache.keyCache[layer] {
			if tensor != nil {
				total += tensor.Numel() * 4
			}
		}
		for _, tensor := range cache.valueCache[layer] {
			if tensor != nil {
				total += tensor.Numel() * 4
			}
		}
	}
	if cache.compressedKey != nil {
		for layer := 0; layer < cache.layerCount; layer++ {
			for _, tensor := range cache.compressedKey[layer] {
				if tensor != nil {
					total += tensor.Numel() * 4
				}
			}
		}
	}
	if cache.compressedValue != nil {
		for layer := 0; layer < cache.layerCount; layer++ {
			for _, tensor := range cache.compressedValue[layer] {
				if tensor != nil {
					total += tensor.Numel() * 4
				}
			}
		}
	}
	if cache.indexerKey != nil {
		for layer := 0; layer < cache.layerCount; layer++ {
			for _, tensor := range cache.indexerKey[layer] {
				if tensor != nil {
					total += tensor.Numel() * 4
				}
			}
		}
	}
	if cache.tailHidden != nil {
		for layer := 0; layer < cache.layerCount; layer++ {
			for _, row := range cache.tailHidden[layer] {
				total += len(row) * 4
			}
			for _, row := range cache.tailKey[layer] {
				total += len(row) * 4
			}
			for _, row := range cache.tailValue[layer] {
				total += len(row) * 4
			}
		}
	}
	if cache.mlaLatent != nil {
		for layer := 0; layer < cache.layerCount; layer++ {
			for _, tensor := range cache.mlaLatent[layer] {
				if tensor != nil {
					total += tensor.Numel() * 4
				}
			}
			for _, tensor := range cache.mlaRopeKey[layer] {
				if tensor != nil {
					total += tensor.Numel() * 4
				}
			}
		}
	}
	if cache.previousEmbedding != nil {
		total += cache.previousEmbedding.Numel() * 4
	}
	return total
}

// RawStripped reports whether the raw key/value buffers were dropped from this
// snapshot and must be rebuilt by bounded replay.
func (cache *KVBuffer) RawStripped() bool { return cache.rawStripped }

// StripRaw drops the raw key/value buffers and the local sliding-window tail
// state, keeping only the global compressed KV, the cached indexer keys, and the
// buffered compression tail. It implements the paper's persistent-cache policy
// of not storing SWA KV; the local rows are rebuilt by ReplaySWA on a hit. It is
// only meaningful for compressed caches, which are the only ones with a separate
// global state.
func (cache *KVBuffer) StripRaw() {
	if cache.compressionRatio <= 1 {
		return
	}
	for layer := 0; layer < cache.layerCount; layer++ {
		for index := range cache.keyCache[layer] {
			if cache.keyCache[layer][index] != nil {
				clear(cache.keyCache[layer][index].Data)
			}
			if cache.valueCache[layer][index] != nil {
				clear(cache.valueCache[layer][index].Data)
			}
		}
	}
	cache.rawStripped = true
}

// copyIndexerKeys replicates a layer's cached indexer-key projections from a
// batch-1 source across every destination row.
func copyIndexerKeys(destination, source *KVBuffer, layer, count int) {
	if destination.indexerKey == nil || source.indexerKey == nil {
		return
	}
	if destination.indexerKey[layer] == nil || source.indexerKey[layer] == nil {
		return
	}
	for head := 0; head < destination.keyValueHeadCount; head++ {
		indexerWidth := source.indexerKey[layer][head].Shape[1]
		for batch := 0; batch < destination.batchSize; batch++ {
			destinationIndex := batch*destination.keyValueHeadCount + head
			copy(destination.indexerKey[layer][destinationIndex].Data[:count*indexerWidth], source.indexerKey[layer][head].Data[:count*indexerWidth])
		}
	}
}
