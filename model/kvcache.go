package model

import "github.com/cookiengineer/gonano/tensors"

// KVBuffer is the key/value cache used during autoregressive inference. It
// owns the per-sequence bookkeeping (position, compression tails, counters and
// the smear embedding); the token rows live in a pluggable kvStorage backend.
// The default backend is paged: rows are stored in 128-token blocks, allowing
// sharing and copy-on-write.
type KVBuffer struct {
	batchSize             int
	maximumSequenceLength int
	layerCount            int
	keyValueHeadCount     int
	headDimension         int

	// sequenceLength is the uniform sequence length. rowPositions, when set,
	// carries one logical position per batch row so independent sequences can
	// decode together.
	sequenceLength int32
	rowPositions   []int32

	// previousEmbedding caches the previous token's post-norm embedding for the
	// smear mechanism during single-token decode.
	previousEmbedding *tensors.Tensor

	// HCA-style dense compression state, allocated by EnableCompression.
	// layerRatios carries the per-layer compression layout; compressionRatio
	// and compressedMaxBlocks are retained as uniform/legacy summaries.
	compressionEnabled     bool
	compressionRatio       int
	compressedMaxBlocks    int
	layerRatios            []int
	maxBlocksByLayer       []int
	compressedEmbeddingDim int
	compressedKVWidth      int
	// tailHidden[layer][batch] is [ratio, d] flattened.
	tailHidden [][][]float32
	// tailKey/tailValue[layer][batch] are [ratio, kvWidth] flattened.
	tailKey   [][][]float32
	tailValue [][][]float32
	// tailLength[layer][batch] is the number of buffered rows.
	tailLength [][]int
	// compressedCount[layer][batch] is the number of complete blocks.
	compressedCount [][]int

	// indexerKeyWidth is the cached indexer-key projection width, or zero.
	indexerKeyWidth int
	indexerEnabled  bool
	// compressedAlloc/indexerAlloc record which layers own the corresponding
	// paged pools. Reuse layers read the producer's data instead.
	compressedAlloc []bool
	indexerAlloc    []bool

	// MLA latent cache (DeepSeek-V4.1 §2.3/§4.2.1): the shared key/value latent
	// c_kv is one row per token (not per head), and the decoupled rotary key is
	// per key/value head. Allocated by EnableMLA.
	mlaLatentWidth  int
	mlaRopeKeyWidth int
	mlaEnabled      bool

	// rawStripped marks a snapshot whose raw key/value buffers were dropped
	// (DeepSeek-V4.1 §3.2.1): only the global compressed/indexer state and the
	// buffered tail survive. Bounded replay rebuilds the local sliding-window
	// rows from the cached tokens.
	rawStripped bool

	storage kvStorage

	// Bounded sliding-window rings. Layers whose RingWindows entry is positive
	// store only the last window of raw K/V instead of every position.
	ringWindow  []int
	ringEnabled []bool
	ringSize    int
	ringKey     [][][][]float32
	ringValue   [][][][]float32

	ringScratchKey   [][][]float32
	ringScratchValue [][][]float32
}

// kvRingBacktrack is the extra ring slack that lets speculative verification
// append drafts without overwriting rows needed after a partial rollback.
const kvRingBacktrack = 8

// NewKVBuffer allocates a zeroed paged KV cache for the given model geometry.
func NewKVBuffer(batchSize, maximumSequenceLength, layerCount, keyValueHeadCount, headDimension int) *KVBuffer {
	return NewPagedKVBuffer(nil, batchSize, maximumSequenceLength, layerCount, keyValueHeadCount, headDimension)
}

// NewPagedKVBuffer allocates a paged KV cache backed by manager. A nil manager
// creates a private pool sized for batchSize independent full-length rows.
func NewPagedKVBuffer(manager *KVBlockManager, batchSize, maximumSequenceLength, layerCount, keyValueHeadCount, headDimension int) *KVBuffer {
	return NewPagedKVBufferWithPools(manager, nil, batchSize, maximumSequenceLength, layerCount, keyValueHeadCount, headDimension)
}

// NewPagedKVBufferWithPools allocates a paged cache that shares block data with
// every other cache using the same pool set.
func NewPagedKVBufferWithPools(manager *KVBlockManager, pools *KVBlockPools, batchSize, maximumSequenceLength, layerCount, keyValueHeadCount, headDimension int) *KVBuffer {
	cache := &KVBuffer{
		batchSize:             batchSize,
		maximumSequenceLength: maximumSequenceLength,
		layerCount:            layerCount,
		keyValueHeadCount:     keyValueHeadCount,
		headDimension:         headDimension,
	}
	cache.storage = newPagedKVStorageWithPools(manager, pools, batchSize, maximumSequenceLength, layerCount, keyValueHeadCount, headDimension)
	return cache
}

// NewDenseKVBuffer allocates the historical contiguous KV cache. It is used as
// a reference implementation by parity tests.
func NewDenseKVBuffer(batchSize, maximumSequenceLength, layerCount, keyValueHeadCount, headDimension int) *KVBuffer {
	cache := &KVBuffer{
		batchSize:             batchSize,
		maximumSequenceLength: maximumSequenceLength,
		layerCount:            layerCount,
		keyValueHeadCount:     keyValueHeadCount,
		headDimension:         headDimension,
	}
	cache.storage = newDenseKVStorage(batchSize, maximumSequenceLength, layerCount, keyValueHeadCount, headDimension)
	return cache
}

// Position returns the current sequence length (row 0 when rows differ).
func (cache *KVBuffer) Position() int { return cache.PositionAt(0) }

// PositionAt returns one batch row's logical position.
func (cache *KVBuffer) PositionAt(row int) int {
	if cache.rowPositions != nil && row >= 0 && row < len(cache.rowPositions) {
		return int(cache.rowPositions[row])
	}
	return int(cache.sequenceLength)
}

// Positions returns the logical position of every row.
func (cache *KVBuffer) Positions(count int) []int32 {
	positions := make([]int32, count)
	for row := 0; row < count; row++ {
		positions[row] = int32(cache.PositionAt(row))
	}
	return positions
}

// SetPositions installs one logical position per batch row.
func (cache *KVBuffer) SetPositions(positions []int32) {
	cache.rowPositions = append([]int32(nil), positions...)
	if len(positions) > 0 {
		cache.sequenceLength = positions[0]
	}
}

// SetPositionAt sets one row's logical position.
func (cache *KVBuffer) SetPositionAt(row, position int) {
	cache.ensureRowPositions()
	if row < 0 || row >= len(cache.rowPositions) {
		panic("model: KV cache row out of range")
	}
	if position < 0 || position > cache.maximumSequenceLength {
		panic("model: KV cache position out of range")
	}
	cache.rowPositions[row] = int32(position)
	cache.sequenceLength = cache.rowPositions[0]
}

// Advance moves every row forward by tokenCount tokens.
func (cache *KVBuffer) Advance(tokenCount int) {
	if cache.rowPositions != nil {
		for row := range cache.rowPositions {
			cache.rowPositions[row] += int32(tokenCount)
		}
		if len(cache.rowPositions) > 0 {
			cache.sequenceLength = cache.rowPositions[0]
		}
		return
	}
	cache.sequenceLength += int32(tokenCount)
}

// AdvanceAt moves one row forward by tokenCount tokens.
func (cache *KVBuffer) AdvanceAt(row, tokenCount int) {
	cache.ensureRowPositions()
	if row < 0 || row >= len(cache.rowPositions) {
		panic("model: KV cache row out of range")
	}
	cache.rowPositions[row] += int32(tokenCount)
	cache.sequenceLength = cache.rowPositions[0]
}

func (cache *KVBuffer) ensureRowPositions() {
	if cache.rowPositions != nil {
		return
	}
	cache.rowPositions = make([]int32, cache.batchSize)
	for row := range cache.rowPositions {
		cache.rowPositions[row] = cache.sequenceLength
	}
}

// SetPosition rewinds or advances the cache position to position. Speculative
// verification uses it to discard rejected draft tokens: entries beyond the
// position are ignored and overwritten by the next forward.
func (cache *KVBuffer) SetPosition(position int) {
	if position < 0 || position > cache.maximumSequenceLength {
		panic("model: KV cache position out of range")
	}
	cache.sequenceLength = int32(position)
	if cache.rowPositions != nil {
		for row := range cache.rowPositions {
			cache.rowPositions[row] = int32(position)
		}
	}
}

// NumLayers returns the number of layers.
func (cache *KVBuffer) NumLayers() int { return cache.layerCount }

// BatchSize returns the number of batch rows.
func (cache *KVBuffer) BatchSize() int { return cache.batchSize }

// Reset clears the cache, frees any paged blocks, and resets the smear state.
func (cache *KVBuffer) Reset() {
	cache.sequenceLength = 0
	cache.rowPositions = nil
	cache.previousEmbedding = nil
	cache.rawStripped = false
	cache.ringKey = nil
	cache.ringValue = nil
	cache.storage.reset()
}

// PrevEmbedding returns the cached previous-token embedding (may be nil).
func (cache *KVBuffer) PrevEmbedding() *tensors.Tensor {
	if cache.previousEmbedding != nil {
		cache.ensurePreviousEmbedding(cache.previousEmbedding.Shape[2])
	}
	return cache.previousEmbedding
}

// SetPrevEmbedding stores the previous-token embedding for the smear step.
func (cache *KVBuffer) SetPrevEmbedding(embedding *tensors.Tensor) {
	cache.previousEmbedding = embedding
}

// SetPreviousEmbeddingRows updates the leading rows of the previous-embedding
// tensor without discarding rows for inactive or later batch entries.
func (cache *KVBuffer) SetPreviousEmbeddingRows(embedding *tensors.Tensor) {
	if embedding == nil || embedding.Numel() == 0 {
		return
	}
	rows := embedding.Shape[0]
	channels := embedding.Shape[len(embedding.Shape)-1]
	previous := tensors.New(cache.batchSize, 1, channels)
	if cache.previousEmbedding != nil && cache.previousEmbedding.Shape[2] == channels {
		existing := cache.previousEmbedding.Shape[0]
		if existing > cache.batchSize {
			existing = cache.batchSize
		}
		copy(previous.Data[:existing*channels], cache.previousEmbedding.Data[:existing*channels])
	}
	limit := rows
	if limit > cache.batchSize {
		limit = cache.batchSize
	}
	copy(previous.Data[:limit*channels], embedding.Data[:limit*channels])
	cache.previousEmbedding = previous
}

func (cache *KVBuffer) ensurePreviousEmbedding(channels int) *tensors.Tensor {
	if channels <= 0 {
		return cache.previousEmbedding
	}
	if cache.previousEmbedding != nil && cache.previousEmbedding.Shape[0] >= cache.batchSize && cache.previousEmbedding.Shape[2] == channels {
		return cache.previousEmbedding
	}
	previous := tensors.New(cache.batchSize, 1, channels)
	if cache.previousEmbedding != nil && cache.previousEmbedding.Shape[2] == channels {
		rows := cache.previousEmbedding.Shape[0]
		if rows > cache.batchSize {
			rows = cache.batchSize
		}
		copy(previous.Data[:rows*channels], cache.previousEmbedding.Data[:rows*channels])
	}
	cache.previousEmbedding = previous
	return previous
}

// EnableSWARings configures bounded raw K/V rings. layerWindows[layer] is the
// local window width to keep for that layer; zero keeps the full-context raw
// cache. It must be called before the first token is appended.
func (cache *KVBuffer) EnableSWARings(layerWindows []int) {
	cache.ringWindow = make([]int, cache.layerCount)
	cache.ringEnabled = make([]bool, cache.layerCount)
	maxWindow := 0
	for layer := 0; layer < cache.layerCount; layer++ {
		window := 0
		if layer < len(layerWindows) {
			window = layerWindows[layer]
		}
		if window < 0 {
			window = 0
		}
		cache.ringWindow[layer] = window
		if window > 0 {
			cache.ringEnabled[layer] = true
			if window > maxWindow {
				maxWindow = window
			}
		}
	}
	if maxWindow == 0 {
		cache.ringSize = 0
		return
	}
	cache.ringSize = maxWindow + kvRingBacktrack
	if cache.ringSize > cache.maximumSequenceLength {
		cache.ringSize = cache.maximumSequenceLength
	}
	if cache.ringSize < 1 {
		cache.ringSize = 1
	}
}

// RingEnabled reports whether a layer stores raw K/V in a bounded ring.
func (cache *KVBuffer) RingEnabled(layer int) bool {
	return layer >= 0 && layer < len(cache.ringEnabled) && cache.ringEnabled[layer]
}

// appendRawKV writes raw K/V to the layer's ring or the paged store.
func (cache *KVBuffer) appendRawKV(layer, row, head, position int, key, value []float32) {
	if cache.RingEnabled(layer) {
		cache.appendRing(layer, row, head, position, key, value)
		return
	}
	cache.storage.appendRaw(layer, row, head, position, key, value)
}

func (cache *KVBuffer) appendRing(layer, row, head, position int, key, value []float32) {
	if cache.ringSize <= 0 {
		return
	}
	if cache.ringKey == nil {
		cache.ringKey = make([][][][]float32, cache.layerCount)
		cache.ringValue = make([][][][]float32, cache.layerCount)
	}
	if cache.ringKey[layer] == nil {
		cache.ringKey[layer] = make([][][]float32, cache.batchSize)
		cache.ringValue[layer] = make([][][]float32, cache.batchSize)
	}
	if cache.ringKey[layer][row] == nil {
		cache.ringKey[layer][row] = make([][]float32, cache.keyValueHeadCount)
		cache.ringValue[layer][row] = make([][]float32, cache.keyValueHeadCount)
	}
	if cache.ringKey[layer][row][head] == nil {
		cache.ringKey[layer][row][head] = make([]float32, cache.ringSize*cache.headDimension)
		cache.ringValue[layer][row][head] = make([]float32, cache.ringSize*cache.headDimension)
	}
	tokenCount := len(key) / cache.headDimension
	for token := 0; token < tokenCount; token++ {
		slot := (position + token) % cache.ringSize
		copy(cache.ringKey[layer][row][head][slot*cache.headDimension:], key[token*cache.headDimension:(token+1)*cache.headDimension])
		copy(cache.ringValue[layer][row][head][slot*cache.headDimension:], value[token*cache.headDimension:(token+1)*cache.headDimension])
	}
}

func (cache *KVBuffer) ringScratch(row, head, length int) ([]float32, []float32) {
	if cache.ringScratchKey == nil {
		cache.ringScratchKey = make([][][]float32, cache.batchSize)
		cache.ringScratchValue = make([][][]float32, cache.batchSize)
	}
	if cache.ringScratchKey[row] == nil {
		cache.ringScratchKey[row] = make([][]float32, cache.keyValueHeadCount)
		cache.ringScratchValue[row] = make([][]float32, cache.keyValueHeadCount)
	}
	if len(cache.ringScratchKey[row][head]) < length {
		cache.ringScratchKey[row][head] = make([]float32, length)
		cache.ringScratchValue[row][head] = make([]float32, length)
	}
	return cache.ringScratchKey[row][head][:length], cache.ringScratchValue[row][head][:length]
}

func (cache *KVBuffer) readRing(layer, row, head, start, end int, key, value []float32) {
	for position := start; position < end; position++ {
		slot := position % cache.ringSize
		base := (position - start) * cache.headDimension
		if cache.ringKey == nil || cache.ringKey[layer] == nil || cache.ringKey[layer][row] == nil || cache.ringKey[layer][row][head] == nil {
			continue
		}
		copy(key[base:base+cache.headDimension], cache.ringKey[layer][row][head][slot*cache.headDimension:(slot+1)*cache.headDimension])
		copy(value[base:base+cache.headDimension], cache.ringValue[layer][row][head][slot*cache.headDimension:(slot+1)*cache.headDimension])
	}
}

// LocalPrefix builds a contiguous local K/V range for a query chunk on a ring
// layer: the cached window rows before firstQuery followed by the chunk's own
// newly computed rows. It returns the relative position of the first query
// within the returned range.
func (cache *KVBuffer) LocalPrefix(layer, row, head, firstQuery, queryLength, window, floor int, newKey, newValue []float32) (key, value []float32, relativeOffset int) {
	localStart := firstQuery - window
	if localStart < floor {
		localStart = floor
	}
	if localStart < 0 {
		localStart = 0
	}
	prefixCount := firstQuery - localStart
	localCount := prefixCount + queryLength
	scratchKey, scratchValue := cache.ringScratch(row, head, localCount*cache.headDimension)
	cache.readRing(layer, row, head, localStart, firstQuery, scratchKey, scratchValue)
	copy(scratchKey[prefixCount*cache.headDimension:], newKey)
	copy(scratchValue[prefixCount*cache.headDimension:], newValue)
	return scratchKey, scratchValue, firstQuery - localStart
}

// LocalWindow returns the raw K/V rows a single query may attend to on a ring
// layer, in chronological order. It is used by the compressed local branch.
func (cache *KVBuffer) LocalWindow(layer, row, head, absolutePosition, window, floor int) (key, value []float32, count int) {
	localStart := absolutePosition - window
	if localStart < floor {
		localStart = floor
	}
	if localStart < 0 {
		localStart = 0
	}
	count = absolutePosition - localStart + 1
	if count < 1 {
		count = 1
	}
	if !cache.RingEnabled(layer) {
		key, value = cache.storage.readRaw(layer, row, head, absolutePosition+1)
		key = key[localStart*cache.headDimension : (absolutePosition+1)*cache.headDimension]
		value = value[localStart*cache.headDimension : (absolutePosition+1)*cache.headDimension]
		return key, value, count
	}
	scratchKey, scratchValue := cache.ringScratch(row, head, count*cache.headDimension)
	cache.readRing(layer, row, head, localStart, absolutePosition+1, scratchKey, scratchValue)
	return scratchKey, scratchValue, count
}

// writeKeyValue copies the new key/value for the given layer, row, and head
// into the cache at the current position, and returns the contiguous
// key/value prefix [position+tokenCount, headDimension] available for
// attention. Ring layers return nil; attention reads their local window
// through LocalPrefix/LocalWindow.
func (cache *KVBuffer) writeKeyValue(layer, row, head int, newKey, newValue []float32) (fullKey, fullValue []float32) {
	position := cache.PositionAt(row)
	tokenCount := len(newKey) / cache.headDimension
	cache.appendRawKV(layer, row, head, position, newKey, newValue)
	if cache.RingEnabled(layer) {
		return nil, nil
	}
	return cache.storage.readRaw(layer, row, head, position+tokenCount)
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
	ratios := make([]int, cache.layerCount)
	maxBlocksByLayer := make([]int, cache.layerCount)
	for layer := 0; layer < cache.layerCount; layer++ {
		ratios[layer] = ratio
		maxBlocksByLayer[layer] = maxBlocks
	}
	cache.EnableCompressionLayout(ratios, embeddingDimension, kvWidth, maxBlocksByLayer, allocate)
}

// EnableCompressionLayout allocates the compression state with a per-layer
// ratio and capacity. ratios[layer] is zero for sliding-window-only layers, one
// for C1 packed layers (no row merging), and above one for compressed layers.
func (cache *KVBuffer) EnableCompressionLayout(ratios []int, embeddingDimension, kvWidth int, maxBlocks []int, allocate []bool) {
	if len(ratios) != cache.layerCount {
		panic("model: compression layout length does not match layer count")
	}
	if len(maxBlocks) != cache.layerCount {
		panic("model: compression max-block layout length does not match layer count")
	}
	cache.compressionEnabled = false
	cache.compressionRatio = 1
	cache.compressedMaxBlocks = 0
	cache.layerRatios = append([]int(nil), ratios...)
	cache.maxBlocksByLayer = append([]int(nil), maxBlocks...)
	for layer, ratio := range ratios {
		if ratio < 0 {
			ratio = 0
			cache.layerRatios[layer] = 0
		}
		if ratio > 0 && KVBlockSize%ratio != 0 {
			panic("model: compression ratio must divide the paged KV block size")
		}
		if ratio > cache.compressionRatio {
			cache.compressionRatio = ratio
		}
		if ratio >= 1 {
			cache.compressionEnabled = true
			if maxBlocks[layer] > cache.compressedMaxBlocks {
				cache.compressedMaxBlocks = maxBlocks[layer]
			}
		}
	}
	cache.compressedEmbeddingDim = embeddingDimension
	cache.compressedKVWidth = kvWidth
	headDimension := kvWidth / cache.keyValueHeadCount

	cache.tailHidden = make([][][]float32, cache.layerCount)
	cache.tailKey = make([][][]float32, cache.layerCount)
	cache.tailValue = make([][][]float32, cache.layerCount)
	cache.tailLength = make([][]int, cache.layerCount)
	cache.compressedCount = make([][]int, cache.layerCount)
	cache.compressedAlloc = make([]bool, cache.layerCount)

	for layer := 0; layer < cache.layerCount; layer++ {
		ratio := cache.layerRatios[layer]
		// Bookkeeping counters are tiny and needed by reindex layers to track
		// how many indexer keys they have cached.
		cache.tailLength[layer] = make([]int, cache.batchSize)
		cache.compressedCount[layer] = make([]int, cache.batchSize)
		if ratio < 1 {
			continue
		}
		if allocate != nil && !allocate[layer] {
			continue
		}
		cache.compressedAlloc[layer] = true
		cache.tailHidden[layer] = make([][]float32, cache.batchSize)
		cache.tailKey[layer] = make([][]float32, cache.batchSize*cache.keyValueHeadCount)
		cache.tailValue[layer] = make([][]float32, cache.batchSize*cache.keyValueHeadCount)
		for batch := 0; batch < cache.batchSize; batch++ {
			cache.tailHidden[layer][batch] = make([]float32, ratio*embeddingDimension)
		}
		for index := range cache.tailKey[layer] {
			cache.tailKey[layer][index] = make([]float32, ratio*headDimension)
			cache.tailValue[layer][index] = make([]float32, ratio*headDimension)
		}
	}
	cache.storage.enableCompression(cache.layerRatios, embeddingDimension, kvWidth, maxBlocks, allocate)
}

// EnableIndexerKeys allocates the cached indexer key projections on every
// layer. See EnableIndexerKeysLayers.
func (cache *KVBuffer) EnableIndexerKeys(width int) {
	cache.EnableIndexerKeysLayers(width, nil)
}

// EnableIndexerKeysLayers enables the cached indexer key projections, one
// buffer per (layer, row, kv-head) whose allocate flag is true.
// EnableCompression must be called first. A nil allocate allocates every layer.
func (cache *KVBuffer) EnableIndexerKeysLayers(width int, allocate []bool) {
	cache.indexerKeyWidth = width
	cache.indexerEnabled = true
	cache.indexerAlloc = make([]bool, cache.layerCount)
	for layer := 0; layer < cache.layerCount; layer++ {
		cache.indexerAlloc[layer] = allocate == nil || allocate[layer]
	}
	cache.storage.enableIndexer(width, allocate)
}

// EnableMLA allocates the MLA latent and rope-key buffers: one shared latent row
// and one rope-key row per token per (layer, batch).
func (cache *KVBuffer) EnableMLA(latentWidth, ropeKeyWidth int) {
	cache.mlaLatentWidth = latentWidth
	cache.mlaRopeKeyWidth = ropeKeyWidth
	cache.mlaEnabled = true
	cache.storage.enableMLA(latentWidth, ropeKeyWidth)
}

// MLAEnabled reports whether the MLA latent buffers were allocated.
func (cache *KVBuffer) MLAEnabled() bool { return cache.mlaEnabled }

// MLALatentWidth returns the shared key/value latent width.
func (cache *KVBuffer) MLALatentWidth() int { return cache.mlaLatentWidth }

// MLARopeKeyWidth returns the decoupled rotary key width (NumKVHead*ropeDim).
func (cache *KVBuffer) MLARopeKeyWidth() int { return cache.mlaRopeKeyWidth }

// AppendMLALatentAt writes one token's shared latent at position+row.
func (cache *KVBuffer) AppendMLALatentAt(layer, batch, row int, latent []float32) {
	position := cache.PositionAt(batch) + row
	cache.storage.appendMLALatent(layer, batch, position, latent)
}

// AppendMLARopeKeyAt writes one token's decoupled rotary key at position+row.
func (cache *KVBuffer) AppendMLARopeKeyAt(layer, batch, row int, key []float32) {
	position := cache.PositionAt(batch) + row
	cache.storage.appendMLARope(layer, batch, position, key)
}

// MLALatent returns the first count shared latent rows.
func (cache *KVBuffer) MLALatent(layer, batch, count int) []float32 {
	return cache.storage.readMLALatent(layer, batch, count)
}

// MLARopeKey returns the first count decoupled rotary key rows.
func (cache *KVBuffer) MLARopeKey(layer, batch, count int) []float32 {
	return cache.storage.readMLARope(layer, batch, count)
}

// AppendIndexerKey stores the indexer key projection of one compressed entry.
func (cache *KVBuffer) AppendIndexerKey(layer, batch, head int, key []float32) {
	if !cache.indexerEnabled {
		return
	}
	count := cache.compressedCount[layer][batch]
	cache.storage.appendIndexer(layer, batch, head, count, key)
}

// IndexerKey returns the first count cached indexer key projections for a
// (row, kv-head).
func (cache *KVBuffer) IndexerKey(layer, batch, head, count int) []float32 {
	if !cache.indexerEnabled {
		return nil
	}
	return cache.storage.readIndexer(layer, batch, head, count)
}

// CompressedBlock returns one compressed key entry [headDim] for a (row, head).
func (cache *KVBuffer) CompressedBlock(layer, batch, head, block int) []float32 {
	return cache.storage.readCompressedBlock(layer, batch, head, block)
}

// CompressionEnabled reports whether any layer uses the packed/compressed KV
// path.
func (cache *KVBuffer) CompressionEnabled() bool { return cache.compressionEnabled }

// CompressionEnabledAt reports whether one layer uses the packed/compressed KV
// path.
func (cache *KVBuffer) CompressionEnabledAt(layer int) bool {
	return layer >= 0 && layer < len(cache.layerRatios) && cache.layerRatios[layer] >= 1
}

// CompressionRatio returns the maximum configured compression ratio.
func (cache *KVBuffer) CompressionRatio() int { return cache.compressionRatio }

// CompressionRatioAt returns one layer's configured compression ratio.
func (cache *KVBuffer) CompressionRatioAt(layer int) int {
	if layer < 0 || layer >= len(cache.layerRatios) {
		return 0
	}
	return cache.layerRatios[layer]
}

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
	cache.storage.appendCompressed(layer, batch, head, count, key, value)
}

// AdvanceCompressedCount marks that every (batch, head) gained one entry.
func (cache *KVBuffer) AdvanceCompressedCount(layer int) {
	for batch := 0; batch < cache.batchSize; batch++ {
		cache.compressedCount[layer][batch]++
	}
}

// AdvanceCompressedCountAt marks that one row gained one compressed entry.
func (cache *KVBuffer) AdvanceCompressedCountAt(layer, batch int) {
	cache.compressedCount[layer][batch]++
}

// CompressedCount returns the number of complete blocks for a row.
func (cache *KVBuffer) CompressedCount(layer, batch int) int {
	return cache.compressedCount[layer][batch]
}

// CompressedKey returns the first count compressed key rows for a (row, head).
func (cache *KVBuffer) CompressedKey(layer, batch, head, count int) []float32 {
	key, _ := cache.storage.readCompressed(layer, batch, head, count)
	return key
}

// CompressedValue returns the first count compressed value rows for a (row, head).
func (cache *KVBuffer) CompressedValue(layer, batch, head, count int) []float32 {
	_, value := cache.storage.readCompressed(layer, batch, head, count)
	return value
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
				key, value := source.storage.readRaw(layer, 0, head, position)
				destination.storage.appendRaw(layer, row, head, 0, key, value)
			}
		}
	}
	if destination.CompressionEnabled() {
		for layer := 0; layer < destination.layerCount; layer++ {
			sourceCount := 0
			if layer < len(source.compressedCount) && len(source.compressedCount[layer]) > 0 {
				sourceCount = source.compressedCount[layer][0]
			}
			for batch := 0; batch < destination.batchSize; batch++ {
				destination.compressedCount[layer][batch] = sourceCount
			}
			if destination.tailHidden[layer] != nil && source.tailHidden[layer] != nil {
				for batch := 0; batch < destination.batchSize; batch++ {
					copy(destination.tailHidden[layer][batch], source.tailHidden[layer][0])
					destination.tailLength[layer][batch] = source.tailLength[layer][0]
				}
				for index := range destination.tailKey[layer] {
					sourceIndex := index % destination.keyValueHeadCount
					copy(destination.tailKey[layer][index], source.tailKey[layer][sourceIndex])
					copy(destination.tailValue[layer][index], source.tailValue[layer][sourceIndex])
				}
			}
			for head := 0; head < destination.keyValueHeadCount; head++ {
				keyData, valueData := source.storage.readCompressed(layer, 0, head, sourceCount)
				width := 0
				if sourceCount > 0 {
					width = len(keyData) / sourceCount
				}
				for entry := 0; entry < sourceCount; entry++ {
					key := keyData[entry*width : (entry+1)*width]
					value := valueData[entry*width : (entry+1)*width]
					for batch := 0; batch < destination.batchSize; batch++ {
						destination.storage.appendCompressed(layer, batch, head, entry, key, value)
					}
				}
				indexerWidth := source.indexerKeyWidth
				if source.indexerEnabled && sourceCount > 0 {
					keys := source.storage.readIndexer(layer, 0, head, sourceCount)
					for entry := 0; entry < sourceCount; entry++ {
						part := keys[entry*indexerWidth : (entry+1)*indexerWidth]
						for batch := 0; batch < destination.batchSize; batch++ {
							destination.storage.appendIndexer(layer, batch, head, entry, part)
						}
					}
				}
			}
		}
	}

	if destination.mlaEnabled {
		latentWidth := destination.mlaLatentWidth
		ropeWidth := destination.mlaRopeKeyWidth
		for layer := 0; layer < destination.layerCount; layer++ {
			for row := 0; row < destination.batchSize; row++ {
				latent := source.storage.readMLALatent(layer, 0, position)
				rope := source.storage.readMLARope(layer, 0, position)
				for token := 0; token < position; token++ {
					destination.storage.appendMLALatent(layer, row, token, latent[token*latentWidth:(token+1)*latentWidth])
					destination.storage.appendMLARope(layer, row, token, rope[token*ropeWidth:(token+1)*ropeWidth])
				}
			}
		}
	}

	if source.ringSize > 0 {
		if destination.ringSize <= 0 {
			destination.ringWindow = append([]int(nil), source.ringWindow...)
			destination.ringEnabled = append([]bool(nil), source.ringEnabled...)
			destination.ringSize = source.ringSize
		}
		destination.ringKey = make([][][][]float32, destination.layerCount)
		destination.ringValue = make([][][][]float32, destination.layerCount)
		for layer := 0; layer < destination.layerCount; layer++ {
			if source.ringKey == nil || source.ringKey[layer] == nil {
				continue
			}
			destination.ringKey[layer] = make([][][]float32, destination.batchSize)
			destination.ringValue[layer] = make([][][]float32, destination.batchSize)
			for row := 0; row < destination.batchSize; row++ {
				destination.ringKey[layer][row] = make([][]float32, destination.keyValueHeadCount)
				destination.ringValue[layer][row] = make([][]float32, destination.keyValueHeadCount)
				for head := 0; head < destination.keyValueHeadCount; head++ {
					if source.ringKey[layer][0] == nil || source.ringKey[layer][0][head] == nil {
						continue
					}
					destination.ringKey[layer][row][head] = make([]float32, destination.ringSize*destination.headDimension)
					destination.ringValue[layer][row][head] = make([]float32, destination.ringSize*destination.headDimension)
					for token := 0; token < position; token++ {
						sourceSlot := token % source.ringSize
						destinationSlot := token % destination.ringSize
						copy(destination.ringKey[layer][row][head][destinationSlot*destination.headDimension:],
							source.ringKey[layer][0][head][sourceSlot*destination.headDimension:sourceSlot*destination.headDimension+destination.headDimension])
						copy(destination.ringValue[layer][row][head][destinationSlot*destination.headDimension:],
							source.ringValue[layer][0][head][sourceSlot*destination.headDimension:sourceSlot*destination.headDimension+destination.headDimension])
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

// IsPaged reports whether the cache uses block paging.
func (cache *KVBuffer) IsPaged() bool {
	_, ok := cache.storage.(*pagedKVStorage)
	return ok
}

// BlockTableSize returns the number of physical blocks currently allocated for
// one row.
func (cache *KVBuffer) BlockTableSize(row int) int {
	if storage, ok := cache.storage.(*pagedKVStorage); ok {
		if row >= 0 && row < len(storage.tables) {
			return len(storage.tables[row])
		}
	}
	return 0
}

// RawBlock returns one physical raw K/V block for a (layer, row, head).
func (cache *KVBuffer) RawBlock(layer, row, head, block int) ([]float32, []float32) {
	if storage, ok := cache.storage.(*pagedKVStorage); ok {
		return storage.rawPhysicalBlock(layer, row, head, block)
	}
	return nil, nil
}

// CompressedBlockData returns one physical compressed K/V block for a
// (layer, row, head).
func (cache *KVBuffer) CompressedBlockData(layer, row, head, block int) ([]float32, []float32) {
	if storage, ok := cache.storage.(*pagedKVStorage); ok {
		return storage.compressedPhysicalBlock(layer, row, head, block)
	}
	return nil, nil
}

// CompressedEntriesPerBlock returns how many compressed entries one physical
// block holds on a layer.
func (cache *KVBuffer) CompressedEntriesPerBlock(layer int) int {
	if storage, ok := cache.storage.(*pagedKVStorage); ok {
		return storage.entriesPerBlock(layer)
	}
	return 0
}

// MLABlock returns one physical MLA latent and rope-key block for a row.
func (cache *KVBuffer) MLABlock(layer, row, block int) ([]float32, []float32) {
	if storage, ok := cache.storage.(*pagedKVStorage); ok {
		return storage.mlaPhysicalBlock(layer, row, block)
	}
	return nil, nil
}

// BlockManager returns the paged block manager, or nil for a dense cache.
func (cache *KVBuffer) BlockManager() *KVBlockManager {
	if storage, ok := cache.storage.(*pagedKVStorage); ok {
		return storage.manager
	}
	return nil
}

// AdoptRow moves the state of source row 0 into destination row. Both caches
// must be paged and share a block manager. It is used when a prefilled
// sequence joins a batched decode cache without copying its KV rows.
func (cache *KVBuffer) AdoptRow(row int, source *KVBuffer) bool {
	return cache.AdoptRowFrom(row, source, 0)
}

// AdoptRowFrom moves the state of one source row into destination row.
func (cache *KVBuffer) AdoptRowFrom(row int, source *KVBuffer, sourceRow int) bool {
	if source == nil || row < 0 || row >= cache.batchSize || sourceRow < 0 || sourceRow >= source.batchSize {
		return false
	}
	destinationStorage, ok := cache.storage.(*pagedKVStorage)
	if !ok {
		return false
	}
	sourceStorage, ok := source.storage.(*pagedKVStorage)
	if !ok || destinationStorage.manager != sourceStorage.manager || destinationStorage.pools != sourceStorage.pools {
		return false
	}
	if cache.layerCount != source.layerCount || cache.keyValueHeadCount != source.keyValueHeadCount || cache.headDimension != source.headDimension {
		return false
	}
	if cache.compressionEnabled != source.compressionEnabled || cache.mlaEnabled != source.mlaEnabled || cache.indexerKeyWidth != source.indexerKeyWidth || cache.ringSize != source.ringSize {
		return false
	}
	cache.ReleaseRow(row)
	destinationStorage.tables[row] = sourceStorage.tables[sourceRow]
	sourceStorage.tables[sourceRow] = nil

	if cache.compressionEnabled && source.compressionEnabled {
		for layer := 0; layer < cache.layerCount; layer++ {
			cache.compressedCount[layer][row] = source.compressedCount[layer][sourceRow]
			cache.tailLength[layer][row] = source.tailLength[layer][sourceRow]
			if cache.tailHidden[layer] != nil && source.tailHidden[layer] != nil {
				copy(cache.tailHidden[layer][row], source.tailHidden[layer][sourceRow])
				for head := 0; head < cache.keyValueHeadCount; head++ {
					sourceIndex := sourceRow*cache.keyValueHeadCount + head
					destinationIndex := row*cache.keyValueHeadCount + head
					copy(cache.tailKey[layer][destinationIndex], source.tailKey[layer][sourceIndex])
					copy(cache.tailValue[layer][destinationIndex], source.tailValue[layer][sourceIndex])
				}
			}
		}
	}

	cache.moveRingRow(row, source, sourceRow)
	cache.copyPreviousEmbeddingRow(row, source, sourceRow)
	cache.SetPositionAt(row, source.PositionAt(sourceRow))
	return true
}

// MoveRow moves the state of source row into destination row within this cache.
func (cache *KVBuffer) MoveRow(destination, source int) bool {
	if destination < 0 || destination >= cache.batchSize || source < 0 || source >= cache.batchSize || destination == source {
		return false
	}
	storage, ok := cache.storage.(*pagedKVStorage)
	if !ok {
		return false
	}
	cache.ReleaseRow(destination)
	storage.moveRow(destination, source)

	if cache.compressionEnabled {
		for layer := 0; layer < cache.layerCount; layer++ {
			cache.compressedCount[layer][destination] = cache.compressedCount[layer][source]
			cache.tailLength[layer][destination] = cache.tailLength[layer][source]
			cache.compressedCount[layer][source] = 0
			cache.tailLength[layer][source] = 0
			if cache.tailHidden[layer] != nil {
				copy(cache.tailHidden[layer][destination], cache.tailHidden[layer][source])
				clear(cache.tailHidden[layer][source])
				for head := 0; head < cache.keyValueHeadCount; head++ {
					sourceIndex := source*cache.keyValueHeadCount + head
					destinationIndex := destination*cache.keyValueHeadCount + head
					copy(cache.tailKey[layer][destinationIndex], cache.tailKey[layer][sourceIndex])
					copy(cache.tailValue[layer][destinationIndex], cache.tailValue[layer][sourceIndex])
					clear(cache.tailKey[layer][sourceIndex])
					clear(cache.tailValue[layer][sourceIndex])
				}
			}
		}
	}

	cache.moveRingRow(destination, cache, source)
	cache.copyPreviousEmbeddingRow(destination, cache, source)
	cache.SetPositionAt(destination, cache.PositionAt(source))
	cache.SetPositionAt(source, 0)
	return true
}

// ReleaseRow frees one paged row and clears its per-row state.
func (cache *KVBuffer) ReleaseRow(row int) {
	if row < 0 || row >= cache.batchSize {
		return
	}
	if storage, ok := cache.storage.(*pagedKVStorage); ok {
		storage.releaseRow(row)
	}
	if cache.compressionEnabled {
		for layer := 0; layer < cache.layerCount; layer++ {
			cache.compressedCount[layer][row] = 0
			cache.tailLength[layer][row] = 0
			if cache.tailHidden[layer] != nil {
				clear(cache.tailHidden[layer][row])
				for head := 0; head < cache.keyValueHeadCount; head++ {
					index := row*cache.keyValueHeadCount + head
					clear(cache.tailKey[layer][index])
					clear(cache.tailValue[layer][index])
				}
			}
		}
	}
	if cache.ringKey != nil {
		for layer := 0; layer < cache.layerCount; layer++ {
			if cache.ringKey[layer] == nil || row >= len(cache.ringKey[layer]) || cache.ringKey[layer][row] == nil {
				continue
			}
			if len(cache.ringKey[layer][row]) < cache.keyValueHeadCount {
				continue
			}
			for head := 0; head < cache.keyValueHeadCount; head++ {
				cache.ringKey[layer][row][head] = nil
				if cache.ringValue[layer] != nil && row < len(cache.ringValue[layer]) && cache.ringValue[layer][row] != nil {
					cache.ringValue[layer][row][head] = nil
				}
			}
		}
	}
	if cache.previousEmbedding != nil && row < cache.previousEmbedding.Shape[0] {
		channels := cache.previousEmbedding.Shape[2]
		clear(cache.previousEmbedding.Data[row*channels : (row+1)*channels])
	}
	if cache.rowPositions == nil {
		cache.rowPositions = make([]int32, cache.batchSize)
		for candidate := range cache.rowPositions {
			cache.rowPositions[candidate] = cache.sequenceLength
		}
	}
	if row < len(cache.rowPositions) {
		cache.rowPositions[row] = 0
	}
}

func (cache *KVBuffer) moveRingRow(destination int, source *KVBuffer, sourceRow int) {
	if cache.ringSize <= 0 {
		return
	}
	if cache.ringKey == nil {
		cache.ringKey = make([][][][]float32, cache.layerCount)
		cache.ringValue = make([][][][]float32, cache.layerCount)
	}
	for layer := 0; layer < cache.layerCount; layer++ {
		if source.ringKey == nil || source.ringKey[layer] == nil || sourceRow >= len(source.ringKey[layer]) {
			continue
		}
		if len(source.ringKey[layer][sourceRow]) < cache.keyValueHeadCount {
			continue
		}
		if cache.ringKey[layer] == nil {
			cache.ringKey[layer] = make([][][]float32, cache.batchSize)
			cache.ringValue[layer] = make([][][]float32, cache.batchSize)
		}
		if cache.ringKey[layer][destination] == nil {
			cache.ringKey[layer][destination] = make([][]float32, cache.keyValueHeadCount)
			cache.ringValue[layer][destination] = make([][]float32, cache.keyValueHeadCount)
		}
		for head := 0; head < cache.keyValueHeadCount; head++ {
			cache.ringKey[layer][destination][head] = source.ringKey[layer][sourceRow][head]
			cache.ringValue[layer][destination][head] = source.ringValue[layer][sourceRow][head]
			source.ringKey[layer][sourceRow][head] = nil
			source.ringValue[layer][sourceRow][head] = nil
		}
	}
}

func (cache *KVBuffer) copyPreviousEmbeddingRow(destination int, source *KVBuffer, sourceRow int) {
	if source.previousEmbedding == nil || sourceRow >= source.previousEmbedding.Shape[0] {
		return
	}
	channels := source.previousEmbedding.Shape[2]
	previous := cache.ensurePreviousEmbedding(channels)
	if destination >= previous.Shape[0] {
		return
	}
	copy(previous.Data[destination*channels:(destination+1)*channels],
		source.previousEmbedding.Data[sourceRow*channels:(sourceRow+1)*channels])
}

// CloneRow returns a cache that owns one row's state. The returned cache keeps
// the source batch width but every other row is empty; row 0 is the requested
// row. It is used to snapshot one row of a shared prefill cache.
func (cache *KVBuffer) CloneRow(row int) *KVBuffer {
	if row < 0 || row >= cache.batchSize {
		return nil
	}
	clone := cache.Clone()
	for candidate := 0; candidate < clone.batchSize; candidate++ {
		if candidate != row {
			clone.ReleaseRow(candidate)
		}
	}
	if row != 0 && !clone.MoveRow(0, row) {
		clone.Release()
		return nil
	}
	return clone
}

// Clone returns a copy of the cache. Paged clones share blocks through the
// block manager's refcounts; compression/indexer/MLA state and the smear
// embedding are deep-copied.
func (cache *KVBuffer) Clone() *KVBuffer {
	clone := &KVBuffer{
		batchSize:              cache.batchSize,
		maximumSequenceLength:  cache.maximumSequenceLength,
		layerCount:             cache.layerCount,
		keyValueHeadCount:      cache.keyValueHeadCount,
		headDimension:          cache.headDimension,
		sequenceLength:         cache.sequenceLength,
		rowPositions:           append([]int32(nil), cache.rowPositions...),
		rawStripped:            cache.rawStripped,
		compressionEnabled:     cache.compressionEnabled,
		compressionRatio:       cache.compressionRatio,
		compressedMaxBlocks:    cache.compressedMaxBlocks,
		compressedEmbeddingDim: cache.compressedEmbeddingDim,
		compressedKVWidth:      cache.compressedKVWidth,
		indexerKeyWidth:        cache.indexerKeyWidth,
		indexerEnabled:         cache.indexerEnabled,
		compressedAlloc:        append([]bool(nil), cache.compressedAlloc...),
		indexerAlloc:           append([]bool(nil), cache.indexerAlloc...),
		mlaLatentWidth:         cache.mlaLatentWidth,
		mlaRopeKeyWidth:        cache.mlaRopeKeyWidth,
		mlaEnabled:             cache.mlaEnabled,
		layerRatios:            append([]int(nil), cache.layerRatios...),
		maxBlocksByLayer:       append([]int(nil), cache.maxBlocksByLayer...),
		storage:                cache.storage.clone(),
		ringWindow:             append([]int(nil), cache.ringWindow...),
		ringEnabled:            append([]bool(nil), cache.ringEnabled...),
		ringSize:               cache.ringSize,
	}
	if cache.compressionEnabled {
		clone.tailHidden = make([][][]float32, cache.layerCount)
		clone.tailKey = make([][][]float32, cache.layerCount)
		clone.tailValue = make([][][]float32, cache.layerCount)
		clone.tailLength = make([][]int, cache.layerCount)
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
		}
	}

	if cache.ringKey != nil {
		clone.ringKey = make([][][][]float32, cache.layerCount)
		clone.ringValue = make([][][][]float32, cache.layerCount)
		for layer := 0; layer < cache.layerCount; layer++ {
			if cache.ringKey[layer] == nil {
				continue
			}
			clone.ringKey[layer] = make([][][]float32, len(cache.ringKey[layer]))
			clone.ringValue[layer] = make([][][]float32, len(cache.ringValue[layer]))
			for row := range cache.ringKey[layer] {
				if cache.ringKey[layer][row] == nil || len(cache.ringKey[layer][row]) < cache.keyValueHeadCount {
					continue
				}
				clone.ringKey[layer][row] = make([][]float32, len(cache.ringKey[layer][row]))
				clone.ringValue[layer][row] = make([][]float32, len(cache.ringValue[layer][row]))
				for head := range cache.ringKey[layer][row] {
					if cache.ringKey[layer][row][head] == nil {
						continue
					}
					clone.ringKey[layer][row][head] = append([]float32(nil), cache.ringKey[layer][row][head]...)
					clone.ringValue[layer][row][head] = append([]float32(nil), cache.ringValue[layer][row][head]...)
				}
			}
		}
	}

	if cache.previousEmbedding != nil {
		clone.previousEmbedding = cache.previousEmbedding.Clone()
	}
	return clone
}

// Release frees the paged blocks owned by this cache. The cache must not be
// used after Release.
func (cache *KVBuffer) Release() { cache.storage.release() }

// CompressedBytesAllocated returns the bytes allocated for compressed key/value
// storage and cached indexer keys across every layer.
func (cache *KVBuffer) CompressedBytesAllocated() int {
	return cache.storage.compressedBytesAllocated()
}

// BytesAllocated returns the approximate bytes allocated for every key/value,
// compression, indexer, tail, and previous-embedding buffer.
func (cache *KVBuffer) BytesAllocated() int {
	total := cache.storage.bytesAllocated()
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
	for layer := 0; layer < cache.layerCount; layer++ {
		if cache.ringKey == nil || cache.ringKey[layer] == nil {
			continue
		}
		for _, row := range cache.ringKey[layer] {
			for _, block := range row {
				total += len(block) * 4
			}
		}
		for _, row := range cache.ringValue[layer] {
			for _, block := range row {
				total += len(block) * 4
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
func (cache *KVBuffer) RawStripped() bool { return cache.rawStripped || cache.storage.rawStripped() }

// StripRaw drops the raw key/value rows and the local sliding-window state while
// keeping the global compressed KV, cached indexer keys, and compression tail.
// Paged storage records the policy instead of mutating shared blocks, so the
// source snapshot stays intact.
func (cache *KVBuffer) StripRaw() {
	if !cache.compressionEnabled {
		return
	}
	cache.storage.stripRaw()
	cache.ringKey = nil
	cache.ringValue = nil
	cache.rawStripped = true
}

// MarkRawRebuilt records that a stripped snapshot's local rows have been rebuilt
// by bounded replay, so subsequent appends no longer clear them.
func (cache *KVBuffer) MarkRawRebuilt() {
	cache.rawStripped = false
	cache.storage.unstripRaw()
}

// KVMark is a lightweight snapshot of the mutable cache state needed to roll
// back speculative or rejected tokens. Paged block tables are captured so
// blocks appended or copied during verification can be released.
type KVMark struct {
	position          int32
	rowPositions      []int32
	previousEmbedding *tensors.Tensor
	compressedCount   [][]int
	tailLength        [][]int
	tailHidden        [][][]float32
	tailKey           [][][]float32
	tailValue         [][][]float32
	tables            [][]int
}

// Mark snapshots the mutable cache state.
func (cache *KVBuffer) Mark() KVMark {
	mark := KVMark{position: cache.sequenceLength, rowPositions: append([]int32(nil), cache.rowPositions...), tables: cache.storage.markTables()}
	if cache.previousEmbedding != nil {
		mark.previousEmbedding = cache.previousEmbedding.Clone()
	}
	if !cache.compressionEnabled {
		return mark
	}
	mark.compressedCount = make([][]int, cache.layerCount)
	mark.tailLength = make([][]int, cache.layerCount)
	mark.tailHidden = make([][][]float32, cache.layerCount)
	mark.tailKey = make([][][]float32, cache.layerCount)
	mark.tailValue = make([][][]float32, cache.layerCount)
	for layer := 0; layer < cache.layerCount; layer++ {
		mark.compressedCount[layer] = append([]int(nil), cache.compressedCount[layer]...)
		mark.tailLength[layer] = append([]int(nil), cache.tailLength[layer]...)
		if cache.tailHidden[layer] == nil {
			continue
		}
		mark.tailHidden[layer] = make([][]float32, len(cache.tailHidden[layer]))
		for batch := range cache.tailHidden[layer] {
			mark.tailHidden[layer][batch] = append([]float32(nil), cache.tailHidden[layer][batch]...)
		}
		mark.tailKey[layer] = make([][]float32, len(cache.tailKey[layer]))
		mark.tailValue[layer] = make([][]float32, len(cache.tailValue[layer]))
		for index := range cache.tailKey[layer] {
			mark.tailKey[layer][index] = append([]float32(nil), cache.tailKey[layer][index]...)
			mark.tailValue[layer][index] = append([]float32(nil), cache.tailValue[layer][index]...)
		}
	}
	return mark
}

// Rollback restores a state captured by Mark and releases blocks that were
// appended or copied for rejected tokens.
func (cache *KVBuffer) Rollback(mark KVMark) {
	cache.sequenceLength = mark.position
	cache.rowPositions = append([]int32(nil), mark.rowPositions...)
	cache.previousEmbedding = mark.previousEmbedding
	cache.storage.rollbackTables(mark.tables)
	if !cache.compressionEnabled {
		return
	}
	for layer := 0; layer < cache.layerCount; layer++ {
		if layer < len(mark.compressedCount) {
			copy(cache.compressedCount[layer], mark.compressedCount[layer])
		}
		if layer < len(mark.tailLength) {
			copy(cache.tailLength[layer], mark.tailLength[layer])
		}
		if layer >= len(mark.tailHidden) || mark.tailHidden[layer] == nil {
			continue
		}
		for batch := range mark.tailHidden[layer] {
			copy(cache.tailHidden[layer][batch], mark.tailHidden[layer][batch])
		}
		for index := range mark.tailKey[layer] {
			copy(cache.tailKey[layer][index], mark.tailKey[layer][index])
			copy(cache.tailValue[layer][index], mark.tailValue[layer][index])
		}
	}
}
