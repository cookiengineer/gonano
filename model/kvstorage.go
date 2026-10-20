package model

import (
	"sync"

	"github.com/cookiengineer/gonano/tensors"
)

// kvStorage is the pluggable backing store for the large KV arrays. KVBuffer
// owns the per-sequence bookkeeping (position, compression tails, counters and
// previous embedding); the storage owns the per-token rows.
//
// Positions are logical token positions. Compressed and indexer positions are
// logical compressed-entry positions. Implementations must tolerate appends
// that skip forward within the current block (prefix reuse and replay).
type kvStorage interface {
	enableCompression(ratios []int, embeddingDimension, kvWidth int, maxBlocks []int, allocate []bool)
	enableIndexer(width int, allocate []bool)
	enableMLA(latentWidth, ropeKeyWidth int)

	appendRaw(layer, row, head, position int, key, value []float32)
	readRaw(layer, row, head, end int) (key, value []float32)

	appendCompressed(layer, row, head, position int, key, value []float32)
	readCompressed(layer, row, head, end int) (key, value []float32)
	readCompressedBlock(layer, row, head, block int) []float32

	appendIndexer(layer, row, head, position int, key []float32)
	readIndexer(layer, row, head, end int) []float32

	appendMLALatent(layer, row, position int, latent []float32)
	appendMLARope(layer, row, position int, key []float32)
	readMLALatent(layer, row, end int) []float32
	readMLARope(layer, row, end int) []float32

	clone() kvStorage
	release()
	reset()
	stripRaw()
	unstripRaw()
	rawStripped() bool

	bytesAllocated() int
	compressedBytesAllocated() int

	markTables() [][]int
	rollbackTables(mark [][]int)
}

// denseKVStorage is the historical contiguous backing store. It is retained as
// the reference implementation for parity tests and for callers that explicitly
// request it.
type denseKVStorage struct {
	batchSize             int
	maximumSequenceLength int
	layerCount            int
	keyValueHeadCount     int
	headDimension         int

	keyCache   [][]*tensors.Tensor
	valueCache [][]*tensors.Tensor

	compressionEnabled bool
	layerRatios        []int
	maxBlocksByLayer   []int
	embeddingDimension int
	kvWidth            int

	compressedKey   [][]*tensors.Tensor
	compressedValue [][]*tensors.Tensor

	indexerKeyWidth int
	indexerKey      [][]*tensors.Tensor

	mlaLatentWidth  int
	mlaRopeKeyWidth int
	mlaLatent       [][]*tensors.Tensor
	mlaRopeKey      [][]*tensors.Tensor

	rawStrippedFlag bool
	rawCleared      map[int]bool
}

func newDenseKVStorage(batchSize, maximumSequenceLength, layerCount, keyValueHeadCount, headDimension int) *denseKVStorage {
	storage := &denseKVStorage{
		batchSize:             batchSize,
		maximumSequenceLength: maximumSequenceLength,
		layerCount:            layerCount,
		keyValueHeadCount:     keyValueHeadCount,
		headDimension:         headDimension,
		keyCache:              make([][]*tensors.Tensor, layerCount),
		valueCache:            make([][]*tensors.Tensor, layerCount),
	}
	for layer := 0; layer < layerCount; layer++ {
		storage.keyCache[layer] = make([]*tensors.Tensor, batchSize*keyValueHeadCount)
		storage.valueCache[layer] = make([]*tensors.Tensor, batchSize*keyValueHeadCount)
		for index := 0; index < batchSize*keyValueHeadCount; index++ {
			storage.keyCache[layer][index] = tensors.New(maximumSequenceLength, headDimension)
			storage.valueCache[layer][index] = tensors.New(maximumSequenceLength, headDimension)
		}
	}
	return storage
}

func (storage *denseKVStorage) enableCompression(ratios []int, embeddingDimension, kvWidth int, maxBlocks []int, allocate []bool) {
	storage.compressionEnabled = true
	storage.layerRatios = append([]int(nil), ratios...)
	storage.maxBlocksByLayer = append([]int(nil), maxBlocks...)
	storage.embeddingDimension = embeddingDimension
	storage.kvWidth = kvWidth
	storage.compressedKey = make([][]*tensors.Tensor, storage.layerCount)
	storage.compressedValue = make([][]*tensors.Tensor, storage.layerCount)
	headDimension := kvWidth / storage.keyValueHeadCount
	for layer := 0; layer < storage.layerCount; layer++ {
		ratio := 1
		if layer < len(ratios) {
			ratio = ratios[layer]
		}
		if ratio < 1 {
			continue
		}
		if allocate != nil && !allocate[layer] {
			continue
		}
		blocks := 0
		if layer < len(maxBlocks) {
			blocks = maxBlocks[layer]
		}
		storage.compressedKey[layer] = make([]*tensors.Tensor, storage.batchSize*storage.keyValueHeadCount)
		storage.compressedValue[layer] = make([]*tensors.Tensor, storage.batchSize*storage.keyValueHeadCount)
		for index := range storage.compressedKey[layer] {
			storage.compressedKey[layer][index] = tensors.New(blocks, headDimension)
			storage.compressedValue[layer][index] = tensors.New(blocks, headDimension)
		}
	}
}

func (storage *denseKVStorage) enableIndexer(width int, allocate []bool) {
	storage.indexerKeyWidth = width
	storage.indexerKey = make([][]*tensors.Tensor, storage.layerCount)
	for layer := 0; layer < storage.layerCount; layer++ {
		if allocate != nil && !allocate[layer] {
			continue
		}
		blocks := 0
		if layer < len(storage.maxBlocksByLayer) {
			blocks = storage.maxBlocksByLayer[layer]
		}
		storage.indexerKey[layer] = make([]*tensors.Tensor, storage.batchSize*storage.keyValueHeadCount)
		for index := range storage.indexerKey[layer] {
			storage.indexerKey[layer][index] = tensors.New(blocks, width)
		}
	}
}

func (storage *denseKVStorage) enableMLA(latentWidth, ropeKeyWidth int) {
	storage.mlaLatentWidth = latentWidth
	storage.mlaRopeKeyWidth = ropeKeyWidth
	storage.mlaLatent = make([][]*tensors.Tensor, storage.layerCount)
	storage.mlaRopeKey = make([][]*tensors.Tensor, storage.layerCount)
	for layer := 0; layer < storage.layerCount; layer++ {
		storage.mlaLatent[layer] = make([]*tensors.Tensor, storage.batchSize)
		storage.mlaRopeKey[layer] = make([]*tensors.Tensor, storage.batchSize)
		for batch := 0; batch < storage.batchSize; batch++ {
			storage.mlaLatent[layer][batch] = tensors.New(storage.maximumSequenceLength, latentWidth)
			storage.mlaRopeKey[layer][batch] = tensors.New(storage.maximumSequenceLength, ropeKeyWidth)
		}
	}
}

func (storage *denseKVStorage) appendRaw(layer, row, head, position int, key, value []float32) {
	index := row*storage.keyValueHeadCount + head
	tokenCount := len(key) / storage.headDimension
	keyBuffer := storage.keyCache[layer][index].Data
	valueBuffer := storage.valueCache[layer][index].Data
	copy(keyBuffer[position*storage.headDimension:], key)
	copy(valueBuffer[position*storage.headDimension:], value)
	_ = tokenCount
}

func (storage *denseKVStorage) readRaw(layer, row, head, end int) ([]float32, []float32) {
	index := row*storage.keyValueHeadCount + head
	limit := end * storage.headDimension
	return storage.keyCache[layer][index].Data[:limit], storage.valueCache[layer][index].Data[:limit]
}

func (storage *denseKVStorage) appendCompressed(layer, row, head, position int, key, value []float32) {
	if storage.compressedKey[layer] == nil {
		return
	}
	index := row*storage.keyValueHeadCount + head
	headDimension := storage.compressedKey[layer][index].Shape[1]
	copy(storage.compressedKey[layer][index].Data[position*headDimension:], key)
	copy(storage.compressedValue[layer][index].Data[position*headDimension:], value)
}

func (storage *denseKVStorage) readCompressed(layer, row, head, end int) ([]float32, []float32) {
	if storage.compressedKey == nil || storage.compressedKey[layer] == nil {
		return nil, nil
	}
	index := row*storage.keyValueHeadCount + head
	headDimension := storage.compressedKey[layer][index].Shape[1]
	limit := end * headDimension
	return storage.compressedKey[layer][index].Data[:limit], storage.compressedValue[layer][index].Data[:limit]
}

func (storage *denseKVStorage) readCompressedBlock(layer, row, head, block int) []float32 {
	if storage.compressedKey == nil || storage.compressedKey[layer] == nil {
		return nil
	}
	index := row*storage.keyValueHeadCount + head
	headDimension := storage.compressedKey[layer][index].Shape[1]
	return storage.compressedKey[layer][index].Data[block*headDimension : (block+1)*headDimension]
}

func (storage *denseKVStorage) appendIndexer(layer, row, head, position int, key []float32) {
	if storage.indexerKey == nil || storage.indexerKey[layer] == nil {
		return
	}
	index := row*storage.keyValueHeadCount + head
	copy(storage.indexerKey[layer][index].Data[position*storage.indexerKeyWidth:], key)
}

func (storage *denseKVStorage) readIndexer(layer, row, head, end int) []float32 {
	if storage.indexerKey == nil || storage.indexerKey[layer] == nil {
		return nil
	}
	index := row*storage.keyValueHeadCount + head
	return storage.indexerKey[layer][index].Data[:end*storage.indexerKeyWidth]
}

func (storage *denseKVStorage) appendMLALatent(layer, row, position int, latent []float32) {
	if storage.mlaLatent == nil {
		return
	}
	copy(storage.mlaLatent[layer][row].Data[position*storage.mlaLatentWidth:], latent)
}

func (storage *denseKVStorage) appendMLARope(layer, row, position int, key []float32) {
	if storage.mlaRopeKey == nil {
		return
	}
	copy(storage.mlaRopeKey[layer][row].Data[position*storage.mlaRopeKeyWidth:], key)
}

func (storage *denseKVStorage) readMLALatent(layer, row, end int) []float32 {
	if storage.mlaLatent == nil {
		return nil
	}
	return storage.mlaLatent[layer][row].Data[:end*storage.mlaLatentWidth]
}

func (storage *denseKVStorage) readMLARope(layer, row, end int) []float32 {
	if storage.mlaRopeKey == nil {
		return nil
	}
	return storage.mlaRopeKey[layer][row].Data[:end*storage.mlaRopeKeyWidth]
}

func (storage *denseKVStorage) clone() kvStorage {
	clone := &denseKVStorage{
		batchSize:             storage.batchSize,
		maximumSequenceLength: storage.maximumSequenceLength,
		layerCount:            storage.layerCount,
		keyValueHeadCount:     storage.keyValueHeadCount,
		headDimension:         storage.headDimension,
		compressionEnabled:    storage.compressionEnabled,
		embeddingDimension:    storage.embeddingDimension,
		kvWidth:               storage.kvWidth,
		indexerKeyWidth:       storage.indexerKeyWidth,
		mlaLatentWidth:        storage.mlaLatentWidth,
		mlaRopeKeyWidth:       storage.mlaRopeKeyWidth,
		rawStrippedFlag:       storage.rawStrippedFlag,
		layerRatios:           append([]int(nil), storage.layerRatios...),
		maxBlocksByLayer:      append([]int(nil), storage.maxBlocksByLayer...),
		keyCache:              cloneTensorGrid(storage.keyCache),
		valueCache:            cloneTensorGrid(storage.valueCache),
		compressedKey:         cloneTensorGrid(storage.compressedKey),
		compressedValue:       cloneTensorGrid(storage.compressedValue),
		indexerKey:            cloneTensorGrid(storage.indexerKey),
		mlaLatent:             cloneTensorGrid(storage.mlaLatent),
		mlaRopeKey:            cloneTensorGrid(storage.mlaRopeKey),
	}
	return clone
}

func cloneTensorGrid(grid [][]*tensors.Tensor) [][]*tensors.Tensor {
	if grid == nil {
		return nil
	}
	clone := make([][]*tensors.Tensor, len(grid))
	for index := range grid {
		if grid[index] == nil {
			continue
		}
		clone[index] = make([]*tensors.Tensor, len(grid[index]))
		for element := range grid[index] {
			if grid[index][element] != nil {
				clone[index][element] = grid[index][element].Clone()
			}
		}
	}
	return clone
}

func (storage *denseKVStorage) release() {}

func (storage *denseKVStorage) reset() {}

func (storage *denseKVStorage) stripRaw() {
	if !storage.compressionEnabled {
		return
	}
	for layer := 0; layer < storage.layerCount; layer++ {
		for index := range storage.keyCache[layer] {
			clear(storage.keyCache[layer][index].Data)
			clear(storage.valueCache[layer][index].Data)
		}
	}
	storage.rawStrippedFlag = true
}

func (storage *denseKVStorage) unstripRaw() { storage.rawStrippedFlag = false }

func (storage *denseKVStorage) rawStripped() bool { return storage.rawStrippedFlag }

func (storage *denseKVStorage) bytesAllocated() int {
	total := 0
	for _, grid := range [][][]*tensors.Tensor{storage.keyCache, storage.valueCache, storage.compressedKey, storage.compressedValue, storage.indexerKey, storage.mlaLatent, storage.mlaRopeKey} {
		for _, row := range grid {
			for _, tensor := range row {
				if tensor != nil {
					total += tensor.Numel() * 4
				}
			}
		}
	}
	return total
}

func (storage *denseKVStorage) compressedBytesAllocated() int {
	total := 0
	for _, grid := range [][][]*tensors.Tensor{storage.compressedKey, storage.compressedValue, storage.indexerKey} {
		for _, row := range grid {
			for _, tensor := range row {
				if tensor != nil {
					total += tensor.Numel() * 4
				}
			}
		}
	}
	return total
}

func (storage *denseKVStorage) markTables() [][]int { return nil }

func (storage *denseKVStorage) rollbackTables([][]int) {}

// KVBlockPools holds the per-layer block data shared by every cache that uses
// the same KVBlockManager and model layout. Sharing the pools is what lets a
// prefilled row move into a batched decode cache without copying KV rows.
type KVBlockPools struct {
	mutex sync.Mutex

	manager           *KVBlockManager
	blockCount        int
	layerCount        int
	keyValueHeadCount int
	headDimension     int

	rawKey      [][][][]float32
	rawValue    [][][][]float32
	compressedK [][][][]float32
	compressedV [][][][]float32
	indexerK    [][][][]float32
	mlaLatent   [][][]float32
	mlaRopeKey  [][][]float32
}

// NewKVBlockPools creates a shared block-data pool set for a manager and model
// geometry.
func NewKVBlockPools(manager *KVBlockManager, layerCount, keyValueHeadCount, headDimension int) *KVBlockPools {
	if manager == nil {
		manager = NewKVBlockManager(1)
	}
	return &KVBlockPools{
		manager:           manager,
		blockCount:        manager.BlockCount(),
		layerCount:        layerCount,
		keyValueHeadCount: keyValueHeadCount,
		headDimension:     headDimension,
		rawKey:            make([][][][]float32, layerCount),
		rawValue:          make([][][][]float32, layerCount),
		compressedK:       make([][][][]float32, layerCount),
		compressedV:       make([][][][]float32, layerCount),
		indexerK:          make([][][][]float32, layerCount),
		mlaLatent:         make([][][]float32, layerCount),
		mlaRopeKey:        make([][][]float32, layerCount),
	}
}

// Manager returns the shared block manager.
func (pools *KVBlockPools) Manager() *KVBlockManager { return pools.manager }

// pagedKVStorage stores token rows in 128-token blocks. A block ID is shared
// across every layer pool; writing to a shared block copies it first.
type pagedKVStorage struct {
	manager *KVBlockManager
	pools   *KVBlockPools

	batchSize             int
	maximumSequenceLength int
	layerCount            int
	keyValueHeadCount     int
	headDimension         int
	blockCount            int

	tables [][]int

	layerRatios      []int
	maxBlocksByLayer []int
	embeddingDim     int
	kvWidth          int
	indexerKeyWidth  int
	mlaLatentWidth   int
	mlaRopeKeyWidth  int

	rawKey      [][][][]float32
	rawValue    [][][][]float32
	compressedK [][][][]float32
	compressedV [][][][]float32
	indexerK    [][][][]float32
	mlaLatent   [][][]float32
	mlaRopeKey  [][][]float32

	rawScratchKey   [][][]float32
	rawScratchValue [][][]float32
	compScratchKey  [][][]float32
	compScratchVal  [][][]float32
	idxScratch      [][][]float32
	latentScratch   [][]float32
	ropeScratch     [][]float32

	rawStrippedFlag bool
	rawCleared      map[int]bool
}

func newPagedKVStorage(manager *KVBlockManager, batchSize, maximumSequenceLength, layerCount, keyValueHeadCount, headDimension int) *pagedKVStorage {
	return newPagedKVStorageWithPools(manager, nil, batchSize, maximumSequenceLength, layerCount, keyValueHeadCount, headDimension)
}

func newPagedKVStorageWithPools(manager *KVBlockManager, pools *KVBlockPools, batchSize, maximumSequenceLength, layerCount, keyValueHeadCount, headDimension int) *pagedKVStorage {
	if manager == nil && pools != nil {
		manager = pools.manager
	}
	if manager == nil {
		blocksPerRow := (maximumSequenceLength + KVBlockSize - 1) / KVBlockSize
		if blocksPerRow < 1 {
			blocksPerRow = 1
		}
		// Spare blocks leave room for copy-on-write when cloned caches diverge
		// from a shared partial block.
		manager = NewKVBlockManager(blocksPerRow*batchSize + 8*batchSize)
	}
	if pools == nil {
		pools = NewKVBlockPools(manager, layerCount, keyValueHeadCount, headDimension)
	}
	return &pagedKVStorage{
		manager:               manager,
		pools:                 pools,
		batchSize:             batchSize,
		maximumSequenceLength: maximumSequenceLength,
		layerCount:            layerCount,
		keyValueHeadCount:     keyValueHeadCount,
		headDimension:         headDimension,
		blockCount:            manager.BlockCount(),
		tables:                make([][]int, batchSize),
		rawKey:                pools.rawKey,
		rawValue:              pools.rawValue,
		compressedK:           pools.compressedK,
		compressedV:           pools.compressedV,
		indexerK:              pools.indexerK,
		mlaLatent:             pools.mlaLatent,
		mlaRopeKey:            pools.mlaRopeKey,
	}
}

func (storage *pagedKVStorage) enableCompression(ratios []int, embeddingDimension, kvWidth int, maxBlocks []int, allocate []bool) {
	storage.layerRatios = append([]int(nil), ratios...)
	storage.maxBlocksByLayer = append([]int(nil), maxBlocks...)
	storage.embeddingDim = embeddingDimension
	storage.kvWidth = kvWidth
}

func (storage *pagedKVStorage) enableIndexer(width int, allocate []bool) {
	storage.indexerKeyWidth = width
}

func (storage *pagedKVStorage) enableMLA(latentWidth, ropeKeyWidth int) {
	storage.mlaLatentWidth = latentWidth
	storage.mlaRopeKeyWidth = ropeKeyWidth
}

func (storage *pagedKVStorage) ratioAt(layer int) int {
	if layer >= 0 && layer < len(storage.layerRatios) {
		if ratio := storage.layerRatios[layer]; ratio >= 1 {
			return ratio
		}
	}
	return 1
}

func (storage *pagedKVStorage) entriesPerBlock(layer int) int {
	ratio := storage.ratioAt(layer)
	if ratio < 1 {
		ratio = 1
	}
	entries := KVBlockSize / ratio
	if entries < 1 {
		entries = 1
	}
	return entries
}

func (storage *pagedKVStorage) ensureTable(row, blockIndex int) int {
	if row < 0 || row >= storage.batchSize {
		panic("model: KV cache row out of range")
	}
	for len(storage.tables[row]) <= blockIndex {
		block := storage.manager.Allocate()
		if block < 0 {
			panic("model: KV block pool exhausted")
		}
		storage.tables[row] = append(storage.tables[row], block)
	}
	return storage.tables[row][blockIndex]
}

// cowBlock makes blockIndex private for row and returns the new block ID.
func (storage *pagedKVStorage) cowBlock(row, blockIndex int) int {
	oldBlock := storage.tables[row][blockIndex]
	if storage.manager.RefCount(oldBlock) <= 1 {
		return oldBlock
	}
	newBlock := storage.manager.Allocate()
	if newBlock < 0 {
		panic("model: KV block pool exhausted during copy-on-write")
	}
	storage.copyBlock(oldBlock, newBlock, row)
	storage.tables[row][blockIndex] = newBlock
	storage.manager.Free(oldBlock)
	return newBlock
}

func (storage *pagedKVStorage) privateBlock(row, blockIndex int) int {
	storage.ensureTable(row, blockIndex)
	block := storage.cowBlock(row, blockIndex)
	if storage.rawStrippedFlag {
		storage.clearRawBlock(block)
	}
	return block
}

// clearRawBlock zeros the raw rows of a block once, after a stripped snapshot is
// replayed into it. Compressed, indexer and MLA rows are left untouched.
func (storage *pagedKVStorage) clearRawBlock(block int) {
	if storage.rawCleared == nil {
		storage.rawCleared = make(map[int]bool)
	}
	if storage.rawCleared[block] {
		return
	}
	storage.rawCleared[block] = true
	for layer := 0; layer < storage.layerCount; layer++ {
		if storage.rawKey != nil && storage.rawKey[layer] != nil {
			for head := 0; head < storage.keyValueHeadCount; head++ {
				if storage.rawKey[layer][head] != nil && block < len(storage.rawKey[layer][head]) {
					clear(storage.rawKey[layer][head][block])
				}
				if storage.rawValue != nil && storage.rawValue[layer] != nil && storage.rawValue[layer][head] != nil && block < len(storage.rawValue[layer][head]) {
					clear(storage.rawValue[layer][head][block])
				}
			}
		}
	}
}

func (storage *pagedKVStorage) copyBlock(source, destination, row int) {
	if storage.pools != nil {
		storage.pools.mutex.Lock()
		defer storage.pools.mutex.Unlock()
	}
	copyData := func(data [][]float32) {
		if source < 0 || source >= len(data) || destination < 0 || destination >= len(data) {
			return
		}
		if block := data[source]; block != nil {
			data[destination] = append([]float32(nil), block...)
		} else {
			data[destination] = nil
		}
	}
	headCount := storage.keyValueHeadCount
	for layer := 0; layer < storage.layerCount; layer++ {
		for head := 0; head < headCount; head++ {
			if storage.rawKey != nil && storage.rawKey[layer] != nil {
				copyData(storage.rawKey[layer][head])
			}
			if storage.rawValue != nil && storage.rawValue[layer] != nil {
				copyData(storage.rawValue[layer][head])
			}
			if storage.compressedK != nil && storage.compressedK[layer] != nil {
				copyData(storage.compressedK[layer][head])
			}
			if storage.compressedV != nil && storage.compressedV[layer] != nil {
				copyData(storage.compressedV[layer][head])
			}
			if storage.indexerK != nil && storage.indexerK[layer] != nil {
				copyData(storage.indexerK[layer][head])
			}
		}
		if storage.mlaLatent != nil && storage.mlaLatent[layer] != nil {
			copyData(storage.mlaLatent[layer])
		}
		if storage.mlaRopeKey != nil && storage.mlaRopeKey[layer] != nil {
			copyData(storage.mlaRopeKey[layer])
		}
	}
}

func (storage *pagedKVStorage) ensureRaw(layer, head, block int) []float32 {
	if storage.pools != nil {
		storage.pools.mutex.Lock()
		defer storage.pools.mutex.Unlock()
	}
	if storage.rawKey == nil {
		storage.rawKey = make([][][][]float32, storage.layerCount)
		storage.rawValue = make([][][][]float32, storage.layerCount)
	}
	if storage.rawKey[layer] == nil {
		storage.rawKey[layer] = make([][][]float32, storage.keyValueHeadCount)
		storage.rawValue[layer] = make([][][]float32, storage.keyValueHeadCount)
	}
	if storage.rawKey[layer][head] == nil {
		storage.rawKey[layer][head] = make([][]float32, storage.blockCount)
		storage.rawValue[layer][head] = make([][]float32, storage.blockCount)
	}
	if storage.rawKey[layer][head][block] == nil {
		storage.rawKey[layer][head][block] = make([]float32, KVBlockSize*storage.headDimension)
		storage.rawValue[layer][head][block] = make([]float32, KVBlockSize*storage.headDimension)
	}
	return storage.rawKey[layer][head][block]
}

func (storage *pagedKVStorage) rawBlock(layer, row, head, block int) []float32 {
	if storage.rawKey == nil || storage.rawKey[layer] == nil || storage.rawKey[layer][head] == nil {
		return nil
	}
	if block < 0 || block >= len(storage.rawKey[layer][head]) {
		return nil
	}
	return storage.rawKey[layer][head][block]
}

func (storage *pagedKVStorage) rawPhysicalBlock(layer, row, head, block int) ([]float32, []float32) {
	if block < 0 || block >= len(storage.tables[row]) {
		return nil, nil
	}
	id := storage.tables[row][block]
	if storage.rawKey == nil || storage.rawKey[layer] == nil || storage.rawKey[layer][head] == nil {
		return nil, nil
	}
	if block < 0 || block >= len(storage.rawKey[layer][head]) {
		return nil, nil
	}
	key := storage.rawKey[layer][head][id]
	if key == nil {
		return nil, nil
	}
	var value []float32
	if storage.rawValue != nil && storage.rawValue[layer] != nil && storage.rawValue[layer][head] != nil {
		value = storage.rawValue[layer][head][id]
	}
	return key, value
}

func (storage *pagedKVStorage) compressedPhysicalBlock(layer, row, head, block int) ([]float32, []float32) {
	if block < 0 || block >= len(storage.tables[row]) {
		return nil, nil
	}
	id := storage.tables[row][block]
	key := storage.compressedBlockData(layer, head, id)
	if key == nil {
		return nil, nil
	}
	var value []float32
	if storage.compressedV != nil && storage.compressedV[layer] != nil && storage.compressedV[layer][head] != nil {
		value = storage.compressedV[layer][head][id]
	}
	return key, value
}

func (storage *pagedKVStorage) mlaPhysicalBlock(layer, row, block int) ([]float32, []float32) {
	if block < 0 || block >= len(storage.tables[row]) {
		return nil, nil
	}
	id := storage.tables[row][block]
	if storage.mlaLatent == nil || storage.mlaLatent[layer] == nil {
		return nil, nil
	}
	latent := storage.mlaLatent[layer][id]
	if latent == nil {
		return nil, nil
	}
	var rope []float32
	if storage.mlaRopeKey != nil && storage.mlaRopeKey[layer] != nil {
		rope = storage.mlaRopeKey[layer][id]
	}
	return latent, rope
}

func (storage *pagedKVStorage) rawScratchAt(row, head, length int) ([]float32, []float32) {
	if storage.rawScratchKey == nil {
		storage.rawScratchKey = make([][][]float32, storage.batchSize)
		storage.rawScratchValue = make([][][]float32, storage.batchSize)
	}
	if storage.rawScratchKey[row] == nil {
		storage.rawScratchKey[row] = make([][]float32, storage.keyValueHeadCount)
		storage.rawScratchValue[row] = make([][]float32, storage.keyValueHeadCount)
	}
	if len(storage.rawScratchKey[row][head]) < length {
		storage.rawScratchKey[row][head] = make([]float32, length)
		storage.rawScratchValue[row][head] = make([]float32, length)
	}
	return storage.rawScratchKey[row][head][:length], storage.rawScratchValue[row][head][:length]
}

func (storage *pagedKVStorage) appendRaw(layer, row, head, position int, key, value []float32) {
	headDimension := storage.headDimension
	tokenCount := len(key) / headDimension
	for token := 0; token < tokenCount; token++ {
		absolute := position + token
		blockIndex := absolute / KVBlockSize
		offset := absolute % KVBlockSize
		block := storage.privateBlock(row, blockIndex)
		target := storage.ensureRaw(layer, head, block)
		copy(target[offset*headDimension:(offset+1)*headDimension], key[token*headDimension:(token+1)*headDimension])
		if storage.rawValue[layer][head] == nil || storage.rawValue[layer][head][block] == nil {
			storage.rawValue[layer][head][block] = make([]float32, KVBlockSize*headDimension)
		}
		copy(storage.rawValue[layer][head][block][offset*headDimension:(offset+1)*headDimension], value[token*headDimension:(token+1)*headDimension])
	}
}

func (storage *pagedKVStorage) readRaw(layer, row, head, end int) ([]float32, []float32) {
	headDimension := storage.headDimension
	length := end * headDimension
	key, value := storage.rawScratchAt(row, head, length)
	if storage.rawStrippedFlag {
		clear(key)
		clear(value)
	}
	for position := 0; position < end; position++ {
		blockIndex := position / KVBlockSize
		offset := position % KVBlockSize
		if blockIndex >= len(storage.tables[row]) {
			break
		}
		block := storage.tables[row][blockIndex]
		if storage.rawStrippedFlag && !storage.rawCleared[block] {
			continue
		}
		sourceKey := storage.rawBlock(layer, row, head, block)
		if sourceKey == nil {
			continue
		}
		copy(key[position*headDimension:(position+1)*headDimension], sourceKey[offset*headDimension:(offset+1)*headDimension])
		sourceValue := storage.rawValue[layer][head][block]
		if sourceValue != nil {
			copy(value[position*headDimension:(position+1)*headDimension], sourceValue[offset*headDimension:(offset+1)*headDimension])
		}
	}
	return key, value
}

func (storage *pagedKVStorage) ensureCompressed(layer, head, block int) []float32 {
	if storage.pools != nil {
		storage.pools.mutex.Lock()
		defer storage.pools.mutex.Unlock()
	}
	if storage.compressedK == nil {
		storage.compressedK = make([][][][]float32, storage.layerCount)
		storage.compressedV = make([][][][]float32, storage.layerCount)
	}
	if storage.compressedK[layer] == nil {
		storage.compressedK[layer] = make([][][]float32, storage.keyValueHeadCount)
		storage.compressedV[layer] = make([][][]float32, storage.keyValueHeadCount)
	}
	if storage.compressedK[layer][head] == nil {
		storage.compressedK[layer][head] = make([][]float32, storage.blockCount)
		storage.compressedV[layer][head] = make([][]float32, storage.blockCount)
	}
	entries := storage.entriesPerBlock(layer)
	if storage.compressedK[layer][head][block] == nil {
		storage.compressedK[layer][head][block] = make([]float32, entries*storage.headDimension)
		storage.compressedV[layer][head][block] = make([]float32, entries*storage.headDimension)
	}
	return storage.compressedK[layer][head][block]
}

func (storage *pagedKVStorage) compressedBlockData(layer, head, block int) []float32 {
	if storage.compressedK == nil || storage.compressedK[layer] == nil || storage.compressedK[layer][head] == nil {
		return nil
	}
	if block < 0 || block >= len(storage.compressedK[layer][head]) {
		return nil
	}
	return storage.compressedK[layer][head][block]
}

func (storage *pagedKVStorage) appendCompressed(layer, row, head, position int, key, value []float32) {
	entries := storage.entriesPerBlock(layer)
	blockIndex := position / entries
	offset := position % entries
	block := storage.privateBlock(row, blockIndex)
	target := storage.ensureCompressed(layer, head, block)
	copy(target[offset*storage.headDimension:(offset+1)*storage.headDimension], key)
	if storage.compressedV[layer][head][block] == nil {
		storage.compressedV[layer][head][block] = make([]float32, entries*storage.headDimension)
	}
	copy(storage.compressedV[layer][head][block][offset*storage.headDimension:(offset+1)*storage.headDimension], value)
}

func (storage *pagedKVStorage) readCompressed(layer, row, head, end int) ([]float32, []float32) {
	headDimension := storage.headDimension
	entries := storage.entriesPerBlock(layer)
	length := end * headDimension
	if storage.compScratchKey == nil {
		storage.compScratchKey = make([][][]float32, storage.batchSize)
		storage.compScratchVal = make([][][]float32, storage.batchSize)
	}
	if storage.compScratchKey[row] == nil {
		storage.compScratchKey[row] = make([][]float32, storage.keyValueHeadCount)
		storage.compScratchVal[row] = make([][]float32, storage.keyValueHeadCount)
	}
	if len(storage.compScratchKey[row][head]) < length {
		storage.compScratchKey[row][head] = make([]float32, length)
		storage.compScratchVal[row][head] = make([]float32, length)
	}
	key := storage.compScratchKey[row][head][:length]
	value := storage.compScratchVal[row][head][:length]
	for entry := 0; entry < end; entry++ {
		blockIndex := entry / entries
		offset := entry % entries
		if blockIndex >= len(storage.tables[row]) {
			break
		}
		block := storage.tables[row][blockIndex]
		source := storage.compressedBlockData(layer, head, block)
		if source == nil {
			continue
		}
		copy(key[entry*headDimension:(entry+1)*headDimension], source[offset*headDimension:(offset+1)*headDimension])
		if storage.compressedV[layer][head][block] != nil {
			copy(value[entry*headDimension:(entry+1)*headDimension], storage.compressedV[layer][head][block][offset*headDimension:(offset+1)*headDimension])
		}
	}
	return key, value
}

func (storage *pagedKVStorage) readCompressedBlock(layer, row, head, block int) []float32 {
	entries := storage.entriesPerBlock(layer)
	blockIndex := block / entries
	offset := block % entries
	if blockIndex < 0 || blockIndex >= len(storage.tables[row]) {
		return nil
	}
	id := storage.tables[row][blockIndex]
	source := storage.compressedBlockData(layer, head, id)
	if source == nil {
		return nil
	}
	return source[offset*storage.headDimension : (offset+1)*storage.headDimension]
}

func (storage *pagedKVStorage) ensureIndexer(layer, head, block int) []float32 {
	if storage.pools != nil {
		storage.pools.mutex.Lock()
		defer storage.pools.mutex.Unlock()
	}
	if storage.indexerK == nil {
		storage.indexerK = make([][][][]float32, storage.layerCount)
	}
	if storage.indexerK[layer] == nil {
		storage.indexerK[layer] = make([][][]float32, storage.keyValueHeadCount)
	}
	if storage.indexerK[layer][head] == nil {
		storage.indexerK[layer][head] = make([][]float32, storage.blockCount)
	}
	entries := storage.entriesPerBlock(layer)
	if storage.indexerK[layer][head][block] == nil {
		storage.indexerK[layer][head][block] = make([]float32, entries*storage.indexerKeyWidth)
	}
	return storage.indexerK[layer][head][block]
}

func (storage *pagedKVStorage) appendIndexer(layer, row, head, position int, key []float32) {
	entries := storage.entriesPerBlock(layer)
	blockIndex := position / entries
	offset := position % entries
	block := storage.privateBlock(row, blockIndex)
	target := storage.ensureIndexer(layer, head, block)
	copy(target[offset*storage.indexerKeyWidth:(offset+1)*storage.indexerKeyWidth], key)
}

func (storage *pagedKVStorage) readIndexer(layer, row, head, end int) []float32 {
	width := storage.indexerKeyWidth
	entries := storage.entriesPerBlock(layer)
	length := end * width
	if storage.idxScratch == nil {
		storage.idxScratch = make([][][]float32, storage.batchSize)
	}
	if storage.idxScratch[row] == nil {
		storage.idxScratch[row] = make([][]float32, storage.keyValueHeadCount)
	}
	if len(storage.idxScratch[row][head]) < length {
		storage.idxScratch[row][head] = make([]float32, length)
	}
	result := storage.idxScratch[row][head][:length]
	for entry := 0; entry < end; entry++ {
		blockIndex := entry / entries
		offset := entry % entries
		if blockIndex >= len(storage.tables[row]) {
			break
		}
		block := storage.tables[row][blockIndex]
		if storage.indexerK == nil || storage.indexerK[layer] == nil || storage.indexerK[layer][head] == nil {
			continue
		}
		source := storage.indexerK[layer][head][block]
		if source == nil {
			continue
		}
		copy(result[entry*width:(entry+1)*width], source[offset*width:(offset+1)*width])
	}
	return result
}

func (storage *pagedKVStorage) ensureMLALatent(layer, block int) []float32 {
	if storage.pools != nil {
		storage.pools.mutex.Lock()
		defer storage.pools.mutex.Unlock()
	}
	if storage.mlaLatent == nil {
		storage.mlaLatent = make([][][]float32, storage.layerCount)
		storage.mlaRopeKey = make([][][]float32, storage.layerCount)
	}
	if storage.mlaLatent[layer] == nil {
		storage.mlaLatent[layer] = make([][]float32, storage.blockCount)
		storage.mlaRopeKey[layer] = make([][]float32, storage.blockCount)
	}
	if storage.mlaLatent[layer][block] == nil {
		storage.mlaLatent[layer][block] = make([]float32, KVBlockSize*storage.mlaLatentWidth)
		storage.mlaRopeKey[layer][block] = make([]float32, KVBlockSize*storage.mlaRopeKeyWidth)
	}
	return storage.mlaLatent[layer][block]
}

func (storage *pagedKVStorage) appendMLALatent(layer, row, position int, latent []float32) {
	blockIndex := position / KVBlockSize
	offset := position % KVBlockSize
	block := storage.privateBlock(row, blockIndex)
	target := storage.ensureMLALatent(layer, block)
	copy(target[offset*storage.mlaLatentWidth:(offset+1)*storage.mlaLatentWidth], latent)
}

func (storage *pagedKVStorage) appendMLARope(layer, row, position int, key []float32) {
	blockIndex := position / KVBlockSize
	offset := position % KVBlockSize
	block := storage.privateBlock(row, blockIndex)
	storage.ensureMLALatent(layer, block)
	if storage.mlaRopeKey[layer][block] == nil {
		storage.mlaRopeKey[layer][block] = make([]float32, KVBlockSize*storage.mlaRopeKeyWidth)
	}
	copy(storage.mlaRopeKey[layer][block][offset*storage.mlaRopeKeyWidth:(offset+1)*storage.mlaRopeKeyWidth], key)
}

func (storage *pagedKVStorage) readMLALatent(layer, row, end int) []float32 {
	width := storage.mlaLatentWidth
	length := end * width
	if storage.latentScratch == nil {
		storage.latentScratch = make([][]float32, storage.batchSize)
	}
	if len(storage.latentScratch[row]) < length {
		storage.latentScratch[row] = make([]float32, length)
	}
	result := storage.latentScratch[row][:length]
	for position := 0; position < end; position++ {
		blockIndex := position / KVBlockSize
		offset := position % KVBlockSize
		if blockIndex >= len(storage.tables[row]) {
			break
		}
		block := storage.tables[row][blockIndex]
		if storage.mlaLatent == nil || storage.mlaLatent[layer] == nil || storage.mlaLatent[layer][block] == nil {
			continue
		}
		copy(result[position*width:(position+1)*width], storage.mlaLatent[layer][block][offset*width:(offset+1)*width])
	}
	return result
}

func (storage *pagedKVStorage) readMLARope(layer, row, end int) []float32 {
	width := storage.mlaRopeKeyWidth
	length := end * width
	if storage.ropeScratch == nil {
		storage.ropeScratch = make([][]float32, storage.batchSize)
	}
	if len(storage.ropeScratch[row]) < length {
		storage.ropeScratch[row] = make([]float32, length)
	}
	result := storage.ropeScratch[row][:length]
	for position := 0; position < end; position++ {
		blockIndex := position / KVBlockSize
		offset := position % KVBlockSize
		if blockIndex >= len(storage.tables[row]) {
			break
		}
		block := storage.tables[row][blockIndex]
		if storage.mlaRopeKey == nil || storage.mlaRopeKey[layer] == nil || storage.mlaRopeKey[layer][block] == nil {
			continue
		}
		copy(result[position*width:(position+1)*width], storage.mlaRopeKey[layer][block][offset*width:(offset+1)*width])
	}
	return result
}

func (storage *pagedKVStorage) clone() kvStorage {
	clone := &pagedKVStorage{
		manager:               storage.manager,
		pools:                 storage.pools,
		batchSize:             storage.batchSize,
		maximumSequenceLength: storage.maximumSequenceLength,
		layerCount:            storage.layerCount,
		keyValueHeadCount:     storage.keyValueHeadCount,
		headDimension:         storage.headDimension,
		blockCount:            storage.blockCount,
		tables:                make([][]int, storage.batchSize),
		layerRatios:           append([]int(nil), storage.layerRatios...),
		maxBlocksByLayer:      append([]int(nil), storage.maxBlocksByLayer...),
		embeddingDim:          storage.embeddingDim,
		kvWidth:               storage.kvWidth,
		indexerKeyWidth:       storage.indexerKeyWidth,
		mlaLatentWidth:        storage.mlaLatentWidth,
		mlaRopeKeyWidth:       storage.mlaRopeKeyWidth,
		rawStrippedFlag:       storage.rawStrippedFlag,
		rawKey:                storage.rawKey,
		rawValue:              storage.rawValue,
		compressedK:           storage.compressedK,
		compressedV:           storage.compressedV,
		indexerK:              storage.indexerK,
		mlaLatent:             storage.mlaLatent,
		mlaRopeKey:            storage.mlaRopeKey,
	}
	for row := range storage.tables {
		clone.tables[row] = append([]int(nil), storage.tables[row]...)
		for _, block := range clone.tables[row] {
			storage.manager.Retain(block)
		}
	}
	return clone
}

func (storage *pagedKVStorage) release() {
	for row := range storage.tables {
		for _, block := range storage.tables[row] {
			storage.manager.Free(block)
		}
		storage.tables[row] = nil
	}
}

func (storage *pagedKVStorage) reset() { storage.release() }

func (storage *pagedKVStorage) releaseRow(row int) {
	if row < 0 || row >= storage.batchSize {
		return
	}
	for _, block := range storage.tables[row] {
		storage.manager.Free(block)
	}
	storage.tables[row] = nil
}

func (storage *pagedKVStorage) moveRow(destination, source int) {
	if destination < 0 || destination >= storage.batchSize || source < 0 || source >= storage.batchSize || destination == source {
		return
	}
	storage.releaseRow(destination)
	storage.tables[destination] = storage.tables[source]
	storage.tables[source] = nil
}

func (storage *pagedKVStorage) stripRaw() {
	storage.rawStrippedFlag = true
	storage.rawCleared = nil
}

func (storage *pagedKVStorage) unstripRaw() {
	storage.rawStrippedFlag = false
	storage.rawCleared = nil
}

func (storage *pagedKVStorage) rawStripped() bool { return storage.rawStrippedFlag }

func (storage *pagedKVStorage) bytesAllocated() int {
	total := 0
	for _, grid := range [][][][][]float32{storage.rawKey, storage.rawValue, storage.compressedK, storage.compressedV, storage.indexerK} {
		for _, layer := range grid {
			for _, head := range layer {
				for _, block := range head {
					total += len(block) * 4
				}
			}
		}
	}
	for _, layer := range storage.mlaLatent {
		for _, block := range layer {
			total += len(block) * 4
		}
	}
	for _, layer := range storage.mlaRopeKey {
		for _, block := range layer {
			total += len(block) * 4
		}
	}
	return total
}

func (storage *pagedKVStorage) compressedBytesAllocated() int {
	total := 0
	for _, grid := range [][][][][]float32{storage.compressedK, storage.compressedV, storage.indexerK} {
		for _, layer := range grid {
			for _, head := range layer {
				for _, block := range head {
					total += len(block) * 4
				}
			}
		}
	}
	return total
}

func (storage *pagedKVStorage) markTables() [][]int {
	tables := make([][]int, len(storage.tables))
	for row := range storage.tables {
		tables[row] = append([]int(nil), storage.tables[row]...)
	}
	return tables
}

func (storage *pagedKVStorage) rollbackTables(mark [][]int) {
	if mark == nil {
		return
	}
	for row := 0; row < storage.batchSize; row++ {
		var marked []int
		if row < len(mark) {
			marked = mark[row]
		}
		current := storage.tables[row]
		for index := 0; index < len(current); index++ {
			if index >= len(marked) {
				storage.manager.Free(current[index])
				continue
			}
			if current[index] != marked[index] {
				storage.manager.Free(current[index])
				storage.manager.Retain(marked[index])
			}
		}
		for index := len(current); index < len(marked); index++ {
			storage.manager.Retain(marked[index])
		}
		storage.tables[row] = append([]int(nil), marked...)
	}
}
