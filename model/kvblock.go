package model

import "sync"

// KVBlockSize is the number of token positions stored in one paged KV block.
// DeepSeek-V4.1 aligns its persistent KV cache to 128-token blocks, and every
// compression ratio used by the model layout must divide this size.
const KVBlockSize = 128

// KVBlockManager owns a fixed pool of paged KV blocks. Block IDs are global
// across layers: a block ID names the same logical position range in every
// layer's block pool. Refcounts allow multiple sequences (or batch rows) to
// share a block until one of them writes to it, at which point the writer must
// copy the block first (copy-on-write).
//
// It is safe for concurrent use.
type KVBlockManager struct {
	mutex sync.Mutex

	blockCount int
	free       []int
	refCount   []int

	allocated int
	peak      int
}

// NewKVBlockManager builds a manager with blockCount blocks. At least one block
// is always available so a cache can hold one token.
func NewKVBlockManager(blockCount int) *KVBlockManager {
	if blockCount < 1 {
		blockCount = 1
	}
	manager := &KVBlockManager{
		blockCount: blockCount,
		free:       make([]int, blockCount),
		refCount:   make([]int, blockCount),
	}
	for block := 0; block < blockCount; block++ {
		manager.free[block] = blockCount - 1 - block
	}
	return manager
}

// Allocate reserves one free block and returns its ID, or -1 when the pool is
// exhausted.
func (manager *KVBlockManager) Allocate() int {
	manager.mutex.Lock()
	defer manager.mutex.Unlock()
	if len(manager.free) == 0 {
		return -1
	}
	block := manager.free[len(manager.free)-1]
	manager.free = manager.free[:len(manager.free)-1]
	manager.refCount[block] = 1
	manager.allocated++
	if manager.allocated > manager.peak {
		manager.peak = manager.allocated
	}
	return block
}

// Retain increments a block's reference count without allocating.
func (manager *KVBlockManager) Retain(block int) {
	manager.mutex.Lock()
	defer manager.mutex.Unlock()
	if block < 0 || block >= manager.blockCount || manager.refCount[block] <= 0 {
		panic("model: retain of an unallocated KV block")
	}
	manager.refCount[block]++
}

// Free releases one reference to a block. The block returns to the free list
// when its last reference is released.
func (manager *KVBlockManager) Free(block int) {
	manager.mutex.Lock()
	defer manager.mutex.Unlock()
	if block < 0 || block >= manager.blockCount || manager.refCount[block] <= 0 {
		panic("model: free of an unallocated KV block")
	}
	manager.refCount[block]--
	if manager.refCount[block] == 0 {
		manager.free = append(manager.free, block)
		manager.allocated--
	}
}

// RefCount returns the number of references to a block.
func (manager *KVBlockManager) RefCount(block int) int {
	manager.mutex.Lock()
	defer manager.mutex.Unlock()
	if block < 0 || block >= manager.blockCount {
		return 0
	}
	return manager.refCount[block]
}

// BlockCount returns the total number of blocks in the pool.
func (manager *KVBlockManager) BlockCount() int { return manager.blockCount }

// FreeBlocks returns the number of currently unused blocks.
func (manager *KVBlockManager) FreeBlocks() int {
	manager.mutex.Lock()
	defer manager.mutex.Unlock()
	return len(manager.free)
}

// InUse returns the number of allocated blocks.
func (manager *KVBlockManager) InUse() int {
	manager.mutex.Lock()
	defer manager.mutex.Unlock()
	return manager.allocated
}

// Peak returns the high-water mark of allocated blocks.
func (manager *KVBlockManager) Peak() int {
	manager.mutex.Lock()
	defer manager.mutex.Unlock()
	return manager.peak
}

// Reset releases every block and clears accounting. It is only safe when no
// cache still references the manager.
func (manager *KVBlockManager) Reset() {
	manager.mutex.Lock()
	defer manager.mutex.Unlock()
	manager.free = manager.free[:manager.blockCount]
	for block := 0; block < manager.blockCount; block++ {
		manager.refCount[block] = 0
		manager.free[block] = manager.blockCount - 1 - block
	}
	manager.allocated = 0
	manager.peak = 0
}
