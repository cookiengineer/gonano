package inference

import (
	"crypto/sha256"
	"encoding/binary"
	"sync"

	"github.com/cookiengineer/gonano/model"
)

// prefixHash is a block-chain hash: each 128-token block hashes the previous
// block's hash together with its own token ids.
type prefixHash [sha256.Size]byte

type blockPrefixEntry struct {
	hash   prefixHash
	length int
	state  *model.KVBuffer
	size   int
}

// BlockPrefixCache stores KV snapshots at complete 128-token boundaries and
// shares their blocks through copy-on-write refcounts. It only matches strict,
// block-aligned prefixes, so a hit always ends on a block boundary and the
// compression tail is empty.
//
// It is safe for concurrent use.
type BlockPrefixCache struct {
	mutex      sync.Mutex
	maxEntries int
	maxBytes   int
	entries    map[prefixHash]*blockPrefixEntry
	order      []prefixHash
	bytes      int
	hits       int
	misses     int
}

// NewBlockPrefixCache builds a block-aligned prefix cache. Zero limits are
// unbounded.
func NewBlockPrefixCache(maxEntries, maxBytes int) *BlockPrefixCache {
	return &BlockPrefixCache{
		maxEntries: maxEntries,
		maxBytes:   maxBytes,
		entries:    make(map[prefixHash]*blockPrefixEntry),
	}
}

// Lookup returns a cloned KV state and the matched prefix length when a stored
// block-aligned prefix strictly precedes tokens. It returns nil, 0 on a miss.
func (cache *BlockPrefixCache) Lookup(tokens []int) (*model.KVBuffer, int) {
	cache.mutex.Lock()
	defer cache.mutex.Unlock()

	maxBlocks := (len(tokens) - 1) / model.KVBlockSize
	if maxBlocks <= 0 {
		cache.misses++
		return nil, 0
	}
	var hash prefixHash
	best := (*blockPrefixEntry)(nil)
	bestLength := 0
	for block := 0; block < maxBlocks; block++ {
		start := block * model.KVBlockSize
		hash = chainBlockHash(hash, tokens[start:start+model.KVBlockSize])
		if entry := cache.entries[hash]; entry != nil {
			best = entry
			bestLength = entry.length
		}
	}
	if best == nil {
		cache.misses++
		return nil, 0
	}
	cache.hits++
	cache.touchLocked(best.hash)
	return best.state.Clone(), bestLength
}

// Store snapshots a cache at a complete block boundary. It is a no-op when
// tokens is not a positive multiple of the block size or the cache is not at
// that position.
func (cache *BlockPrefixCache) Store(tokens []int, state *model.KVBuffer) {
	if state == nil || len(tokens) == 0 || len(tokens)%model.KVBlockSize != 0 || state.Position() != len(tokens) {
		return
	}
	hash := prefixHash{}
	for block := 0; block < len(tokens)/model.KVBlockSize; block++ {
		start := block * model.KVBlockSize
		hash = chainBlockHash(hash, tokens[start:start+model.KVBlockSize])
	}

	cache.mutex.Lock()
	defer cache.mutex.Unlock()
	if entry, exists := cache.entries[hash]; exists {
		cache.touchLocked(entry.hash)
		return
	}
	clone := state.Clone()
	size := clone.BytesAllocated()
	if cache.maxBytes > 0 && size > cache.maxBytes {
		clone.Release()
		return
	}
	entry := &blockPrefixEntry{hash: hash, length: len(tokens), state: clone, size: size}
	cache.entries[hash] = entry
	cache.order = append(cache.order, hash)
	cache.bytes += size
	cache.evictLocked()
}

// Invalidate drops every cached snapshot.
func (cache *BlockPrefixCache) Invalidate() {
	cache.mutex.Lock()
	defer cache.mutex.Unlock()
	for _, entry := range cache.entries {
		entry.state.Release()
	}
	cache.entries = make(map[prefixHash]*blockPrefixEntry)
	cache.order = nil
	cache.bytes = 0
}

// Len returns the number of cached snapshots.
func (cache *BlockPrefixCache) Len() int {
	cache.mutex.Lock()
	defer cache.mutex.Unlock()
	return len(cache.entries)
}

// MaxEntries returns the configured entry limit (zero means unbounded).
func (cache *BlockPrefixCache) MaxEntries() int { return cache.maxEntries }

// HeldBlocks returns the number of block IDs referenced by cached snapshots.
// Snapshots can share blocks, so this is an upper bound used for admission.
func (cache *BlockPrefixCache) HeldBlocks() int {
	cache.mutex.Lock()
	defer cache.mutex.Unlock()
	total := 0
	for _, entry := range cache.entries {
		total += entry.length / model.KVBlockSize
	}
	return total
}

// Stats returns hit and miss counters.
func (cache *BlockPrefixCache) Stats() (hits, misses int) {
	cache.mutex.Lock()
	defer cache.mutex.Unlock()
	return cache.hits, cache.misses
}

func (cache *BlockPrefixCache) touchLocked(hash prefixHash) {
	for index, candidate := range cache.order {
		if candidate == hash {
			cache.order = append(cache.order[:index], cache.order[index+1:]...)
			break
		}
	}
	cache.order = append(cache.order, hash)
}

func (cache *BlockPrefixCache) evictLocked() {
	for cache.maxEntries > 0 && len(cache.entries) > cache.maxEntries && len(cache.order) > 0 {
		cache.removeOldestLocked()
	}
	for cache.maxBytes > 0 && cache.bytes > cache.maxBytes && len(cache.order) > 0 {
		cache.removeOldestLocked()
	}
}

func (cache *BlockPrefixCache) removeOldestLocked() {
	hash := cache.order[0]
	cache.order = cache.order[1:]
	entry, exists := cache.entries[hash]
	if !exists {
		return
	}
	delete(cache.entries, hash)
	cache.bytes -= entry.size
	entry.state.Release()
}

func chainBlockHash(previous prefixHash, tokens []int) prefixHash {
	hasher := sha256.New()
	hasher.Write(previous[:])
	var buffer [4]byte
	for _, token := range tokens {
		binary.LittleEndian.PutUint32(buffer[:], uint32(token))
		hasher.Write(buffer[:])
	}
	var result prefixHash
	copy(result[:], hasher.Sum(nil))
	return result
}
