package inference

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/cookiengineer/gonano/model"
)

// CacheOptions configures a CacheManager.
type CacheOptions struct {
	// MaxEntries bounds the number of cached prefixes; zero means unbounded.
	MaxEntries int
	// MaxBytes bounds the approximate allocated size of the in-memory entries;
	// zero means unbounded.
	MaxBytes int
	// TTL expires an entry after this duration; zero disables expiry.
	TTL time.Duration
	// DiskDir enables a persistent on-disk tier: entries are serialized to this
	// directory and only their metadata is kept in memory. Zero keeps every
	// entry in memory.
	DiskDir string
}

// cacheEntry is one cached prefill.
type cacheEntry struct {
	tokens   []int
	state    *model.KVBuffer // nil when the entry is disk-backed
	diskPath string
	size     int
	storedAt time.Time
	usedAt   time.Time
}

// CacheManager is a multi-entry KV prefix cache with LRU and TTL eviction and
// an optional persistent on-disk tier (DeepSeek-V4.1 §3.2.1). Like the
// single-entry PrefixCache it only hits when a stored token sequence is a strict
// prefix of the request, so reused KV state stays exactly consistent with a
// full prefill.
//
// It is safe for concurrent use.
type CacheManager struct {
	mutex   sync.Mutex
	options CacheOptions
	entries map[string]*cacheEntry
	order   []string // LRU order, oldest first
	bytes   int
	clock   func() time.Time
}

// NewCacheManager builds a cache manager. When DiskDir is set it is created if
// missing; if creation fails the manager falls back to the in-memory tier.
func NewCacheManager(options CacheOptions) *CacheManager {
	if options.DiskDir != "" {
		if err := os.MkdirAll(options.DiskDir, 0o755); err != nil {
			options.DiskDir = ""
		}
	}
	return &CacheManager{
		options: options,
		entries: make(map[string]*cacheEntry),
	}
}

// Lookup returns a fresh KV state and the matched prefix length when a stored
// sequence is a strict prefix of tokens. It returns nil, 0 on a miss. When
// several prefixes match, the longest is returned.
func (cache *CacheManager) Lookup(tokens []int) (*model.KVBuffer, int) {
	cache.mutex.Lock()
	defer cache.mutex.Unlock()
	now := cache.now()
	cache.expireLocked(now)

	bestKey := ""
	bestLength := 0
	for key, entry := range cache.entries {
		if len(entry.tokens) == 0 || len(entry.tokens) >= len(tokens) || len(entry.tokens) <= bestLength {
			continue
		}
		matched := true
		for index := range entry.tokens {
			if entry.tokens[index] != tokens[index] {
				matched = false
				break
			}
		}
		if matched {
			bestKey = key
			bestLength = len(entry.tokens)
		}
	}
	if bestKey == "" {
		return nil, 0
	}
	entry := cache.entries[bestKey]
	entry.usedAt = now
	cache.touchLocked(bestKey)
	if entry.state != nil {
		return entry.state.Clone(), bestLength
	}
	state, err := loadKVFile(entry.diskPath)
	if err != nil {
		cache.removeEntryLocked(bestKey, entry)
		return nil, 0
	}
	return state, bestLength
}

// Store snapshots the KV state for the given token sequence, evicting the least
// recently used entries if the cache exceeds its limits.
func (cache *CacheManager) Store(tokens []int, state *model.KVBuffer) {
	if state == nil {
		return
	}
	cache.mutex.Lock()
	defer cache.mutex.Unlock()

	key := cacheKey(tokens)
	if existing, ok := cache.entries[key]; ok {
		cache.removeEntryLocked(key, existing)
	}
	now := cache.now()
	entry := &cacheEntry{
		tokens:   append([]int(nil), tokens...),
		size:     state.BytesAllocated(),
		storedAt: now,
		usedAt:   now,
	}
	if cache.options.DiskDir != "" {
		path := filepath.Join(cache.options.DiskDir, key+".kv")
		if err := saveKVFile(path, state); err == nil {
			entry.diskPath = path
		} else {
			entry.state = state.Clone()
		}
	} else {
		entry.state = state.Clone()
	}
	cache.entries[key] = entry
	cache.order = append(cache.order, key)
	cache.bytes += entry.size
	cache.evictLocked()
}

// Invalidate drops every cached entry and deletes any disk-backed snapshots.
func (cache *CacheManager) Invalidate() {
	cache.mutex.Lock()
	defer cache.mutex.Unlock()
	for key, entry := range cache.entries {
		if entry.diskPath != "" {
			os.Remove(entry.diskPath)
		}
		delete(cache.entries, key)
	}
	cache.order = nil
	cache.bytes = 0
}

// Len returns the number of cached entries.
func (cache *CacheManager) Len() int {
	cache.mutex.Lock()
	defer cache.mutex.Unlock()
	return len(cache.entries)
}

func (cache *CacheManager) now() time.Time {
	if cache.clock != nil {
		return cache.clock()
	}
	return time.Now()
}

// touchLocked moves a key to the most-recently-used end of the LRU order.
func (cache *CacheManager) touchLocked(key string) {
	for index, candidate := range cache.order {
		if candidate == key {
			cache.order = append(cache.order[:index], cache.order[index+1:]...)
			break
		}
	}
	cache.order = append(cache.order, key)
}

// removeEntryLocked removes an entry, deletes its disk snapshot, and updates the
// LRU order and byte accounting.
func (cache *CacheManager) removeEntryLocked(key string, entry *cacheEntry) {
	delete(cache.entries, key)
	cache.bytes -= entry.size
	if entry.diskPath != "" {
		os.Remove(entry.diskPath)
	}
	for index, candidate := range cache.order {
		if candidate == key {
			cache.order = append(cache.order[:index], cache.order[index+1:]...)
			break
		}
	}
}

// evictLocked removes expired and least-recently-used entries until the cache
// fits its configured limits.
func (cache *CacheManager) evictLocked() {
	for cache.options.MaxEntries > 0 && len(cache.entries) > cache.options.MaxEntries && len(cache.order) > 0 {
		cache.evictOldestLocked()
	}
	for cache.options.MaxBytes > 0 && cache.bytes > cache.options.MaxBytes && len(cache.order) > 0 {
		cache.evictOldestLocked()
	}
}

func (cache *CacheManager) evictOldestLocked() {
	key := cache.order[0]
	cache.order = cache.order[1:]
	if entry, ok := cache.entries[key]; ok {
		cache.removeEntryLocked(key, entry)
	}
}

// expireLocked removes entries whose TTL has elapsed.
func (cache *CacheManager) expireLocked(now time.Time) {
	if cache.options.TTL <= 0 {
		return
	}
	var expired []string
	for key, entry := range cache.entries {
		if now.Sub(entry.storedAt) > cache.options.TTL {
			expired = append(expired, key)
		}
	}
	for _, key := range expired {
		if entry, ok := cache.entries[key]; ok {
			cache.removeEntryLocked(key, entry)
		}
	}
}

// cacheKey hashes a token sequence into a stable key.
func cacheKey(tokens []int) string {
	hash := sha256.New()
	var buffer [4]byte
	for _, token := range tokens {
		binary.LittleEndian.PutUint32(buffer[:], uint32(token))
		hash.Write(buffer[:])
	}
	return hex.EncodeToString(hash.Sum(nil))
}

// saveKVFile writes a serialized KV cache to path atomically.
func saveKVFile(path string, state *model.KVBuffer) error {
	data, err := state.MarshalBinary()
	if err != nil {
		return err
	}
	temporary := path + ".tmp"
	if err := os.WriteFile(temporary, data, 0o644); err != nil {
		return err
	}
	return os.Rename(temporary, path)
}

// loadKVFile reads and decodes a serialized KV cache from path.
func loadKVFile(path string) (*model.KVBuffer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return model.UnmarshalKVBuffer(data)
}
