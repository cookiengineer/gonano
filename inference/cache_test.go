package inference

import (
	"os"
	"testing"
	"time"

	"github.com/cookiengineer/gonano/model"
)

func TestCacheManagerLookupStore(t *testing.T) {
	cache := NewCacheManager(CacheOptions{})
	if state, hit := cache.Lookup([]int{1, 2, 3}); state != nil || hit != 0 {
		t.Fatal("empty cache should miss")
	}

	kv := model.NewKVBuffer(1, 8, 2, 2, 4)
	kv.Advance(3)
	cache.Store([]int{1, 2, 3}, kv)

	if state, hit := cache.Lookup([]int{1, 2, 3, 4}); state == nil || hit != 3 {
		t.Fatalf("extension lookup = (%v, %d), want a state and 3", state != nil, hit)
	}
	if state, _ := cache.Lookup([]int{1, 2}); state != nil {
		t.Fatal("shorter request should miss")
	}
	if state, _ := cache.Lookup([]int{1, 2, 3}); state != nil {
		t.Fatal("exact-length request should miss")
	}
	if cache.Len() != 1 {
		t.Fatalf("len = %d, want 1", cache.Len())
	}

	cache.Invalidate()
	if state, _ := cache.Lookup([]int{1, 2, 3, 4}); state != nil {
		t.Fatal("invalidated cache should miss")
	}
}

func TestCacheManagerLongestPrefix(t *testing.T) {
	cache := NewCacheManager(CacheOptions{})
	short := model.NewKVBuffer(1, 8, 1, 1, 4)
	short.Advance(2)
	long := model.NewKVBuffer(1, 8, 1, 1, 4)
	long.Advance(4)
	cache.Store([]int{1, 2}, short)
	cache.Store([]int{1, 2, 3, 4}, long)

	state, hit := cache.Lookup([]int{1, 2, 3, 4, 5})
	if state == nil || hit != 4 {
		t.Fatalf("lookup hit = (%v, %d), want the 4-token prefix", state != nil, hit)
	}
}

func TestCacheManagerLRUEviction(t *testing.T) {
	cache := NewCacheManager(CacheOptions{MaxEntries: 1})
	first := model.NewKVBuffer(1, 8, 1, 1, 4)
	first.Advance(2)
	second := model.NewKVBuffer(1, 8, 1, 1, 4)
	second.Advance(2)
	cache.Store([]int{1, 2}, first)
	cache.Store([]int{9, 9}, second)

	if cache.Len() != 1 {
		t.Fatalf("len = %d, want 1 after eviction", cache.Len())
	}
	if state, _ := cache.Lookup([]int{1, 2, 3}); state != nil {
		t.Fatal("oldest entry should have been evicted")
	}
	if state, hit := cache.Lookup([]int{9, 9, 3}); state == nil || hit != 2 {
		t.Fatal("newest entry should still hit")
	}
}

func TestCacheManagerTTL(t *testing.T) {
	cache := NewCacheManager(CacheOptions{TTL: time.Minute})
	current := time.Unix(1000, 0)
	cache.clock = func() time.Time { return current }

	kv := model.NewKVBuffer(1, 8, 1, 1, 4)
	kv.Advance(2)
	cache.Store([]int{1, 2}, kv)
	if state, _ := cache.Lookup([]int{1, 2, 3}); state == nil {
		t.Fatal("fresh entry should hit")
	}

	current = current.Add(2 * time.Minute)
	if state, _ := cache.Lookup([]int{1, 2, 3}); state != nil {
		t.Fatal("expired entry should miss")
	}
	if cache.Len() != 0 {
		t.Fatalf("len = %d, want 0 after expiry", cache.Len())
	}
}

func TestCacheManagerDiskTier(t *testing.T) {
	dir := t.TempDir()
	cache := NewCacheManager(CacheOptions{DiskDir: dir})
	cache.Store([]int{1, 2, 3}, seededKV())

	if len(cache.entries) != 1 {
		t.Fatalf("entries = %d, want 1", len(cache.entries))
	}
	for _, entry := range cache.entries {
		if entry.state != nil {
			t.Fatal("disk-backed entry must not hold state in memory")
		}
		if entry.diskPath == "" {
			t.Fatal("disk-backed entry must record a path")
		}
		if _, err := os.Stat(entry.diskPath); err != nil {
			t.Fatalf("disk snapshot missing: %v", err)
		}
	}

	state, hit := cache.Lookup([]int{1, 2, 3, 4})
	if state == nil || hit != 3 {
		t.Fatalf("disk lookup = (%v, %d), want a state and 3", state != nil, hit)
	}
	if state.Position() != 3 {
		t.Fatalf("disk state position = %d, want 3", state.Position())
	}

	cache.Invalidate()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read dir: %v", err)
	}
	if len(entries) != 0 {
		t.Fatalf("disk files remain after invalidate: %d", len(entries))
	}
}

// seededKV returns a KV buffer with a non-trivial used prefix.
func seededKV() *model.KVBuffer {
	kv := model.NewKVBuffer(1, 8, 2, 2, 4)
	kv.Advance(3)
	return kv
}

func TestCacheManagerGenerationMatchesFullPrefill(t *testing.T) {
	transformer, tokenizerImpl := testPrefixEngineModel()
	cachedEngine := NewEngine(transformer, tokenizerImpl)
	cachedEngine.Cache = NewCacheManager(CacheOptions{MaxEntries: 4})

	prompt1 := []int{1, 5, 2, 8, 3, 7}
	prompt2 := append(append([]int(nil), prompt1...), 4, 6, 9)

	cachedEngine.GenerateBatch(prompt1, 1, 4, 0, 0, 7)
	got, _ := cachedEngine.GenerateBatch(prompt2, 1, 6, 0, 0, 11)

	referenceEngine := NewEngine(transformer, tokenizerImpl)
	want, _ := referenceEngine.GenerateBatch(prompt2, 1, 6, 0, 0, 11)
	assertSameTokens(t, got[0], want[0])
}

func TestCacheManagerDiskGenerationMatchesFullPrefill(t *testing.T) {
	transformer, tokenizerImpl := testPrefixEngineModel()
	cachedEngine := NewEngine(transformer, tokenizerImpl)
	cachedEngine.Cache = NewCacheManager(CacheOptions{MaxEntries: 4, DiskDir: t.TempDir()})

	prompt1 := []int{1, 5, 2, 8, 3, 7}
	prompt2 := append(append([]int(nil), prompt1...), 4, 6, 9)

	cachedEngine.GenerateBatch(prompt1, 1, 4, 0, 0, 7)
	got, _ := cachedEngine.GenerateBatch(prompt2, 1, 6, 0, 0, 11)

	referenceEngine := NewEngine(transformer, tokenizerImpl)
	want, _ := referenceEngine.GenerateBatch(prompt2, 1, 6, 0, 0, 11)
	assertSameTokens(t, got[0], want[0])
}

func assertSameTokens(t *testing.T, got, want []int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("lengths differ: got %d want %d", len(got), len(want))
	}
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("token %d: cached=%d reference=%d", index, got[index], want[index])
		}
	}
}
