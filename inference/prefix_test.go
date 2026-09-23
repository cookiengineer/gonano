package inference

import (
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
)

func testPrefixEngineModel() (*model.Transformer, *tokenizer.Tokenizer) {
	config := model.Config{
		SequenceLen: 32, VocabSize: 32, NumLayer: 3, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L", CompressionRatio: 2, SWAWindow: 4,
		SparseTopK: 2, IndexerDim: 4, IndexerPool: 2, ReusePattern: "FR",
	}
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	ranks := make(map[string]int, 256)
	for index := 0; index < 256; index++ {
		ranks[string([]byte{byte(index)})] = index
	}
	return transformer, tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
}

func TestPrefixCacheLookupStore(t *testing.T) {
	cache := NewPrefixCache(0)
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
	if state, _ := cache.Lookup([]int{1, 2, 9, 4}); state != nil {
		t.Fatal("divergent prefix should miss")
	}
	if state, _ := cache.Lookup([]int{1, 2, 3}); state != nil {
		t.Fatal("exact-length request should miss (no suffix to prefill)")
	}

	cache.Invalidate()
	if state, _ := cache.Lookup([]int{1, 2, 3, 4}); state != nil {
		t.Fatal("invalidated cache should miss")
	}
}

// TestPrefixCacheGenerationMatchesFullPrefill verifies that prefill-prefix reuse
// produces exactly the same continuation as a full prefill.
func TestPrefixCacheGenerationMatchesFullPrefill(t *testing.T) {
	transformer, tokenizerImpl := testPrefixEngineModel()
	cachedEngine := NewEngine(transformer, tokenizerImpl)
	cachedEngine.Prefix = NewPrefixCache(transformer.Config.SequenceLen)

	prompt1 := []int{1, 5, 2, 8, 3, 7}
	prompt2 := append(append([]int(nil), prompt1...), 4, 6, 9)

	// Seed the cache with the first prompt.
	cachedEngine.GenerateBatch(prompt1, 1, 4, 0, 0, 7)

	got, _ := cachedEngine.GenerateBatch(prompt2, 1, 6, 0, 0, 11)

	referenceEngine := NewEngine(transformer, tokenizerImpl)
	want, _ := referenceEngine.GenerateBatch(prompt2, 1, 6, 0, 0, 11)

	if len(got) != len(want) || len(got[0]) != len(want[0]) {
		t.Fatalf("lengths differ: got %v want %v", len(got[0]), len(want[0]))
	}
	for index := range want[0] {
		if got[0][index] != want[0][index] {
			t.Fatalf("token %d: cached=%d reference=%d", index, got[0][index], want[0][index])
		}
	}
}

// TestPrefixCacheMissFallsBack verifies that a divergent prompt re-prefills and
// still matches the reference.
func TestPrefixCacheMissFallsBack(t *testing.T) {
	transformer, tokenizerImpl := testPrefixEngineModel()
	engine := NewEngine(transformer, tokenizerImpl)
	engine.Prefix = NewPrefixCache(transformer.Config.SequenceLen)

	engine.GenerateBatch([]int{1, 2, 3, 4}, 1, 3, 0, 0, 1)

	prompt := []int{9, 8, 7, 6, 5}
	got, _ := engine.GenerateBatch(prompt, 1, 4, 0, 0, 2)

	referenceEngine := NewEngine(transformer, tokenizerImpl)
	want, _ := referenceEngine.GenerateBatch(prompt, 1, 4, 0, 0, 2)

	for index := range want[0] {
		if got[0][index] != want[0][index] {
			t.Fatalf("token %d: cached=%d reference=%d", index, got[0][index], want[0][index])
		}
	}
}

// TestPrefixCacheCloneIndependence verifies that a lookup clone can be mutated
// without corrupting the stored snapshot.
func TestPrefixCacheCloneIndependence(t *testing.T) {
	cache := NewPrefixCache(0)
	kv := model.NewKVBuffer(1, 8, 1, 1, 2)
	kv.Advance(2)
	cache.Store([]int{1, 2}, kv)

	first, hit := cache.Lookup([]int{1, 2, 3})
	if first == nil || hit != 2 {
		t.Fatal("expected a hit")
	}
	// Mutate the clone's key cache; the stored snapshot must be unaffected.
	first.Advance(1)
	first.Reset()

	second, hit := cache.Lookup([]int{1, 2, 3})
	if second == nil || hit != 2 {
		t.Fatal("second lookup should still hit")
	}
	if second.Position() != 2 {
		t.Fatalf("stored snapshot position = %d, want 2", second.Position())
	}
}
