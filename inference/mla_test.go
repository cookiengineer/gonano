package inference

import (
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
)

func testMLAEngineModel() (*model.Transformer, *tokenizer.Tokenizer) {
	config := model.Config{
		SequenceLen: 16, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L", MLALatent: 8, MLARotaryDims: 8,
	}
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	ranks := make(map[string]int, 256)
	for index := 0; index < 256; index++ {
		ranks[string([]byte{byte(index)})] = index
	}
	return transformer, tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
}

// mlaReference greedily generates by recomputing the full forward from scratch
// with a fresh latent cache at every step, providing an independent reference
// for the incremental engine cache.
func mlaReference(transformer *model.Transformer, tokens []int, maxTokens int) []int {
	config := transformer.Config
	ids := append([]int(nil), tokens...)
	for step := 0; step < maxTokens; step++ {
		cache := model.NewKVBuffer(1, len(ids), config.NumLayer, config.NumKVHead, config.HeadDim())
		cache.EnableMLA(config.MLALatent, config.NumKVHead*config.MLARotaryDimension())
		logits := transformer.Forward(tensors.NewInt32sWithData([]int{1, len(ids)}, toI32(ids)), cache)
		flat := logits.Reshape(len(ids), config.VocabSize)
		row := flat.Data[(len(ids)-1)*config.VocabSize:]
		ids = append(ids, argmaxSlice(row))
	}
	return ids
}

// TestEngineMatchesNaiveGenerateMLA verifies the incremental MLA engine against
// a full-recompute reference.
func TestEngineMatchesNaiveGenerateMLA(t *testing.T) {
	transformer, tokenizerImpl := testMLAEngineModel()
	engine := NewEngine(transformer, tokenizerImpl)
	prompt := []int{1, 5, 2, 8}

	got, _ := engine.GenerateBatch(prompt, 1, 6, 0, 0, 7)
	want := mlaReference(transformer, prompt, 6)
	if len(got[0]) != len(want) {
		t.Fatalf("length %d, want %d", len(got[0]), len(want))
	}
	for index := range want {
		if got[0][index] != want[index] {
			t.Fatalf("token %d: engine=%d reference=%d", index, got[0][index], want[index])
		}
	}
}

// TestMLACacheManagerGenerationMatchesFullPrefill verifies the multi-entry
// persistent cache (including the on-disk codec) with MLA buffers.
func TestMLACacheManagerGenerationMatchesFullPrefill(t *testing.T) {
	transformer, tokenizerImpl := testMLAEngineModel()
	engine := NewEngine(transformer, tokenizerImpl)
	engine.Cache = NewCacheManager(CacheOptions{MaxEntries: 4, DiskDir: t.TempDir()})

	prompt1 := []int{1, 5, 2, 8, 3, 7}
	prompt2 := append(append([]int(nil), prompt1...), 4, 6, 9)

	engine.GenerateBatch(prompt1, 1, 4, 0, 0, 7)
	got, _ := engine.GenerateBatch(prompt2, 1, 6, 0, 0, 11)

	referenceEngine := NewEngine(transformer, tokenizerImpl)
	want, _ := referenceEngine.GenerateBatch(prompt2, 1, 6, 0, 0, 11)
	assertSameTokens(t, got[0], want[0])
}

// TestMLAPrefixCacheGenerationMatchesFullPrefill covers the single-entry
// in-memory prefix cache with MLA.
func TestMLAPrefixCacheGenerationMatchesFullPrefill(t *testing.T) {
	transformer, tokenizerImpl := testMLAEngineModel()
	engine := NewEngine(transformer, tokenizerImpl)
	engine.Prefix = NewPrefixCache(transformer.Config.SequenceLen)

	prompt1 := []int{1, 5, 2, 8, 3, 7}
	prompt2 := append(append([]int(nil), prompt1...), 4, 6, 9)

	engine.GenerateBatch(prompt1, 1, 4, 0, 0, 7)
	got, _ := engine.GenerateBatch(prompt2, 1, 6, 0, 0, 11)

	referenceEngine := NewEngine(transformer, tokenizerImpl)
	want, _ := referenceEngine.GenerateBatch(prompt2, 1, 6, 0, 0, 11)
	assertSameTokens(t, got[0], want[0])
}
