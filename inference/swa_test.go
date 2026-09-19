package inference

import (
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
)

// newByteTokenizer builds the byte-level tokenizer used by the SWA fixtures.
func newByteTokenizer() *tokenizer.Tokenizer {
	ranks := make(map[string]int, 256)
	for index := 0; index < 256; index++ {
		ranks[string([]byte{byte(index)})] = index
	}
	return tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
}

// TestEngineMatchesRecomputeCompressedSWA verifies the incremental compressed
// cache with an active local sliding-window branch, for both dense and sparse
// global attention.
func TestEngineMatchesRecomputeCompressedSWA(t *testing.T) {
	for _, sparseTopK := range []int{0, 2} {
		config := model.Config{
			SequenceLen: 16, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
			EmbedDim: 32, WindowPattern: "L", CompressionRatio: 2, SWAWindow: 4,
			SparseTopK: sparseTopK, IndexerDim: 4,
		}
		transformer := model.NewTransformer(config)
		transformer.InitWeights(tensors.NewRNG(42))
		tokenizerImpl := newByteTokenizer()
		engine := NewEngine(transformer, tokenizerImpl)
		prompt := []int{1, 5, 2, 8, 3, 7, 4, 6}

		want := compressedReference(transformer, prompt, 6)
		got, _ := engine.GenerateBatch(prompt, 1, 6, 0, 0, 7)

		limit := len(want)
		if len(got[0]) < limit {
			limit = len(got[0])
		}
		for index := 0; index < limit; index++ {
			if got[0][index] != want[index] {
				t.Fatalf("sparseTopK=%d token %d: engine=%d recompute=%d", sparseTopK, index, got[0][index], want[index])
			}
		}
	}
}
