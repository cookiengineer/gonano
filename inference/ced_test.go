package inference

import (
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
)

// testCEDEngineModel exercises the causal encoder-decoder split during
// incremental inference. The bottom half of the layers is the encoder; the
// decoder's global compressed keys/values are projected from the encoder's
// final hidden state.
func testCEDEngineModel(sparseTopK int, reusePattern string) (*model.Transformer, *tokenizer.Tokenizer) {
	config := model.Config{
		SequenceLen: 16, VocabSize: 32, NumLayer: 4, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L", CompressionRatio: 2, SWAWindow: 4,
		CED: true, SparseTopK: sparseTopK, IndexerDim: 4, ReusePattern: reusePattern,
	}
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	ranks := make(map[string]int, 256)
	for index := 0; index < 256; index++ {
		ranks[string([]byte{byte(index)})] = index
	}
	tokenizerImpl := tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
	return transformer, tokenizerImpl
}

// cedPrefillReference greedily decodes with a batch-1 cache driven directly by
// PrefillCED and single-token Forward calls. It is the incremental-cache
// reference for the engine's own prefill + replicated-decode path.
func cedPrefillReference(transformer *model.Transformer, tokens []int, maxTokens int) []int {
	config := transformer.Config
	headDim := config.HeadDim()
	cacheLength := len(tokens) + maxTokens
	cache := model.NewKVBuffer(1, cacheLength, config.NumLayer, config.NumKVHead, headDim)
	ratio := config.Compression()
	cache.EnableCompression(ratio, config.EmbedDim, config.NumKVHead*headDim, cacheLength/ratio+1)
	if config.SparseTopK > 0 {
		dim := config.IndexerDim
		if dim <= 0 {
			dim = 64
		}
		heads := config.IndexerHeads
		if heads <= 0 {
			heads = 1
		}
		cache.EnableIndexerKeys(dim * heads)
	}

	ids := append([]int(nil), tokens...)
	inputIDs := tensors.NewInt32sWithData([]int{1, len(ids)}, toI32(ids))
	logits := transformer.PrefillCED(inputIDs, cache)
	lastRow := func(logits *tensors.Tensor) *tensors.Tensor {
		rows := logits.Shape[1]
		flat := logits.Reshape(rows, config.VocabSize)
		row := append([]float32(nil), flat.Data[(rows-1)*config.VocabSize:rows*config.VocabSize]...)
		return tensors.NewWithData([]int{1, config.VocabSize}, row)
	}
	next := int(SampleNextToken(lastRow(logits), tensors.NewRNG(1), 0, 0).Data[0])
	ids = append(ids, next)
	for step := 1; step < maxTokens; step++ {
		nextIDs := tensors.NewInt32sWithData([]int{1, 1}, toI32([]int{next}))
		logits = transformer.Forward(nextIDs, cache)
		next = int(SampleNextToken(lastRow(logits), tensors.NewRNG(1), 0, 0).Data[0])
		ids = append(ids, next)
	}
	return ids
}

func TestEngineMatchesCEDPrefill(t *testing.T) {
	cases := []struct {
		name         string
		sparseTopK   int
		reusePattern string
	}{
		{"dense", 0, ""},
		{"sparse", 2, ""},
		{"dense-reuse", 0, "FR"},
		{"sparse-reuse", 2, "FR"},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			transformer, tokenizerImpl := testCEDEngineModel(testCase.sparseTopK, testCase.reusePattern)
			engine := NewEngine(transformer, tokenizerImpl)
			prompt := []int{1, 5, 2, 8, 3, 7, 4, 6}

			want := cedPrefillReference(transformer, prompt, 6)
			got, _ := engine.GenerateBatch(prompt, 1, 6, 0, 0, 7)

			limit := len(want)
			if len(got[0]) < limit {
				limit = len(got[0])
			}
			for index := 0; index < limit; index++ {
				if got[0][index] != want[index] {
					t.Fatalf("token %d: engine=%d reference=%d", index, got[0][index], want[index])
				}
			}
		})
	}
}

// TestEngineMatchesRecomputeCED keeps a full-forward comparison for prompts
// that fit inside the replay window, where the bounded replay is exact.
func TestEngineMatchesRecomputeCED(t *testing.T) {
	config := model.Config{
		SequenceLen: 16, VocabSize: 32, NumLayer: 4, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L", CompressionRatio: 2, SWAWindow: 16, CED: true,
	}
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	ranks := make(map[string]int, 256)
	for index := 0; index < 256; index++ {
		ranks[string([]byte{byte(index)})] = index
	}
	tokenizerImpl := tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
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
			t.Fatalf("token %d: engine=%d recompute=%d", index, got[0][index], want[index])
		}
	}
}
