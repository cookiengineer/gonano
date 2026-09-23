package inference

import (
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
)

func tinyBlendModel(vocab int, seed uint64) *model.Transformer {
	configuration := model.Config{
		SequenceLen: 16, VocabSize: vocab, NumLayer: 1, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	transformer := model.NewTransformer(configuration)
	transformer.InitWeights(tensors.NewRNG(seed))
	return transformer
}

// TestEnsembleSingleMemberMatchesEngine verifies that a one-member blend is
// bit-identical to the plain single-model engine, so blending is a strict
// generalization.
func TestEnsembleSingleMemberMatchesEngine(t *testing.T) {
	tok := byteTokenizerForTest()
	base := tinyBlendModel(tok.VocabSize(), 1)

	engine := NewEngine(base, tok)
	want, _ := engine.GenerateBatch([]int{1, 2, 3}, 1, 4, 0, 0, 42)

	ensemble := NewEnsemble([]ModelWeight{{Model: base, Weight: 1}}, tok)
	got, _ := ensemble.GenerateBatch([]int{1, 2, 3}, 1, 4, 0, 0, 42)

	if len(got) != len(want) || len(got[0]) != len(want[0]) {
		t.Fatalf("shape mismatch: got %v want %v", got, want)
	}
	for index := range want[0] {
		if got[0][index] != want[0][index] {
			t.Fatalf("token %d differs: got %d want %d", index, got[0][index], want[0][index])
		}
	}
}

// TestEnsembleBlendIsDeterministic checks a two-member blend is finite and
// reproducible for a fixed seed.
func TestEnsembleBlendIsDeterministic(t *testing.T) {
	tok := byteTokenizerForTest()
	first := tinyBlendModel(tok.VocabSize(), 1)
	second := tinyBlendModel(tok.VocabSize(), 2)

	ensemble := NewEnsemble([]ModelWeight{
		{Model: first, Weight: 0.5, Domain: "a"},
		{Model: second, Weight: 0.5, Domain: "b"},
	}, tok)
	runA, _ := ensemble.GenerateBatch([]int{1, 2, 3}, 1, 6, 0.8, 10, 7)
	runB, _ := ensemble.GenerateBatch([]int{1, 2, 3}, 1, 6, 0.8, 10, 7)
	if len(runA) != 1 || len(runA[0]) < 4 {
		t.Fatalf("unexpected blend output %v", runA)
	}
	if len(runA[0]) != len(runB[0]) {
		t.Fatalf("non-deterministic length: %v vs %v", runA, runB)
	}
	for index := range runA[0] {
		if runA[0][index] != runB[0][index] {
			t.Fatalf("non-deterministic token at %d: %v vs %v", index, runA[0], runB[0])
		}
	}
}

func byteTokenizerForTest() *tokenizer.Tokenizer {
	ranks := make(map[string]int, 256)
	for index := 0; index < 256; index++ {
		ranks[string([]byte{byte(index)})] = index
	}
	return tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
}
