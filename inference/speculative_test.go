package inference

import (
	"path/filepath"
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/model/checkpoint"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
)

func testSpeculativeModel() (*model.Transformer, *model.Transformer, *tokenizer.Tokenizer) {
	config := model.Config{
		SequenceLen: 32, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	backbone := model.NewTransformer(config)
	backbone.InitWeights(tensors.NewRNG(42))
	perturbSpeculative(backbone, 11)
	drafter := model.NewDrafter(backbone)
	drafter.InitWeights(tensors.NewRNG(99))
	perturbSpeculative(drafter, 22)

	ranks := make(map[string]int, 256)
	for index := 0; index < 256; index++ {
		ranks[string([]byte{byte(index)})] = index
	}
	return backbone, drafter, tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
}

func perturbSpeculative(transformer *model.Transformer, seed uint64) {
	rng := tensors.NewRNG(seed)
	for _, parameter := range transformer.Parameters() {
		for index := range parameter.Data {
			parameter.Data[index] += rng.NormFloat32() * 0.2
		}
	}
}

// greedyReference decodes greedily with the target model, one token per step.
func greedyReference(transformer *model.Transformer, prompt []int, maxTokens int) []int {
	cache := model.NewKVBuffer(1, transformer.Config.SequenceLen, transformer.Config.NumLayer, transformer.Config.NumKVHead, transformer.Config.HeadDim())
	ids := tensors.NewInt32sWithData([]int{1, len(prompt)}, toI32(prompt))
	logits := transformer.Forward(ids, cache)
	vocab := transformer.Config.VocabSize
	last := logits.Data[(len(prompt)-1)*vocab : len(prompt)*vocab]

	output := make([]int, 0, maxTokens)
	for index := 0; index < maxTokens; index++ {
		token := argmaxSlice(last)
		output = append(output, token)
		next := tensors.NewInt32sWithData([]int{1, 1}, []int32{int32(token)})
		logits = transformer.Forward(next, cache)
		last = logits.Data
	}
	return output
}

// TestSpeculativeMatchesGreedy verifies that exact greedy speculative decoding
// reproduces the target model's greedy output.
func TestSpeculativeMatchesGreedy(t *testing.T) {
	backbone, drafter, tokenizerImpl := testSpeculativeModel()
	engine := NewEngine(backbone, tokenizerImpl)
	engine.Drafter = drafter
	engine.Speculative = true
	engine.DraftLength = 5

	prompt := []int{1, 5, 2, 8, 3, 7}
	got, _ := engine.GenerateBatch(prompt, 1, 12, 0, 0, 0)
	if len(got[0]) < len(prompt) {
		t.Fatalf("generation shorter than the prompt: %d", len(got[0]))
	}
	assertSameTokens(t, got[0][len(prompt):], greedyReference(backbone, prompt, 12))
}

// TestSpeculativeMatchesGreedyWithLongDraft checks the same property with a
// draft block longer than the remaining token budget.
func TestSpeculativeMatchesGreedyWithLongDraft(t *testing.T) {
	backbone, drafter, tokenizerImpl := testSpeculativeModel()
	engine := NewEngine(backbone, tokenizerImpl)
	engine.Drafter = drafter
	engine.Speculative = true
	engine.DraftLength = 16

	prompt := []int{1, 2, 3, 4}
	got, _ := engine.GenerateBatch(prompt, 1, 5, 0, 0, 0)
	assertSameTokens(t, got[0][len(prompt):], greedyReference(backbone, prompt, 5))
}

// TestLoadedDrafterSpeculation verifies that a drafter round-trips through a
// checkpoint and still yields exact greedy speculation.
func TestLoadedDrafterSpeculation(t *testing.T) {
	backbone, drafter, tokenizerImpl := testSpeculativeModel()
	path := filepath.Join(t.TempDir(), "drafter.gn")
	meta := checkpoint.Meta{Step: 1, ModelConfig: drafter.Config}
	if err := checkpoint.Save(path, meta, drafter.NamedParameters()); err != nil {
		t.Fatalf("save drafter: %v", err)
	}
	loadedMeta, loadedParams, err := checkpoint.Load(path)
	if err != nil {
		t.Fatalf("load drafter: %v", err)
	}
	loaded := checkpoint.LoadModel(loadedMeta, loadedParams)

	engine := NewEngine(backbone, tokenizerImpl)
	engine.Drafter = loaded
	engine.Speculative = true
	prompt := []int{1, 5, 2, 8, 3, 7}
	got, _ := engine.GenerateBatch(prompt, 1, 8, 0, 0, 0)
	assertSameTokens(t, got[0][len(prompt):], greedyReference(backbone, prompt, 8))
}

func TestSpeculativeEligibility(t *testing.T) {
	backbone, drafter, tokenizerImpl := testSpeculativeModel()
	engine := NewEngine(backbone, tokenizerImpl)
	engine.Drafter = drafter
	engine.Speculative = true
	if !engine.speculativeEligible(1, 0) {
		t.Fatal("greedy single-row uncompressed request should be eligible")
	}
	if engine.speculativeEligible(2, 0) {
		t.Fatal("batched requests must not speculate")
	}
	if engine.speculativeEligible(1, 0.5) {
		t.Fatal("non-greedy requests must not speculate")
	}

	compressed, compressedTokenizer := testPrefixEngineModel()
	compressedEngine := NewEngine(compressed, compressedTokenizer)
	compressedEngine.Drafter = model.NewDrafter(compressed)
	compressedEngine.Speculative = true
	if compressedEngine.speculativeEligible(1, 0) {
		t.Fatal("compressed models must not speculate")
	}
}

// TestSpeculativeWithDSparkMatchesGreedy verifies that the DSpark confidence
// scheduler preserves exactness.
func TestSpeculativeWithDSparkMatchesGreedy(t *testing.T) {
	backbone, _, tokenizerImpl := testSpeculativeModel()
	dspark := model.NewDSpark(backbone)
	dspark.InitWeights(tensors.NewRNG(123))

	engine := NewEngine(backbone, tokenizerImpl)
	engine.DSpark = dspark
	engine.Speculative = true
	engine.DraftLength = 5

	prompt := []int{1, 5, 2, 8, 3, 7}
	got, _ := engine.GenerateBatch(prompt, 1, 12, 0, 0, 0)
	assertSameTokens(t, got[0][len(prompt):], greedyReference(backbone, prompt, 12))
}

func TestScheduledLength(t *testing.T) {
	if got := scheduledLength(nil, 0.5); got != 1 {
		t.Fatalf("empty confidence length = %d, want 1", got)
	}
	if got := scheduledLength([]float32{0.9, 0.9, 0.1}, 0.5); got != 3 {
		t.Fatalf("length = %d, want 3", got)
	}
	if got := scheduledLength([]float32{0.1, 0.9}, 0.5); got != 1 {
		t.Fatalf("length = %d, want 1", got)
	}
}
