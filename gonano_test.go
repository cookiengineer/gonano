package gonano_test

import (
	"path/filepath"
	"testing"

	"github.com/cookiengineer/gonano/inference"
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/model/checkpoint"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
	"github.com/cookiengineer/gonano/trainer"
)

// TestEndToEnd exercises the full pipeline: train a tiny model, save it, load
// it back, and generate from it.
func TestEndToEnd(t *testing.T) {
	cfg := model.Config{
		SequenceLen: 16, VocabSize: 265, NumLayer: 1, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	m := model.NewTransformer(cfg)
	m.InitWeights(tensors.NewRNG(0))
	groups := m.SetupOptimizer(0.01, 0.1, 0.01, 0.0, 0.1, false)
	tr := trainer.NewTrainer(m, groups, 1)
	tr.WarmupSteps = 2
	tr.WarmdownRatio = 0

	x := tensors.NewInt32sWithData([]int{1, 8}, []int32{1, 2, 3, 4, 5, 6, 7, 8})
	y := tensors.NewInt32sWithData([]int{1, 8}, []int32{2, 3, 4, 5, 6, 7, 8, 9})
	for step := 0; step < 20; step++ {
		tr.TrainStep(x, y)
		tr.StepOptimizer(step, 20)
	}

	// Save + load.
	path := filepath.Join(t.TempDir(), "model_000020.gn")
	if err := checkpoint.Save(path, checkpoint.Meta{Step: 20, ModelConfig: cfg}, m.NamedParameters()); err != nil {
		t.Fatalf("save: %v", err)
	}
	meta, params, err := checkpoint.Load(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	loaded := checkpoint.LoadModel(meta, params)

	// Tokenizer + engine + generation.
	ranks := make(map[string]int, 256)
	for i := 0; i < 256; i++ {
		ranks[string([]byte{byte(i)})] = i
	}
	tok := tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
	engine := inference.NewEngine(loaded, tok)

	prompt := []int{1, 2, 3}
	results, _ := engine.GenerateBatch(prompt, 1, 4, 0.0, 0, 42)
	if len(results) != 1 || len(results[0]) < len(prompt) {
		t.Fatalf("unexpected generation result %v", results)
	}
	decoded := tok.Decode(results[0])
	if decoded == "" {
		t.Fatal("empty generation")
	}
}
