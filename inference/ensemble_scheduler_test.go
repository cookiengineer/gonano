package inference

import (
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
)

func TestEnsembleSchedulerMatchesSequential(t *testing.T) {
	config := model.Config{
		SequenceLen: 32, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	first := model.NewTransformer(config)
	first.InitWeights(tensors.NewRNG(42))
	second := model.NewTransformer(config)
	second.InitWeights(tensors.NewRNG(99))
	perturbSpeculative(first, 11)
	perturbSpeculative(second, 22)
	tokenizerImpl := testSpeculativeTokenizer()
	ensemble := NewEnsemble([]ModelWeight{
		{Model: first, Weight: 0.6, Domain: "math"},
		{Model: second, Weight: 0.4, Domain: "physics"},
	}, tokenizerImpl)

	prompts := [][]int{
		{1, 5, 2, 8, 3, 7},
		{1, 2},
	}
	scheduler := NewEnsembleScheduler(ensemble, 2)
	scheduler.PrefillChunk = 2
	for index, prompt := range prompts {
		scheduler.Submit(Request{ID: string(rune('a' + index)), Prompt: prompt, MaxTokens: 6, Seed: uint64(index + 1)})
	}
	finished := scheduler.Run()
	if len(finished) != len(prompts) {
		t.Fatalf("finished = %d, want %d", len(finished), len(prompts))
	}
	SortFinished(finished)
	for index, sequence := range finished {
		want, _ := ensemble.GenerateBatch(prompts[index], 1, 6, 0, 0, uint64(index+1))
		assertSameTokens(t, sequence.Generated, want[0][len(prompts[index]):])
	}
	for _, cache := range scheduler.caches {
		if cache.BlockManager().InUse() != 0 {
			t.Fatalf("ensemble member leaked blocks: %d", cache.BlockManager().InUse())
		}
	}
	stats := scheduler.Stats()
	if stats.Waiting != 0 || stats.Running != 0 {
		t.Fatalf("scheduler did not drain: %+v", stats)
	}
	if stats.Generated == 0 || stats.PrefillTokens == 0 || stats.DecodeTokens == 0 || stats.KVBlocksInUse != 0 {
		t.Fatalf("ensemble stats = %+v", stats)
	}
}

func TestEnsembleSchedulerThinkingBudget(t *testing.T) {
	transformer, tokenizerImpl := testThinkingModel()
	ensemble := NewEnsemble([]ModelWeight{
		{Model: transformer, Weight: 1, Domain: "a"},
		{Model: transformer, Weight: 1, Domain: "b"},
	}, tokenizerImpl)
	scheduler := NewEnsembleScheduler(ensemble, 1)
	thinkStart := tokenizerImpl.EncodeSpecial("<|think_start|>")
	thinkEnd := tokenizerImpl.EncodeSpecial("<|think_end|>")
	prompt := []int{tokenizerImpl.BOSTokenID(), tokenizerImpl.EncodeSpecial("<|assistant_start|>"), thinkStart}
	scheduler.Submit(Request{ID: "think", Prompt: prompt, MaxTokens: 6, Seed: 1, Options: GenerateOptions{Thinking: true, ThinkingBudget: 2}})
	var events []Event
	for scheduler.Pending() {
		events = append(events, scheduler.Step()...)
	}
	forced := false
	for _, event := range events {
		if event.TokenMask == 0 && event.Token == thinkEnd {
			forced = true
		}
	}
	if !forced {
		t.Fatal("ensemble scheduler never forced <|think_end|>")
	}
}

func TestEnsembleSchedulerLogprobs(t *testing.T) {
	transformer, tokenizerImpl := testThinkingModel()
	ensemble := NewEnsemble([]ModelWeight{
		{Model: transformer, Weight: 1, Domain: "a"},
		{Model: transformer, Weight: 1, Domain: "b"},
	}, tokenizerImpl)
	scheduler := NewEnsembleScheduler(ensemble, 1)
	scheduler.Submit(Request{
		ID: "logprobs", Prompt: []int{1, 2, 3}, MaxTokens: 4, Temperature: 0.7, TopK: 5, Seed: 1,
		Options: GenerateOptions{Logprobs: true, TopLogprobs: 2},
	})
	finished := scheduler.Run()
	if len(finished) != 1 || len(finished[0].Logprobs) != len(finished[0].Generated) {
		t.Fatalf("logprobs = %d generated = %d", len(finished[0].Logprobs), len(finished[0].Generated))
	}
	for _, entry := range finished[0].Logprobs {
		if len(entry.Top) != 2 {
			t.Fatalf("top logprobs = %d", len(entry.Top))
		}
	}
}

func TestEnsembleSchedulerRegexConstraint(t *testing.T) {
	tokenizerImpl := jsonTestTokenizer()
	config := model.Config{
		SequenceLen: 16, VocabSize: tokenizerImpl.VocabSize(), NumLayer: 1, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	first := model.NewTransformer(config)
	first.InitWeights(tensors.NewRNG(1))
	second := model.NewTransformer(config)
	second.InitWeights(tensors.NewRNG(2))
	ensemble := NewEnsemble([]ModelWeight{{Model: first, Weight: 1, Domain: "a"}, {Model: second, Weight: 1, Domain: "b"}}, tokenizerImpl)
	scheduler := NewEnsembleScheduler(ensemble, 1)
	constraint, err := NewRegexConstraint(`ab`)
	if err != nil {
		t.Fatalf("regex: %v", err)
	}
	scheduler.Submit(Request{ID: "regex", Prompt: []int{tokenizerImpl.EncodeSpecial("<|bos|>")}, MaxTokens: 8, Seed: 1, Constraint: constraint})
	finished := scheduler.Run()
	if len(finished) != 1 {
		t.Fatalf("finished = %d", len(finished))
	}
	text := tokenizerImpl.Decode(finished[0].Generated)
	if text != "ab" {
		t.Fatalf("constrained output = %q, want ab", text)
	}
}
