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

func TestEnsembleSchedulerDecoupledPrefillDecode(t *testing.T) {
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
	scheduler := NewEnsembleScheduler(ensemble, 2)
	scheduler.PrefillChunk = 2
	scheduler.Submit(Request{ID: "a", Prompt: []int{1, 5, 2, 8, 3, 7}, MaxTokens: 6, Seed: 1})
	scheduler.Submit(Request{ID: "b", Prompt: []int{1, 2}, MaxTokens: 6, Seed: 2})
	scheduler.Step()

	var prefill *Sequence
	var decode *Sequence
	for _, sequence := range scheduler.running {
		switch sequence.Status {
		case SequencePrefill:
			prefill = sequence
		case SequenceDecode:
			decode = sequence
		}
	}
	if prefill == nil || decode == nil {
		t.Fatalf("expected mixed prefill/decode rows, got %v", scheduler.running)
	}
	if prefill.Cache != scheduler.prefillCaches[0] {
		t.Fatal("prefill row is not in the detached ensemble prefill cache")
	}
	if decode.Cache != scheduler.caches[0] {
		t.Fatal("decode row is not in the ensemble decode cache")
	}
	prefilled := prefill.Prefilled
	generated := len(decode.Generated)

	scheduler.Step()
	if prefill.Prefilled != prefilled+2 {
		t.Fatalf("prefill advanced %d tokens, want a chunk of 2", prefill.Prefilled-prefilled)
	}
	if len(decode.Generated) != generated+1 {
		t.Fatalf("decode generated %d tokens, want 1", len(decode.Generated)-generated)
	}
}

func TestEnsembleSchedulerPreemptsAndReplays(t *testing.T) {
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
	scheduler := NewEnsembleScheduler(ensemble, 2)
	scheduler.Preemption = PreemptLongest
	scheduler.Blocks = []*model.KVBlockManager{
		model.NewKVBlockManager(9),
		model.NewKVBlockManager(9),
	}
	promptA := []int{1, 5, 2, 8, 3, 7}
	promptB := []int{1, 2}
	scheduler.Submit(Request{ID: "a", Prompt: promptA, MaxTokens: 4, Seed: 1})
	scheduler.Step()
	scheduler.Step()
	scheduler.Submit(Request{ID: "b", Prompt: promptB, MaxTokens: 4, Seed: 2})
	finished := scheduler.Run()
	if len(finished) != 2 {
		t.Fatalf("finished = %d, want 2", len(finished))
	}
	SortFinished(finished)
	for index, prompt := range [][]int{promptA, promptB} {
		want, _ := ensemble.GenerateBatch(prompt, 1, 4, 0, 0, uint64(index+1))
		assertSameTokens(t, finished[index].Generated, want[0][len(prompt):])
	}
	if scheduler.Stats().Preemptions == 0 {
		t.Fatal("ensemble scheduler did not report a preemption")
	}
	for index, manager := range scheduler.Blocks {
		if manager.InUse() != 0 {
			t.Fatalf("ensemble member %d leaked %d blocks", index, manager.InUse())
		}
	}
}

func TestEnsembleSchedulerCEDMatchesSequential(t *testing.T) {
	config := model.ConfigForPreset(model.PresetFlash, 4, 32, 16, 32, 64, "SSSL")
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
	scheduler := NewEnsembleScheduler(ensemble, 2)
	prompts := [][]int{{1, 5, 2, 8, 3, 7}, {1, 2}}
	for index, prompt := range prompts {
		scheduler.Submit(Request{ID: string(rune('a' + index)), Prompt: prompt, MaxTokens: 6, Seed: uint64(index + 1)})
	}
	finished := scheduler.Run()
	if len(finished) != 2 {
		t.Fatalf("finished = %d, want 2", len(finished))
	}
	SortFinished(finished)
	for index, prompt := range prompts {
		want, _ := ensemble.GenerateBatch(prompt, 1, 6, 0, 0, uint64(index+1))
		assertSameTokens(t, finished[index].Generated, want[0][len(prompt):])
	}
	for index, manager := range scheduler.Blocks {
		if manager.InUse() != 0 {
			t.Fatalf("ensemble member %d leaked %d blocks", index, manager.InUse())
		}
	}
}
