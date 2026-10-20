package inference

import (
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
)

func scheduleComparison(t *testing.T, backbone *model.Transformer, tokenizerImpl *tokenizer.Tokenizer, prompts [][]int, maxTokens, maxActive, chunk int, reference func(*model.Transformer, []int, int) []int) {
	t.Helper()
	engine := NewEngine(backbone, tokenizerImpl)
	scheduler := NewScheduler(engine, maxActive)
	scheduler.PrefillChunk = chunk
	for index, prompt := range prompts {
		scheduler.Submit(Request{
			ID:        string(rune('a' + index)),
			Prompt:    prompt,
			MaxTokens: maxTokens,
			Seed:      uint64(index + 1),
		})
	}
	finished := scheduler.Run()
	if len(finished) != len(prompts) {
		t.Fatalf("finished %d requests, want %d", len(finished), len(prompts))
	}
	SortFinished(finished)
	for index, sequence := range finished {
		want := reference(backbone, prompts[index], maxTokens)
		assertSameTokens(t, sequence.Generated, want)
	}
	if stats := scheduler.Stats(); stats.Waiting != 0 || stats.Running != 0 {
		t.Fatalf("scheduler did not drain: %+v", stats)
	}
}

func TestSchedulerMatchesSequentialGeneration(t *testing.T) {
	backbone, _, tokenizerImpl := testSpeculativeModel()
	prompts := [][]int{
		{1, 5, 2, 8, 3, 7},
		{1, 2},
		{1, 9, 4, 6},
	}
	scheduleComparison(t, backbone, tokenizerImpl, prompts, 8, 2, 2, func(transformer *model.Transformer, prompt []int, maxTokens int) []int {
		engine := NewEngine(transformer, tokenizerImpl)
		got, _ := engine.GenerateBatch(prompt, 1, maxTokens, 0, 0, 1)
		return got[0][len(prompt):]
	})
}

func TestSchedulerSerializedAndChunked(t *testing.T) {
	backbone, _, tokenizerImpl := testSpeculativeModel()
	prompts := [][]int{{1, 5, 2, 8, 3, 7}, {1, 2, 3}}
	scheduleComparison(t, backbone, tokenizerImpl, prompts, 6, 1, 3, func(transformer *model.Transformer, prompt []int, maxTokens int) []int {
		engine := NewEngine(transformer, tokenizerImpl)
		got, _ := engine.GenerateBatch(prompt, 1, maxTokens, 0, 0, 1)
		return got[0][len(prompt):]
	})
}

func TestSchedulerCompressedAndMLA(t *testing.T) {
	config := model.ConfigForPreset(model.PresetFlash, 4, 32, 16, 32, 64, "SSSL")
	backbone := model.NewTransformer(config)
	backbone.InitWeights(tensors.NewRNG(42))
	perturbSpeculative(backbone, 11)
	prompts := [][]int{{1, 5, 2, 8}, {1, 3, 7}}
	scheduleComparison(t, backbone, testSpeculativeTokenizer(), prompts, 6, 2, 2, greedyReferenceCompressed)
}

func TestSchedulerJSONConstraint(t *testing.T) {
	tokenizerImpl := jsonTestTokenizer()
	config := model.Config{
		SequenceLen: 32, VocabSize: tokenizerImpl.VocabSize(), NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	backbone := model.NewTransformer(config)
	backbone.InitWeights(tensors.NewRNG(42))
	engine := NewEngine(backbone, tokenizerImpl)
	scheduler := NewScheduler(engine, 1)
	scheduler.Submit(Request{
		ID: "json", Prompt: []int{tokenizerImpl.EncodeSpecial("<|bos|>")}, MaxTokens: 12,
		Seed: 1, Constraint: NewJSONConstraint(),
	})
	finished := scheduler.Run()
	if len(finished) != 1 {
		t.Fatalf("finished = %d", len(finished))
	}
	constraint := NewJSONConstraint()
	for _, token := range finished[0].Generated {
		bytes := tokenizerImpl.DecodeSingleTokenBytes(token)
		if tokenizerImpl.IsSpecial(token) {
			break
		}
		if !constraint.Allowed(bytes) {
			t.Fatalf("scheduler emitted disallowed token %q", string(bytes))
		}
		constraint.Advance(bytes)
	}
}

func TestSchedulerBatchesDecodeRows(t *testing.T) {
	backbone, _, tokenizerImpl := testSpeculativeModel()
	engine := NewEngine(backbone, tokenizerImpl)
	scheduler := NewScheduler(engine, 2)
	scheduler.Submit(Request{ID: "a", Prompt: []int{1, 5, 2, 8, 3, 7}, MaxTokens: 8, Seed: 1})
	scheduler.Submit(Request{ID: "b", Prompt: []int{1, 2}, MaxTokens: 8, Seed: 2})
	scheduler.Step()

	active := 0
	for _, sequence := range scheduler.rows {
		if sequence != nil {
			active++
		}
	}
	if active != 2 {
		t.Fatalf("active work rows = %d, want 2", active)
	}
	if scheduler.rows[0] == nil || scheduler.rows[1] == nil {
		t.Fatal("work rows are not contiguous after batching")
	}
	if scheduler.rows[0].Cache != scheduler.workCache || scheduler.rows[1].Cache != scheduler.workCache {
		t.Fatal("sequences do not share the mixed work cache")
	}

	scheduler.Run()
	if blocks := scheduler.Blocks.InUse(); blocks != 0 {
		t.Fatalf("scheduler leaked %d blocks", blocks)
	}
}

func TestSchedulerQueuesWhenBlocksExhausted(t *testing.T) {
	backbone, _, tokenizerImpl := testSpeculativeModel()
	engine := NewEngine(backbone, tokenizerImpl)
	scheduler := NewScheduler(engine, 2)
	scheduler.Blocks = model.NewKVBlockManager(9)
	scheduler.Submit(Request{ID: "a", Prompt: []int{1, 5, 2, 8, 3, 7}, MaxTokens: 4, Seed: 1})
	scheduler.Submit(Request{ID: "b", Prompt: []int{1, 2}, MaxTokens: 4, Seed: 2})

	finished := scheduler.Run()
	if len(finished) != 2 {
		t.Fatalf("finished = %d, want 2", len(finished))
	}
	if scheduler.reserved != 0 {
		t.Fatalf("reserved blocks = %d, want 0", scheduler.reserved)
	}
	if scheduler.Blocks.InUse() != 0 {
		t.Fatalf("blocks leaked: inUse=%d", scheduler.Blocks.InUse())
	}
}

func TestSchedulerReusesBatchedRows(t *testing.T) {
	backbone, _, tokenizerImpl := testSpeculativeModel()
	engine := NewEngine(backbone, tokenizerImpl)
	scheduler := NewScheduler(engine, 2)
	scheduler.Submit(Request{ID: "a", Prompt: []int{1, 5, 2, 8, 3, 7}, MaxTokens: 6, Seed: 1})
	scheduler.Submit(Request{ID: "b", Prompt: []int{1, 2}, MaxTokens: 1, Seed: 2})
	scheduler.Submit(Request{ID: "c", Prompt: []int{1, 9, 4}, MaxTokens: 4, Seed: 3})

	finished := scheduler.Run()
	if len(finished) != 3 {
		t.Fatalf("finished = %d, want 3", len(finished))
	}
	if scheduler.Blocks.InUse() != 0 {
		t.Fatalf("blocks leaked: inUse=%d", scheduler.Blocks.InUse())
	}
}

func TestSchedulerThinkingBudget(t *testing.T) {
	transformer, tokenizerImpl := testThinkingModel()
	engine := NewEngine(transformer, tokenizerImpl)
	scheduler := NewScheduler(engine, 1)
	thinkStart := tokenizerImpl.EncodeSpecial("<|think_start|>")
	thinkEnd := tokenizerImpl.EncodeSpecial("<|think_end|>")
	prompt := []int{tokenizerImpl.BOSTokenID(), tokenizerImpl.EncodeSpecial("<|assistant_start|>"), thinkStart}
	scheduler.Submit(Request{
		ID: "think", Prompt: prompt, MaxTokens: 6, Seed: 1,
		Options: GenerateOptions{Thinking: true, ThinkingBudget: 2},
	})

	var events []Event
	for len(scheduler.waiting) > 0 || len(scheduler.running) > 0 {
		events = append(events, scheduler.Step()...)
	}
	forced := false
	for _, event := range events {
		if event.TokenMask == 0 && event.Token == thinkEnd {
			forced = true
		}
	}
	if !forced {
		t.Fatal("scheduler never forced <|think_end|> after the thinking budget")
	}
}

func TestSchedulerLogprobs(t *testing.T) {
	backbone, _, tokenizerImpl := testSpeculativeModel()
	engine := NewEngine(backbone, tokenizerImpl)
	scheduler := NewScheduler(engine, 2)
	callbacks := 0
	scheduler.Submit(Request{
		ID: "lp", Prompt: []int{1, 2, 3}, MaxTokens: 4, Temperature: 0.7, TopK: 5, Seed: 1,
		Options: GenerateOptions{
			Logprobs: true, TopLogprobs: 2,
			OnLogprobs: func(rows []TokenLogprob) { callbacks += len(rows) },
		},
	})
	finished := scheduler.Run()
	if len(finished) != 1 {
		t.Fatalf("finished = %d", len(finished))
	}
	sequence := finished[0]
	if len(sequence.Logprobs) != len(sequence.Generated) {
		t.Fatalf("logprobs = %d, generated = %d", len(sequence.Logprobs), len(sequence.Generated))
	}
	if callbacks != len(sequence.Generated) {
		t.Fatalf("callbacks = %d, generated = %d", callbacks, len(sequence.Generated))
	}
	for _, entry := range sequence.Logprobs {
		if len(entry.Top) != 2 {
			t.Fatalf("top logprobs = %d, want 2", len(entry.Top))
		}
	}
}

func TestSchedulerCompletesToolCall(t *testing.T) {
	backbone, _, tokenizerImpl := testSpeculativeModel()
	engine := NewEngine(backbone, tokenizerImpl)
	scheduler := NewScheduler(engine, 1)
	sequence := scheduler.Submit(Request{ID: "tool", Prompt: []int{1}, MaxTokens: 4, Seed: 1})
	sequence.inToolCall = true
	sequence.toolCallTokens = tokenizerImpl.Encode("2+2")
	completeToolCall(engine.Tools, tokenizerImpl, sequence)
	if len(sequence.forcedTokens) < 3 {
		t.Fatalf("forced tokens = %v", sequence.forcedTokens)
	}
	if sequence.forcedTokens[0] != tokenizerImpl.EncodeSpecial("<|tool_output_start|>") {
		t.Fatal("tool output did not start with <|tool_output_start|>")
	}
	if sequence.forcedTokens[len(sequence.forcedTokens)-1] != tokenizerImpl.EncodeSpecial("<|tool_output_end|>") {
		t.Fatal("tool output did not end with <|tool_output_end|>")
	}
	decoded := tokenizerImpl.Decode(sequence.forcedTokens[1 : len(sequence.forcedTokens)-1])
	if decoded != "4" {
		t.Fatalf("tool result = %q, want 4", decoded)
	}
}

func TestSchedulerSpeculativeMatchesGreedy(t *testing.T) {
	backbone, drafter, tokenizerImpl := testSpeculativeModel()
	engine := NewEngine(backbone, tokenizerImpl)
	engine.Speculative = true
	engine.Drafter = drafter
	engine.DraftLength = 4
	scheduler := NewScheduler(engine, 2)
	prompt := []int{1, 5, 2, 8, 3, 7}
	scheduler.Submit(Request{ID: "spec", Prompt: prompt, MaxTokens: 8, Seed: 1})
	finished := scheduler.Run()
	if len(finished) != 1 {
		t.Fatalf("finished = %d", len(finished))
	}
	want := greedyReference(backbone, prompt, 8)
	if len(finished[0].Generated) != len(want) {
		t.Fatalf("generated %v, want %v", finished[0].Generated, want)
	}
	for index := range want {
		if finished[0].Generated[index] != want[index] {
			t.Fatalf("token %d = %d, want %d", index, finished[0].Generated[index], want[index])
		}
	}
	if scheduler.Blocks.InUse() != 0 {
		t.Fatalf("blocks leaked: inUse=%d", scheduler.Blocks.InUse())
	}
}

func TestSchedulerBlockPrefixReuse(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*model.Config)
	}{
		{"plain", func(config *model.Config) {}},
		{"compressed", func(config *model.Config) { config.CompressionRatio = 2; config.SWAWindow = 4 }},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			config := model.Config{
				SequenceLen: 256, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
				EmbedDim: 32, WindowPattern: "L",
			}
			testCase.mutate(&config)
			backbone := model.NewTransformer(config)
			backbone.InitWeights(tensors.NewRNG(42))
			tokenizerImpl := testSpeculativeTokenizer()

			prompt1 := make([]int, 200)
			for index := range prompt1 {
				prompt1[index] = index%11 + 1
			}
			prompt2 := make([]int, 160)
			copy(prompt2, prompt1[:model.KVBlockSize])
			for index := model.KVBlockSize; index < len(prompt2); index++ {
				prompt2[index] = index%9 + 2
			}

			prefix := NewBlockPrefixCache(4, 0)
			firstEngine := NewEngine(backbone, tokenizerImpl)
			first := NewScheduler(firstEngine, 2)
			first.Prefix = prefix
			first.Submit(Request{ID: "first", Prompt: prompt1, MaxTokens: 2, Seed: 1})
			first.Run()
			if prefix.Len() == 0 {
				t.Fatal("scheduler stored no block-aligned snapshot")
			}

			secondEngine := NewEngine(backbone, tokenizerImpl)
			second := NewScheduler(secondEngine, 2)
			second.Prefix = prefix
			second.Submit(Request{ID: "second", Prompt: prompt2, MaxTokens: 4, Seed: 7})
			finished := second.Run()
			reference, _ := secondEngine.GenerateBatch(prompt2, 1, 4, 0, 0, 7)
			if len(finished) != 1 || len(reference) != 1 {
				t.Fatalf("finished/reference = %d/%d", len(finished), len(reference))
			}
			if len(finished[0].Generated) != len(reference[0][len(prompt2):]) {
				t.Fatalf("generated %v, want %v", finished[0].Generated, reference[0][len(prompt2):])
			}
			for index, token := range reference[0][len(prompt2):] {
				if finished[0].Generated[index] != token {
					t.Fatalf("token %d = %d, want %d", index, finished[0].Generated[index], token)
				}
			}
			if hits, _ := prefix.Stats(); hits == 0 {
				t.Fatal("scheduler prefix cache was never hit")
			}
		})
	}
}

func TestSchedulerMixedPrefillDecodeForward(t *testing.T) {
	backbone, _, tokenizerImpl := testSpeculativeModel()
	engine := NewEngine(backbone, tokenizerImpl)
	scheduler := NewScheduler(engine, 2)
	scheduler.PrefillChunk = 2
	scheduler.Submit(Request{ID: "a", Prompt: []int{1, 5, 2, 8, 3, 7}, MaxTokens: 8, Seed: 1})
	scheduler.Submit(Request{ID: "b", Prompt: []int{1, 2}, MaxTokens: 8, Seed: 2})

	scheduler.Step()
	var prefill *Sequence
	var decode *Sequence
	for _, sequence := range scheduler.running {
		if !sequence.Solo && sequence.Status == SequencePrefill {
			prefill = sequence
		}
		if !sequence.Solo && sequence.Status == SequenceDecode {
			decode = sequence
		}
	}
	if prefill == nil || decode == nil {
		t.Fatalf("expected mixed prefill/decode rows, got %v", scheduler.running)
	}
	if prefill.Cache != scheduler.workCache || decode.Cache != scheduler.workCache {
		t.Fatal("mixed rows do not share the work cache")
	}
	prefilled := prefill.Prefilled
	generated := len(decode.Generated)

	scheduler.Step()
	if prefill.Prefilled != prefilled+1 {
		t.Fatalf("prefill advanced %d tokens, want 1 in mixed mode", prefill.Prefilled-prefilled)
	}
	if len(decode.Generated) != generated+1 {
		t.Fatalf("decode generated %d tokens, want 1", len(decode.Generated)-generated)
	}
}

func TestSchedulerStatsReportWorkAndKVUsage(t *testing.T) {
	backbone, _, tokenizerImpl := testSpeculativeModel()
	engine := NewEngine(backbone, tokenizerImpl)
	scheduler := NewScheduler(engine, 2)
	prompt := []int{1, 5, 2, 8, 3, 7}
	scheduler.Submit(Request{ID: "stats", Prompt: prompt, MaxTokens: 4, Seed: 1})
	scheduler.Run()
	stats := scheduler.Stats()
	if stats.Generated == 0 || stats.PrefillTokens < int64(len(prompt)) || stats.DecodeTokens == 0 {
		t.Fatalf("stats = %+v", stats)
	}
	if stats.KVBlocksInUse != 0 || stats.KVBlocksFree == 0 {
		t.Fatalf("KV stats = inUse %d free %d", stats.KVBlocksInUse, stats.KVBlocksFree)
	}
	if stats.Finished != 1 {
		t.Fatalf("finished = %d", stats.Finished)
	}
}

func TestSchedulerPreemptsAndReplays(t *testing.T) {
	backbone, _, tokenizerImpl := testSpeculativeModel()
	engine := NewEngine(backbone, tokenizerImpl)
	scheduler := NewScheduler(engine, 2)
	scheduler.Blocks = model.NewKVBlockManager(9)
	scheduler.Preemption = PreemptLongest
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
		want, _ := engine.GenerateBatch(prompt, 1, 4, 0, 0, uint64(index+1))
		assertSameTokens(t, finished[index].Generated, want[0][len(prompt):])
	}
	if scheduler.Stats().Preemptions == 0 {
		t.Fatal("scheduler did not report a preemption")
	}
	if scheduler.Blocks.InUse() != 0 {
		t.Fatalf("blocks leaked: %d", scheduler.Blocks.InUse())
	}
}

func TestSchedulerPreemptsCompressedAndReplays(t *testing.T) {
	config := model.Config{
		SequenceLen: 32, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L", CompressionRatio: 2, SWAWindow: 4,
	}
	backbone := model.NewTransformer(config)
	backbone.InitWeights(tensors.NewRNG(42))
	perturbSpeculative(backbone, 11)
	tokenizerImpl := testSpeculativeTokenizer()
	engine := NewEngine(backbone, tokenizerImpl)
	scheduler := NewScheduler(engine, 2)
	scheduler.Blocks = model.NewKVBlockManager(9)
	scheduler.Preemption = PreemptLongest
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
		want, _ := engine.GenerateBatch(prompt, 1, 4, 0, 0, uint64(index+1))
		assertSameTokens(t, finished[index].Generated, want[0][len(prompt):])
	}
	if scheduler.Stats().Preemptions == 0 {
		t.Fatal("compressed replay did not preempt")
	}
	if scheduler.Blocks.InUse() != 0 {
		t.Fatalf("blocks leaked: %d", scheduler.Blocks.InUse())
	}
}

func TestSchedulerCompressedMultiRequestMatchesSequential(t *testing.T) {
	config := model.Config{
		SequenceLen: 32, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L", CompressionRatio: 2, SWAWindow: 4,
	}
	backbone := model.NewTransformer(config)
	backbone.InitWeights(tensors.NewRNG(42))
	perturbSpeculative(backbone, 11)
	tokenizerImpl := testSpeculativeTokenizer()
	engine := NewEngine(backbone, tokenizerImpl)
	scheduler := NewScheduler(engine, 2)
	promptA := []int{1, 5, 2, 8, 3, 7}
	promptB := []int{1, 2}
	scheduler.Submit(Request{ID: "a", Prompt: promptA, MaxTokens: 4, Seed: 1})
	scheduler.Submit(Request{ID: "b", Prompt: promptB, MaxTokens: 4, Seed: 2})
	finished := scheduler.Run()
	SortFinished(finished)
	for index, prompt := range [][]int{promptA, promptB} {
		want, _ := engine.GenerateBatch(prompt, 1, 4, 0, 0, uint64(index+1))
		assertSameTokens(t, finished[index].Generated, want[0][len(prompt):])
	}
}

func TestSchedulerCompressedSerializedMatchesSequential(t *testing.T) {
	config := model.Config{
		SequenceLen: 32, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L", CompressionRatio: 2, SWAWindow: 4,
	}
	backbone := model.NewTransformer(config)
	backbone.InitWeights(tensors.NewRNG(42))
	perturbSpeculative(backbone, 11)
	tokenizerImpl := testSpeculativeTokenizer()
	engine := NewEngine(backbone, tokenizerImpl)
	scheduler := NewScheduler(engine, 1)
	promptA := []int{1, 5, 2, 8, 3, 7}
	promptB := []int{1, 2}
	scheduler.Submit(Request{ID: "a", Prompt: promptA, MaxTokens: 4, Seed: 1})
	scheduler.Submit(Request{ID: "b", Prompt: promptB, MaxTokens: 4, Seed: 2})
	finished := scheduler.Run()
	SortFinished(finished)
	for index, prompt := range [][]int{promptA, promptB} {
		want, _ := engine.GenerateBatch(prompt, 1, 4, 0, 0, uint64(index+1))
		assertSameTokens(t, finished[index].Generated, want[0][len(prompt):])
	}
}

func TestSchedulerShortCompressedMatchesReferences(t *testing.T) {
	config := model.Config{
		SequenceLen: 32, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L", CompressionRatio: 2, SWAWindow: 4,
	}
	backbone := model.NewTransformer(config)
	backbone.InitWeights(tensors.NewRNG(42))
	perturbSpeculative(backbone, 11)
	tokenizerImpl := testSpeculativeTokenizer()
	engine := NewEngine(backbone, tokenizerImpl)
	scheduler := NewScheduler(engine, 1)
	promptB := []int{1, 2}
	scheduler.Submit(Request{ID: "b", Prompt: promptB, MaxTokens: 4, Seed: 2})
	finished := scheduler.Run()
	want, _ := engine.GenerateBatch(promptB, 1, 4, 0, 0, 2)
	truth := compressedReference(backbone, promptB, 4)
	assertSameTokens(t, finished[0].Generated, want[0][len(promptB):])
	assertSameTokens(t, finished[0].Generated, truth[len(promptB):])
}

func TestSchedulerRunnerCancelReleasesState(t *testing.T) {
	backbone, _, tokenizerImpl := testSpeculativeModel()
	engine := NewEngine(backbone, tokenizerImpl)
	scheduler := NewScheduler(engine, 1)
	runner := NewSchedulerRunner(scheduler)
	submission := runner.Submit(Request{ID: "cancel", Prompt: []int{1, 5, 2, 8, 3, 7}, MaxTokens: 100, Seed: 1})
	if _, ok := submission.Next(); !ok {
		t.Fatal("submission closed before producing a token")
	}
	submission.Cancel()
	for {
		if _, ok := submission.Next(); !ok {
			break
		}
	}
	runner.Stop()
	if scheduler.Pending() {
		t.Fatal("cancelled scheduler still has work")
	}
	if scheduler.Blocks.InUse() != 0 {
		t.Fatalf("cancelled request leaked blocks: %d", scheduler.Blocks.InUse())
	}
}

func TestSchedulerRegexConstraintEnforced(t *testing.T) {
	tokenizerImpl := jsonTestTokenizer()
	config := model.Config{
		SequenceLen: 16, VocabSize: tokenizerImpl.VocabSize(), NumLayer: 1, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	backbone := model.NewTransformer(config)
	backbone.InitWeights(tensors.NewRNG(1))
	engine := NewEngine(backbone, tokenizerImpl)
	scheduler := NewScheduler(engine, 1)
	constraint, err := NewRegexConstraint(`ab`)
	if err != nil {
		t.Fatalf("regex: %v", err)
	}
	scheduler.Submit(Request{ID: "regex", Prompt: []int{tokenizerImpl.EncodeSpecial("<|bos|>")}, MaxTokens: 8, Seed: 1, Constraint: constraint})
	finished := scheduler.Run()
	if len(finished) != 1 {
		t.Fatalf("finished = %d", len(finished))
	}
	if text := tokenizerImpl.Decode(finished[0].Generated); text != "ab" {
		t.Fatalf("constrained output = %q, want ab", text)
	}
}
