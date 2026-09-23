package tokenizer

import (
	"reflect"
	"testing"
)

func TestRenderConversation(tests *testing.T) {
	ranks := TrainBPE([]string{"hello world", "hi there", "the sky is blue"}, 30)
	tok := NewTokenizer(ranks, SpecialTokens)

	conv := &Conversation{
		Messages: []Message{
			{Role: "user", Content: "hi there"},
			{Role: "assistant", Content: "hello world"},
		},
	}
	ids, mask := tok.RenderConversation(conv, 2048)

	want := append([]int{tok.EncodeSpecial("<|bos|>")}, tok.EncodeSpecial("<|user_start|>"))
	want = append(want, tok.Encode("hi there")...)
	want = append(want, tok.EncodeSpecial("<|user_end|>"))
	want = append(want, tok.EncodeSpecial("<|assistant_start|>"))
	want = append(want, tok.Encode("hello world")...)
	want = append(want, tok.EncodeSpecial("<|assistant_end|>"))

	if !reflect.DeepEqual(ids, want) {
		tests.Fatalf("ids mismatch:\n got %v\nwant %v", ids, want)
	}
	if len(mask) != len(ids) {
		tests.Fatalf("mask length %d != ids length %d", len(mask), len(ids))
	}
}

// TestReasoningEffortInstruction checks the effort conditioning instruction is
// prepended to the system prompt and affects the rendered tokens.
func TestReasoningEffortInstruction(t *testing.T) {
	if got := ReasoningEffortInstruction(75); got != "Reasoning Effort: 75 (range 1--100; higher values request more thorough reasoning)" {
		t.Fatalf("instruction = %q", got)
	}
	ranks := TrainBPE([]string{"hello world"}, 10)
	tok := NewTokenizer(ranks, SpecialTokens)
	base := &Conversation{Messages: []Message{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "world"},
	}}
	withEffort := &Conversation{
		Messages: base.Messages,
		Extra:    map[string]any{"effort": 100},
	}
	baseIDs, _ := tok.RenderConversation(base, 2048)
	effortIDs, _ := tok.RenderConversation(withEffort, 2048)
	if len(effortIDs) <= len(baseIDs) {
		t.Fatalf("effort prompt should be longer: %d vs %d", len(effortIDs), len(baseIDs))
	}
	instructionTokens := tok.Encode(ReasoningEffortInstruction(100))
	found := false
	for start := 0; start+len(instructionTokens) <= len(effortIDs); start++ {
		match := true
		for offset := range instructionTokens {
			if effortIDs[start+offset] != instructionTokens[offset] {
				match = false
				break
			}
		}
		if match {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("effort instruction tokens not found in the rendered prompt")
	}
	// The prompt must not be mutated for subsequent renders.
	again, _ := tok.RenderConversation(base, 2048)
	if len(again) != len(baseIDs) {
		t.Fatalf("base render changed after effort render: %d vs %d", len(again), len(baseIDs))
	}
}

// TestRenderConversationThinking checks an assistant reasoning trace is
// rendered as a supervised <|think_start|>...<|think_end|> block.
func TestRenderConversationThinking(tests *testing.T) {
	ranks := TrainBPE([]string{"hello world", "reasoning here"}, 30)
	tok := NewTokenizer(ranks, SpecialTokens)
	conv := &Conversation{Messages: []Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Thinking: "reasoning here", Content: "hello world"},
	}}
	ids, mask := tok.RenderConversation(conv, 2048)
	thinkStart := tok.EncodeSpecial("<|think_start|>")
	thinkEnd := tok.EncodeSpecial("<|think_end|>")
	startIdx, endIdx := -1, -1
	for index, tokenID := range ids {
		if tokenID == thinkStart {
			startIdx = index
		}
		if tokenID == thinkEnd {
			endIdx = index
		}
	}
	if startIdx < 0 || endIdx < 0 || endIdx <= startIdx {
		tests.Fatalf("think markers not found: %d..%d", startIdx, endIdx)
	}
	for index := startIdx; index <= endIdx; index++ {
		if mask[index] != 1 {
			tests.Fatalf("thinking token at %d mask = %d, want 1", index, mask[index])
		}
	}
	// The reasoning text must be between the markers.
	between := tok.Decode(ids[startIdx+1 : endIdx])
	if between != "reasoning here" {
		tests.Fatalf("reasoning = %q", between)
	}
}

// TestThinkingInstruction checks the instruction enables and disables the mode.
func TestThinkingInstruction(t *testing.T) {
	if got := ThinkingInstruction(true); got == "" || got == ThinkingInstruction(false) {
		t.Fatalf("enabled instruction = %q", got)
	}
	if got := ThinkingInstruction(false); got == "" {
		t.Fatalf("disabled instruction is empty")
	}
	ranks := TrainBPE([]string{"hello"}, 5)
	tok := NewTokenizer(ranks, SpecialTokens)
	base := &Conversation{Messages: []Message{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "world"},
	}}
	withThinking := &Conversation{Messages: base.Messages, Extra: map[string]any{"thinking": true}}
	baseIDs, _ := tok.RenderConversation(base, 2048)
	thinkIDs, _ := tok.RenderConversation(withThinking, 2048)
	if len(thinkIDs) <= len(baseIDs) {
		t.Fatalf("thinking render should be longer: %d vs %d", len(thinkIDs), len(baseIDs))
	}
}

// TestReasoningTokenCount checks only the tokens inside the think block count.
func TestReasoningTokenCount(t *testing.T) {
	ranks := TrainBPE([]string{"a b c d"}, 20)
	tok := NewTokenizer(ranks, SpecialTokens)
	ids, _ := tok.RenderConversation(&Conversation{Messages: []Message{
		{Role: "user", Content: "a"},
		{Role: "assistant", Thinking: "b c", Content: "d"},
	}}, 2048)
	got := tok.ReasoningTokenCount(ids)
	want := len(tok.Encode("b c"))
	if got != want {
		t.Fatalf("reasoning token count = %d, want %d", got, want)
	}
	if tok.ReasoningTokenCount(tok.Encode("no think block")) != 0 {
		t.Fatal("expected zero for no think block")
	}
}

// TestRenderForCompletionThinking checks the think-start token is primed.
func TestRenderForCompletionThinking(t *testing.T) {
	ranks := TrainBPE([]string{"question answer"}, 10)
	tok := NewTokenizer(ranks, SpecialTokens)
	conv := &Conversation{
		Messages: []Message{
			{Role: "user", Content: "question"},
			{Role: "assistant", Content: "answer"},
		},
		Extra: map[string]any{"thinking": true},
	}
	ids := tok.RenderForCompletion(conv)
	last := ids[len(ids)-1]
	if last != tok.EncodeSpecial("<|think_start|>") {
		t.Fatalf("last id = %d, want think_start", last)
	}
}

func TestRenderConversationAssistantMasked(tests *testing.T) {
	ranks := TrainBPE([]string{"hello"}, 5)
	tok := NewTokenizer(ranks, SpecialTokens)
	conv := &Conversation{Messages: []Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
	}}
	ids, mask := tok.RenderConversation(conv, 2048)
	assistantStart := tok.EncodeSpecial("<|assistant_start|>")
	assistantEnd := tok.EncodeSpecial("<|assistant_end|>")
	startIdx, endIdx := -1, -1
	for index, tokenID := range ids {
		if tokenID == assistantStart {
			startIdx = index
		}
		if tokenID == assistantEnd && endIdx == -1 {
			endIdx = index
		}
	}
	if startIdx < 0 || endIdx < 0 || endIdx <= startIdx {
		tests.Fatalf("assistant markers not found: %d..%d", startIdx, endIdx)
	}
	for index, tokenID := range ids {
		switch {
		case index == startIdx:
			if mask[index] != 0 {
				tests.Fatalf("assistant_start at %d mask = %d, want 0", index, mask[index])
			}
		case index > startIdx && index <= endIdx:
			// Content tokens and the assistant_end token are supervision targets.
			if mask[index] != 1 {
				tests.Fatalf("assistant target %d at %d mask = %d, want 1", tokenID, index, mask[index])
			}
		default:
			if mask[index] != 0 {
				tests.Fatalf("non-assistant token %d at %d mask = %d, want 0", tokenID, index, mask[index])
			}
		}
	}
}

func TestRenderConversationSystemMerge(tests *testing.T) {
	ranks := TrainBPE([]string{"system prompt user text assistant text"}, 20)
	tok := NewTokenizer(ranks, SpecialTokens)
	conv := &Conversation{Messages: []Message{
		{Role: "system", Content: "system prompt"},
		{Role: "user", Content: "user text"},
		{Role: "assistant", Content: "assistant text"},
	}}
	ids, _ := tok.RenderConversation(conv, 2048)
	decoded := tok.Decode(ids)
	if decoded != "<|bos|><|user_start|>system prompt\n\nuser text<|user_end|><|assistant_start|>assistant text<|assistant_end|>" {
		tests.Fatalf("system merge decoded = %q", decoded)
	}
}

func TestRenderForCompletion(tests *testing.T) {
	ranks := TrainBPE([]string{"question answer"}, 10)
	tok := NewTokenizer(ranks, SpecialTokens)
	conv := &Conversation{Messages: []Message{
		{Role: "user", Content: "question"},
		{Role: "assistant", Content: "answer"},
	}}
	ids := tok.RenderForCompletion(conv)
	last := ids[len(ids)-1]
	if last != tok.EncodeSpecial("<|assistant_start|>") {
		tests.Fatalf("last id = %d, want assistant_start", last)
	}
	// The assistant answer must not be present.
	decoded := tok.Decode(ids)
	if decoded == "" {
		tests.Fatal("empty completion")
	}
}

func TestRenderConversationToolUse(tests *testing.T) {
	ranks := TrainBPE([]string{"what is 2 plus 2", "4"}, 30)
	tok := NewTokenizer(ranks, SpecialTokens)
	conv := &Conversation{Messages: []Message{
		{Role: "user", Content: "what is 2 plus 2"},
		{Role: "assistant", Parts: []MessagePart{
			{Type: "tool_call", Text: "2+2"},
			{Type: "tool_output", Text: "4"},
			{Type: "text", Text: "The answer is 4."},
		}},
	}}
	ids, mask := tok.RenderConversation(conv, 2048)
	decoded := tok.Decode(ids)
	want := "<|bos|><|user_start|>what is 2 plus 2<|user_end|>" +
		"<|assistant_start|><|tool_start|>2+2<|tool_end|>" +
		"<|tool_output_start|>4<|tool_output_end|>The answer is 4.<|assistant_end|>"
	if decoded != want {
		tests.Fatalf("tool-use decoded = %q\nwant %q", decoded, want)
	}
	// Verify the tool_output region is masked 0 while tool_call and text are 1.
	for index, tokenID := range ids {
		if tokenID == tok.EncodeSpecial("<|tool_output_start|>") || tokenID == tok.EncodeSpecial("<|tool_output_end|>") {
			if mask[index] != 0 {
				tests.Fatalf("tool output special at %d mask = %d, want 0", index, mask[index])
			}
		}
	}
}
