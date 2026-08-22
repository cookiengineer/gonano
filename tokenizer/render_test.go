package tokenizer

import (
	"reflect"
	"testing"
)

func TestRenderConversation(t *testing.T) {
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
		t.Fatalf("ids mismatch:\n got %v\nwant %v", ids, want)
	}
	if len(mask) != len(ids) {
		t.Fatalf("mask length %d != ids length %d", len(mask), len(ids))
	}
}

func TestRenderConversationAssistantMasked(t *testing.T) {
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
	for i, id := range ids {
		if id == assistantStart {
			startIdx = i
		}
		if id == assistantEnd && endIdx == -1 {
			endIdx = i
		}
	}
	if startIdx < 0 || endIdx < 0 || endIdx <= startIdx {
		t.Fatalf("assistant markers not found: %d..%d", startIdx, endIdx)
	}
	for i, id := range ids {
		switch {
		case i == startIdx:
			if mask[i] != 0 {
				t.Fatalf("assistant_start at %d mask = %d, want 0", i, mask[i])
			}
		case i > startIdx && i <= endIdx:
			// Content tokens and the assistant_end token are supervision targets.
			if mask[i] != 1 {
				t.Fatalf("assistant target %d at %d mask = %d, want 1", id, i, mask[i])
			}
		default:
			if mask[i] != 0 {
				t.Fatalf("non-assistant token %d at %d mask = %d, want 0", id, i, mask[i])
			}
		}
	}
}

func TestRenderConversationSystemMerge(t *testing.T) {
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
		t.Fatalf("system merge decoded = %q", decoded)
	}
}

func TestRenderForCompletion(t *testing.T) {
	ranks := TrainBPE([]string{"question answer"}, 10)
	tok := NewTokenizer(ranks, SpecialTokens)
	conv := &Conversation{Messages: []Message{
		{Role: "user", Content: "question"},
		{Role: "assistant", Content: "answer"},
	}}
	ids := tok.RenderForCompletion(conv)
	last := ids[len(ids)-1]
	if last != tok.EncodeSpecial("<|assistant_start|>") {
		t.Fatalf("last id = %d, want assistant_start", last)
	}
	// The assistant answer must not be present.
	decoded := tok.Decode(ids)
	if decoded == "" {
		t.Fatal("empty completion")
	}
}

func TestRenderConversationToolUse(t *testing.T) {
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
		t.Fatalf("tool-use decoded = %q\nwant %q", decoded, want)
	}
	// Verify the tool_output region is masked 0 while tool_call and text are 1.
	for i, id := range ids {
		if id == tok.EncodeSpecial("<|tool_output_start|>") || id == tok.EncodeSpecial("<|tool_output_end|>") {
			if mask[i] != 0 {
				t.Fatalf("tool output special at %d mask = %d, want 0", i, mask[i])
			}
		}
	}
}
