package data

import (
	"os"
	"path/filepath"
	"testing"
)

func TestOpenConversationsReasoningShape(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "reasoning.jsonl")
	content := `{"instruction":"what is 2+2?","thinking":"I think it is four.","response":"4","intent":"The user wants the sum of two and two.","language":"EN"}
{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello","reasoning_content":"The user greets me."}]}
`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	source, err := OpenConversations(path, 10)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()

	batch, ok := source.Next()
	if !ok {
		t.Fatal("no conversations")
	}
	if len(batch) != 2 {
		t.Fatalf("batch = %d, want 2", len(batch))
	}

	first := batch[0]
	if first.Messages[1].Thinking != "The user wants the sum of two and two.\n\nI think it is four." {
		t.Fatalf("thinking = %q", first.Messages[1].Thinking)
	}
	if enabled, _ := first.Extra["thinking"].(bool); !enabled {
		t.Fatalf("expected thinking Extra, got %v", first.Extra)
	}

	second := batch[1]
	if second.Messages[1].Thinking != "The user greets me." {
		t.Fatalf("reasoning_content = %q", second.Messages[1].Thinking)
	}

	tok := newSimpleTokenizer()
	ids, mask := tok.RenderConversation(first, 1<<20)
	start := tok.EncodeSpecial("<|think_start|>")
	end := tok.EncodeSpecial("<|think_end|>")
	sawStart, sawEnd := false, false
	for index, id := range ids {
		if id == start {
			sawStart = true
			if mask[index] != 1 {
				t.Fatalf("think_start mask = %d, want 1", mask[index])
			}
		}
		if id == end {
			sawEnd = true
		}
	}
	if !sawStart || !sawEnd {
		t.Fatalf("rendered conversation missing think block (start=%v end=%v)", sawStart, sawEnd)
	}
}

func TestParseConversationLineSkipsGarbage(t *testing.T) {
	if _, ok := parseConversationLine([]byte("not json")); ok {
		t.Fatal("expected garbage line to be skipped")
	}
	if _, ok := parseConversationLine([]byte(`{"response":"no instruction"}`)); ok {
		t.Fatal("expected instruction-less record to be skipped")
	}
}
