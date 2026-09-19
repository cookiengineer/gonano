package data

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cookiengineer/gonano/tokenizer"
)

func TestTrainTokenizerFromMarkdown(tests *testing.T) {
	dir := tests.TempDir()
	os.WriteFile(filepath.Join(dir, "a.md"), []byte("the quick brown fox jumps over the lazy dog"), 0o644)
	os.WriteFile(filepath.Join(dir, "b.md"), []byte("machine learning is the study of algorithms"), 0o644)

	src := NewMarkdownSource(dir, 128)
	ranks := TrainTokenizer(func() ([]string, State) { return src.Next() }, 512, 1_000_000)

	// vocab = 256 bytes + merges + 9 specials; ranks only holds the mergeable
	// tokens, so it should have 256 + (some merges).
	if len(ranks) < 256 {
		tests.Fatalf("ranks = %d, want >= 256", len(ranks))
	}
	if len(ranks) > 512 {
		tests.Fatalf("ranks = %d, want <= 512", len(ranks))
	}

	tok := tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
	for _, text := range []string{"the quick brown fox", "machine learning"} {
		if got := tok.Decode(tok.Encode(text)); got != text {
			tests.Fatalf("roundtrip %q -> %q", text, got)
		}
	}
}

func TestTrainTokenizerFromParquet(tests *testing.T) {
	path := buildParquetFile(tests, []string{"hello world hello", "goodbye world"})
	src := NewParquetSource([]string{path}, 128)
	ranks := TrainTokenizer(func() ([]string, State) { return src.Next() }, 512, 1_000_000)
	if len(ranks) < 256 {
		tests.Fatalf("ranks = %d, want >= 256", len(ranks))
	}
	tok := tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
	if got := tok.Decode(tok.Encode("hello world")); got != "hello world" {
		tests.Fatalf("roundtrip = %q", got)
	}
}
