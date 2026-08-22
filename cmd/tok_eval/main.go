// Command tok_eval evaluates a tokenizer's compression rate.
package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/cookiengineer/gonano/data"
	"github.com/cookiengineer/gonano/logging"
	"github.com/cookiengineer/gonano/tokenizer"
)

func main() {
	baseDir := flag.String("base-dir", "", "tokenizer directory (default ~/.cache/gonano)")
	flag.Parse()

	logger := logging.Default(slog.LevelInfo)
	if *baseDir == "" {
		*baseDir = data.BaseDir()
	}
	tok, err := tokenizer.LoadTokenizer(filepath.Join(*baseDir, "tokenizer", "tokenizer.json"))
	if err != nil {
		logger.Error("load tokenizer", "err", err)
		os.Exit(1)
	}

	texts := []string{
		"Hello world! This is a test.",
		"The quick brown fox jumps over the lazy dog.",
		"Machine learning is the study of algorithms that improve with experience.",
	}
	var totalChars, totalTokens int
	for _, text := range texts {
		ids := tok.Encode(text)
		totalChars += len(text)
		totalTokens += len(ids)
		back := tok.Decode(ids)
		if back != text {
			logger.Error("round-trip mismatch", "text", text, "decoded", back)
			os.Exit(1)
		}
	}
	fmt.Printf("vocab size: %d\n", tok.VocabSize())
	fmt.Printf("characters: %d\n", totalChars)
	fmt.Printf("tokens: %d\n", totalTokens)
	fmt.Printf("compression ratio: %.2f chars/token\n", float64(totalChars)/float64(totalTokens))
}
