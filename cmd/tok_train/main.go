// Command tok_train trains a byte-level BPE tokenizer on training data
// (Parquet shards or Markdown files).
package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/cookiengineer/gonano/data"
	"github.com/cookiengineer/gonano/internal/logging"
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tokenizer"
)

func main() {
	dataDir := flag.String("data-dir", "", "directory of training data (.parquet or .md) (required)")
	dataFormat := flag.String("data-format", "parquet", "data format: parquet|markdown")
	vocabSize := flag.Int("vocab-size", model.DefaultVocabSize, "vocabulary size")
	maxChars := flag.Int("max-chars", 2000000, "max characters to train on")
	baseDir := flag.String("base-dir", "", "output directory (default ~/.cache/gonano)")
	flag.Parse()

	logger := logging.Default(slog.LevelInfo)
	if *dataDir == "" {
		fmt.Fprintln(os.Stderr, "usage: tok_train --data-dir <dir> [--data-format parquet|markdown]")
		os.Exit(1)
	}
	if *baseDir == "" {
		*baseDir = data.BaseDir()
	}

	provider, err := newDocProvider(*dataDir, *dataFormat)
	if err != nil {
		logger.Error("data", "err", err)
		os.Exit(1)
	}

	logger.Info("training tokenizer", "format", *dataFormat, "vocab", *vocabSize, "max_chars", *maxChars)
	ranks := data.TrainTokenizer(provider, *vocabSize, *maxChars)
	tokenizer := tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)

	outputDir := filepath.Join(*baseDir, "tokenizer")
	os.MkdirAll(outputDir, 0o755)
	outputPath := filepath.Join(outputDir, "tokenizer.json")
	if err := tokenizer.Save(outputPath); err != nil {
		logger.Error("save tokenizer", "err", err)
		os.Exit(1)
	}
	logger.Info("saved tokenizer", "path", outputPath, "vocab", tokenizer.VocabSize())
}

// newDocProvider builds a document provider for the given directory + format.
func newDocProvider(dir, format string) (data.DocProvider, error) {
	switch format {
	case "markdown":
		src := data.NewMarkdownSource(dir, 128)
		if src.NumFiles() == 0 {
			return nil, fmt.Errorf("no .md files found in %s", dir)
		}
		return func() ([]string, data.State) { return src.Next() }, nil
	case "parquet":
		paths := data.ListParquetFiles(dir)
		if len(paths) == 0 {
			return nil, fmt.Errorf("no .parquet files found in %s", dir)
		}
		src := data.NewParquetSource(paths, 128)
		return func() ([]string, data.State) { return src.Next() }, nil
	default:
		return nil, fmt.Errorf("unknown data format %q (want parquet|markdown)", format)
	}
}
