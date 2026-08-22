// Command tok_train trains a byte-level BPE tokenizer on Parquet text shards.
package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/cookiengineer/gonano/data"
	"github.com/cookiengineer/gonano/data/parquet"
	"github.com/cookiengineer/gonano/logging"
	"github.com/cookiengineer/gonano/tokenizer"
)

func main() {
	dataDir := flag.String("data-dir", "", "directory of .parquet text shards (required)")
	vocabSize := flag.Int("vocab-size", 32768, "vocabulary size")
	maxChars := flag.Int("max-chars", 2000000, "max characters to train on")
	baseDir := flag.String("base-dir", "", "output directory (default ~/.cache/gonano)")
	flag.Parse()

	logger := logging.Default(slog.LevelInfo)
	if *dataDir == "" {
		fmt.Fprintln(os.Stderr, "usage: tok_train --data-dir <dir>")
		os.Exit(1)
	}
	if *baseDir == "" {
		*baseDir = data.BaseDir()
	}

	paths := data.ListParquetFiles(*dataDir)
	if len(paths) == 0 {
		logger.Error("no parquet files", "dir", *dataDir)
		os.Exit(1)
	}

	// Collect document pieces (regex-split) up to maxChars.
	var pieces []string
	total := 0
outer:
	for _, path := range paths {
		r, err := parquet.Open(path)
		if err != nil {
			logger.Error("open parquet", "path", path, "err", err)
			os.Exit(1)
		}
		for rg := 0; rg < r.NumRowGroups(); rg++ {
			docs, err := r.ReadColumnStrings(rg, "text")
			if err != nil {
				logger.Error("read parquet", "err", err)
				os.Exit(1)
			}
			for _, doc := range docs {
				for _, p := range tokenizer.SplitPieces(doc) {
					pieces = append(pieces, p)
					total += len(p)
					if total >= *maxChars {
						break outer
					}
				}
			}
		}
		r.Close()
	}

	numMerges := *vocabSize - 256 - len(tokenizer.SpecialTokens)
	logger.Info("training tokenizer", "pieces", len(pieces), "chars", total, "merges", numMerges)
	ranks := tokenizer.TrainBPE(pieces, numMerges)
	tok := tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)

	out := filepath.Join(*baseDir, "tokenizer")
	os.MkdirAll(out, 0o755)
	path := filepath.Join(out, "tokenizer.json")
	if err := tok.Save(path); err != nil {
		logger.Error("save tokenizer", "err", err)
		os.Exit(1)
	}
	logger.Info("saved tokenizer", "path", path, "vocab", tok.VocabSize())
}
