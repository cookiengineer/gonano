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
	"github.com/cookiengineer/gonano/internal/logging"
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
	totalChars := 0
outer:
	for _, shardPath := range paths {
		reader, err := parquet.Open(shardPath)
		if err != nil {
			logger.Error("open parquet", "path", shardPath, "err", err)
			os.Exit(1)
		}
		for rowGroupIndex := 0; rowGroupIndex < reader.NumRowGroups(); rowGroupIndex++ {
			docs, err := reader.ReadColumnStrings(rowGroupIndex, "text")
			if err != nil {
				logger.Error("read parquet", "err", err)
				os.Exit(1)
			}
			for _, document := range docs {
				for _, piece := range tokenizer.SplitPieces(document) {
					pieces = append(pieces, piece)
					totalChars += len(piece)
					if totalChars >= *maxChars {
						break outer
					}
				}
			}
		}
		reader.Close()
	}

	numMerges := *vocabSize - 256 - len(tokenizer.SpecialTokens)
	logger.Info("training tokenizer", "pieces", len(pieces), "chars", totalChars, "merges", numMerges)
	ranks := tokenizer.TrainBPE(pieces, numMerges)
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
