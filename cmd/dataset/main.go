// Command dataset downloads Parquet text shards from a HuggingFace dataset
// (e.g. HuggingFaceFW/fineweb-edu) into a local directory, for use with
// base_train --data-dir.
package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/cookiengineer/gonano/data"
	"github.com/cookiengineer/gonano/internal/logging"
)

func main() {
	repo := flag.String("repo", "HuggingFaceFW/fineweb-edu", "HuggingFace dataset repo")
	config := flag.String("config", "default", "dataset config")
	split := flag.String("split", "train", "dataset split")
	num := flag.Int("num", 10, "number of shards to download")
	workers := flag.Int("workers", 4, "parallel download workers")
	outputDir := flag.String("out", "", "output directory (default ~/.cache/gonano/base_data)")
	flag.Parse()

	logger := logging.Default(slog.LevelInfo)
	if *outputDir == "" {
		*outputDir = filepath.Join(data.BaseDir(), "base_data")
	}
	os.MkdirAll(*outputDir, 0o755)

	shards, err := data.ListHFParquetShards(*repo, *config, *split)
	if err != nil {
		logger.Error("list shards", "err", err)
		os.Exit(1)
	}
	if len(shards) == 0 {
		logger.Error("no shards found for dataset", "repo", *repo)
		os.Exit(1)
	}
	shardCount := *num
	if shardCount < 0 || shardCount > len(shards) {
		shardCount = len(shards)
	}
	logger.Info("downloading shards", "total", shardCount, "repo", *repo)

	results := make(chan error, shardCount)
	semaphore := make(chan struct{}, *workers)
	for index := 0; index < shardCount; index++ {
		semaphore <- struct{}{}
		go func(shardURL string, shardIndex int) {
			defer func() { <-semaphore }()
			name := fmt.Sprintf("shard_%05d.parquet", shardIndex)
			results <- data.DownloadFile(shardURL, filepath.Join(*outputDir, name))
		}(shards[index], index)
	}
	for index := 0; index < shardCount; index++ {
		if err := <-results; err != nil {
			logger.Error("download failed", "err", err)
			os.Exit(1)
		}
	}
	logger.Info("download complete", "dir", *outputDir)
}
