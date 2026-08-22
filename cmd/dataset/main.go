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
	"github.com/cookiengineer/gonano/logging"
)

func main() {
	repo := flag.String("repo", "HuggingFaceFW/fineweb-edu", "HuggingFace dataset repo")
	config := flag.String("config", "default", "dataset config")
	split := flag.String("split", "train", "dataset split")
	num := flag.Int("num", 10, "number of shards to download")
	workers := flag.Int("workers", 4, "parallel download workers")
	out := flag.String("out", "", "output directory (default ~/.cache/gonano/base_data)")
	flag.Parse()

	logger := logging.Default(slog.LevelInfo)
	if *out == "" {
		*out = filepath.Join(data.BaseDir(), "base_data")
	}
	os.MkdirAll(*out, 0o755)

	shards, err := data.ListHFParquetShards(*repo, *config, *split)
	if err != nil {
		logger.Error("list shards", "err", err)
		os.Exit(1)
	}
	if len(shards) == 0 {
		logger.Error("no shards found for dataset", "repo", *repo)
		os.Exit(1)
	}
	n := *num
	if n < 0 || n > len(shards) {
		n = len(shards)
	}
	logger.Info("downloading shards", "total", n, "repo", *repo)

	results := make(chan error, n)
	sem := make(chan struct{}, *workers)
	for i := 0; i < n; i++ {
		sem <- struct{}{}
		go func(shardURL string, idx int) {
			defer func() { <-sem }()
			name := fmt.Sprintf("shard_%05d.parquet", idx)
			results <- data.DownloadFile(shardURL, filepath.Join(*out, name))
		}(shards[i], i)
	}
	for i := 0; i < n; i++ {
		if err := <-results; err != nil {
			logger.Error("download failed", "err", err)
			os.Exit(1)
		}
	}
	logger.Info("download complete", "dir", *out)
}
