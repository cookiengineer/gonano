// Command base_eval evaluates a pretrained model: bits-per-byte and samples.
package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/cookiengineer/gonano/data"
	"github.com/cookiengineer/gonano/evaluator"
	"github.com/cookiengineer/gonano/inference"
	"github.com/cookiengineer/gonano/internal/logging"
	"github.com/cookiengineer/gonano/model/checkpoint"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
)

func main() {
	modelPath := flag.String("model", "", "path to a .gn checkpoint (required)")
	dataDir := flag.String("data-dir", "", "directory of .parquet text shards (for BPB)")
	batchSize := flag.Int("batch-size", 1, "batch size for BPB")
	steps := flag.Int("steps", 8, "BPB evaluation batches")
	baseDir := flag.String("base-dir", "", "tokenizer directory")
	flag.Parse()

	logger := logging.Default(slog.LevelInfo)
	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: base_eval --model <path>")
		os.Exit(1)
	}
	if *baseDir == "" {
		*baseDir = data.BaseDir()
	}
	tokenizer, err := tokenizer.LoadTokenizer(filepath.Join(*baseDir, "tokenizer", "tokenizer.json"))
	if err != nil {
		logger.Error("load tokenizer", "err", err)
		os.Exit(1)
	}
	meta, params, err := checkpoint.Load(*modelPath)
	if err != nil {
		logger.Error("load checkpoint", "err", err)
		os.Exit(1)
	}
	model := checkpoint.LoadModel(meta, params)

	// Samples.
	engine := inference.NewEngine(model, tokenizer)
	prompts := []string{"The capital of France is", "The chemical symbol of gold is"}
	for _, prompt := range prompts {
		ids := append([]int{tokenizer.BOSTokenID()}, tokenizer.Encode(prompt)...)
		results, _ := engine.GenerateBatch(ids, 1, 8, 0, 0, 42)
		fmt.Printf("%s%s\n", prompt, tokenizer.Decode(results[0][len(ids):]))
	}

	// BPB (requires data).
	if *dataDir != "" {
		paths := data.ListParquetFiles(*dataDir)
		if len(paths) == 0 {
			logger.Error("no parquet files", "dir", *dataDir)
			os.Exit(1)
		}
		tokenBytes := make([]int32, tokenizer.VocabSize())
		for index := 0; index < tokenizer.VocabSize(); index++ {
			if tokenizer.IsSpecial(index) {
				tokenBytes[index] = 0
			} else {
				tokenBytes[index] = int32(len(tokenizer.DecodeSingleTokenBytes(index)))
			}
		}
		src := data.NewParquetSource(paths, 128)
		loader := data.NewPretrainLoader(tokenizer, *batchSize, meta.ModelConfig.SequenceLen, func() ([]string, data.State) {
			return src.Next()
		}, 1000)
		batches := func() (*tensors.Int32s, *tensors.Int32s) {
			inputs, targets, _ := loader.Next()
			return inputs, targets
		}
		bitsPerByte := evaluator.BitsPerByte(model, batches, *steps, tokenBytes)
		fmt.Printf("bits per byte: %.6f\n", bitsPerByte)
	}
}
