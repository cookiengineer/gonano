// Command base_eval evaluates a pretrained model: bits-per-byte and samples.
package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/cookiengineer/gonano/checkpoint"
	"github.com/cookiengineer/gonano/data"
	"github.com/cookiengineer/gonano/eval"
	"github.com/cookiengineer/gonano/infer"
	"github.com/cookiengineer/gonano/logging"
	"github.com/cookiengineer/gonano/tensor"
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
	tok, err := tokenizer.LoadTokenizer(filepath.Join(*baseDir, "tokenizer", "tokenizer.json"))
	if err != nil {
		logger.Error("load tokenizer", "err", err)
		os.Exit(1)
	}
	meta, params, err := checkpoint.Load(*modelPath)
	if err != nil {
		logger.Error("load checkpoint", "err", err)
		os.Exit(1)
	}
	m := checkpoint.LoadModel(meta, params)

	// Samples.
	engine := infer.NewEngine(m, tok)
	prompts := []string{"The capital of France is", "The chemical symbol of gold is"}
	for _, p := range prompts {
		ids := append([]int{tok.BOSTokenID()}, tok.Encode(p)...)
		results, _ := engine.GenerateBatch(ids, 1, 8, 0, 0, 42)
		fmt.Printf("%s%s\n", p, tok.Decode(results[0][len(ids):]))
	}

	// BPB (requires data).
	if *dataDir != "" {
		paths := data.ListParquetFiles(*dataDir)
		if len(paths) == 0 {
			logger.Error("no parquet files", "dir", *dataDir)
			os.Exit(1)
		}
		tokenBytes := make([]int32, tok.VocabSize())
		for i := 0; i < tok.VocabSize(); i++ {
			if tok.IsSpecial(i) {
				tokenBytes[i] = 0
			} else {
				tokenBytes[i] = int32(len(tok.DecodeSingleTokenBytes(i)))
			}
		}
		src := data.NewParquetSource(paths, 128)
		loader := data.NewPretrainLoader(tok, *batchSize, meta.ModelConfig.SequenceLen, func() ([]string, data.State) {
			return src.Next()
		}, 1000)
		batches := func() (*tensor.Int32s, *tensor.Int32s) {
			x, y, _ := loader.Next()
			return x, y
		}
		bpb := eval.BitsPerByte(m, batches, *steps, tokenBytes)
		fmt.Printf("bits per byte: %.6f\n", bpb)
	}
}
