// Command base_train pretrains a nanochat transformer. It derives all
// hyperparameters from a single dial (--depth) using the scaling laws in the
// train package.
package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/cookiengineer/gonano/checkpoint"
	"github.com/cookiengineer/gonano/data"
	"github.com/cookiengineer/gonano/logging"
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensor"
	"github.com/cookiengineer/gonano/tokenizer"
	"github.com/cookiengineer/gonano/train"
)

func main() {
	depth := flag.Int("depth", 12, "transformer depth (complexity dial)")
	maxSeqLen := flag.Int("max-seq-len", 512, "context length")
	vocabSize := flag.Int("vocab-size", 32768, "vocabulary size")
	numIterations := flag.Int("num-iterations", 50, "optimization steps")
	deviceBatchSize := flag.Int("device-batch-size", 1, "per-step batch size")
	totalBatchSize := flag.Int("total-batch-size", -1, "total batch tokens (-1 = auto)")
	dataDir := flag.String("data-dir", "", "directory of .parquet text shards (empty = synthetic)")
	dataFormat := flag.String("data-format", "parquet", "data format: parquet|markdown")
	baseDir := flag.String("base-dir", "", "checkpoint/tokenizer directory (default ~/.cache/gonano)")
	modelTag := flag.String("model-tag", "", "checkpoint directory name (default d<depth>)")
	flag.Parse()

	if *baseDir == "" {
		*baseDir = data.BaseDir()
	}
	logger := logging.Default(slog.LevelInfo)

	// Tokenizer: load a trained tokenizer if present, else fall back to a
	// byte-level tokenizer for smoke runs.
	tok, err := tokenizer.LoadTokenizer(filepath.Join(*baseDir, "tokenizer", "tokenizer.json"))
	if err != nil {
		logger.Warn("no trained tokenizer, using byte-level tokenizer", "err", err)
		tok = byteTokenizer()
		*vocabSize = tok.VocabSize()
		// Persist the tokenizer so downstream commands can load it.
		tokDir := filepath.Join(*baseDir, "tokenizer")
		os.MkdirAll(tokDir, 0o755)
		if err := tok.Save(filepath.Join(tokDir, "tokenizer.json")); err != nil {
			logger.Warn("could not save tokenizer", "err", err)
		}
	}

	cfg := model.ConfigForDepth(*depth, tok.VocabSize(), 64, 128, *maxSeqLen, "SSSL")
	m := model.NewTransformer(cfg)
	m.InitWeights(tensor.NewRNG(42))

	// Optimizer groups and training hyperparameters.
	groups := m.SetupOptimizer(0.01, 0.1, 0.02, 0.28, 0.5)

	// Data source.
	var provider data.DocProvider
	if *dataDir != "" {
		switch *dataFormat {
		case "markdown":
			src := data.NewMarkdownSource(*dataDir, 128)
			if src.NumFiles() == 0 {
				logger.Error("no .md files found", "dir", *dataDir)
				os.Exit(1)
			}
			provider = func() ([]string, data.State) { return src.Next() }
		default: // parquet
			paths := data.ListParquetFiles(*dataDir)
			if len(paths) == 0 {
				logger.Error("no parquet files found", "dir", *dataDir)
				os.Exit(1)
			}
			src := data.NewParquetSource(paths, 128)
			provider = func() ([]string, data.State) { return src.Next() }
		}
	} else {
		logger.Warn("no --data-dir; training on synthetic documents")
		provider = syntheticProvider()
	}

	loader := data.NewPretrainLoader(tok, *deviceBatchSize, *maxSeqLen, provider, 1000)

	gradAccum := *totalBatchSize / (*deviceBatchSize * *maxSeqLen)
	if gradAccum < 1 {
		gradAccum = 1
	}
	tr := train.NewTrainer(m, groups, gradAccum)

	// Training loop.
	outputDir := filepath.Join(*baseDir, "base_checkpoints", modelTagOrDepth(*modelTag, *depth))
	os.MkdirAll(outputDir, 0o755)

	logger.Info("training", "depth", *depth, "dim", cfg.EmbedDim, "params", m.TotalParams(), "steps", *numIterations)
	for step := 0; step < *numIterations; step++ {
		x, y, _ := loader.Next()
		loss := tr.TrainStep(x, y)
		tr.StepOptimizer(step, *numIterations)
		if step%10 == 0 || step == *numIterations-1 {
			logger.Info("step", "step", step, "loss", fmt.Sprintf("%.4f", loss))
		}
	}

	// Save the final checkpoint.
	meta := checkpoint.Meta{Step: *numIterations, ModelConfig: cfg}
	if err := checkpoint.Save(checkpoint.ModelPath(outputDir, *numIterations), meta, m.NamedParameters()); err != nil {
		logger.Error("save checkpoint", "err", err)
		os.Exit(1)
	}
	logger.Info("saved checkpoint", "path", checkpoint.ModelPath(outputDir, *numIterations))
}

func modelTagOrDepth(tag string, depth int) string {
	if tag != "" {
		return tag
	}
	return fmt.Sprintf("d%d", depth)
}

func byteTokenizer() *tokenizer.Tokenizer {
	ranks := make(map[string]int, 256)
	for i := 0; i < 256; i++ {
		ranks[string([]byte{byte(i)})] = i
	}
	return tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
}

func syntheticProvider() data.DocProvider {
	docs := []string{
		"the quick brown fox jumps over the lazy dog",
		"machine learning is the study of algorithms that improve with experience",
		"the capital of France is Paris",
	}
	i := 0
	return func() ([]string, data.State) {
		d := docs[i%len(docs)]
		i++
		return []string{d}, data.State{PQIndex: 0, RGIndex: 0, Epoch: 1}
	}
}
