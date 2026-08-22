// Command trainer is a conventional, end-to-end training CLI. It accepts a
// data directory (Parquet shards or Markdown files), sets up a tokenizer
// (training one on the data, loading an existing one, or falling back to a
// byte-level default), and trains a model.
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
	var (
		dataDir   string
		format    string
		tokenizerPath string
		trainTok  bool
		vocabSize int
		maxChars  int
		depth     int
		maxSeqLen int
		numIter   int
		batchSize int
		modelTag  string
		baseDir   string
	)
	flag.StringVar(&dataDir, "data-dir", "", "directory of training data (.parquet or .md) (required)")
	flag.StringVar(&format, "format", "parquet", "data format: parquet|markdown")
	flag.StringVar(&tokenizerPath, "tokenizer", "", "path to a tokenizer.json (overrides auto-setup)")
	flag.BoolVar(&trainTok, "train-tokenizer", false, "train a new tokenizer on the data (ignored if --tokenizer is set)")
	flag.IntVar(&vocabSize, "vocab-size", 32768, "vocabulary size when training a tokenizer")
	flag.IntVar(&maxChars, "max-chars", 2000000000, "max characters for tokenizer training")
	flag.IntVar(&depth, "depth", 12, "transformer depth (the complexity dial)")
	flag.IntVar(&maxSeqLen, "max-seq-len", 512, "context length")
	flag.IntVar(&numIter, "num-iterations", 50, "optimization steps")
	flag.IntVar(&batchSize, "device-batch-size", 1, "sequences per step")
	flag.StringVar(&modelTag, "model-tag", "", "checkpoint directory name (default d<depth>)")
	flag.StringVar(&baseDir, "base-dir", "", "checkpoint/tokenizer directory (default ~/.cache/gonano)")
	flag.Parse()

	logger := logging.Default(slog.LevelInfo)
	if dataDir == "" {
		fmt.Fprintln(os.Stderr, "usage: trainer --data-dir <dir> [--format parquet|markdown] [--train-tokenizer] [--depth N] ...")
		os.Exit(1)
	}
	if format != "parquet" && format != "markdown" {
		logger.Error("--format must be 'parquet' or 'markdown'", "got", format)
		os.Exit(1)
	}
	if baseDir == "" {
		baseDir = data.BaseDir()
	}

	// 1) Tokenizer setup.
	tok := setupTokenizer(logger, baseDir, dataDir, format, tokenizerPath, trainTok, vocabSize, maxChars)

	// 2) Model.
	cfg := model.ConfigForDepth(depth, tok.VocabSize(), 64, 128, maxSeqLen, "SSSL")
	m := model.NewTransformer(cfg)
	m.InitWeights(tensor.NewRNG(42))
	groups := m.SetupOptimizer(0.01, 0.1, 0.02, 0.28, 0.5)

	// 3) Data source.
	source, err := newDocProvider(dataDir, format)
	if err != nil {
		logger.Error("data", "err", err)
		os.Exit(1)
	}

	loader := data.NewPretrainLoader(tok, batchSize, maxSeqLen, source, 1000)
	gradAccum := 1
	tr := train.NewTrainer(m, groups, gradAccum)

	outputDir := filepath.Join(baseDir, "base_checkpoints", tagOrDepth(modelTag, depth))
	os.MkdirAll(outputDir, 0o755)

	logger.Info("training", "depth", depth, "dim", cfg.EmbedDim, "vocab", tok.VocabSize(),
		"format", format, "params", m.TotalParams(), "steps", numIter)
	for step := 0; step < numIter; step++ {
		x, y, _ := loader.Next()
		loss := tr.TrainStep(x, y)
		tr.StepOptimizer(step, numIter)
		if step%10 == 0 || step == numIter-1 {
			logger.Info("step", "step", step, "loss", fmt.Sprintf("%.4f", loss))
		}
	}

	// 4) Save.
	meta := checkpoint.Meta{Step: numIter, ModelConfig: cfg}
	if err := checkpoint.Save(checkpoint.ModelPath(outputDir, numIter), meta, m.NamedParameters()); err != nil {
		logger.Error("save", "err", err)
		os.Exit(1)
	}
	logger.Info("saved checkpoint", "path", checkpoint.ModelPath(outputDir, numIter))
}

// setupTokenizer resolves the tokenizer in this order:
//  1. --tokenizer <path> (if given),
//  2. --train-tokenizer (train a fresh one on the data),
//  3. an existing tokenizer.json in the base dir,
//  4. a byte-level default (written to the base dir).
func setupTokenizer(logger *slog.Logger, baseDir, dataDir, format, tokenizerPath string, trainTok bool, vocabSize, maxChars int) *tokenizer.Tokenizer {
	tokDir := filepath.Join(baseDir, "tokenizer")
	os.MkdirAll(tokDir, 0o755)
	defaultPath := filepath.Join(tokDir, "tokenizer.json")

	if tokenizerPath != "" {
		tok, err := tokenizer.LoadTokenizer(tokenizerPath)
		if err != nil {
			logger.Error("load --tokenizer", "err", err)
			os.Exit(1)
		}
		return tok
	}

	if trainTok {
		logger.Info("training tokenizer on data", "format", format, "vocab", vocabSize)
		provider, err := newDocProvider(dataDir, format)
		if err != nil {
			logger.Error("data", "err", err)
			os.Exit(1)
		}
		ranks := data.TrainTokenizer(provider, vocabSize, maxChars)
		tok := tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
		if err := tok.Save(defaultPath); err != nil {
			logger.Error("save tokenizer", "err", err)
			os.Exit(1)
		}
		logger.Info("saved tokenizer", "path", defaultPath, "vocab", tok.VocabSize())
		return tok
	}

	if tok, err := tokenizer.LoadTokenizer(defaultPath); err == nil {
		return tok
	}

	// Fallback: a byte-level tokenizer.
	logger.Warn("no tokenizer found; using a byte-level default", "path", defaultPath)
	tok := byteTokenizer()
	if err := tok.Save(defaultPath); err != nil {
		logger.Warn("could not save default tokenizer", "err", err)
	}
	return tok
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
	default:
		paths := data.ListParquetFiles(dir)
		if len(paths) == 0 {
			return nil, fmt.Errorf("no .parquet files found in %s", dir)
		}
		src := data.NewParquetSource(paths, 128)
		return func() ([]string, data.State) { return src.Next() }, nil
	}
}

func tagOrDepth(tag string, depth int) string {
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
