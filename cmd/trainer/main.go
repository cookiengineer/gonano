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

	"github.com/cookiengineer/gonano/data"
	"github.com/cookiengineer/gonano/internal/logging"
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/model/checkpoint"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
	"github.com/cookiengineer/gonano/trainer"
)

func main() {
	var (
		dataDir           string
		format            string
		tokenizerPath     string
		trainTok          bool
		vocabSize         int
		maxChars          int
		depth             int
		maxSeqLen         int
		kvHeadRatio       int
		compressionRatio  int
		sparseTopK        int
		indexerDim        int
		indexerPool       int
		indexerCandidates int
		reusePattern      string
		swaWindow         int
		ced               bool
		headWiseMuon      bool
		queryCompression  int
		kvLatentDim       int
		numIterations     int
		batchSize         int
		modelTag          string
		baseDir           string
	)
	flag.StringVar(&dataDir, "data-dir", "", "directory of training data (.parquet or .md) (required)")
	flag.StringVar(&format, "format", "parquet", "data format: parquet|markdown")
	flag.StringVar(&tokenizerPath, "tokenizer", "", "path to a tokenizer.json (overrides auto-setup)")
	flag.BoolVar(&trainTok, "train-tokenizer", false, "train a new tokenizer on the data (ignored if --tokenizer is set)")
	flag.IntVar(&vocabSize, "vocab-size", 32768, "vocabulary size when training a tokenizer")
	flag.IntVar(&maxChars, "max-chars", 2000000000, "max characters for tokenizer training")
	flag.IntVar(&depth, "depth", 12, "transformer depth (the complexity dial)")
	flag.IntVar(&maxSeqLen, "max-seq-len", 512, "context length")
	flag.IntVar(&kvHeadRatio, "kv-head-ratio", 1, "query heads per key/value head (1 = MHA, >1 = GQA)")
	flag.IntVar(&compressionRatio, "compression-ratio", 0, "HCA-style dense KV compression ratio (0/1 disables)")
	flag.IntVar(&sparseTopK, "sparse-topk", 0, "CSA sparse attention top-k compressed blocks (0 disables)")
	flag.IntVar(&indexerDim, "indexer-dim", 64, "lightning indexer per-head dimension")
	flag.IntVar(&indexerPool, "indexer-pool", 0, "hierarchical indexer super-block size (0 disables)")
	flag.IntVar(&indexerCandidates, "indexer-candidates", 0, "hierarchical indexer candidate budget (0 = auto)")
	flag.StringVar(&reusePattern, "reuse-pattern", "", "cross-layer reuse pattern of F/R/U per layer (empty = all full)")
	flag.IntVar(&swaWindow, "swa-window", 0, "local sliding-window branch width on compressed layers (0 disables)")
	flag.BoolVar(&ced, "ced", false, "causal encoder-decoder split (decoder global KV from the encoder hidden state; requires --compression-ratio and --swa-window)")
	flag.BoolVar(&headWiseMuon, "head-wise-muon", false, "split query/key projection weights by attention head for the Muon update")
	flag.IntVar(&queryCompression, "query-compression-dim", 0, "low-rank query bottleneck width (0 = full-rank)")
	flag.IntVar(&kvLatentDim, "kv-latent-dim", 0, "shared low-rank KV latent width (0 = full-rank)")
	flag.IntVar(&numIterations, "num-iterations", 50, "optimization steps")
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
	tokenizer := setupTokenizer(logger, baseDir, dataDir, format, tokenizerPath, trainTok, vocabSize, maxChars)

	// 2) Model.
	configuration := model.ConfigForDepthRatio(depth, tokenizer.VocabSize(), 64, 128, maxSeqLen, "SSSL", kvHeadRatio)
	configuration.CompressionRatio = compressionRatio
	configuration.SparseTopK = sparseTopK
	configuration.IndexerDim = indexerDim
	configuration.IndexerPool = indexerPool
	configuration.IndexerCandidates = indexerCandidates
	configuration.ReusePattern = reusePattern
	configuration.SWAWindow = swaWindow
	configuration.CED = ced
	configuration.HeadWiseMuon = headWiseMuon
	configuration.QueryCompressionDim = queryCompression
	configuration.KVLatentDim = kvLatentDim
	model := model.NewTransformer(configuration)
	model.InitWeights(tensors.NewRNG(42))
	groups := model.SetupOptimizer(0.01, 0.1, 0.02, 0.28, 0.5)

	// 3) Data source.
	source, err := newDocProvider(dataDir, format)
	if err != nil {
		logger.Error("data", "err", err)
		os.Exit(1)
	}

	loader := data.NewPretrainLoader(tokenizer, batchSize, maxSeqLen, source, 1000)
	gradAccum := 1
	trainer := trainer.NewTrainer(model, groups, gradAccum)

	outputDir := filepath.Join(baseDir, "base_checkpoints", tagOrDepth(modelTag, depth))
	os.MkdirAll(outputDir, 0o755)

	logger.Info("training", "depth", depth, "dim", configuration.EmbedDim, "vocab", tokenizer.VocabSize(),
		"format", format, "params", model.TotalParams(), "steps", numIterations)
	for step := 0; step < numIterations; step++ {
		inputs, targets, _ := loader.Next()
		loss := trainer.TrainStep(inputs, targets)
		trainer.StepOptimizer(step, numIterations)
		if step%10 == 0 || step == numIterations-1 {
			logger.Info("step", "step", step, "loss", fmt.Sprintf("%.4f", loss))
		}
	}

	// 4) Save.
	meta := checkpoint.Meta{Step: numIterations, ModelConfig: configuration}
	if err := checkpoint.Save(checkpoint.ModelPath(outputDir, numIterations), meta, model.NamedParameters()); err != nil {
		logger.Error("save", "err", err)
		os.Exit(1)
	}
	logger.Info("saved checkpoint", "path", checkpoint.ModelPath(outputDir, numIterations))
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
		tokenizer, err := tokenizer.LoadTokenizer(tokenizerPath)
		if err != nil {
			logger.Error("load --tokenizer", "err", err)
			os.Exit(1)
		}
		return tokenizer
	}

	if trainTok {
		logger.Info("training tokenizer on data", "format", format, "vocab", vocabSize)
		provider, err := newDocProvider(dataDir, format)
		if err != nil {
			logger.Error("data", "err", err)
			os.Exit(1)
		}
		ranks := data.TrainTokenizer(provider, vocabSize, maxChars)
		tokenizer := tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
		if err := tokenizer.Save(defaultPath); err != nil {
			logger.Error("save tokenizer", "err", err)
			os.Exit(1)
		}
		logger.Info("saved tokenizer", "path", defaultPath, "vocab", tokenizer.VocabSize())
		return tokenizer
	}

	if tokenizer, err := tokenizer.LoadTokenizer(defaultPath); err == nil {
		return tokenizer
	}

	// Fallback: a byte-level tokenizer.
	logger.Warn("no tokenizer found; using a byte-level default", "path", defaultPath)
	tokenizer := byteTokenizer()
	if err := tokenizer.Save(defaultPath); err != nil {
		logger.Warn("could not save default tokenizer", "err", err)
	}
	return tokenizer
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
	for index := 0; index < 256; index++ {
		ranks[string([]byte{byte(index)})] = index
	}
	return tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
}
