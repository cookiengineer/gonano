// Command base_train pretrains a nanochat transformer. It derives all
// hyperparameters from a single dial (--depth) using the scaling laws in the
// trainer package.
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
	depth := flag.Int("depth", 12, "transformer depth (complexity dial)")
	maxSeqLen := flag.Int("max-seq-len", 512, "context length")
	kvHeadRatio := flag.Int("kv-head-ratio", 1, "query heads per key/value head (1 = MHA, >1 = GQA)")
	compressionRatio := flag.Int("compression-ratio", 0, "HCA-style dense KV compression ratio (0/1 disables)")
	sparseTopK := flag.Int("sparse-topk", 0, "CSA sparse attention top-k compressed blocks (0 disables)")
	indexerDim := flag.Int("indexer-dim", 64, "lightning indexer per-head dimension")
	indexerPool := flag.Int("indexer-pool", 0, "hierarchical indexer super-block size (0 disables)")
	indexerCandidates := flag.Int("indexer-candidates", 0, "hierarchical indexer candidate budget (0 = auto)")
	reusePattern := flag.String("reuse-pattern", "", "cross-layer reuse pattern of F/R/U per layer (empty = all full)")
	swaWindow := flag.Int("swa-window", 0, "local sliding-window branch width on compressed layers (0 disables)")
	ced := flag.Bool("ced", false, "causal encoder-decoder split (decoder global KV from the encoder hidden state; requires --compression-ratio and --swa-window)")
	headWiseMuon := flag.Bool("head-wise-muon", false, "split query/key projection weights by attention head for the Muon update")
	sinkhornEmbeddings := flag.Bool("sinkhorn-embeddings", false, "Sinkhorn-balanced momentum update for the embedding table, lm_head, and value embeddings")
	queryCompressionDim := flag.Int("query-compression-dim", 0, "low-rank query bottleneck width (0 = full-rank)")
	kvLatentDim := flag.Int("kv-latent-dim", 0, "shared low-rank KV latent width (0 = full-rank)")
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
	tokenizer, err := tokenizer.LoadTokenizer(filepath.Join(*baseDir, "tokenizer", "tokenizer.json"))
	if err != nil {
		logger.Warn("no trained tokenizer, using byte-level tokenizer", "err", err)
		tokenizer = byteTokenizer()
		*vocabSize = tokenizer.VocabSize()
		// Persist the tokenizer so downstream commands can load it.
		tokDir := filepath.Join(*baseDir, "tokenizer")
		os.MkdirAll(tokDir, 0o755)
		if err := tokenizer.Save(filepath.Join(tokDir, "tokenizer.json")); err != nil {
			logger.Warn("could not save tokenizer", "err", err)
		}
	}

	configuration := model.ConfigForDepthRatio(*depth, tokenizer.VocabSize(), 64, 128, *maxSeqLen, "SSSL", *kvHeadRatio)
	configuration.CompressionRatio = *compressionRatio
	configuration.SparseTopK = *sparseTopK
	configuration.IndexerDim = *indexerDim
	configuration.IndexerPool = *indexerPool
	configuration.IndexerCandidates = *indexerCandidates
	configuration.ReusePattern = *reusePattern
	configuration.SWAWindow = *swaWindow
	configuration.CED = *ced
	configuration.HeadWiseMuon = *headWiseMuon
	configuration.QueryCompressionDim = *queryCompressionDim
	configuration.KVLatentDim = *kvLatentDim
	model := model.NewTransformer(configuration)
	model.InitWeights(tensors.NewRNG(42))

	// Optimizer groups and training hyperparameters.
	groups := model.SetupOptimizer(0.01, 0.1, 0.02, 0.28, 0.5, *sinkhornEmbeddings)

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

	loader := data.NewPretrainLoader(tokenizer, *deviceBatchSize, *maxSeqLen, provider, 1000)

	gradAccum := *totalBatchSize / (*deviceBatchSize * *maxSeqLen)
	if gradAccum < 1 {
		gradAccum = 1
	}
	trainer := trainer.NewTrainer(model, groups, gradAccum)

	// Training loop.
	outputDir := filepath.Join(*baseDir, "base_checkpoints", modelTagOrDepth(*modelTag, *depth))
	os.MkdirAll(outputDir, 0o755)

	logger.Info("training", "depth", *depth, "dim", configuration.EmbedDim, "params", model.TotalParams(), "steps", *numIterations)
	for step := 0; step < *numIterations; step++ {
		inputs, targets, _ := loader.Next()
		loss := trainer.TrainStep(inputs, targets)
		trainer.StepOptimizer(step, *numIterations)
		if step%10 == 0 || step == *numIterations-1 {
			logger.Info("step", "step", step, "loss", fmt.Sprintf("%.4f", loss))
		}
	}

	// Save the final checkpoint.
	meta := checkpoint.Meta{Step: *numIterations, ModelConfig: configuration}
	if err := checkpoint.Save(checkpoint.ModelPath(outputDir, *numIterations), meta, model.NamedParameters()); err != nil {
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
	for index := 0; index < 256; index++ {
		ranks[string([]byte{byte(index)})] = index
	}
	return tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
}

func syntheticProvider() data.DocProvider {
	docs := []string{
		"the quick brown fox jumps over the lazy dog",
		"machine learning is the study of algorithms that improve with experience",
		"the capital of France is Paris",
	}
	index := 0
	return func() ([]string, data.State) {
		document := docs[index%len(docs)]
		index++
		return []string{document}, data.State{PQIndex: 0, RGIndex: 0, Epoch: 1}
	}
}
