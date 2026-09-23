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
	"github.com/cookiengineer/gonano/internal/device"
	"github.com/cookiengineer/gonano/internal/logging"
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/model/checkpoint"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
	"github.com/cookiengineer/gonano/trainer"
)

func main() {
	depth := flag.Int("depth", model.DefaultDepth, "transformer depth (complexity dial)")
	maxSeqLen := flag.Int("max-seq-len", 512, "context length")
	presetName := flag.String("preset", string(model.DefaultPreset), "architecture preset: flash (DeepSeek-V4.1 long-context + MoE), latent (absorbed MLA + MoE), dense (classic)")
	vocabSize := flag.Int("vocab-size", model.DefaultVocabSize, "vocabulary size")
	headDim := flag.Int("head-dim", model.DefaultHeadDim, "attention head dimension")
	numIterations := flag.Int("num-iterations", 50, "optimization steps")
	deviceBatchSize := flag.Int("device-batch-size", 1, "per-step batch size")
	totalBatchSize := flag.Int("total-batch-size", -1, "total batch tokens (-1 = auto)")
	dataDir := flag.String("data-dir", "", "directory of .parquet text shards (empty = synthetic)")
	dataFormat := flag.String("data-format", "parquet", "data format: parquet|markdown")
	baseDir := flag.String("base-dir", "", "checkpoint/tokenizer directory (default ~/.cache/gonano)")
	modelTag := flag.String("model-tag", "", "checkpoint directory name (default d<depth>)")
	sampleMasking := flag.Bool("sample-masking", true, "mask attention across packed documents (DeepSeek-V4.1 §4.2.2)")
	flag.Parse()

	if *baseDir == "" {
		*baseDir = data.BaseDir()
	}
	logger := logging.Default(slog.LevelInfo)

	preset, err := model.ParsePreset(*presetName)
	if err != nil {
		logger.Error("invalid preset", "err", err)
		os.Exit(1)
	}

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

	// The preset selects the whole architecture: compression, sparsity,
	// cross-layer reuse, SWA, CED, MoE, GQA, head-wise Muon, and Sinkhorn.
	configuration := model.ConfigForPreset(preset, *depth, tokenizer.VocabSize(), 64, *headDim, *maxSeqLen, "SSSL")

	// Preflight: refuse to build a training run that cannot fit in RAM. The
	// trainer is fully in-RAM (weights + gradients + optimizer state plus
	// activations), so estimate the requirement and compare it against the
	// kernel's available memory before allocating anything.
	requiredBytes := model.EstimatedTrainingMemoryBytes(configuration, *deviceBatchSize, *maxSeqLen)
	if availableBytes, ok := device.AvailableMemoryBytes(); ok && uint64(requiredBytes) > availableBytes {
		logger.Error("insufficient memory for training",
			"required", humanGiB(requiredBytes),
			"available", humanGiB(int64(availableBytes)),
			"params", model.TotalParamsForConfig(configuration),
			"depth", *depth, "head_dim", *headDim, "vocab", tokenizer.VocabSize(),
			"device_batch_size", *deviceBatchSize, "max_seq_len", *maxSeqLen)
		fmt.Fprintf(os.Stderr,
			"gonano: estimated training memory %s exceeds the available %s.\n"+
				"  The trainer is fully in-RAM: %d bytes/parameter for weights, gradients, and optimizer state, plus activations.\n"+
				"  Lower --depth, --device-batch-size, or --max-seq-len, or run on a host with more RAM.\n"+
				"  (Weights alone are %.1f GiB; checkpoints go to disk, but the training working set cannot.)\n",
			humanGiB(requiredBytes), humanGiB(int64(availableBytes)), model.TrainingBytesPerParameter,
			float64(model.TotalParamsForConfig(configuration))*4/(1<<30))
		os.Exit(1)
	}

	model := model.NewTransformer(configuration)
	model.InitWeights(tensors.NewRNG(42))

	// Optimizer groups and training hyperparameters.
	groups := model.SetupOptimizer(0.01, 0.1, 0.02, 0.28, 0.5, preset.UsesSinkhorn())

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
		var loss float32
		if *sampleMasking {
			inputs, targets, segments, _ := loader.NextSegments()
			loss = trainer.TrainStepSegments(inputs, targets, segments)
		} else {
			inputs, targets, _ := loader.Next()
			loss = trainer.TrainStep(inputs, targets)
		}
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

// humanGiB formats a byte count as GiB for the memory preflight messages.
func humanGiB(bytes int64) string {
	return fmt.Sprintf("%.1f GiB", float64(bytes)/(1<<30))
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
