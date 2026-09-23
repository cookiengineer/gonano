// Command dspark_train trains a DSpark drafter (DeepSeek-V4.1 §2.4.3): a small
// transformer trunk distilled from a frozen backbone, plus a Markov head and a
// confidence head trained on top of the frozen trunk. The backbone must be
// uncompressed, because the drafter learns from the backbone's full-vocabulary
// logits.
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
	"github.com/cookiengineer/gonano/optimizer"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
	"github.com/cookiengineer/gonano/trainer"
)

func main() {
	modelPath := flag.String("model", "", "path to the frozen backbone .gn checkpoint (required)")
	dataDir := flag.String("data-dir", "", "directory of training data (.parquet or .md)")
	dataFormat := flag.String("data-format", "parquet", "data format: parquet|markdown")
	numIterations := flag.Int("num-iterations", 200, "trunk distillation steps")
	headSteps := flag.Int("head-steps", 100, "head training steps")
	headLR := flag.Float64("head-lr", 1e-3, "head learning rate")
	draftPositions := flag.Int("draft-positions", model.DefaultDraftPositions(), "semi-autoregressive draft positions trained in parallel")
	deviceBatchSize := flag.Int("device-batch-size", 1, "per-step batch size")
	distillTemperature := flag.Float64("distill-temperature", 1.0, "distillation softmax temperature")
	outPath := flag.String("out", "", "output drafter checkpoint path")
	baseDir := flag.String("base-dir", "", "tokenizer directory (default ~/.cache/gonano)")
	flag.Parse()

	logger := logging.Default(slog.LevelInfo)
	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: dspark_train --model <backbone.gn> --data-dir <dir>")
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
		logger.Error("load backbone", "err", err)
		os.Exit(1)
	}
	backbone := checkpoint.LoadModel(meta, params)
	if backbone.Config.Compression() > 1 {
		logger.Error("dspark requires an uncompressed backbone")
		os.Exit(1)
	}

	dspark := model.NewDSpark(backbone)
	dspark.InitWeights(tensors.NewRNG(42))
	drafter := dspark.Drafter
	logger.Info("drafter", "layers", drafter.Config.NumLayer, "dim", drafter.Config.EmbedDim,
		"context", drafter.Config.SequenceLen, "params", len(dspark.Parameters()))

	// The drafter only ever sees SequenceLen tokens, so train on windows of
	// exactly that length.
	sequenceLen := drafter.Config.SequenceLen
	var provider data.DocProvider
	if *dataDir != "" {
		switch *dataFormat {
		case "markdown":
			source := data.NewMarkdownSource(*dataDir, 128)
			if source.NumFiles() == 0 {
				logger.Error("no .md files found", "dir", *dataDir)
				os.Exit(1)
			}
			provider = func() ([]string, data.State) { return source.Next() }
		default:
			paths := data.ListParquetFiles(*dataDir)
			if len(paths) == 0 {
				logger.Error("no parquet files found", "dir", *dataDir)
				os.Exit(1)
			}
			source := data.NewParquetSource(paths, 128)
			provider = func() ([]string, data.State) { return source.Next() }
		}
	} else {
		logger.Error("--data-dir is required")
		os.Exit(1)
	}

	loader := data.NewPretrainLoader(tokenizer, *deviceBatchSize, sequenceLen, provider, 1000)

	// Phase 1: distill the trunk from the frozen backbone.
	trunkGroups := drafter.SetupOptimizer(0.008, 0.2, 0.02, 0.0, 0.5, true)
	distillNext := func() (*tensors.Int32s, *tensors.Int32s, bool) {
		inputs, _, _ := loader.Next()
		return inputs, nil, true
	}
	trainer.Distill(drafter, backbone, trunkGroups, distillNext, *numIterations, float32(*distillTemperature), func(step int, loss float32) {
		if step%20 == 0 {
			logger.Info("dspark-trunk", "step", step, "loss", fmt.Sprintf("%.4f", loss))
		}
	})

	// Phase 2: jointly train the drafter trunk, the draft-mask embedding, and
	// both heads under the semi-autoregressive placeholder forward
	// (DeepSeek-V4.1 §2.4.3), so the mask embedding is optimized rather than
	// held fixed.
	jointGroups := dspark.SetupTrainingOptimizer(0.008, 0.2, 0.02, 0.0, 0.5, float32(*headLR))
	jointOptimizer := optimizer.NewMuonAdamW(jointGroups)
	for step := 0; step < *headSteps; step++ {
		inputs, targets, _ := loader.Next()
		loss := dspark.TrainStep(inputs, targets, *draftPositions)
		jointOptimizer.Step()
		jointOptimizer.ZeroGrad()
		if step%20 == 0 {
			logger.Info("dspark-joint", "step", step, "loss", fmt.Sprintf("%.4f", loss))
		}
	}

	if *outPath == "" {
		*outPath = filepath.Join(*baseDir, "dspark_checkpoints", fmt.Sprintf("dspark_%06d.gn", *numIterations))
	}
	os.MkdirAll(filepath.Dir(*outPath), 0o755)
	saveMeta := checkpoint.Meta{
		Step:        *numIterations,
		ModelConfig: drafter.Config,
		UserConfig:  map[string]any{"dspark": true},
	}
	if err := checkpoint.Save(*outPath, saveMeta, dspark.NamedParameters()); err != nil {
		logger.Error("save drafter", "err", err)
		os.Exit(1)
	}
	logger.Info("saved drafter", "path", *outPath)
}
