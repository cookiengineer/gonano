// Command dflash_train trains a DeepSeek-V4.1 DFlash/MTP drafter from a frozen
// backbone. The drafter is conditioned on the backbone's final hidden state and
// keeps the Markov correction and confidence heads used by speculative decoding.
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
)

func main() {
	modelPath := flag.String("model", "", "path to the frozen backbone .gn checkpoint (required)")
	dataDir := flag.String("data-dir", "", "directory of training data (.parquet or .md)")
	dataFormat := flag.String("data-format", "parquet", "data format: parquet|markdown")
	headSteps := flag.Int("head-steps", 200, "joint DFlash training steps")
	headLR := flag.Float64("head-lr", 1e-3, "DFlash head learning rate")
	draftPositions := flag.Int("draft-positions", 5, "draft positions trained in parallel")
	deviceBatchSize := flag.Int("device-batch-size", 1, "per-step batch size")
	outPath := flag.String("out", "", "output drafter checkpoint path")
	baseDir := flag.String("base-dir", "", "tokenizer directory (default ~/.cache/gonano)")
	flag.Parse()

	logger := logging.Default(slog.LevelInfo)
	if *modelPath == "" || *dataDir == "" {
		fmt.Fprintln(os.Stderr, "usage: dflash_train --model <backbone.gn> --data-dir <dir>")
		os.Exit(1)
	}
	if *baseDir == "" {
		*baseDir = data.BaseDir()
	}
	tokenizerImpl, err := tokenizer.LoadTokenizer(filepath.Join(*baseDir, "tokenizer", "tokenizer.json"))
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
	dflash := model.NewDFlashFromConfig(backbone.Config)
	dflash.BlockSize = *draftPositions
	dflash.InitWeights(tensors.NewRNG(42))
	drafter := dflash.Drafter

	logger.Info("dflash", "layers", drafter.Config.NumLayer, "dim", drafter.Config.EmbedDim,
		"context", drafter.Config.SequenceLen, "targets", dflash.TargetIDs, "params", len(dflash.Parameters()))

	sequenceLen := drafter.Config.SequenceLen
	if sequenceLen < 8 {
		sequenceLen = 8
	}
	var provider data.DocProvider
	switch *dataFormat {
	case "markdown":
		source := data.NewMarkdownSource(*dataDir, 128)
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
	loader := data.NewPretrainLoader(tokenizerImpl, *deviceBatchSize, sequenceLen, provider, 1000)

	groups := dflash.SetupTrainingOptimizer(0.008, 0.2, 0.02, 0.0, 0.5, float32(*headLR))
	joint := optimizer.NewMuonAdamW(groups)
	for step := 0; step < *headSteps; step++ {
		inputs, targets, _ := loader.Next()
		_, context := backbone.TrainForward(inputs)
		hidden := model.TargetHiddenFromContext(context, dflash.TargetIDs)
		loss := dflash.TrainStep(inputs, targets, hidden, *draftPositions)
		joint.Step()
		joint.ZeroGrad()
		if step%20 == 0 {
			logger.Info("dflash", "step", step, "loss", fmt.Sprintf("%.4f", loss))
		}
	}

	if *outPath == "" {
		*outPath = filepath.Join(*baseDir, "dflash_checkpoints", fmt.Sprintf("dflash_%06d.gn", *headSteps))
	}
	os.MkdirAll(filepath.Dir(*outPath), 0o755)
	saveMeta := checkpoint.Meta{
		Step:        *headSteps,
		ModelConfig: backbone.Config,
		UserConfig:  map[string]any{"dflash": true},
	}
	if err := checkpoint.Save(*outPath, saveMeta, dflash.NamedParameters()); err != nil {
		logger.Error("save drafter", "err", err)
		os.Exit(1)
	}
	logger.Info("saved dflash drafter", "path", *outPath)
}
