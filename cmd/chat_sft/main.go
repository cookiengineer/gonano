// Command chat_sft runs supervised fine-tuning over conversation datasets.
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
	"github.com/cookiengineer/gonano/train"
	"github.com/cookiengineer/gonano/tokenizer"
)

func main() {
	modelPath := flag.String("model", "", "path to a .gn base checkpoint (required)")
	numIterations := flag.Int("num-iterations", 200, "optimization steps")
	batchSize := flag.Int("batch-size", 1, "batch size")
	maxSeqLen := flag.Int("max-seq-len", 512, "max sequence length")
	outPath := flag.String("out", "", "output checkpoint path")
	baseDir := flag.String("base-dir", "", "tokenizer directory")
	flag.Parse()

	logger := logging.Default(slog.LevelInfo)
	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: chat_sft --model <path> --out <path>")
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

	// Synthetic conversation provider for demonstration; a production setup
	// loads SmolTalk/MMLU/GSM8K via the tasks package.
	convs := []*tokenizer.Conversation{
		{Messages: []tokenizer.Message{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "hello"}}},
		{Messages: []tokenizer.Message{{Role: "user", Content: "how are you"}, {Role: "assistant", Content: "fine thanks"}}},
	}
	i := 0
	provider := func() ([]*tokenizer.Conversation, bool) {
		c := convs[i%len(convs)]
		i++
		return []*tokenizer.Conversation{c}, true
	}
	loader := data.NewSFTLoader(tok, *batchSize, *maxSeqLen, provider, 100)

	groups := m.SetupOptimizer(0.008, 0.2, 0.02, 0.0, 0.5)
	train.TrainSFT(m, groups, loader, *numIterations, func(step int, loss float32) {
		if step%20 == 0 {
			logger.Info("sft", "step", step, "loss", fmt.Sprintf("%.4f", loss))
		}
	})

	if *outPath != "" {
		outMeta := checkpoint.Meta{Step: *numIterations, ModelConfig: m.Config}
		if err := checkpoint.Save(*outPath, outMeta, m.NamedParameters()); err != nil {
			logger.Error("save", "err", err)
			os.Exit(1)
		}
		logger.Info("saved SFT checkpoint", "path", *outPath)
	}
}
