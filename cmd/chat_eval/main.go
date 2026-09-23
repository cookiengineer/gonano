// Command chat_eval evaluates a chat model with the ChatCORE metric.
package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/cookiengineer/gonano/data"
	"github.com/cookiengineer/gonano/evaluator"
	"github.com/cookiengineer/gonano/evaluator/tasks"
	"github.com/cookiengineer/gonano/inference"
	"github.com/cookiengineer/gonano/internal/logging"
	"github.com/cookiengineer/gonano/model/checkpoint"
	"github.com/cookiengineer/gonano/tokenizer"
)

func main() {
	modelPath := flag.String("model", "", "path to a .gn checkpoint (required)")
	baseDir := flag.String("base-dir", "", "tokenizer directory")
	domain := flag.String("domain", "", "domain label to report for this evaluation (optional)")
	flag.Parse()

	logger := logging.Default(slog.LevelInfo)
	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: chat_eval --model <path> [--domain <name>]")
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
	engine := inference.NewEngine(model, tokenizer)
	if *domain != "" {
		logger.Info("evaluating domain", "domain", *domain, "model", *modelPath)
	}

	// Synthetic categorical task for demonstration.
	mmlu := tasks.NewMMLUFromRows([]tasks.MMLURow{
		{Question: "What is 2+2?", Choices: []string{"3", "4", "5", "6"}, Answer: 1},
		{Question: "Capital of France?", Choices: []string{"London", "Paris", "Rome", "Berlin"}, Answer: 1},
	})
	categoricalAccuracy := evaluator.CategoricalAccuracy(mmlu, model, tokenizer)
	fmt.Printf("MMLU (synthetic) accuracy: %.2f%%\n", 100*categoricalAccuracy)

	gsm8k := tasks.NewGSM8KFromRows([]tasks.GSM8KRow{
		{Question: "What is 2+2?", Answer: "#### 4"},
	})
	generativeAccuracy := evaluator.GenerativeAccuracy(gsm8k, model, tokenizer, engine, 1, 16, 0.0, 50)
	fmt.Printf("GSM8K (synthetic) accuracy: %.2f%%\n", 100*generativeAccuracy)
}
