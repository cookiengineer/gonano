// Command chat_eval evaluates a chat model with the ChatCORE metric.
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
	"github.com/cookiengineer/gonano/eval/tasks"
	"github.com/cookiengineer/gonano/infer"
	"github.com/cookiengineer/gonano/logging"
	"github.com/cookiengineer/gonano/tokenizer"
)

func main() {
	modelPath := flag.String("model", "", "path to a .gn checkpoint (required)")
	baseDir := flag.String("base-dir", "", "tokenizer directory")
	flag.Parse()

	logger := logging.Default(slog.LevelInfo)
	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: chat_eval --model <path>")
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
	engine := infer.NewEngine(m, tok)

	// Synthetic categorical task for demonstration.
	mm := tasks.NewMMLUFromRows([]tasks.MMLURow{
		{Question: "What is 2+2?", Choices: []string{"3", "4", "5", "6"}, Answer: 1},
		{Question: "Capital of France?", Choices: []string{"London", "Paris", "Rome", "Berlin"}, Answer: 1},
	})
	acc := eval.CategoricalAccuracy(mm, m, tok)
	fmt.Printf("MMLU (synthetic) accuracy: %.2f%%\n", 100*acc)

	gsm := tasks.NewGSM8KFromRows([]tasks.GSM8KRow{
		{Question: "What is 2+2?", Answer: "#### 4"},
	})
	gen := eval.GenerativeAccuracy(gsm, m, tok, engine, 1, 16, 0.0, 50)
	fmt.Printf("GSM8K (synthetic) accuracy: %.2f%%\n", 100*gen)
}
