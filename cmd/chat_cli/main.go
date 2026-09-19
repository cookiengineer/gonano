// Command chat_cli talks to a trained model over the command line.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/cookiengineer/gonano/data"
	"github.com/cookiengineer/gonano/inference"
	"github.com/cookiengineer/gonano/internal/logging"
	"github.com/cookiengineer/gonano/model/checkpoint"
	"github.com/cookiengineer/gonano/tokenizer"
)

func main() {
	modelPath := flag.String("model", "", "path to a .gn checkpoint (required)")
	temperature := flag.Float64("temperature", 0.6, "sampling temperature")
	topK := flag.Int("top-k", 50, "top-k sampling")
	maxTokens := flag.Int("max-tokens", 256, "max tokens per response")
	prompt := flag.String("prompt", "", "single-shot prompt (empty = interactive)")
	baseDir := flag.String("base-dir", "", "tokenizer directory (default ~/.cache/gonano)")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: chat_cli --model <path>")
		os.Exit(1)
	}
	logger := logging.Default(slog.LevelInfo)
	if *baseDir == "" {
		*baseDir = data.BaseDir()
	}

	tokenizer, err := tokenizer.LoadTokenizer(filepath.Join(*baseDir, "tokenizer", "tokenizer.json"))
	if err != nil {
		logger.Error("load tokenizer", "err", err)
		os.Exit(1)
	}

	meta, params, err := checkpoint.LoadAny(*modelPath)
	if err != nil {
		logger.Error("load checkpoint", "err", err)
		os.Exit(1)
	}
	model := checkpoint.LoadModel(meta, params)
	engine := inference.NewEngine(model, tokenizer)

	bos := tokenizer.BOSTokenID()
	userStart := tokenizer.EncodeSpecial("<|user_start|>")
	userEnd := tokenizer.EncodeSpecial("<|user_end|>")
	assistantStart := tokenizer.EncodeSpecial("<|assistant_start|>")
	assistantEnd := tokenizer.EncodeSpecial("<|assistant_end|>")

	respond := func(text string) {
		conversation := []int{bos, userStart}
		conversation = append(conversation, tokenizer.Encode(text)...)
		conversation = append(conversation, userEnd, assistantStart)
		fmt.Print("Assistant: ")
		generator := engine.Generate(conversation, 1, *maxTokens, float32(*temperature), *topK, 42)
		generator(func(column, mask []int) bool {
			token := column[0]
			if token == assistantEnd {
				return false
			}
			fmt.Print(tokenizer.Decode([]int{token}))
			return true
		})
		fmt.Println()
	}

	if *prompt != "" {
		respond(*prompt)
		return
	}

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("User: ")
		if !scanner.Scan() {
			break
		}
		text := scanner.Text()
		switch text {
		case "quit", "exit":
			return
		}
		respond(text)
	}
}
