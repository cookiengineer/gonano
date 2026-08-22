// Command chat_cli talks to a trained model over the command line.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/cookiengineer/gonano/checkpoint"
	"github.com/cookiengineer/gonano/data"
	"github.com/cookiengineer/gonano/infer"
	"github.com/cookiengineer/gonano/logging"
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

	tok, err := tokenizer.LoadTokenizer(filepath.Join(*baseDir, "tokenizer", "tokenizer.json"))
	if err != nil {
		logger.Error("load tokenizer", "err", err)
		os.Exit(1)
	}

	meta, params, err := checkpoint.LoadAny(*modelPath)
	if err != nil {
		logger.Error("load checkpoint", "err", err)
		os.Exit(1)
	}
	m := checkpoint.LoadModel(meta, params)
	engine := infer.NewEngine(m, tok)

	bos := tok.BOSTokenID()
	userStart := tok.EncodeSpecial("<|user_start|>")
	userEnd := tok.EncodeSpecial("<|user_end|>")
	assistantStart := tok.EncodeSpecial("<|assistant_start|>")
	assistantEnd := tok.EncodeSpecial("<|assistant_end|>")

	respond := func(text string) {
		conv := []int{bos, userStart}
		conv = append(conv, tok.Encode(text)...)
		conv = append(conv, userEnd, assistantStart)
		fmt.Print("Assistant: ")
		gen := engine.Generate(conv, 1, *maxTokens, float32(*temperature), *topK, 42)
		gen(func(column, mask []int) bool {
			token := column[0]
			if token == assistantEnd {
				return false
			}
			fmt.Print(tok.Decode([]int{token}))
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
