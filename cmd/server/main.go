// Command server runs an OpenAI-compatible HTTP API for a gonano checkpoint,
// with tool-call support. Tools are executed server-side in Go.
//
// Example:
//
//	go run ./cmd/server --model ~/.cache/gonano/base_checkpoints/d4/model_000050.gn --addr :8080
//	curl http://localhost:8080/v1/chat/completions \
//	  -d '{"model":"gonano","messages":[{"role":"user","content":"what is 2+2?"}]}'
package main

import (
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/cookiengineer/gonano/data"
	"github.com/cookiengineer/gonano/inference"
	"github.com/cookiengineer/gonano/internal/logging"
	"github.com/cookiengineer/gonano/model/checkpoint"
	"github.com/cookiengineer/gonano/server"
	"github.com/cookiengineer/gonano/tokenizer"
)

func main() {
	modelPath := flag.String("model", "", "path to a .gn or .gguf checkpoint (required)")
	tokenizerPath := flag.String("tokenizer", "", "path to tokenizer.json (default: <base-dir>/tokenizer/tokenizer.json)")
	baseDir := flag.String("base-dir", "", "base directory (default ~/.cache/gonano)")
	addr := flag.String("addr", ":8080", "listen address")
	modelName := flag.String("model-name", "gonano", "model id reported to clients")
	flag.Parse()

	logger := logging.Default(slog.LevelInfo)
	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: server --model <path> [--addr :8080]")
		os.Exit(1)
	}
	if *baseDir == "" {
		*baseDir = data.BaseDir()
	}

	tokenizer, err := tokenizer.LoadTokenizer(resolveTokenizer(*tokenizerPath, *baseDir))
	if err != nil {
		logger.Warn("no tokenizer found; using a byte-level default", "err", err)
		tokenizer = byteTokenizer()
	}

	meta, params, err := checkpoint.LoadAny(*modelPath)
	if err != nil {
		logger.Error("load checkpoint", "err", err)
		os.Exit(1)
	}
	model := checkpoint.LoadModel(meta, params)

	// Register the built-in calculator plus a demo "now" tool. Add more Go
	// tools here to extend the model's abilities.
	tools := inference.NewCalculator()
	tools.Register(nowTool{})

	httpServer := server.NewServer(model, tokenizer, tools, *modelName)
	logger.Info("serving OpenAI-compatible API", "addr", *addr, "model", *modelName, "vocab", tokenizer.VocabSize())
	if err := http.ListenAndServe(*addr, httpServer.Handler()); err != nil {
		logger.Error("serve", "err", err)
		os.Exit(1)
	}
}

func resolveTokenizer(path, baseDir string) string {
	if path != "" {
		return path
	}
	return filepath.Join(baseDir, "tokenizer", "tokenizer.json")
}

// nowTool returns the current UTC time when the model calls "now".
type nowTool struct{}

func (nowTool) Name() string { return "now" }
func (nowTool) Call(expr string) (string, bool) {
	if expr != "now" {
		return "", false
	}
	return time.Now().UTC().Format(time.RFC3339), true
}

func byteTokenizer() *tokenizer.Tokenizer {
	ranks := make(map[string]int, 256)
	for index := 0; index < 256; index++ {
		ranks[string([]byte{byte(index)})] = index
	}
	return tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
}
