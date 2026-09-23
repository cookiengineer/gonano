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

	"github.com/cookiengineer/gonano/bank"
	"github.com/cookiengineer/gonano/data"
	"github.com/cookiengineer/gonano/inference"
	"github.com/cookiengineer/gonano/internal/logging"
	"github.com/cookiengineer/gonano/model/checkpoint"
	"github.com/cookiengineer/gonano/router"
	"github.com/cookiengineer/gonano/server"
	"github.com/cookiengineer/gonano/tokenizer"
)

func main() {
	modelPath := flag.String("model", "", "path to a .gn or .gguf checkpoint (required unless --bank is set)")
	bankPath := flag.String("bank", "", "path to a domain bank manifest (enables domain routing)")
	routerPath := flag.String("router", "", "router bundle path (default: the manifest's router field)")
	maxDomains := flag.Int("max-domains", 1, "maximum domains blended per request when --bank is set")
	minDomainScore := flag.Float64("min-domain-score", 0, "minimum router probability to include an extra domain")
	routeScope := flag.String("route-scope", "last-turn", "what the router classifies: last-turn|full-prompt")
	tokenizerPath := flag.String("tokenizer", "", "path to tokenizer.json (default: <base-dir>/tokenizer/tokenizer.json)")
	baseDir := flag.String("base-dir", "", "base directory (default ~/.cache/gonano)")
	addr := flag.String("addr", ":8080", "listen address")
	modelName := flag.String("model-name", "gonano", "model id reported to clients")
	prefixCacheSize := flag.Int("prefix-cache-size", 8, "KV prefix cache entries (0 disables)")
	prefixCacheDir := flag.String("prefix-cache-dir", "", "persist the KV prefix cache to this directory")
	prefixCacheTTL := flag.Duration("prefix-cache-ttl", 0, "KV prefix cache entry TTL (0 = no expiry)")
	flag.Parse()

	logger := logging.Default(slog.LevelInfo)
	if *baseDir == "" {
		*baseDir = data.BaseDir()
	}

	// Domain bank mode: route each request to the relevant domain models.
	if *bankPath != "" {
		serveBank(logger, *bankPath, *routerPath, *tokenizerPath, *baseDir, *addr, *modelName, *maxDomains, float32(*minDomainScore), *routeScope)
		return
	}

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: server --model <path> [--addr :8080]")
		fmt.Fprintln(os.Stderr, "   or: server --bank <manifest> [--addr :8080]")
		os.Exit(1)
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
	if *prefixCacheSize > 0 || *prefixCacheDir != "" {
		httpServer.Engine.Cache = inference.NewCacheManager(inference.CacheOptions{
			MaxEntries: *prefixCacheSize,
			TTL:        *prefixCacheTTL,
			DiskDir:    *prefixCacheDir,
		})
	}
	logger.Info("serving OpenAI-compatible API", "addr", *addr, "model", *modelName, "vocab", tokenizer.VocabSize())
	if err := http.ListenAndServe(*addr, httpServer.Handler()); err != nil {
		logger.Error("serve", "err", err)
		os.Exit(1)
	}
}

// serveBank starts the server in domain-bank mode.
func serveBank(logger *slog.Logger, bankPath, routerPath, tokenizerPath, baseDir, addr, modelName string, maxDomains int, minDomainScore float32, routeScope string) {
	manifest, err := bank.LoadManifest(bankPath)
	if err != nil {
		logger.Error("load bank manifest", "err", err)
		os.Exit(1)
	}
	tokenizerFile := tokenizerPath
	if tokenizerFile == "" {
		tokenizerFile = manifest.Tokenizer
	}
	if tokenizerFile == "" {
		tokenizerFile = resolveTokenizer("", baseDir)
	}
	tokenizer, err := tokenizer.LoadTokenizer(tokenizerFile)
	if err != nil {
		logger.Error("load tokenizer", "path", tokenizerFile, "err", err)
		os.Exit(1)
	}
	bundlePath := routerPath
	if bundlePath == "" {
		bundlePath = manifest.Router
	}
	if bundlePath == "" {
		logger.Error("no router bundle configured; set --router or manifest.router")
		os.Exit(1)
	}
	domainRouter, err := router.LoadRouterAuto(bundlePath)
	if err != nil {
		logger.Error("load router", "err", err)
		os.Exit(1)
	}
	domainBank := bank.NewBank(manifest, tokenizer, bank.BankOptions{})

	// The first domain provides the single-model engine used as a fallback.
	firstID := manifest.IDs()[0]
	firstModel, err := domainBank.Acquire(firstID)
	if err != nil {
		logger.Error("load first domain", "domain", firstID, "err", err)
		os.Exit(1)
	}

	tools := inference.NewCalculator()
	tools.Register(nowTool{})
	httpServer := server.NewServer(firstModel, tokenizer, tools, modelName)
	httpServer.Bank = domainBank
	httpServer.Router = domainRouter
	httpServer.MaxDomains = maxDomains
	httpServer.MinDomainScore = minDomainScore
	httpServer.RouteScope = routeScope

	logger.Info("serving domain bank", "addr", addr, "domains", manifest.IDs(), "max_domains", maxDomains, "route_scope", routeScope)
	if err := http.ListenAndServe(addr, httpServer.Handler()); err != nil {
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
