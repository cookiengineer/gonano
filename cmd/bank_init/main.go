// Command bank_init writes a domain bank manifest (bank.json) from a list of
// domain checkpoints, so cmd/server --bank can route requests to them.
//
// Example:
//
//	go run ./cmd/bank_init \
//	  --domain physics=~/.cache/gonano/domains/physics/base_checkpoints/d2/model_000200.gn \
//	  --domain math=~/.cache/gonano/domains/math/base_checkpoints/d2/model_000200.gn \
//	  --data physics=~/data/physics --data math=~/data/math \
//	  --tokenizer ~/.cache/gonano/tokenizer/tokenizer.json \
//	  --router ~/.cache/gonano/router/router.gn \
//	  --out ~/.cache/gonano/bank/bank.json
package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/cookiengineer/gonano/bank"
	"github.com/cookiengineer/gonano/data"
	"github.com/cookiengineer/gonano/internal/logging"
)

// nameValue collects repeated --domain/--data name=value pairs.
type nameValue struct {
	kind   string
	values map[string]string
	order  []string
}

func newNameValue(kind string) *nameValue {
	return &nameValue{kind: kind, values: map[string]string{}}
}

func (flagValue *nameValue) String() string {
	parts := make([]string, 0, len(flagValue.order))
	for _, name := range flagValue.order {
		parts = append(parts, name+"="+flagValue.values[name])
	}
	return strings.Join(parts, ",")
}

func (flagValue *nameValue) Set(value string) error {
	parts := strings.SplitN(value, "=", 2)
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return fmt.Errorf("expected name=path, got %q", value)
	}
	if _, exists := flagValue.values[parts[0]]; !exists {
		flagValue.order = append(flagValue.order, parts[0])
	}
	flagValue.values[parts[0]] = parts[1]
	return nil
}

func main() {
	domains := newNameValue("domain")
	datasets := newNameValue("data")
	flag.Var(domains, "domain", "domain spec name=checkpoint (repeatable, at least two)")
	flag.Var(datasets, "data", "optional domain data dir name=dir (repeatable)")
	tokenizerPath := flag.String("tokenizer", "", "path to tokenizer.json (default <base-dir>/tokenizer/tokenizer.json)")
	routerPath := flag.String("router", "", "path to a router bundle")
	baseDir := flag.String("base-dir", "", "base directory (default ~/.cache/gonano)")
	outPath := flag.String("out", "", "manifest output (default <base-dir>/bank/bank.json)")
	flag.Parse()

	if *baseDir == "" {
		*baseDir = data.BaseDir()
	}
	logger := logging.Default(slog.LevelInfo)

	if len(domains.order) < 2 {
		logger.Error("at least two --domain name=checkpoint entries are required")
		os.Exit(1)
	}
	if *tokenizerPath == "" {
		*tokenizerPath = filepath.Join(*baseDir, "tokenizer", "tokenizer.json")
	}
	if *outPath == "" {
		*outPath = filepath.Join(*baseDir, "bank", "bank.json")
	}

	manifest := &bank.Manifest{Version: 1, Tokenizer: *tokenizerPath, Router: *routerPath}
	for _, name := range domains.order {
		domain := bank.Domain{ID: name, Name: name, Checkpoint: domains.values[name]}
		if dir, ok := datasets.values[name]; ok {
			domain.DataDirs = []string{dir}
		}
		manifest.Domains = append(manifest.Domains, domain)
	}

	if err := os.MkdirAll(filepath.Dir(*outPath), 0o755); err != nil {
		logger.Error("create output directory", "err", err)
		os.Exit(1)
	}
	if err := manifest.Save(*outPath); err != nil {
		logger.Error("save manifest", "err", err)
		os.Exit(1)
	}
	logger.Info("saved bank manifest", "path", *outPath, "domains", domains.order)
}
