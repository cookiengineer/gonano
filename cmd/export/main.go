// Command export converts a gonano checkpoint (.gn) to the GGUF container
// format. Because nanochat is a custom architecture, the resulting GGUF is a
// faithful weight container with a "nanochat" architecture tag; see the export
// guide for details on consuming it.
package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"

	"github.com/cookiengineer/gonano/internal/logging"
	"github.com/cookiengineer/gonano/model/checkpoint"
)

func main() {
	modelPath := flag.String("model", "", "path to a .gn checkpoint (required)")
	outputPath := flag.String("out", "", "output .gguf path (required)")
	flag.Parse()

	logger := logging.Default(slog.LevelInfo)
	if *modelPath == "" || *outputPath == "" {
		fmt.Fprintln(os.Stderr, "usage: export --model <path.gn> --out <path.gguf>")
		os.Exit(1)
	}

	meta, params, err := checkpoint.Load(*modelPath)
	if err != nil {
		logger.Error("load checkpoint", "err", err)
		os.Exit(1)
	}
	if err := checkpoint.ExportGGUF(*outputPath, meta, params); err != nil {
		logger.Error("export", "err", err)
		os.Exit(1)
	}
	logger.Info("exported GGUF", "path", *outputPath, "tensors", len(params))
}
