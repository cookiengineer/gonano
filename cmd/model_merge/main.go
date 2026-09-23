// Command model_merge averages several checkpoints into one. It is the
// lightweight model-merging reinitialization DeepSeek-V4.1 §5.1.2 uses between
// successive RL runs.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/cookiengineer/gonano/model/checkpoint"
)

func main() {
	models := flag.String("models", "", "comma-separated checkpoint paths (required)")
	weights := flag.String("weights", "", "comma-separated merge weights (default uniform)")
	out := flag.String("out", "", "output checkpoint path (required)")
	flag.Parse()

	if *models == "" || *out == "" {
		fmt.Fprintln(os.Stderr, "usage: model_merge --models a.gn,b.gn [--weights 0.5,0.5] --out merged.gn")
		os.Exit(1)
	}
	paths := splitList(*models)
	parsedWeights, err := parseWeights(*weights, len(paths))
	if err != nil {
		fmt.Fprintln(os.Stderr, "model_merge:", err)
		os.Exit(1)
	}
	if directory := filepath.Dir(*out); directory != "." {
		if err := os.MkdirAll(directory, 0o755); err != nil {
			fmt.Fprintln(os.Stderr, "model_merge:", err)
			os.Exit(1)
		}
	}
	if err := checkpoint.MergeFiles(paths, parsedWeights, *out); err != nil {
		fmt.Fprintln(os.Stderr, "model_merge:", err)
		os.Exit(1)
	}
	fmt.Printf("merged %d checkpoints into %s\n", len(paths), *out)
}

// splitList splits a comma-separated list, dropping empty fields.
func splitList(spec string) []string {
	fields := strings.Split(spec, ",")
	out := make([]string, 0, len(fields))
	for _, field := range fields {
		field = strings.TrimSpace(field)
		if field != "" {
			out = append(out, field)
		}
	}
	return out
}

// parseWeights parses a comma-separated weight list. An empty spec returns nil
// (uniform weights); otherwise the count must match the model count.
func parseWeights(spec string, modelCount int) ([]float32, error) {
	if strings.TrimSpace(spec) == "" {
		return nil, nil
	}
	fields := splitList(spec)
	if len(fields) != modelCount {
		return nil, fmt.Errorf("%d weights for %d models", len(fields), modelCount)
	}
	weights := make([]float32, len(fields))
	for index, field := range fields {
		value, err := strconv.ParseFloat(field, 32)
		if err != nil {
			return nil, fmt.Errorf("invalid weight %q", field)
		}
		weights[index] = float32(value)
	}
	return weights, nil
}
