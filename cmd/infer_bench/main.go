// Command infer_bench measures inference latency and throughput for a
// checkpoint across decode batch sizes.
package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"time"

	"github.com/cookiengineer/gonano/data"
	"github.com/cookiengineer/gonano/inference"
	"github.com/cookiengineer/gonano/internal/logging"
	"github.com/cookiengineer/gonano/model/checkpoint"
	"github.com/cookiengineer/gonano/tokenizer"
)

func main() {
	modelPath := flag.String("model", "", "path to a .gn checkpoint (required)")
	promptTokens := flag.Int("prompt-tokens", 128, "prompt length")
	decodeTokens := flag.Int("decode-tokens", 64, "tokens to generate per row")
	batchSizes := flag.String("batch-sizes", "1,4,16", "comma-separated batch sizes")
	baseDir := flag.String("base-dir", "", "tokenizer directory (default ~/.cache/gonano)")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: infer_bench --model <path>")
		os.Exit(1)
	}
	logger := logging.Default(slog.LevelInfo)
	if *baseDir == "" {
		*baseDir = data.BaseDir()
	}

	tokenizer, err := tokenizer.LoadTokenizer(filepath.Join(*baseDir, "tokenizer", "tokenizer.json"))
	if err != nil {
		// The benchmark uses a synthetic prompt and never decodes text, so a
		// real tokenizer is not required — fall back to a byte-level one.
		logger.Warn("no tokenizer found; using a byte-level tokenizer for the benchmark", "err", err)
		tokenizer = byteTokenizer()
	}
	meta, params, err := checkpoint.LoadAny(*modelPath)
	if err != nil {
		logger.Error("load checkpoint", "err", err)
		os.Exit(1)
	}
	model := checkpoint.LoadModel(meta, params)
	engine := inference.NewEngine(model, tokenizer)

	// Clamp the prompt so prompt+decode fits the model context.
	maxPrompt := meta.ModelConfig.SequenceLen - *decodeTokens
	if *promptTokens > maxPrompt {
		*promptTokens = maxPrompt
	}
	prompt := []int{tokenizer.BOSTokenID()}
	for len(prompt) < *promptTokens {
		prompt = append(prompt, 1)
	}

	fmt.Printf("%-6s %-10s %-12s %-10s %-12s\n", "batch", "TTFT(ms)", "TPOT(ms)", "tok/s", "decode tok/s")
	for _, batchSize := range parseBatchSizes(*batchSizes) {
		measurement := inference.Measure(engine, prompt, batchSize, *decodeTokens, 0, 0, 42)
		var totalStepTime time.Duration
		for _, stepTime := range measurement.StepTimes {
			totalStepTime += stepTime
		}
		timePerOutputToken := time.Duration(0)
		if len(measurement.StepTimes) > 0 {
			timePerOutputToken = totalStepTime / time.Duration(len(measurement.StepTimes))
		}
		// Overall throughput includes the prefill (TTFT).
		tokensPerSecond := float64(batchSize*measurement.NumTokens) / (measurement.TTFT + totalStepTime).Seconds()
		// Pure decode throughput excludes the prefill and is the number to
		// watch when serving (decode is memory-bandwidth-bound).
		decodeRate := "-"
		if totalStepTime > 0 {
			decodeTokensPerSecond := float64(batchSize*len(measurement.StepTimes)) / totalStepTime.Seconds()
			decodeRate = fmt.Sprintf("%-12.0f", decodeTokensPerSecond)
		}
		fmt.Printf("%-6d %-10.2f %-12.3f %-10.0f %s\n",
			batchSize, measurement.TTFT.Seconds()*1000, timePerOutputToken.Seconds()*1000, tokensPerSecond, decodeRate)
	}
}

func parseBatchSizes(spec string) []int {
	var sizes []int
	value := 0
	for _, character := range spec {
		if character == ',' {
			sizes = append(sizes, value)
			value = 0
			continue
		}
		value = value*10 + int(character-'0')
	}
	sizes = append(sizes, value)
	return sizes
}

// byteTokenizer builds a byte-level BPE tokenizer (256 single-byte ranks plus
// the standard special tokens), used only when no trained tokenizer exists.
func byteTokenizer() *tokenizer.Tokenizer {
	ranks := make(map[string]int, 256)
	for index := 0; index < 256; index++ {
		ranks[string([]byte{byte(index)})] = index
	}
	return tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
}
