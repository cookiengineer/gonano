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
	modelpkg "github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/model/checkpoint"
	"github.com/cookiengineer/gonano/tokenizer"
)

func main() {
	modelPath := flag.String("model", "", "path to a .gn checkpoint (required)")
	promptTokens := flag.Int("prompt-tokens", 128, "prompt length")
	decodeTokens := flag.Int("decode-tokens", 64, "tokens to generate per row")
	batchSizes := flag.String("batch-sizes", "1,4,16", "comma-separated batch sizes")
	prefixCache := flag.Bool("prefix-cache", false, "enable the multi-entry KV prefix cache")
	drafterPath := flag.String("drafter", "", "path to a DSpark drafter checkpoint")
	speculative := flag.Bool("speculative", false, "use exact greedy speculative decoding with --drafter")
	draftLength := flag.Int("draft-length", 5, "tokens drafted per speculative round")
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
	if *prefixCache {
		engine.Cache = inference.NewCacheManager(inference.CacheOptions{MaxEntries: 8})
	}
	if *drafterPath != "" {
		drafterMeta, drafterParams, err := checkpoint.Load(*drafterPath)
		if err != nil {
			logger.Error("load drafter", "err", err)
			os.Exit(1)
		}
		if checkpoint.IsDSpark(drafterMeta) {
			engine.DSpark = modelpkg.LoadDSpark(drafterMeta.ModelConfig, drafterParams)
		} else {
			engine.Drafter = checkpoint.LoadModel(drafterMeta, drafterParams)
		}
		engine.DraftLength = *draftLength
	}
	engine.Speculative = *speculative

	// Clamp the prompt so prompt+decode fits the model context.
	maxPrompt := meta.ModelConfig.SequenceLen - *decodeTokens
	if *promptTokens > maxPrompt {
		*promptTokens = maxPrompt
	}
	prompt := []int{tokenizer.BOSTokenID()}
	for len(prompt) < *promptTokens {
		prompt = append(prompt, 1)
	}

	fmt.Printf("%-6s %-9s %-10s %-9s %-13s %-11s %-11s %-8s %-7s\n",
		"batch", "TTFT(ms)", "TPOT(ms)", "tok/s", "decode tok/s", "weight B/s", "kv B/s", "GB/s", "kv%")
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
		weightPerStep := "-"
		kvPerStep := "-"
		bandwidth := "-"
		kvShare := "-"
		if totalStepTime > 0 && measurement.DecodeSteps > 0 {
			decodeTokensPerSecond := float64(batchSize*len(measurement.StepTimes)) / totalStepTime.Seconds()
			decodeRate = fmt.Sprintf("%-13.0f", decodeTokensPerSecond)

			steps := int64(measurement.DecodeSteps)
			weightPerStep = humanBytes(measurement.WeightBytes / steps)
			kvPerStep = humanBytes(measurement.KVBytes / steps)

			totalBytes := measurement.WeightBytes + measurement.KVBytes
			bandwidth = fmt.Sprintf("%-8.1f", float64(totalBytes)/totalStepTime.Seconds()/1e9)
			kvShare = fmt.Sprintf("%-7.0f", 100*float64(measurement.KVBytes)/float64(totalBytes))
		}
		fmt.Printf("%-6d %-9.2f %-10.3f %-9.0f %-13s %-11s %-11s %-8s %-7s\n",
			batchSize, measurement.TTFT.Seconds()*1000, timePerOutputToken.Seconds()*1000, tokensPerSecond,
			decodeRate, weightPerStep, kvPerStep, bandwidth, kvShare)
	}
}

// humanBytes formats a byte count with a binary unit suffix (e.g. "1.7 MB").
func humanBytes(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	divisor, exponent := int64(unit), 0
	for value := bytes / unit; value >= unit; value /= unit {
		divisor *= unit
		exponent++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(divisor), "KMGTPE"[exponent])
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
