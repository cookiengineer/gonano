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

	"github.com/cookiengineer/gonano/checkpoint"
	"github.com/cookiengineer/gonano/data"
	"github.com/cookiengineer/gonano/infer"
	"github.com/cookiengineer/gonano/logging"
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

	// Clamp the prompt so prompt+decode fits the model context.
	maxPrompt := meta.ModelConfig.SequenceLen - *decodeTokens
	if *promptTokens > maxPrompt {
		*promptTokens = maxPrompt
	}
	prompt := []int{tok.BOSTokenID()}
	for len(prompt) < *promptTokens {
		prompt = append(prompt, 1)
	}

	fmt.Printf("%-6s %-10s %-12s %-12s\n", "batch", "TTFT(ms)", "TPOT(ms)", "tok/s")
	for _, bs := range parseBatchSizes(*batchSizes) {
		meas := infer.Measure(engine, prompt, bs, *decodeTokens, 0, 0, 42)
		var sum time.Duration
		for _, s := range meas.StepTimes {
			sum += s
		}
		tpot := time.Duration(0)
		if len(meas.StepTimes) > 0 {
			tpot = sum / time.Duration(len(meas.StepTimes))
		}
		tokPerSec := float64(bs*meas.NumTokens) / (meas.TTFT + sum).Seconds()
		fmt.Printf("%-6d %-10.2f %-12.3f %-12.0f\n", bs, meas.TTFT.Seconds()*1000, tpot.Seconds()*1000, tokPerSec)
	}
}

func parseBatchSizes(s string) []int {
	var out []int
	num := 0
	for _, r := range s {
		if r == ',' {
			out = append(out, num)
			num = 0
			continue
		}
		num = num*10 + int(r-'0')
	}
	out = append(out, num)
	return out
}
