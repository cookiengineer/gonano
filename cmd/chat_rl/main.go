// Command chat_rl runs reinforcement learning (GRPO/REINFORCE) on GSM8K.
package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/cookiengineer/gonano/checkpoint"
	"github.com/cookiengineer/gonano/data"
	"github.com/cookiengineer/gonano/eval/tasks"
	"github.com/cookiengineer/gonano/infer"
	"github.com/cookiengineer/gonano/logging"
	"github.com/cookiengineer/gonano/optim"
	"github.com/cookiengineer/gonano/tensor"
	"github.com/cookiengineer/gonano/tokenizer"
	"github.com/cookiengineer/gonano/train"
)

func main() {
	modelPath := flag.String("model", "", "path to a .gn checkpoint (required)")
	numSteps := flag.Int("num-steps", 10, "optimization steps")
	numSamples := flag.Int("num-samples", 4, "rollouts per example")
	outPath := flag.String("out", "", "output checkpoint path")
	baseDir := flag.String("base-dir", "", "tokenizer directory")
	flag.Parse()

	logger := logging.Default(slog.LevelInfo)
	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: chat_rl --model <path>")
		os.Exit(1)
	}
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

	gsm := tasks.NewGSM8KFromRows([]tasks.GSM8KRow{
		{Question: "What is 2+2?", Answer: "#### 4"},
		{Question: "What is 3+3?", Answer: "#### 6"},
	})

	groups := m.SetupOptimizer(0.004, 0.2, 0.02, 0.0, 0.5)
	for i := range groups {
		groups[i].LR *= 0.05
	}
	opt := optim.NewMuonAdamW(groups)

	for step := 0; step < *numSteps; step++ {
		conv := gsm.GetExample(step % gsm.NumExamples())
		prompt := tok.RenderForCompletion(conv)
		results, _ := engine.GenerateBatch(prompt, *numSamples, 32, 1.0, 50, uint64(step))

		rewards := make([]float32, len(results))
		for i, r := range results {
			completion := tok.Decode(r[len(prompt):])
			rewards[i] = gsm.Reward(conv, completion)
		}
		var mean float32
		for _, r := range rewards {
			mean += r
		}
		mean /= float32(len(rewards))
		advantages := make([]float32, len(rewards))
		for i, r := range rewards {
			advantages[i] = r - mean
		}

		// Build a padded batch of rollouts.
		maxLen := 0
		for _, r := range results {
			if len(r) > maxLen {
				maxLen = len(r)
			}
		}
		pad := tok.EncodeSpecial("<|assistant_end|>")
		inputs := make([][]int, len(results))
		targets := make([][]int, len(results))
		for i, r := range results {
			padded := append([]int(nil), r...)
			for len(padded) < maxLen {
				padded = append(padded, pad)
			}
			inputs[i] = padded[:len(padded)-1]
			targets[i] = padded[1:]
			// Mask forced/prompt tokens: only train on sampled tokens.
			mask := len(prompt)
			for j := range targets[i] {
				if j < mask {
					targets[i][j] = -1
				}
			}
		}
		x := int32sFrom(inputs)
		y := int32sFrom(targets)
		loss := train.RLStep(m, opt, x, y, advantages, float32(maxLen))
		logger.Info("rl", "step", step, "loss", fmt.Sprintf("%.4f", loss), "mean_reward", fmt.Sprintf("%.2f", mean))
	}

	if *outPath != "" {
		outMeta := checkpoint.Meta{Step: *numSteps, ModelConfig: m.Config}
		if err := checkpoint.Save(*outPath, outMeta, m.NamedParameters()); err != nil {
			logger.Error("save", "err", err)
			os.Exit(1)
		}
	}
}

func int32sFrom(rows [][]int) *tensor.Int32s {
	shape := []int{len(rows), len(rows[0])}
	data := make([]int32, 0, len(rows)*len(rows[0]))
	for _, r := range rows {
		for _, v := range r {
			data = append(data, int32(v))
		}
	}
	return tensor.NewInt32sWithData(shape, data)
}
