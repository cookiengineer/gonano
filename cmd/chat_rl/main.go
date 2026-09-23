// Command chat_rl runs reinforcement learning (GRPO/REINFORCE) on GSM8K.
package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/cookiengineer/gonano/data"
	"github.com/cookiengineer/gonano/evaluator/tasks"
	"github.com/cookiengineer/gonano/inference"
	"github.com/cookiengineer/gonano/internal/logging"
	"github.com/cookiengineer/gonano/model/checkpoint"
	"github.com/cookiengineer/gonano/optimizer"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
	"github.com/cookiengineer/gonano/trainer"
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
	tokenizer, err := tokenizer.LoadTokenizer(filepath.Join(*baseDir, "tokenizer", "tokenizer.json"))
	if err != nil {
		logger.Error("load tokenizer", "err", err)
		os.Exit(1)
	}
	meta, params, err := checkpoint.Load(*modelPath)
	if err != nil {
		logger.Error("load checkpoint", "err", err)
		os.Exit(1)
	}
	model := checkpoint.LoadModel(meta, params)
	engine := inference.NewEngine(model, tokenizer)

	gsm8k := tasks.NewGSM8KFromRows([]tasks.GSM8KRow{
		{Question: "What is 2+2?", Answer: "#### 4"},
		{Question: "What is 3+3?", Answer: "#### 6"},
	})

	groups := model.SetupOptimizer(0.004, 0.2, 0.02, 0.0, 0.5, true)
	for index := range groups {
		groups[index].LR *= 0.05
	}
	optimizer := optimizer.NewMuonAdamW(groups)

	for step := 0; step < *numSteps; step++ {
		conversation := gsm8k.GetExample(step % gsm8k.NumExamples())
		prompt := tokenizer.RenderForCompletion(conversation)
		results, _ := engine.GenerateBatch(prompt, *numSamples, 32, 1.0, 50, uint64(step))

		rewards := make([]float32, len(results))
		for sampleIndex, rollout := range results {
			completion := tokenizer.Decode(rollout[len(prompt):])
			rewards[sampleIndex] = gsm8k.Reward(conversation, completion)
		}
		var mean float32
		for _, reward := range rewards {
			mean += reward
		}
		mean /= float32(len(rewards))
		advantages := make([]float32, len(rewards))
		for sampleIndex, reward := range rewards {
			advantages[sampleIndex] = reward - mean
		}

		// Build a padded batch of rollouts.
		maxLen := 0
		for _, rollout := range results {
			if len(rollout) > maxLen {
				maxLen = len(rollout)
			}
		}
		padToken := tokenizer.EncodeSpecial("<|assistant_end|>")
		inputs := make([][]int, len(results))
		targets := make([][]int, len(results))
		for sampleIndex, rollout := range results {
			padded := append([]int(nil), rollout...)
			for len(padded) < maxLen {
				padded = append(padded, padToken)
			}
			inputs[sampleIndex] = padded[:len(padded)-1]
			targets[sampleIndex] = padded[1:]
			// Mask forced/prompt tokens: only train on sampled tokens.
			promptLength := len(prompt)
			for targetIndex := range targets[sampleIndex] {
				if targetIndex < promptLength {
					targets[sampleIndex][targetIndex] = -1
				}
			}
		}
		inputTensor := int32sFrom(inputs)
		targetTensor := int32sFrom(targets)
		loss := trainer.RLStep(model, optimizer, inputTensor, targetTensor, advantages, float32(maxLen))
		logger.Info("rl", "step", step, "loss", fmt.Sprintf("%.4f", loss), "mean_reward", fmt.Sprintf("%.2f", mean))
	}

	if *outPath != "" {
		outMeta := checkpoint.Meta{Step: *numSteps, ModelConfig: model.Config}
		if err := checkpoint.Save(*outPath, outMeta, model.NamedParameters()); err != nil {
			logger.Error("save", "err", err)
			os.Exit(1)
		}
	}
}

func int32sFrom(rows [][]int) *tensors.Int32s {
	shape := []int{len(rows), len(rows[0])}
	values := make([]int32, 0, len(rows)*len(rows[0]))
	for _, row := range rows {
		for _, value := range row {
			values = append(values, int32(value))
		}
	}
	return tensors.NewInt32sWithData(shape, values)
}
