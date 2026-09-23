// Command chat_rl runs reinforcement learning (GRPO/REINFORCE) on GSM8K.
package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"

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
	effortLevels := flag.String("efforts", "50,75,100", "comma-separated reasoning-effort levels b in [1,100] (empty disables effort conditioning)")
	samplesPerEffort := flag.Int("samples-per-effort", 4, "rollouts per effort level per example")
	penaltyK0 := flag.Float64("penalty-k0", 0.1, "basic length-penalty coefficient at the minimum effort")
	penaltyLambda := flag.Float64("penalty-lambda", 1.0, "rate of exponential penalty decay")
	penaltyCap := flag.Float64("penalty-cap", 0.5, "maximum length deduction per trajectory")
	penaltyNorm := flag.Float64("penalty-norm", 256, "reference reasoning length for the penalty")
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

	efforts := parseEfforts(*effortLevels)
	penaltyConfig := trainer.EffortPenaltyConfig{
		K0: float32(*penaltyK0), Lambda: float32(*penaltyLambda),
		Cap: float32(*penaltyCap), Norm: float32(*penaltyNorm),
		MinEffort: 50, MeanDeltaEffort: 25,
	}
	if len(efforts) > 0 {
		penaltyConfig.MinEffort = efforts[0]
		if len(efforts) > 1 {
			penaltyConfig.MeanDeltaEffort = float32(efforts[len(efforts)-1]-efforts[0]) / float32(len(efforts)-1)
		}
	}

	for step := 0; step < *numSteps; step++ {
		conversation := gsm8k.GetExample(step % gsm8k.NumExamples())

		var results [][]int
		var promptLengths []int
		var rewards []float32
		var groupSizes []int
		if len(efforts) == 0 {
			prompt := tokenizer.RenderForCompletion(conversation)
			rollouts, _ := engine.GenerateBatch(prompt, *numSamples, 32, 1.0, 50, uint64(step))
			for _, rollout := range rollouts {
				completion := tokenizer.Decode(rollout[len(prompt):])
				rewards = append(rewards, gsm8k.Reward(conversation, completion))
				results = append(results, rollout)
				promptLengths = append(promptLengths, len(prompt))
			}
			groupSizes = append(groupSizes, len(rollouts))
		} else {
			for groupIndex, effort := range efforts {
				convCopy := *conversation
				convCopy.Extra = map[string]any{"effort": effort}
				prompt := tokenizer.RenderForCompletion(&convCopy)
				rollouts, _ := engine.GenerateBatch(prompt, *samplesPerEffort, 32, 1.0, 50, uint64(step*1000+groupIndex))
				for _, rollout := range rollouts {
					completion := tokenizer.Decode(rollout[len(prompt):])
					reward := gsm8k.Reward(conversation, completion)
					reward += trainer.ExponentialTokenPenalty(effort, len(rollout)-len(prompt), penaltyConfig)
					rewards = append(rewards, reward)
					results = append(results, rollout)
					promptLengths = append(promptLengths, len(prompt))
				}
				groupSizes = append(groupSizes, len(rollouts))
			}
		}
		if len(results) == 0 {
			continue
		}

		// Group-relative advantages: mean-center within each (example, effort)
		// subgroup (DeepSeek-V4.1 §5.1.4).
		advantages := make([]float32, len(rewards))
		offset := 0
		for _, size := range groupSizes {
			if size == 0 {
				continue
			}
			var groupMean float32
			for index := 0; index < size; index++ {
				groupMean += rewards[offset+index]
			}
			groupMean /= float32(size)
			for index := 0; index < size; index++ {
				advantages[offset+index] = rewards[offset+index] - groupMean
			}
			offset += size
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
			promptLength := promptLengths[sampleIndex]
			for targetIndex := range targets[sampleIndex] {
				if targetIndex < promptLength {
					targets[sampleIndex][targetIndex] = -1
				}
			}
		}
		inputTensor := int32sFrom(inputs)
		targetTensor := int32sFrom(targets)
		loss := trainer.RLStep(model, optimizer, inputTensor, targetTensor, advantages, float32(maxLen))
		logger.Info("rl", "step", step, "loss", fmt.Sprintf("%.4f", loss), "rollouts", len(results))
	}

	if *outPath != "" {
		outMeta := checkpoint.Meta{Step: *numSteps, ModelConfig: model.Config}
		if err := checkpoint.Save(*outPath, outMeta, model.NamedParameters()); err != nil {
			logger.Error("save", "err", err)
			os.Exit(1)
		}
	}
}

// parseEfforts parses a comma-separated list of effort levels. It returns nil
// when the spec is empty, disabling effort conditioning.
func parseEfforts(spec string) []int {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return nil
	}
	efforts := make([]int, 0)
	for _, field := range strings.Split(spec, ",") {
		value, err := strconv.Atoi(strings.TrimSpace(field))
		if err == nil {
			efforts = append(efforts, value)
		}
	}
	return efforts
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
