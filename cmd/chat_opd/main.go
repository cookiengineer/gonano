// Command chat_opd runs full-vocabulary on-policy distillation: the student
// generates rollouts, a frozen teacher scores them, and the student is trained
// to match the teacher's distribution (DeepSeek-V4.1 §5.2.4).
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
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
	"github.com/cookiengineer/gonano/trainer"
)

func main() {
	modelPath := flag.String("model", "", "path to the student .gn checkpoint (required)")
	teacherPath := flag.String("teacher", "", "path to the teacher .gn checkpoint (required)")
	numSteps := flag.Int("num-steps", 10, "optimization steps")
	numSamples := flag.Int("num-samples", 4, "on-policy rollouts per example")
	maxTokens := flag.Int("max-tokens", 32, "tokens to generate per rollout")
	samplingTemperature := flag.Float64("temperature", 1.0, "rollout sampling temperature")
	distillTemperature := flag.Float64("distill-temperature", 1.0, "distillation softmax temperature")
	outPath := flag.String("out", "", "output checkpoint path")
	baseDir := flag.String("base-dir", "", "tokenizer directory")
	domain := flag.String("domain", "", "domain name for the model bank; recorded in the checkpoint (default output goes to domains/<name>/chatopd_checkpoints/)")
	flag.Parse()

	logger := logging.Default(slog.LevelInfo)
	if *modelPath == "" || *teacherPath == "" {
		fmt.Fprintln(os.Stderr, "usage: chat_opd --model <student.gn> --teacher <teacher.gn>")
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

	studentMeta, studentParams, err := checkpoint.Load(*modelPath)
	if err != nil {
		logger.Error("load student", "err", err)
		os.Exit(1)
	}
	student := checkpoint.LoadModel(studentMeta, studentParams)

	teacherMeta, teacherParams, err := checkpoint.Load(*teacherPath)
	if err != nil {
		logger.Error("load teacher", "err", err)
		os.Exit(1)
	}
	teacher := checkpoint.LoadModel(teacherMeta, teacherParams)
	if teacher.Config.VocabSize != student.Config.VocabSize {
		logger.Error("student and teacher vocabularies must match",
			"student", student.Config.VocabSize, "teacher", teacher.Config.VocabSize)
		os.Exit(1)
	}

	engine := inference.NewEngine(student, tokenizer)

	gsm8k := tasks.NewGSM8KFromRows([]tasks.GSM8KRow{
		{Question: "What is 2+2?", Answer: "#### 4"},
		{Question: "What is 3+3?", Answer: "#### 6"},
	})

	groups := student.SetupOptimizer(0.008, 0.2, 0.02, 0.0, 0.5, true)
	for index := range groups {
		groups[index].LR *= 0.05
	}

	taskIndex := 0
	provider := func() (*tensors.Int32s, *tensors.Int32s, bool) {
		conversation := gsm8k.GetExample(taskIndex % gsm8k.NumExamples())
		taskIndex++
		prompt := tokenizer.RenderForCompletion(conversation)
		results, _ := engine.GenerateBatch(prompt, *numSamples, *maxTokens, float32(*samplingTemperature), 50, uint64(taskIndex))
		return rolloutBatch(prompt, results)
	}

	trainer.Distill(student, teacher, groups, provider, *numSteps, float32(*distillTemperature), func(step int, loss float32) {
		logger.Info("opd", "step", step, "loss", fmt.Sprintf("%.4f", loss))
	})

	if *outPath == "" && *domain != "" {
		*outPath = filepath.Join(*baseDir, "domains", *domain, "chatopd_checkpoints", "model_final.gn")
	}
	if *outPath != "" {
		if err := os.MkdirAll(filepath.Dir(*outPath), 0o755); err != nil {
			logger.Error("create output directory", "err", err)
			os.Exit(1)
		}
		outMeta := checkpoint.Meta{Step: *numSteps, ModelConfig: student.Config}
		if *domain != "" {
			outMeta.UserConfig = map[string]any{"domain": *domain}
		}
		if err := checkpoint.Save(*outPath, outMeta, student.NamedParameters()); err != nil {
			logger.Error("save", "err", err)
			os.Exit(1)
		}
		logger.Info("saved distilled checkpoint", "path", *outPath, "domain", *domain)
	}
}

// rolloutBatch turns prompt+completion rollouts into a padded [rows, T] input
// batch and a mask that supervises only the sampled completion tokens.
func rolloutBatch(prompt []int, rollouts [][]int) (*tensors.Int32s, *tensors.Int32s, bool) {
	if len(rollouts) == 0 {
		return nil, nil, false
	}
	maxLen := 0
	for _, rollout := range rollouts {
		if len(rollout) > maxLen {
			maxLen = len(rollout)
		}
	}
	if maxLen < 2 {
		maxLen = 2
	}
	columns := maxLen - 1
	inputs := make([]int32, len(rollouts)*columns)
	mask := make([]int32, len(rollouts)*columns)
	for row, rollout := range rollouts {
		realLen := len(rollout)
		for column := 0; column < columns; column++ {
			index := row*columns + column
			if column < realLen-1 {
				inputs[index] = int32(rollout[column])
			} else {
				inputs[index] = int32(rollout[realLen-1])
			}
			// The target at column+1 is a sampled completion token when it
			// follows the prompt and lies within the generated sequence; the
			// padding tail is masked.
			if column+1 >= len(prompt) && column+1 < realLen {
				mask[index] = 1
			}
		}
	}
	return tensors.NewInt32sWithData([]int{len(rollouts), columns}, inputs),
		tensors.NewInt32sWithData([]int{len(rollouts), columns}, mask), true
}
