// Command chat_sft runs supervised fine-tuning over conversation datasets.
package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/cookiengineer/gonano/data"
	"github.com/cookiengineer/gonano/internal/logging"
	"github.com/cookiengineer/gonano/model/checkpoint"
	tokenizerpkg "github.com/cookiengineer/gonano/tokenizer"
	"github.com/cookiengineer/gonano/trainer"
)

func main() {
	modelPath := flag.String("model", "", "path to a .gn base checkpoint (required)")
	numIterations := flag.Int("num-iterations", 200, "optimization steps")
	batchSize := flag.Int("batch-size", 1, "batch size")
	maxSeqLen := flag.Int("max-seq-len", 512, "max sequence length")
	outPath := flag.String("out", "", "output checkpoint path")
	baseDir := flag.String("base-dir", "", "tokenizer directory")
	dataDir := flag.String("data-dir", "", "JSONL conversation file or directory (reasoning traces); empty uses the synthetic demo set")
	dataFormat := flag.String("data-format", "jsonl", "conversation data format (jsonl)")
	thinkingStyle := flag.String("thinking-style", "", "optional thinking style instruction (e.g. logical); applied to every loaded conversation")
	domain := flag.String("domain", "", "domain name for the model bank; recorded in the checkpoint (default output goes to domains/<name>/chatsft_checkpoints/)")
	flag.Parse()

	if *dataFormat != "jsonl" {
		fmt.Fprintln(os.Stderr, "chat_sft: unsupported --data-format (only jsonl)")
		os.Exit(1)
	}

	logger := logging.Default(slog.LevelInfo)
	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: chat_sft --model <path> --out <path>")
		os.Exit(1)
	}
	if *baseDir == "" {
		*baseDir = data.BaseDir()
	}
	tokenizer, err := tokenizerpkg.LoadTokenizer(filepath.Join(*baseDir, "tokenizer", "tokenizer.json"))
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

	provider, closeSource, err := conversationProvider(*dataDir, *batchSize, *thinkingStyle, logger)
	if err != nil {
		logger.Error("open conversations", "err", err)
		os.Exit(1)
	}
	defer closeSource()
	loader := data.NewSFTLoader(tokenizer, *batchSize, *maxSeqLen, provider, 100)

	groups := model.SetupOptimizer(0.008, 0.2, 0.02, 0.0, 0.5, true)
	trainer.TrainSFT(model, groups, loader, *numIterations, func(step int, loss float32) {
		if step%20 == 0 {
			logger.Info("sft", "step", step, "loss", fmt.Sprintf("%.4f", loss))
		}
	})

	if *outPath == "" && *domain != "" {
		*outPath = filepath.Join(*baseDir, "domains", *domain, "chatsft_checkpoints", "model_final.gn")
	}
	if *outPath != "" {
		if err := os.MkdirAll(filepath.Dir(*outPath), 0o755); err != nil {
			logger.Error("create output directory", "err", err)
			os.Exit(1)
		}
		outMeta := checkpoint.Meta{Step: *numIterations, ModelConfig: model.Config}
		if *domain != "" {
			outMeta.UserConfig = map[string]any{"domain": *domain}
		}
		if err := checkpoint.Save(*outPath, outMeta, model.NamedParameters()); err != nil {
			logger.Error("save", "err", err)
			os.Exit(1)
		}
		logger.Info("saved SFT checkpoint", "path", *outPath, "domain", *domain)
	}
}

// conversationProvider returns the SFT conversation provider. When dataDir is
// empty it falls back to the synthetic demonstration set; otherwise it streams
// the JSONL reasoning conversations, optionally tagging each with a thinking
// style instruction.
func conversationProvider(dataDir string, batchSize int, thinkingStyle string, logger *slog.Logger) (data.ConvProvider, func(), error) {
	if dataDir == "" {
		conversations := []*tokenizerpkg.Conversation{
			{Messages: []tokenizerpkg.Message{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "hello"}}},
			{Messages: []tokenizerpkg.Message{{Role: "user", Content: "how are you"}, {Role: "assistant", Content: "fine thanks"}}},
		}
		index := 0
		provider := func() ([]*tokenizerpkg.Conversation, bool) {
			conversation := conversations[index%len(conversations)]
			index++
			return []*tokenizerpkg.Conversation{conversation}, true
		}
		return provider, func() {}, nil
	}

	source, err := data.OpenConversations(dataDir, batchSize)
	if err != nil {
		return nil, func() {}, err
	}
	provider := func() ([]*tokenizerpkg.Conversation, bool) {
		batch, ok := source.Next()
		if ok && thinkingStyle != "" {
			for _, conversation := range batch {
				if conversation.Extra == nil {
					conversation.Extra = map[string]any{}
				}
				conversation.Extra["thinking_style"] = thinkingStyle
			}
		}
		if err := source.Err(); err != nil {
			logger.Error("conversation source", "err", err)
		}
		return batch, ok
	}
	return provider, func() { source.Close() }, nil
}
