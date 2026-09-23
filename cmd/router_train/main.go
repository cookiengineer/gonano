// Command router_train trains the domain meta-router for the domain model
// bank: it collects labeled examples from per-domain data directories, trains
// the transformer classifier head (optionally after training a small encoder
// LM), distills the n-gram fast path from it, and writes a router bundle.
//
// Example:
//
//	go run ./cmd/router_train \
//	  --domain wikipedia=~/data/wikipedia \
//	  --domain physics=~/data/physics.stackexchange \
//	  --data-format markdown \
//	  --encoder ~/.cache/gonano/router/encoder.gn \
//	  --out ~/.cache/gonano/router/router.gn
package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/cookiengineer/gonano/data"
	"github.com/cookiengineer/gonano/internal/logging"
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/model/checkpoint"
	"github.com/cookiengineer/gonano/router"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
	"github.com/cookiengineer/gonano/trainer"
)

// domainSpec is one `--domain name=dir` entry.
type domainSpec struct {
	name string
	dir  string
}

// domainList collects repeated --domain flags.
type domainList []domainSpec

func (list *domainList) String() string {
	parts := make([]string, len(*list))
	for index, spec := range *list {
		parts[index] = spec.name + "=" + spec.dir
	}
	return strings.Join(parts, ",")
}

func (list *domainList) Set(value string) error {
	parts := strings.SplitN(value, "=", 2)
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return fmt.Errorf("expected name=dir, got %q", value)
	}
	*list = append(*list, domainSpec{name: parts[0], dir: parts[1]})
	return nil
}

func main() {
	var domains domainList
	flag.Var(&domains, "domain", "domain spec name=dir (repeatable)")
	dataFormat := flag.String("data-format", "markdown", "data format: parquet|markdown")
	baseDir := flag.String("base-dir", "", "base directory (default ~/.cache/gonano)")
	tokenizerPath := flag.String("tokenizer", "", "path to tokenizer.json (default <base-dir>/tokenizer/tokenizer.json)")
	encoderPath := flag.String("encoder", "", "path to a pre-trained encoder checkpoint; empty trains a small encoder LM")
	outPath := flag.String("out", "", "router bundle output (default <base-dir>/router/router.gn)")
	maxExamples := flag.Int("max-examples-per-domain", 2000, "maximum labeled examples per domain")
	maxSeqLen := flag.Int("max-seq-len", 128, "router input truncation length")
	headEpochs := flag.Int("head-epochs", 20, "transformer head training epochs")
	ngramEpochs := flag.Int("ngram-epochs", 15, "n-gram distillation epochs")
	ngramBuckets := flag.Int("ngram-buckets", 1<<16, "n-gram hashing table size")
	ngramOnly := flag.Bool("ngram-only", false, "train only the fast path (no transformer head)")
	distill := flag.Bool("distill", false, "train the n-gram fast path by distilling the teacher instead of supervised labels")
	fineTune := flag.Bool("fine-tune", false, "fine-tune the encoder jointly with the head (slower, better features)")
	fineTuneEpochs := flag.Int("fine-tune-epochs", 3, "encoder fine-tuning epochs")
	fineTuneLR := flag.Float64("fine-tune-lr", 1e-3, "encoder fine-tuning learning rate")
	encoderDepth := flag.Int("encoder-depth", 2, "depth of the encoder LM trained when --encoder is empty")
	encoderSteps := flag.Int("encoder-steps", 200, "LM steps for the self-trained encoder")
	flag.Parse()

	if *baseDir == "" {
		*baseDir = data.BaseDir()
	}
	logger := logging.Default(slog.LevelInfo)

	if len(domains) < 2 {
		logger.Error("at least two --domain name=dir entries are required")
		os.Exit(1)
	}
	if *outPath == "" {
		*outPath = filepath.Join(*baseDir, "router", "router.gn")
	}
	if *tokenizerPath == "" {
		*tokenizerPath = filepath.Join(*baseDir, "tokenizer", "tokenizer.json")
	}

	tok, err := tokenizer.LoadTokenizer(*tokenizerPath)
	if err != nil {
		logger.Error("load tokenizer", "path", *tokenizerPath, "err", err)
		os.Exit(1)
	}

	names := make([]string, len(domains))
	examples := make([]router.Example, 0, len(domains)*(*maxExamples))
	sources := make([]func() ([]string, data.State), 0, len(domains))
	for index, spec := range domains {
		names[index] = spec.name
		next, err := domainSource(spec.dir, *dataFormat)
		if err != nil {
			logger.Error("open domain data", "domain", spec.name, "dir", spec.dir, "err", err)
			os.Exit(1)
		}
		sources = append(sources, next)
		collected := collectExamples(tok, next, index, *maxExamples, *maxSeqLen)
		logger.Info("collected examples", "domain", spec.name, "count", len(collected))
		examples = append(examples, collected...)
	}
	if len(examples) < 2 {
		logger.Error("not enough training examples", "count", len(examples))
		os.Exit(1)
	}

	shuffle(examples, 12345)

	ngramConfig := router.NgramConfig{Epochs: *ngramEpochs, Buckets: *ngramBuckets, Seed: 7}
	if *ngramOnly {
		ngram := router.TrainNgram(examples, len(names), ngramConfig)
		bundle := &router.Router{Domains: names, Ngram: ngram}
		if err := bundle.Save(*outPath); err != nil {
			logger.Error("save router", "err", err)
			os.Exit(1)
		}
		logger.Info("saved fast-path router", "path", *outPath, "domains", names)
		return
	}

	encoder, encoderSource, err := resolveEncoder(*encoderPath, *baseDir, *encoderDepth, *encoderSteps, *maxSeqLen, tok, sources, logger)
	if err != nil {
		logger.Error("prepare encoder", "err", err)
		os.Exit(1)
	}

	teacher := router.NewTransformerClassifier(encoder, len(names), *maxSeqLen)
	if *fineTune {
		teacher.FineTune(examples, *fineTuneEpochs, float32(*fineTuneLR), 3)
		saved, saveErr := saveEncoder(*baseDir, encoder)
		if saveErr != nil {
			logger.Error("save fine-tuned encoder", "err", saveErr)
			os.Exit(1)
		}
		encoderSource = saved
		logger.Info("fine-tuned encoder", "path", saved, "epochs", *fineTuneEpochs)
	} else {
		teacher.Train(examples, router.HeadConfig{Epochs: *headEpochs, LearningRate: 0.05, BatchSize: 32, Seed: 3})
	}

	var ngram *router.NgramClassifier
	if *distill {
		ngram = router.DistillNgram(teacher, examples, ngramConfig)
		logger.Info("distilled fast path", "fidelity", fmt.Sprintf("%.3f", router.DistillAccuracy(teacher, ngram, examples)))
	} else {
		// The per-domain directories label the data, so the fast path is
		// trained supervised by default; --distill uses the teacher instead.
		ngram = router.TrainNgram(examples, len(names), ngramConfig)
	}

	bundle := &router.Router{Domains: names, Ngram: ngram, Transformer: teacher, EncoderPath: encoderSource}
	if err := bundle.Save(*outPath); err != nil {
		logger.Error("save router", "err", err)
		os.Exit(1)
	}
	logger.Info("saved router", "path", *outPath, "domains", names)
}

// domainSource builds a document source for one domain directory.
func domainSource(dir, format string) (func() ([]string, data.State), error) {
	switch format {
	case "parquet":
		paths := data.ListParquetFiles(dir)
		if len(paths) == 0 {
			return nil, fmt.Errorf("no parquet files in %s", dir)
		}
		source := data.NewParquetSource(paths, 64)
		return func() ([]string, data.State) { return source.Next() }, nil
	default: // markdown
		source := data.NewMarkdownSource(dir, 64)
		if source.NumFiles() == 0 {
			return nil, fmt.Errorf("no markdown files in %s", dir)
		}
		return func() ([]string, data.State) { return source.Next() }, nil
	}
}

// collectExamples samples up to limit encoded documents from a source.
func collectExamples(tok *tokenizer.Tokenizer, next func() ([]string, data.State), label, limit, maxSeqLen int) []router.Example {
	examples := make([]router.Example, 0, limit)
	guard := 0
	for len(examples) < limit && guard < limit*8 {
		guard++
		documents, _ := next()
		if len(documents) == 0 {
			break
		}
		for _, document := range documents {
			tokens := tok.Encode(document)
			if len(tokens) == 0 {
				continue
			}
			if len(tokens) > maxSeqLen {
				tokens = tokens[:maxSeqLen]
			}
			examples = append(examples, router.Example{Tokens: toI32(tokens), Label: label})
			if len(examples) >= limit {
				break
			}
		}
	}
	return examples
}

// resolveEncoder either loads the given encoder checkpoint or trains a small
// encoder LM on the combined domain documents.
func resolveEncoder(encoderPath, baseDir string, depth, steps, maxSeqLen int, tok *tokenizer.Tokenizer, sources []func() ([]string, data.State), logger *slog.Logger) (*model.Transformer, string, error) {
	if encoderPath != "" {
		meta, parameters, err := checkpoint.LoadAny(encoderPath)
		if err != nil {
			return nil, "", err
		}
		return checkpoint.LoadModel(meta, parameters), encoderPath, nil
	}

	logger.Info("training encoder LM", "depth", depth, "steps", steps)
	configuration := model.ConfigForDepth(depth, tok.VocabSize(), 64, 64, maxSeqLen, "SSSL")
	encoder := model.NewTransformer(configuration)
	encoder.InitWeights(tensors.NewRNG(1))
	groups := encoder.SetupOptimizer(0.01, 0.1, 0.02, 0.28, 0.5, false)
	training := trainer.NewTrainer(encoder, groups, 1)
	loader := data.NewPretrainLoader(tok, 1, maxSeqLen, combinedProvider(sources), 1000)
	for step := 0; step < steps; step++ {
		inputs, targets, _ := loader.Next()
		training.TrainStep(inputs, targets)
		training.StepOptimizer(step, steps)
	}

	path, err := saveEncoder(baseDir, encoder)
	if err != nil {
		return nil, "", err
	}
	return encoder, path, nil
}

// saveEncoder persists an encoder checkpoint under <base-dir>/router/encoder.gn
// so a router bundle can reload the exact encoder it was trained with.
func saveEncoder(baseDir string, encoder *model.Transformer) (string, error) {
	directory := filepath.Join(baseDir, "router")
	if err := os.MkdirAll(directory, 0o755); err != nil {
		return "", err
	}
	path := filepath.Join(directory, "encoder.gn")
	meta := checkpoint.Meta{ModelConfig: encoder.Config}
	if err := checkpoint.Save(path, meta, encoder.NamedParameters()); err != nil {
		return "", err
	}
	return path, nil
}

// combinedProvider cycles documents from every domain source so the encoder LM
// sees all domains.
func combinedProvider(sources []func() ([]string, data.State)) data.DocProvider {
	index := 0
	return func() ([]string, data.State) {
		for tries := 0; tries < len(sources); tries++ {
			source := sources[index%len(sources)]
			index++
			documents, state := source()
			if len(documents) > 0 {
				return documents, state
			}
		}
		return nil, data.State{}
	}
}

// shuffle deterministically permutes examples.
func shuffle(examples []router.Example, seed uint64) {
	random := tensors.NewRNG(seed)
	for index := len(examples) - 1; index > 0; index-- {
		swap := random.IntN(index + 1)
		examples[index], examples[swap] = examples[swap], examples[index]
	}
}

// toI32 converts token ids.
func toI32(ids []int) []int32 {
	converted := make([]int32, len(ids))
	for index, value := range ids {
		converted[index] = int32(value)
	}
	return converted
}
