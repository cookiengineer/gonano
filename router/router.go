// Package router implements the domain meta-router for the gonano domain model
// bank (architecture A): a fast hashed n-gram linear classifier that is
// optionally escalated to a small transformer classifier when its confidence is
// low. The router maps a token sequence to a probability distribution over the
// bank's semantic domains, so only the relevant domain models need to be kept
// resident and executed.
//
// The n-gram fast path is distilled from the transformer teacher; see
// DistillNgram. A Router bundle is serialized with model/checkpoint so it
// shares the project's single persistence format.
package router

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/model/checkpoint"
	"github.com/cookiengineer/gonano/model/layers"
	"github.com/cookiengineer/gonano/tensors"
)

// Example is one labeled router training item. Tokens are token ids from the
// bank's shared tokenizer.
type Example struct {
	Tokens []int32
	Label  int
}

// Score is one domain's routing probability.
type Score struct {
	Domain string
	Value  float32
}

// Router classifies a token sequence into the bank's domains. It always has an
// n-gram fast path; the transformer classifier is optional.
type Router struct {
	// Domains is the ordered domain list; probability indices align with it.
	Domains []string
	// Ngram is the fast path (required).
	Ngram *NgramClassifier
	// Transformer is the meta model (optional). When nil, the router is
	// fast-path only.
	Transformer *TransformerClassifier
	// EncoderPath records the transformer encoder checkpoint so the bundle can
	// be reloaded. It is informational for an in-memory router.
	EncoderPath string
	// Threshold is the top-1 minus top-2 probability margin below which the
	// router escalates from the n-gram fast path to the transformer. Zero
	// disables escalation (fast path always); a negative value always
	// escalates.
	Threshold float32
}

// routerConfig is the JSON metadata stored in a router bundle.
type routerConfig struct {
	Domains       []string `json:"domains"`
	Threshold     float32  `json:"threshold"`
	EncoderPath   string   `json:"encoder_path,omitempty"`
	NgramBuckets  int      `json:"ngram_buckets"`
	NgramUnigrams bool     `json:"ngram_unigrams"`
	NgramBigrams  bool     `json:"ngram_bigrams"`
	NgramTrigrams bool     `json:"ngram_trigrams"`
	MaxLen        int      `json:"max_len,omitempty"`
	HasHead       bool     `json:"has_head"`
}

// NumClasses returns the number of domains.
func (router *Router) NumClasses() int { return len(router.Domains) }

// Predict returns the probability distribution over domains for a token
// sequence. It runs the n-gram fast path and escalates to the transformer when
// configured and the fast path's top-1/top-2 margin is below Threshold.
func (router *Router) Predict(tokens []int32) []float32 {
	if router.Ngram == nil {
		panic("router: no classifier configured")
	}
	probabilities := router.Ngram.Predict(tokens)
	if router.Transformer != nil && router.shouldEscalate(probabilities) {
		transformerProbabilities := router.Transformer.Predict(tokens)
		if len(transformerProbabilities) == len(probabilities) {
			return transformerProbabilities
		}
	}
	return probabilities
}

// shouldEscalate reports whether the transformer should be consulted for the
// given fast-path probabilities.
func (router *Router) shouldEscalate(probabilities []float32) bool {
	if router.Threshold < 0 {
		return true
	}
	if router.Threshold == 0 {
		return false
	}
	first, second := topTwo(probabilities)
	return first-second < router.Threshold
}

// Route returns the domain scores sorted by descending probability.
func (router *Router) Route(tokens []int32) []Score {
	probabilities := router.Predict(tokens)
	scores := make([]Score, len(probabilities))
	for index, value := range probabilities {
		domain := ""
		if index < len(router.Domains) {
			domain = router.Domains[index]
		}
		scores[index] = Score{Domain: domain, Value: value}
	}
	sort.SliceStable(scores, func(left, right int) bool { return scores[left].Value > scores[right].Value })
	return scores
}

// TopDomains returns up to count domains whose probability is at least
// minScore, highest first. It always returns at least the single best domain.
func (router *Router) TopDomains(tokens []int32, count int, minScore float32) []Score {
	scores := router.Route(tokens)
	if len(scores) == 0 {
		return nil
	}
	if count < 1 {
		count = 1
	}
	selected := make([]Score, 0, count)
	for _, score := range scores {
		if len(selected) >= count {
			break
		}
		if len(selected) > 0 && score.Value < minScore {
			break
		}
		selected = append(selected, score)
	}
	return selected
}

// Save writes the router bundle (n-gram weights, optional transformer head, and
// metadata) to path using the shared checkpoint format.
func (router *Router) Save(path string) error {
	if router.Ngram == nil {
		return fmt.Errorf("router: nothing to save")
	}
	parameters := map[string]*tensors.Tensor{
		"router.ngram.weight": router.Ngram.Weight,
		"router.ngram.bias":   router.Ngram.Bias,
	}
	configuration := routerConfig{
		Domains:       router.Domains,
		Threshold:     router.Threshold,
		EncoderPath:   router.EncoderPath,
		NgramBuckets:  router.Ngram.Buckets,
		NgramUnigrams: router.Ngram.Unigrams,
		NgramBigrams:  router.Ngram.Bigrams,
		NgramTrigrams: router.Ngram.Trigrams,
	}
	if router.Transformer != nil {
		parameters["router.head.weight"] = router.Transformer.Head.Weight
		parameters["router.head.bias"] = router.Transformer.Bias
		configuration.MaxLen = router.Transformer.MaxLen
		configuration.HasHead = true
	}
	encoded, err := json.Marshal(configuration)
	if err != nil {
		return err
	}
	meta := checkpoint.Meta{UserConfig: map[string]any{"router_config": string(encoded)}}
	return checkpoint.Save(path, meta, parameters)
}

// LoadRouter reads a router bundle. encoder is required when the bundle carries
// a transformer head and may be nil for a fast-path-only bundle. The encoder
// must match the checkpoint used at training time.
func LoadRouter(path string, encoder *model.Transformer) (*Router, error) {
	meta, parameters, err := checkpoint.Load(path)
	if err != nil {
		return nil, err
	}
	raw, ok := meta.UserConfig["router_config"].(string)
	if !ok {
		return nil, fmt.Errorf("router: %s is not a router bundle", path)
	}
	var configuration routerConfig
	if err := json.Unmarshal([]byte(raw), &configuration); err != nil {
		return nil, err
	}
	weight, ok := parameters["router.ngram.weight"]
	if !ok {
		return nil, fmt.Errorf("router: bundle missing n-gram weights")
	}
	bias := parameters["router.ngram.bias"]
	classifier := &NgramClassifier{
		Classes:  weight.Shape[0],
		Buckets:  weight.Shape[1],
		Unigrams: configuration.NgramUnigrams,
		Bigrams:  configuration.NgramBigrams,
		Trigrams: configuration.NgramTrigrams,
		Weight:   weight,
		Bias:     bias,
	}
	router := &Router{
		Domains:     configuration.Domains,
		Ngram:       classifier,
		Threshold:   configuration.Threshold,
		EncoderPath: configuration.EncoderPath,
	}
	if configuration.HasHead {
		if encoder == nil {
			return nil, fmt.Errorf("router: bundle requires an encoder checkpoint, none supplied")
		}
		headWeight, okWeight := parameters["router.head.weight"]
		headBias, okBias := parameters["router.head.bias"]
		if !okWeight || !okBias {
			return nil, fmt.Errorf("router: bundle missing transformer head")
		}
		if headWeight.Shape[1] != encoder.Config.EmbedDim {
			return nil, fmt.Errorf("router: head dim %d does not match encoder dim %d",
				headWeight.Shape[1], encoder.Config.EmbedDim)
		}
		if headWeight.Shape[0] != len(configuration.Domains) {
			return nil, fmt.Errorf("router: head classes %d does not match %d domains",
				headWeight.Shape[0], len(configuration.Domains))
		}
		router.Transformer = &TransformerClassifier{
			Encoder: encoder,
			Head: &layers.Linear{
				InFeatures:  encoder.Config.EmbedDim,
				OutFeatures: headWeight.Shape[0],
				Weight:      headWeight,
			},
			Bias:   headBias,
			MaxLen: configuration.MaxLen,
		}
	}
	return router, nil
}

// LoadRouterAuto loads a router bundle and, when it carries a transformer head,
// automatically loads the encoder checkpoint recorded in the bundle. It is the
// convenience entry point for serving.
func LoadRouterAuto(path string) (*Router, error) {
	meta, _, err := checkpoint.Load(path)
	if err != nil {
		return nil, err
	}
	raw, ok := meta.UserConfig["router_config"].(string)
	if !ok {
		return nil, fmt.Errorf("router: %s is not a router bundle", path)
	}
	var configuration routerConfig
	if err := json.Unmarshal([]byte(raw), &configuration); err != nil {
		return nil, err
	}
	var encoder *model.Transformer
	if configuration.HasHead {
		if configuration.EncoderPath == "" {
			return nil, fmt.Errorf("router: bundle has a head but no encoder path")
		}
		encoderMeta, encoderParameters, err := checkpoint.LoadAny(configuration.EncoderPath)
		if err != nil {
			return nil, fmt.Errorf("router: load encoder %s: %w", configuration.EncoderPath, err)
		}
		encoder = checkpoint.LoadModel(encoderMeta, encoderParameters)
	}
	return LoadRouter(path, encoder)
}

// softmax returns the numerically stable softmax of logits.
func softmax(logits []float32) []float32 {
	maximum := float32(math.Inf(-1))
	for _, value := range logits {
		if value > maximum {
			maximum = value
		}
	}
	probabilities := make([]float32, len(logits))
	var sum float64
	for index, value := range logits {
		weight := math.Exp(float64(value - maximum))
		probabilities[index] = float32(weight)
		sum += weight
	}
	if sum == 0 {
		return probabilities
	}
	inverse := float32(1 / sum)
	for index := range probabilities {
		probabilities[index] *= inverse
	}
	return probabilities
}

// argMax returns the index of the largest value.
func argMax(values []float32) int {
	best := 0
	for index := 1; index < len(values); index++ {
		if values[index] > values[best] {
			best = index
		}
	}
	return best
}

// topTwo returns the largest and second-largest values (second is -inf when
// there is only one value).
func topTwo(values []float32) (float32, float32) {
	first, second := float32(math.Inf(-1)), float32(math.Inf(-1))
	for _, value := range values {
		switch {
		case value > first:
			second = first
			first = value
		case value > second:
			second = value
		}
	}
	return first, second
}
