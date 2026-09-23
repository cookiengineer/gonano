package router

import (
	"math"

	"github.com/cookiengineer/gonano/tensors"
)

// NgramConfig configures the fastText-style n-gram classifier.
type NgramConfig struct {
	// Buckets is the hashing table size. Zero uses 1<<16.
	Buckets int
	// Epochs is the number of passes over the training set. Zero uses 8.
	Epochs int
	// LearningRate is the Adam learning rate. Zero uses 0.05.
	LearningRate float32
	// L2 is the weight-decay coefficient. Zero uses 1e-5.
	L2 float32
	// Seed seeds the example shuffle and weight initialization.
	Seed uint64
	// Unigrams enables single-token features. Defaults to true when no n-gram
	// order is selected.
	Unigrams bool
	// Bigrams enables adjacent-token-pair features. Defaults to true when no
	// n-gram order is selected.
	Bigrams bool
	// Trigrams enables adjacent-token-triple features. Defaults to true when no
	// n-gram order is selected.
	Trigrams bool
}

// NgramClassifier is a linear softmax classifier over hashed token n-grams. It
// is the distilled fast path of the meta-router.
type NgramClassifier struct {
	Classes  int
	Buckets  int
	Unigrams bool
	Bigrams  bool
	Trigrams bool
	Weight   *tensors.Tensor // [classes, buckets]
	Bias     *tensors.Tensor // [classes]
}

// ngramDefaults fills unset configuration fields with the defaults.
func ngramDefaults(configuration NgramConfig) NgramConfig {
	if configuration.Buckets <= 0 {
		configuration.Buckets = 1 << 16
	}
	if configuration.Epochs <= 0 {
		configuration.Epochs = 8
	}
	if configuration.LearningRate <= 0 {
		configuration.LearningRate = 0.05
	}
	if configuration.L2 <= 0 {
		configuration.L2 = 1e-5
	}
	if !configuration.Unigrams && !configuration.Bigrams && !configuration.Trigrams {
		configuration.Unigrams = true
		configuration.Bigrams = true
		configuration.Trigrams = true
	}
	return configuration
}

// NewNgramClassifier allocates a zero-initialized n-gram classifier.
func NewNgramClassifier(classes int, configuration NgramConfig) *NgramClassifier {
	configuration = ngramDefaults(configuration)
	return &NgramClassifier{
		Classes:  classes,
		Buckets:  configuration.Buckets,
		Unigrams: configuration.Unigrams,
		Bigrams:  configuration.Bigrams,
		Trigrams: configuration.Trigrams,
		Weight:   tensors.New(classes, configuration.Buckets),
		Bias:     tensors.New(classes),
	}
}

// features returns the set of active feature buckets for a token sequence.
// Duplicates collapse, so the feature vector is binary.
func (classifier *NgramClassifier) features(tokens []int32) []int {
	buckets := make(map[int]struct{}, len(tokens)*2)
	if classifier.Unigrams {
		for _, token := range tokens {
			buckets[hashBucket(classifier.Buckets, 1, token)] = struct{}{}
		}
	}
	if classifier.Bigrams {
		for index := 0; index+1 < len(tokens); index++ {
			buckets[hashBucket(classifier.Buckets, 2, tokens[index], tokens[index+1])] = struct{}{}
		}
	}
	if classifier.Trigrams {
		for index := 0; index+2 < len(tokens); index++ {
			buckets[hashBucket(classifier.Buckets, 3, tokens[index], tokens[index+1], tokens[index+2])] = struct{}{}
		}
	}
	features := make([]int, 0, len(buckets))
	for bucket := range buckets {
		features = append(features, bucket)
	}
	return features
}

// logits computes the raw class scores for a feature set.
func (classifier *NgramClassifier) logits(features []int) []float32 {
	classes := classifier.Classes
	buckets := classifier.Buckets
	weight := classifier.Weight.Data
	output := make([]float32, classes)
	for _, feature := range features {
		for class := 0; class < classes; class++ {
			output[class] += weight[class*buckets+feature]
		}
	}
	for class := 0; class < classes; class++ {
		output[class] += classifier.Bias.Data[class]
	}
	return output
}

// Predict returns the softmax distribution over classes for a token sequence.
func (classifier *NgramClassifier) Predict(tokens []int32) []float32 {
	return softmax(classifier.logits(classifier.features(tokens)))
}

// TrainNgram trains an n-gram classifier on the labeled examples with sparse
// Adam updates (only the active feature columns are touched).
func TrainNgram(examples []Example, classes int, configuration NgramConfig) *NgramClassifier {
	configuration = ngramDefaults(configuration)
	classifier := NewNgramClassifier(classes, configuration)
	if len(examples) == 0 {
		return classifier
	}

	weightAdam := newSparseAdam(classifier.Weight.Data, configuration)
	biasAdam := newSparseAdam(classifier.Bias.Data, configuration)

	order := make([]int, len(examples))
	for index := range order {
		order[index] = index
	}
	random := tensors.NewRNG(configuration.Seed)
	for epoch := 0; epoch < configuration.Epochs; epoch++ {
		// Fisher-Yates shuffle with the deterministic RNG.
		for index := len(order) - 1; index > 0; index-- {
			swap := random.IntN(index + 1)
			order[index], order[swap] = order[swap], order[index]
		}
		for _, position := range order {
			example := examples[position]
			features := classifier.features(example.Tokens)
			probabilities := softmax(classifier.logits(features))
			weightAdam.beginStep()
			biasAdam.beginStep()

			buckets := classifier.Buckets
			weight := classifier.Weight.Data
			for class := 0; class < classes; class++ {
				gradient := probabilities[class]
				if class == example.Label {
					gradient -= 1
				}
				biasAdam.apply(gradient, class)
				if gradient == 0 {
					continue
				}
				for _, feature := range features {
					index := class*buckets + feature
					weightGradient := gradient + configuration.L2*weight[index]
					weightAdam.apply(weightGradient, index)
				}
			}
		}
	}
	return classifier
}

// hashBucket hashes a typed integer tuple into [0, buckets).
func hashBucket(buckets int, tag uint32, values ...int32) int {
	hash := uint32(2166136261)
	hash = fnvStep(hash, int32(tag))
	for _, value := range values {
		hash = fnvStep(hash, value)
	}
	return int(hash % uint32(buckets))
}

// fnvStep advances an FNV-1a 32-bit hash with one value.
func fnvStep(hash uint32, value int32) uint32 {
	hash ^= uint32(value)
	hash *= 16777619
	return hash
}

// sparseAdam is a minimal Adam optimizer that applies sparse updates by flat
// index, so a hashed-feature classifier only touches the active columns.
type sparseAdam struct {
	parameter    []float32
	firstMoment  []float32
	secondMoment []float32
	learningRate float32
	beta1        float32
	beta2        float32
	epsilon      float32
	step         int
	inverse1     float32
	inverse2     float32
}

// newSparseAdam allocates Adam state for a parameter slice.
func newSparseAdam(parameter []float32, configuration NgramConfig) *sparseAdam {
	return &sparseAdam{
		parameter:    parameter,
		firstMoment:  make([]float32, len(parameter)),
		secondMoment: make([]float32, len(parameter)),
		learningRate: configuration.LearningRate,
		beta1:        0.9,
		beta2:        0.999,
		epsilon:      1e-8,
	}
}

// beginStep advances the bias-correction schedule once per example.
func (adam *sparseAdam) beginStep() {
	adam.step++
	adam.inverse1 = float32(1 / (1 - math.Pow(float64(adam.beta1), float64(adam.step))))
	adam.inverse2 = float32(1 / (1 - math.Pow(float64(adam.beta2), float64(adam.step))))
}

// apply updates one parameter index with the given gradient.
func (adam *sparseAdam) apply(gradient float32, index int) {
	first := adam.beta1*adam.firstMoment[index] + (1-adam.beta1)*gradient
	second := adam.beta2*adam.secondMoment[index] + (1-adam.beta2)*gradient*gradient
	adam.firstMoment[index] = first
	adam.secondMoment[index] = second
	correctedFirst := first * adam.inverse1
	correctedSecond := second * adam.inverse2
	adam.parameter[index] -= adam.learningRate * correctedFirst / (float32(math.Sqrt(float64(correctedSecond))) + adam.epsilon)
}
