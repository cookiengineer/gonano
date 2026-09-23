package router

import (
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/model/layers"
	"github.com/cookiengineer/gonano/optimizer"
	"github.com/cookiengineer/gonano/tensors"
)

// TransformerClassifier is the transformer meta model: a frozen
// model.Transformer encoder whose mean-pooled final hidden state feeds a
// trained linear classification head (a linear probe).
//
// The encoder can be frozen (Train, a fast linear probe) or fine-tuned jointly
// with the head (FineTune, which backpropagates the classification gradient
// through model.Transformer.TrainClassification).
type TransformerClassifier struct {
	Encoder *model.Transformer
	Head    *layers.Linear  // [classes, EmbedDim]
	Bias    *tensors.Tensor // [classes]
	// MaxLen truncates inputs to this many tokens. Zero uses
	// defaultMaxLen.
	MaxLen int
}

// defaultMaxLen is the default classifier input truncation.
const defaultMaxLen = 128

// HeadConfig configures classification-head training.
type HeadConfig struct {
	Epochs       int
	LearningRate float32
	L2           float32
	BatchSize    int
	Seed         uint64
}

// NewTransformerClassifier allocates a classifier over a frozen encoder.
func NewTransformerClassifier(encoder *model.Transformer, classes, maxLen int) *TransformerClassifier {
	if maxLen <= 0 {
		maxLen = defaultMaxLen
	}
	dim := encoder.Config.EmbedDim
	return &TransformerClassifier{
		Encoder: encoder,
		Head:    layers.NewLinear(dim, classes),
		Bias:    tensors.New(classes),
		MaxLen:  maxLen,
	}
}

// NumClasses returns the number of output classes.
func (classifier *TransformerClassifier) NumClasses() int { return classifier.Head.OutFeatures }

// Featurize runs the frozen encoder over the token sequence and mean-pools the
// final hidden state into a single EmbedDim vector.
func (classifier *TransformerClassifier) Featurize(tokens []int32) []float32 {
	dim := classifier.Encoder.Config.EmbedDim
	truncated := tokens
	if classifier.MaxLen > 0 && len(truncated) > classifier.MaxLen {
		truncated = truncated[:classifier.MaxLen]
	}
	// The cache-less forward requires T > 1; duplicate a single token.
	if len(truncated) == 0 {
		return make([]float32, dim)
	}
	if len(truncated) == 1 {
		truncated = []int32{truncated[0], truncated[0]}
	}
	indexes := tensors.NewInt32sWithData([]int{1, len(truncated)}, append([]int32(nil), truncated...))
	_, hidden := classifier.Encoder.ForwardHidden(indexes, nil) // [1, T, dim]
	sequenceLength := hidden.Shape[1]
	features := make([]float32, dim)
	inverse := 1 / float32(sequenceLength)
	for position := 0; position < sequenceLength; position++ {
		base := position * dim
		for channel := 0; channel < dim; channel++ {
			features[channel] += hidden.Data[base+channel] * inverse
		}
	}
	return features
}

// PredictFeatures returns the softmax distribution for a pooled feature vector.
func (classifier *TransformerClassifier) PredictFeatures(features []float32) []float32 {
	classes := classifier.Head.OutFeatures
	dim := classifier.Head.InFeatures
	weight := classifier.Head.Weight.Data
	logits := make([]float32, classes)
	for class := 0; class < classes; class++ {
		base := class * dim
		sum := classifier.Bias.Data[class]
		for channel := 0; channel < dim; channel++ {
			sum += weight[base+channel] * features[channel]
		}
		logits[class] = sum
	}
	return softmax(logits)
}

// Predict returns the softmax distribution for a token sequence.
func (classifier *TransformerClassifier) Predict(tokens []int32) []float32 {
	return classifier.PredictFeatures(classifier.Featurize(tokens))
}

// truncate returns a mutable token slice within the classifier's input length,
// duplicated as needed so the cache-less forward always sees T > 1.
func (classifier *TransformerClassifier) truncate(tokens []int32) []int32 {
	length := len(tokens)
	if classifier.MaxLen > 0 && length > classifier.MaxLen {
		length = classifier.MaxLen
	}
	truncated := make([]int32, length)
	copy(truncated, tokens[:length])
	if len(truncated) == 0 {
		return []int32{0, 0}
	}
	if len(truncated) == 1 {
		return []int32{truncated[0], truncated[0]}
	}
	return truncated
}

// hiddenGradient computes the classification loss gradient for one example,
// accumulates the head's parameter gradients, and returns dL/d(hidden) so the
// encoder can be fine-tuned.
func (classifier *TransformerClassifier) hiddenGradient(hidden *tensors.Tensor, label int) *tensors.Tensor {
	sequenceLength := hidden.Shape[1]
	dim := classifier.Head.InFeatures
	classes := classifier.Head.OutFeatures
	inverse := 1 / float32(sequenceLength)

	pooled := make([]float32, dim)
	for position := 0; position < sequenceLength; position++ {
		base := position * dim
		for channel := 0; channel < dim; channel++ {
			pooled[channel] += hidden.Data[base+channel] * inverse
		}
	}

	weight := classifier.Head.Weight.Data
	logits := make([]float32, classes)
	for class := 0; class < classes; class++ {
		base := class * dim
		sum := classifier.Bias.Data[class]
		for channel := 0; channel < dim; channel++ {
			sum += weight[base+channel] * pooled[channel]
		}
		logits[class] = sum
	}
	probabilities := softmax(logits)

	gradientWeight := make([]float32, classes*dim)
	gradientBias := make([]float32, classes)
	gradientPooled := make([]float32, dim)
	for class := 0; class < classes; class++ {
		gradient := probabilities[class]
		if class == label {
			gradient -= 1
		}
		gradientBias[class] = gradient
		base := class * dim
		for channel := 0; channel < dim; channel++ {
			gradientWeight[base+channel] = gradient * pooled[channel]
			gradientPooled[channel] += gradient * weight[base+channel]
		}
	}
	classifier.Head.Weight.AddGrad(gradientWeight)
	classifier.Bias.AddGrad(gradientBias)

	gradientHidden := make([]float32, sequenceLength*dim)
	for position := 0; position < sequenceLength; position++ {
		base := position * dim
		for channel := 0; channel < dim; channel++ {
			gradientHidden[base+channel] = gradientPooled[channel] * inverse
		}
	}
	return tensors.NewWithData([]int{1, sequenceLength, dim}, gradientHidden)
}

// FineTune trains the classification head and the encoder jointly. For each
// example it runs a training forward, computes the head loss and its gradient,
// backpropagates through the encoder via model.Transformer.TrainClassification,
// and applies one optimizer step. It is slower than Train (the frozen linear
// probe) but adapts the encoder representation to the domain task.
func (classifier *TransformerClassifier) FineTune(examples []Example, epochs int, learningRate float32, seed uint64) {
	if len(examples) == 0 {
		return
	}
	if epochs <= 0 {
		epochs = 3
	}
	if learningRate <= 0 {
		learningRate = 1e-3
	}

	groups := classifier.Encoder.SetupOptimizer(learningRate, learningRate, learningRate, 0, learningRate, false)
	for index := range groups {
		groups[index].LR = learningRate
	}
	groups = append(groups, optimizer.ParamGroup{
		Kind:   optimizer.KindAdamW,
		Params: []*tensors.Tensor{classifier.Head.Weight, classifier.Bias},
		LR:     learningRate, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8,
	})
	optim := optimizer.NewMuonAdamW(groups)

	order := make([]int, len(examples))
	for index := range order {
		order[index] = index
	}
	random := tensors.NewRNG(seed)
	for epoch := 0; epoch < epochs; epoch++ {
		for index := len(order) - 1; index > 0; index-- {
			swap := random.IntN(index + 1)
			order[index], order[swap] = order[swap], order[index]
		}
		for _, position := range order {
			tokens := classifier.truncate(examples[position].Tokens)
			label := examples[position].Label
			indexes := tensors.NewInt32sWithData([]int{1, len(tokens)}, tokens)
			classifier.Encoder.TrainClassification(indexes, func(hidden *tensors.Tensor) *tensors.Tensor {
				return classifier.hiddenGradient(hidden, label)
			})
			optim.Step()
			optim.ZeroGrad()
		}
	}
}

// Train trains the classification head on the labeled examples, keeping the
// encoder frozen. Features are extracted once and the head is optimized with
// dense Adam on mini-batches. Use FineTune to adapt the encoder as well.
func (classifier *TransformerClassifier) Train(examples []Example, configuration HeadConfig) {
	if configuration.Epochs <= 0 {
		configuration.Epochs = 20
	}
	if configuration.LearningRate <= 0 {
		configuration.LearningRate = 0.01
	}
	if configuration.BatchSize <= 0 {
		configuration.BatchSize = 32
	}
	if len(examples) == 0 {
		return
	}
	features := make([][]float32, len(examples))
	for index, example := range examples {
		features[index] = classifier.Featurize(example.Tokens)
	}

	classes := classifier.Head.OutFeatures
	dim := classifier.Head.InFeatures
	weight := classifier.Head.Weight.Data
	bias := classifier.Bias.Data
	weightAdam := newSparseAdam(weight, NgramConfig{LearningRate: configuration.LearningRate})
	biasAdam := newSparseAdam(bias, NgramConfig{LearningRate: configuration.LearningRate})

	order := make([]int, len(examples))
	for index := range order {
		order[index] = index
	}
	random := tensors.NewRNG(configuration.Seed)
	gradientWeight := make([]float32, len(weight))
	gradientBias := make([]float32, classes)

	for epoch := 0; epoch < configuration.Epochs; epoch++ {
		for index := len(order) - 1; index > 0; index-- {
			swap := random.IntN(index + 1)
			order[index], order[swap] = order[swap], order[index]
		}
		for start := 0; start < len(order); start += configuration.BatchSize {
			end := start + configuration.BatchSize
			if end > len(order) {
				end = len(order)
			}
			for index := range gradientWeight {
				gradientWeight[index] = 0
			}
			for class := 0; class < classes; class++ {
				gradientBias[class] = 0
			}
			for _, position := range order[start:end] {
				feature := features[position]
				logits := make([]float32, classes)
				for class := 0; class < classes; class++ {
					base := class * dim
					sum := bias[class]
					for channel := 0; channel < dim; channel++ {
						sum += weight[base+channel] * feature[channel]
					}
					logits[class] = sum
				}
				probabilities := softmax(logits)
				label := examples[position].Label
				for class := 0; class < classes; class++ {
					gradient := probabilities[class]
					if class == label {
						gradient -= 1
					}
					gradientBias[class] += gradient
					base := class * dim
					for channel := 0; channel < dim; channel++ {
						gradientWeight[base+channel] += gradient * feature[channel]
					}
				}
			}
			scale := 1 / float32(end-start)
			weightAdam.beginStep()
			biasAdam.beginStep()
			for index := range weight {
				weightAdam.apply(gradientWeight[index]*scale+configuration.L2*weight[index], index)
			}
			for class := 0; class < classes; class++ {
				biasAdam.apply(gradientBias[class]*scale, class)
			}
		}
	}
}
