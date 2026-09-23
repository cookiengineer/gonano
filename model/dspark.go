package model

import (
	"math"

	"github.com/cookiengineer/gonano/model/layers"
	"github.com/cookiengineer/gonano/optimizer"
	"github.com/cookiengineer/gonano/tensors"
)

// DrafterLayers is the number of transformer blocks in the DSpark drafter
// (DeepSeek-V4.1 §2.4.3).
const DrafterLayers = 3

// dsparkDraftPositions is the default number of draft positions a single
// semi-autoregressive forward predicts (DeepSeek-V4.1 §2.4.3 uses five).
const dsparkDraftPositions = 5

// DefaultDraftPositions returns the semi-autoregressive draft count DSpark uses
// when none is configured.
func DefaultDraftPositions() int { return dsparkDraftPositions }

// dsparkPlaceholderToken is the input id used for draft-mask positions. Its
// token embedding is replaced by draftMaskEmbedding before the trunk runs, so
// the concrete id only needs to be a valid vocabulary index.
const dsparkPlaceholderToken = 0

// DrafterConfig derives the drafter configuration from a backbone config. The
// drafter shares the backbone's tokenizer, so it keeps the vocabulary and head
// geometry, but it is a small standalone transformer with its own embedding and
// language-model head and a bounded context (the paper's sliding window). The
// context is capped so drafting stays cheap.
func DrafterConfig(backbone Config) Config {
	sequenceLen := backbone.SequenceLen
	if sequenceLen > 128 {
		sequenceLen = 128
	}
	config := Config{
		SequenceLen:   sequenceLen,
		VocabSize:     backbone.VocabSize,
		NumLayer:      DrafterLayers,
		NumHead:       backbone.NumHead,
		NumKVHead:     backbone.NumKVHead,
		EmbedDim:      backbone.EmbedDim,
		WindowPattern: "L",
		RotaryDims:    backbone.RotaryDims,
	}
	config.Validate()
	return config
}

// NewDrafter builds a randomly initialized drafter for a backbone. The caller
// initializes it with InitWeights or trains it by distilling the backbone
// (DeepSeek-V4.1 §2.4.3 trains the drafter separately with the backbone frozen).
func NewDrafter(backbone *Transformer) *Transformer {
	return NewTransformer(DrafterConfig(backbone.Config))
}

// DraftTokens greedily proposes up to count tokens following the context, using
// the model as a draft model. It only sees the last SequenceLen tokens, so the
// per-round drafting cost is bounded independently of the context length. It
// returns fewer than count tokens only when the context is empty.
func (model *Transformer) DraftTokens(context []int, count int) []int {
	if count <= 0 {
		return nil
	}
	window := model.Config.SequenceLen
	if len(context) > window {
		context = context[len(context)-window:]
	}
	if len(context) == 0 {
		return nil
	}

	cache := NewKVBuffer(1, len(context)+count, model.Config.NumLayer, model.Config.NumKVHead, model.Config.HeadDim())
	indexes := tensors.NewInt32sWithData([]int{1, len(context)}, toInt32(context))
	logits := model.Forward(indexes, cache)

	drafts := make([]int, 0, count)
	for len(drafts) < count {
		token := argmaxLastRow(logits, model.Config.VocabSize)
		drafts = append(drafts, token)
		if len(drafts) == count {
			break
		}
		next := tensors.NewInt32sWithData([]int{1, 1}, []int32{int32(token)})
		logits = model.Forward(next, cache)
	}
	return drafts
}

// SetPreviousEmbeddingForToken stores the pre-smear normalized embedding of a
// token as the cache's previous embedding, so a later multi-token forward
// against the partially filled cache smears its first position correctly.
// Speculative verification calls it after rewinding to the last accepted token.
func (model *Transformer) SetPreviousEmbeddingForToken(cache *KVBuffer, token int) {
	if cache == nil {
		return
	}
	indexes := tensors.NewInt32sWithData([]int{1, 1}, []int32{int32(token)})
	cache.SetPrevEmbedding(normalizeLastDim(model.tokenEmbedding.Forward(indexes)))
}

// argmaxLastRow returns the argmax of the last row's first vocabSize logits.
func argmaxLastRow(logits *tensors.Tensor, vocabSize int) int {
	rowCount := logits.Numel() / vocabSize
	base := (rowCount - 1) * vocabSize
	best := 0
	bestValue := logits.Data[base]
	for index := 1; index < vocabSize; index++ {
		if logits.Data[base+index] > bestValue {
			bestValue = logits.Data[base+index]
			best = index
		}
	}
	return best
}

// toInt32 converts token ids to the int32 representation used by the tensors.
func toInt32(values []int) []int32 {
	converted := make([]int32, len(values))
	for index, value := range values {
		converted[index] = int32(value)
	}
	return converted
}

// DSpark is the full DSpark drafter (DeepSeek-V4.1 §2.4.3): a small transformer
// trunk plus a low-rank Markov head and a confidence head. The trunk is trained
// by distillation; the heads are trained on top of the frozen trunk. The Markov
// head biases each drafted token on the previous token, and the confidence head
// predicts whether a drafted token will be accepted, which the engine uses to
// schedule the verification length.
type DSpark struct {
	Drafter    *Transformer
	markovDown *layers.Linear // [markovRank, EmbedDim]
	markovUp   *layers.Linear // [VocabSize, markovRank]
	confHidden *layers.Linear // [confHidden, EmbedDim]
	confOut    *layers.Linear // [1, confHidden]
	// draftMaskEmbedding [1, EmbedDim] replaces the embeddings of the draft
	// positions in the semi-autoregressive forward pass (DeepSeek-V4.1 §2.4.3).
	draftMaskEmbedding *tensors.Tensor
}

// dsparkMarkovRank and dsparkConfHidden cap the head widths for small models.
const (
	dsparkMarkovRankCap = 32
	dsparkConfHiddenCap = 64
)

// NewDSpark builds a DSpark drafter from a backbone, deriving the drafter shape
// from the backbone config.
func NewDSpark(backbone *Transformer) *DSpark {
	return NewDSparkFromConfig(DrafterConfig(backbone.Config))
}

// NewDSparkFromConfig builds a DSpark with the given drafter config.
func NewDSparkFromConfig(config Config) *DSpark {
	drafter := NewTransformer(config)
	rank := config.EmbedDim / 4
	if rank < 4 {
		rank = 4
	}
	if rank > dsparkMarkovRankCap {
		rank = dsparkMarkovRankCap
	}
	confHidden := config.EmbedDim
	if confHidden > dsparkConfHiddenCap {
		confHidden = dsparkConfHiddenCap
	}
	return &DSpark{
		Drafter:            drafter,
		markovDown:         layers.NewLinear(config.EmbedDim, rank),
		markovUp:           layers.NewLinear(rank, config.VocabSize),
		confHidden:         layers.NewLinear(config.EmbedDim, confHidden),
		confOut:            layers.NewLinear(confHidden, 1),
		draftMaskEmbedding: tensors.New(1, config.EmbedDim),
	}
}

// InitWeights initializes the trunk and the heads. The Markov up-projection and
// the confidence output start at zero, so a freshly initialized DSpark drafts
// exactly like its trunk.
func (d *DSpark) InitWeights(rng *tensors.RNG) {
	d.Drafter.InitWeights(rng)
	layers.InitNormal(d.markovDown.Weight, rng, 0.02)
	layers.InitZeros(d.markovUp.Weight)
	layers.InitNormal(d.confHidden.Weight, rng, 0.02)
	layers.InitZeros(d.confOut.Weight)
	layers.InitNormal(d.draftMaskEmbedding, rng, 0.02)
}

// Parameters returns the trunk and head parameters.
func (d *DSpark) Parameters() []*tensors.Tensor {
	parameters := d.Drafter.Parameters()
	parameters = append(parameters, d.markovDown.Weight, d.markovUp.Weight, d.confHidden.Weight, d.confOut.Weight, d.draftMaskEmbedding)
	return parameters
}

// NamedParameters returns every parameter keyed by a stable name, including the
// trunk under a "dspark." prefix so the checkpoint round-trips.
func (d *DSpark) NamedParameters() map[string]*tensors.Tensor {
	parameters := make(map[string]*tensors.Tensor)
	for name, parameter := range d.Drafter.NamedParameters() {
		parameters["dspark."+name] = parameter
	}
	parameters["dspark.markov_down.weight"] = d.markovDown.Weight
	parameters["dspark.markov_up.weight"] = d.markovUp.Weight
	parameters["dspark.conf_hidden.weight"] = d.confHidden.Weight
	parameters["dspark.conf_out.weight"] = d.confOut.Weight
	parameters["dspark.draft_mask.embedding"] = d.draftMaskEmbedding
	return parameters
}

// ZeroGrad zeroes the trunk and head gradients.
func (d *DSpark) ZeroGrad() {
	for _, parameter := range d.Parameters() {
		parameter.ZeroGrad()
	}
}

// SetupHeadOptimizer builds an AdamW group for the two heads. The trunk is
// frozen while the heads are trained.
func (d *DSpark) SetupHeadOptimizer(learningRate, weightDecay float32) []optimizer.ParamGroup {
	return []optimizer.ParamGroup{{
		Kind: optimizer.KindAdamW,
		Params: []*tensors.Tensor{
			d.markovDown.Weight, d.markovUp.Weight, d.confHidden.Weight, d.confOut.Weight,
		},
		LR: learningRate, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8, WeightDecay: weightDecay,
	}}
}

// Draft proposes up to count tokens after the context and returns a confidence
// for each, using the trunk, the Markov head, and the confidence head. It runs
// a single semi-autoregressive trunk forward over the context followed by
// count-1 draft-mask positions and reads the next-token logits of those
// positions in parallel (DeepSeek-V4.1 §2.4.3); the Markov head then chains the
// drafted tokens. It only sees the last SequenceLen tokens.
func (d *DSpark) Draft(context []int, count int) ([]int, []float32) {
	if count <= 0 || len(context) == 0 {
		return nil, nil
	}
	config := d.Drafter.Config
	if count > config.SequenceLen {
		count = config.SequenceLen
	}
	placeholderCount := count - 1
	maxPrefix := config.SequenceLen - placeholderCount
	if maxPrefix < 1 {
		maxPrefix = 1
	}
	if len(context) > maxPrefix {
		context = context[len(context)-maxPrefix:]
	}
	prefixLength := len(context)

	input := make([]int32, 0, prefixLength+placeholderCount)
	for _, token := range context {
		input = append(input, int32(token))
	}
	for index := 0; index < placeholderCount; index++ {
		input = append(input, dsparkPlaceholderToken)
	}

	cache := NewKVBuffer(1, len(input), config.NumLayer, config.NumKVHead, config.HeadDim())
	indexes := tensors.NewInt32sWithData([]int{1, len(input)}, input)
	logits, hidden := d.Drafter.ForwardHiddenSuffix(indexes, cache, d.draftMaskEmbedding, prefixLength)

	vocab := config.VocabSize
	dim := config.EmbedDim
	firstBase := logits.Shape[1] - count
	previous := context[len(context)-1]
	tokens := make([]int, 0, count)
	confidences := make([]float32, 0, count)
	for index := 0; index < count; index++ {
		row := firstBase + index
		baseLogits := logits.Data[row*vocab : (row+1)*vocab]
		rowHidden := hidden.Data[row*dim : (row+1)*dim]

		biased := append([]float32(nil), baseLogits...)
		embedded := d.Drafter.tokenEmbedding.Forward(tensors.NewInt32sWithData([]int{1, 1}, []int32{int32(previous)}))
		bias := d.markovUp.Forward(d.markovDown.Forward(embedded))
		for element := range biased {
			biased[element] += bias.Data[element]
		}

		token := argmaxValues(biased)
		confidenceInput := tensors.NewWithData([]int{1, 1, dim}, rowHidden)
		confidence := sigmoidFloat(d.confOut.Forward(d.confHidden.Forward(confidenceInput)).Data[0])
		tokens = append(tokens, token)
		confidences = append(confidences, confidence)
		previous = token
	}
	return tokens, confidences
}

// TrainHeadsStep updates the Markov and confidence heads on one batch with the
// trunk frozen. inputs/targets are shifted next-token pairs (shape [B,T]);
// targets of -1 are ignored. It runs the same semi-autoregressive placeholder
// forward as Draft so the heads are trained under the distribution they see at
// inference, and returns the mean cross-entropy plus the confidence binary
// cross-entropy. count is the number of parallel draft positions.
func (d *DSpark) TrainHeadsStep(inputs, targets *tensors.Int32s, count int) float32 {
	if count < 1 {
		count = 1
	}
	config := d.Drafter.Config
	vocab := config.VocabSize
	dim := config.EmbedDim
	batchSize := inputs.Shape[0]
	sequenceLength := inputs.Shape[1]
	if count > sequenceLength {
		count = sequenceLength
	}
	placeholderCount := count - 1
	prefixLength := sequenceLength - placeholderCount
	if prefixLength < 1 {
		prefixLength = 1
	}
	supervised := batchSize * count

	masked := make([]int32, batchSize*sequenceLength)
	for batch := 0; batch < batchSize; batch++ {
		for position := 0; position < sequenceLength; position++ {
			index := batch*sequenceLength + position
			if position >= prefixLength {
				masked[index] = dsparkPlaceholderToken
			} else {
				masked[index] = inputs.Data[index]
			}
		}
	}
	maskedInputs := tensors.NewInt32sWithData([]int{batchSize, sequenceLength}, masked)

	d.ZeroGrad()
	// The trunk is frozen: the suffix forward allocates no parameter gradients.
	logits, hidden := d.Drafter.ForwardHiddenSuffix(maskedInputs, nil, d.draftMaskEmbedding, prefixLength)
	flatLogits := logits.Reshape(batchSize*sequenceLength, vocab)
	flatHidden := hidden.Reshape(batchSize*sequenceLength, dim)

	total := tensors.New(supervised, vocab)
	confidenceInput := tensors.New(supervised, dim)
	previousEmbeddings := tensors.New(supervised, dim)
	flatTargets := make([]int32, supervised)
	rowIndex := 0
	for batch := 0; batch < batchSize; batch++ {
		for index := 0; index < count; index++ {
			source := batch*sequenceLength + prefixLength - 1 + index
			target := targets.Data[source]
			flatTargets[rowIndex] = target
			copy(total.Data[rowIndex*vocab:(rowIndex+1)*vocab], flatLogits.Data[source*vocab:(source+1)*vocab])
			copy(confidenceInput.Data[rowIndex*dim:(rowIndex+1)*dim], flatHidden.Data[source*dim:(source+1)*dim])
			previous := inputs.Data[batch*sequenceLength+prefixLength-1]
			if index > 0 {
				previous = targets.Data[source-1]
			}
			embedded := d.Drafter.tokenEmbedding.Forward(tensors.NewInt32sWithData([]int{1, 1}, []int32{previous}))
			copy(previousEmbeddings.Data[rowIndex*dim:(rowIndex+1)*dim], embedded.Data)
			rowIndex++
		}
	}
	targetTensor := tensors.NewInt32sWithData([]int{supervised}, flatTargets)

	markovHidden := d.markovDown.Forward(previousEmbeddings)
	markovBias := d.markovUp.Forward(markovHidden).Reshape(supervised, vocab)
	for index := range total.Data {
		total.Data[index] += markovBias.Data[index]
	}

	loss := tensors.CrossEntropy(total, targetTensor, -1)
	probabilities := tensors.SoftmaxLastDim(total)
	gradTotal := tensors.New(supervised, vocab)
	for row := 0; row < supervised; row++ {
		target := flatTargets[row]
		if target == -1 {
			continue
		}
		for index := 0; index < vocab; index++ {
			gradient := probabilities.Data[row*vocab+index]
			if int32(index) == target {
				gradient -= 1
			}
			gradTotal.Data[row*vocab+index] = gradient / float32(supervised)
		}
	}
	gradMarkovHidden := d.markovUp.Backward(markovHidden, gradTotal)
	d.markovDown.Backward(previousEmbeddings, gradMarkovHidden)

	// Confidence head: label a position as accepted when the biased argmax
	// matches the target.
	scores := d.confHidden.Forward(confidenceInput)
	confidenceLogit := d.confOut.Forward(scores).Reshape(supervised, 1)
	gradConfidence := tensors.New(supervised, 1)
	var binaryCrossEntropy float64
	for row := 0; row < supervised; row++ {
		target := flatTargets[row]
		if target == -1 {
			continue
		}
		label := float32(0)
		if int32(argmaxValues(total.Data[row*vocab:(row+1)*vocab])) == target {
			label = 1
		}
		probability := sigmoidFloat(confidenceLogit.Data[row])
		if label == 1 {
			binaryCrossEntropy += -math.Log(float64(probability) + 1e-9)
		} else {
			binaryCrossEntropy += -math.Log(1 - float64(probability) + 1e-9)
		}
		gradConfidence.Data[row] = (probability - label) / float32(supervised)
	}
	gradConfidenceInput := d.confOut.Backward(scores, gradConfidence)
	d.confHidden.Backward(confidenceInput, gradConfidenceInput)

	return loss + float32(binaryCrossEntropy)/float32(supervised)
}

// LoadDSpark reconstructs a DSpark from a checkpoint's config and parameters.
func LoadDSpark(config Config, parameters map[string]*tensors.Tensor) *DSpark {
	dspark := NewDSparkFromConfig(config)
	named := dspark.NamedParameters()
	for name, parameter := range parameters {
		if target, ok := named[name]; ok {
			copy(target.Data, parameter.Data)
		}
	}
	return dspark
}

// sigmoidFloat is a scalar logistic function.
func sigmoidFloat(value float32) float32 {
	return float32(1 / (1 + math.Exp(-float64(value))))
}

// argmaxValues returns the index of the largest value.
func argmaxValues(values []float32) int {
	best := 0
	for index := 1; index < len(values); index++ {
		if values[index] > values[best] {
			best = index
		}
	}
	return best
}
