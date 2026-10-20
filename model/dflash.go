package model

import (
	"math"

	"github.com/cookiengineer/gonano/model/layers"
	"github.com/cookiengineer/gonano/optimizer"
	"github.com/cookiengineer/gonano/tensors"
)

// DFlash is the DeepSeek-V4.1 MTP/DFlash drafter. It mirrors the DSpark
// semi-autoregressive draft block but conditions the draft positions on a
// projection of the target model's hidden state and keeps a learned Markov
// correction plus a confidence head.
type DFlash struct {
	Drafter    *Transformer
	TargetIDs  []int
	BlockSize  int
	NoiseToken int
	MarkovRank int

	Projection  *layers.Linear  // [EmbedDim, len(TargetIDs)*EmbedDim]
	ProjectNorm *tensors.Tensor // [EmbedDim]

	MarkovDown *layers.Linear // [markovRank, EmbedDim]
	MarkovUp   *layers.Linear // [VocabSize, markovRank]
	ConfHidden *layers.Linear // [confHidden, EmbedDim]
	ConfOut    *layers.Linear // [1, confHidden]

	DraftMaskEmbedding *tensors.Tensor // [1, EmbedDim]
}

// NewDFlash builds a DFlash drafter for a backbone.
func NewDFlash(backbone *Transformer) *DFlash {
	return NewDFlashFromConfig(backbone.Config)
}

// NewDFlashFromConfig builds a DFlash drafter from a backbone config.
func NewDFlashFromConfig(config Config) *DFlash {
	drafterConfig := DrafterConfig(config)
	if config.DFlashBlockSize > 0 && drafterConfig.SequenceLen < config.DFlashBlockSize+1 {
		drafterConfig.SequenceLen = config.DFlashBlockSize + 1
		drafterConfig.Validate()
	}
	targetIDs := append([]int(nil), config.DFlashTargetLayerIDs...)
	if len(targetIDs) == 0 {
		targetIDs = []int{config.NumLayer - 1}
	}
	blockSize := config.DFlashBlockSize
	if blockSize <= 0 {
		blockSize = dsparkDraftPositions
	}
	markovRank := config.DFlashMarkovRank
	if markovRank <= 0 {
		markovRank = config.EmbedDim / 4
		if markovRank < 4 {
			markovRank = 4
		}
		if markovRank > dsparkMarkovRankCap {
			markovRank = dsparkMarkovRankCap
		}
	}
	confHidden := config.EmbedDim
	if confHidden > dsparkConfHiddenCap {
		confHidden = dsparkConfHiddenCap
	}
	noise := config.DFlashNoiseToken
	if noise <= 0 || noise >= config.VocabSize {
		noise = config.VocabSize - 1
	}
	return &DFlash{
		Drafter:            NewTransformer(drafterConfig),
		TargetIDs:          targetIDs,
		BlockSize:          blockSize,
		NoiseToken:         noise,
		MarkovRank:         markovRank,
		Projection:         layers.NewLinear(len(targetIDs)*config.EmbedDim, config.EmbedDim),
		ProjectNorm:        tensors.New(config.EmbedDim),
		MarkovDown:         layers.NewLinear(config.EmbedDim, markovRank),
		MarkovUp:           layers.NewLinear(markovRank, config.VocabSize),
		ConfHidden:         layers.NewLinear(config.EmbedDim, confHidden),
		ConfOut:            layers.NewLinear(confHidden, 1),
		DraftMaskEmbedding: tensors.New(1, config.EmbedDim),
	}
}

// InitWeights initializes the drafter and the MTP heads.
func (d *DFlash) InitWeights(rng *tensors.RNG) {
	d.Drafter.InitWeights(rng)
	layers.InitNormal(d.Projection.Weight, rng, 0.02)
	for index := range d.ProjectNorm.Data {
		d.ProjectNorm.Data[index] = 1
	}
	layers.InitNormal(d.MarkovDown.Weight, rng, 0.02)
	layers.InitZeros(d.MarkovUp.Weight)
	layers.InitNormal(d.ConfHidden.Weight, rng, 0.02)
	layers.InitZeros(d.ConfOut.Weight)
	layers.InitNormal(d.DraftMaskEmbedding, rng, 0.02)
}

// Parameters returns every DFlash parameter.
func (d *DFlash) Parameters() []*tensors.Tensor {
	parameters := d.Drafter.Parameters()
	parameters = append(parameters,
		d.Projection.Weight, d.ProjectNorm,
		d.MarkovDown.Weight, d.MarkovUp.Weight,
		d.ConfHidden.Weight, d.ConfOut.Weight,
		d.DraftMaskEmbedding,
	)
	return parameters
}

// NamedParameters returns the drafter and heads under a dflash. prefix.
func (d *DFlash) NamedParameters() map[string]*tensors.Tensor {
	parameters := make(map[string]*tensors.Tensor)
	for name, parameter := range d.Drafter.NamedParameters() {
		parameters["dflash."+name] = parameter
	}
	parameters["dflash.projection.weight"] = d.Projection.Weight
	parameters["dflash.project_norm"] = d.ProjectNorm
	parameters["dflash.markov_down.weight"] = d.MarkovDown.Weight
	parameters["dflash.markov_up.weight"] = d.MarkovUp.Weight
	parameters["dflash.conf_hidden.weight"] = d.ConfHidden.Weight
	parameters["dflash.conf_out.weight"] = d.ConfOut.Weight
	parameters["dflash.draft_mask.embedding"] = d.DraftMaskEmbedding
	return parameters
}

// ZeroGrad zeroes every DFlash gradient.
func (d *DFlash) ZeroGrad() {
	for _, parameter := range d.Parameters() {
		parameter.ZeroGrad()
	}
}

// SetupTrainingOptimizer builds the drafter's normal optimizer groups plus an
// AdamW group for the MTP heads.
func (d *DFlash) SetupTrainingOptimizer(unembeddingLR, embeddingLR, matrixLR, weightDecay, scalarLR, headLR float32) []optimizer.ParamGroup {
	groups := d.Drafter.SetupOptimizer(unembeddingLR, embeddingLR, matrixLR, weightDecay, scalarLR, true)
	groups = append(groups, optimizer.ParamGroup{
		Kind: optimizer.KindAdamW,
		Params: []*tensors.Tensor{
			d.Projection.Weight, d.ProjectNorm,
			d.MarkovDown.Weight, d.MarkovUp.Weight,
			d.ConfHidden.Weight, d.ConfOut.Weight,
			d.DraftMaskEmbedding,
		},
		LR: headLR, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8, WeightDecay: 0,
	})
	return groups
}

// targetRows selects the hidden row at prefixLength-1 for every batch row.
func targetRows(targetHidden *tensors.Tensor, prefixLength int) *tensors.Tensor {
	batchSize := targetHidden.Shape[0]
	sequenceLength := targetHidden.Shape[1]
	width := targetHidden.Shape[2]
	if prefixLength < 1 {
		prefixLength = 1
	}
	if prefixLength > sequenceLength {
		prefixLength = sequenceLength
	}
	rows := tensors.New(batchSize, 1, width)
	for batch := 0; batch < batchSize; batch++ {
		source := ((batch*sequenceLength + prefixLength - 1) * width)
		copy(rows.Data[batch*width:(batch+1)*width], targetHidden.Data[source:source+width])
	}
	return rows
}

// projectTarget projects the target hidden rows and applies the projection
// norm. It returns the normalized rows and their pre-norm projection.
func (d *DFlash) projectTarget(rows *tensors.Tensor) (projected, normalized *tensors.Tensor) {
	projected = d.Projection.Forward(rows)
	normalized = normalizeLastDim(projected)
	return projected, normalized
}

// scaleNorm applies the learned projection norm scale.
func scaleNorm(normalized *tensors.Tensor, weight *tensors.Tensor) *tensors.Tensor {
	output := normalized.Clone()
	channels := normalized.Shape[len(normalized.Shape)-1]
	rows := normalized.Numel() / channels
	for row := 0; row < rows; row++ {
		for channel := 0; channel < channels; channel++ {
			output.Data[row*channels+channel] *= weight.Data[channel]
		}
	}
	return output
}

// TrainStep trains the drafter, target projection, and heads on one batch.
// inputs and targets are shifted next-token pairs [B,T]; targetHidden is the
// backbone hidden state [B,T,len(TargetIDs)*EmbedDim] at the configured target
// layers. count is the number of draft positions.
func (d *DFlash) TrainStep(inputs, targets *tensors.Int32s, hidden *tensors.Tensor, count int) float32 {
	if count < 1 {
		count = d.BlockSize
	}
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
				masked[index] = int32(d.NoiseToken)
			} else {
				masked[index] = inputs.Data[index]
			}
		}
	}
	maskedInputs := tensors.NewInt32sWithData([]int{batchSize, sequenceLength}, masked)

	// Condition the suffix positions on the target hidden state of the last
	// prefix token, plus the learned draft-mask embedding.
	targetHiddenRows := targetRows(hidden, prefixLength)
	projectedRows, normalizedRows := d.projectTarget(targetHiddenRows)
	suffix := scaleNorm(normalizedRows, d.ProjectNorm)
	for batch := 0; batch < batchSize; batch++ {
		for channel := 0; channel < dim; channel++ {
			suffix.Data[batch*dim+channel] += d.DraftMaskEmbedding.Data[channel]
		}
	}

	d.ZeroGrad()
	logits, context := d.Drafter.TrainForwardSuffix(maskedInputs, suffix, prefixLength)
	hiddenStates := context.finalNorm
	flatLogits := logits.Reshape(batchSize*sequenceLength, vocab)
	flatHidden := hiddenStates.Reshape(batchSize*sequenceLength, dim)

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

	markovHidden := d.MarkovDown.Forward(previousEmbeddings)
	markovBias := d.MarkovUp.Forward(markovHidden).Reshape(supervised, vocab)
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
	gradMarkovHidden := d.MarkovUp.Backward(markovHidden, gradTotal)
	d.MarkovDown.Backward(previousEmbeddings, gradMarkovHidden)

	scores := d.ConfHidden.Forward(confidenceInput)
	confidenceLogit := d.ConfOut.Forward(scores).Reshape(supervised, 1)
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
	gradConfidenceInput := d.ConfOut.Backward(scores, gradConfidence)
	d.ConfHidden.Backward(confidenceInput, gradConfidenceInput)

	gradLogits := tensors.New(batchSize, sequenceLength, vocab)
	rowIndex = 0
	for batch := 0; batch < batchSize; batch++ {
		for index := 0; index < count; index++ {
			source := batch*sequenceLength + prefixLength - 1 + index
			copy(gradLogits.Data[source*vocab:(source+1)*vocab], gradTotal.Data[rowIndex*vocab:(rowIndex+1)*vocab])
			rowIndex++
		}
	}
	d.Drafter.TrainBackward(context, gradLogits)

	// Backpropagate the suffix gradient through the target projection. The
	// suffix rows are [B,D]; the draft mask is shared across the batch.
	suffix.EnsureGrad()
	for batch := 0; batch < batchSize; batch++ {
		for channel := 0; channel < dim; channel++ {
			gradient := suffix.Grad[batch*dim+channel]
			d.DraftMaskEmbedding.Grad[channel] += gradient
			weighted := gradient * d.ProjectNorm.Data[channel]
			d.ProjectNorm.Grad[channel] += gradient * normalizedRows.Data[batch*dim+channel]
			suffix.Grad[batch*dim+channel] = weighted
		}
	}
	gradProjected := normalizeLastDimBackward(projectedRows, suffix)
	d.Projection.Backward(targetHiddenRows, gradProjected)

	return loss + float32(binaryCrossEntropy)/float32(supervised)
}

// Draft proposes up to count tokens conditioned on the target hidden state of
// the last context token.
func (d *DFlash) Draft(context []int, targetHidden *tensors.Tensor, count int) ([]int, []float32) {
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
		input = append(input, int32(d.NoiseToken))
	}

	// targetHidden is [1,1,Din] or [1,Din]; keep one row per target layer.
	rows := targetHidden.Numel() / (len(d.TargetIDs) * d.Drafter.Config.EmbedDim)
	if rows < 1 {
		rows = 1
	}
	targetRowsTensor := tensors.NewWithData([]int{1, 1, targetHidden.Numel()}, targetHidden.Data)
	projected, normalized := d.projectTarget(targetRowsTensor)
	suffix := scaleNorm(normalized, d.ProjectNorm)
	for channel := 0; channel < d.Drafter.Config.EmbedDim; channel++ {
		suffix.Data[channel] += d.DraftMaskEmbedding.Data[channel]
	}

	cache := NewKVBuffer(1, len(input), config.NumLayer, config.NumKVHead, config.HeadDim())
	indexes := tensors.NewInt32sWithData([]int{1, len(input)}, input)
	logits, hidden := d.Drafter.ForwardHiddenSuffix(indexes, cache, suffix, prefixLength)

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
		bias := d.MarkovUp.Forward(d.MarkovDown.Forward(embedded))
		for element := range biased {
			biased[element] += bias.Data[element]
		}

		token := argmaxValues(biased)
		confidenceInput := tensors.NewWithData([]int{1, 1, dim}, rowHidden)
		confidence := sigmoidFloat(d.ConfOut.Forward(d.ConfHidden.Forward(confidenceInput)).Data[0])
		tokens = append(tokens, token)
		confidences = append(confidences, confidence)
		previous = token
	}
	_ = projected
	return tokens, confidences
}

// TargetHiddenFromContext concatenates the backbone hidden states at the
// configured target layers for DFlash training.
func TargetHiddenFromContext(context *trainCtx, layerIDs []int) *tensors.Tensor {
	numLayers := len(context.previousActivations)
	dim := 0
	if numLayers > 0 {
		dim = context.previousActivations[0].Shape[len(context.previousActivations[0].Shape)-1]
	}
	if len(layerIDs) == 0 {
		layerIDs = []int{numLayers - 1}
	}
	batchSize := context.indexes.Shape[0]
	sequenceLength := context.indexes.Shape[1]
	tokens := batchSize * sequenceLength
	width := len(layerIDs) * dim
	output := tensors.New(batchSize, sequenceLength, width)
	for slot, layer := range layerIDs {
		source := context.finalPreNorm
		if layer >= 0 && layer < numLayers-1 {
			source = context.previousActivations[layer+1]
		}
		for token := 0; token < tokens; token++ {
			sourceBase := token * dim
			destinationBase := token*width + slot*dim
			copy(output.Data[destinationBase:destinationBase+dim], source.Data[sourceBase:sourceBase+dim])
		}
	}
	return output
}
