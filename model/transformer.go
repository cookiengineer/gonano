package model

import (
	"math"

	"github.com/cookiengineer/gonano/model/layers"
	"github.com/cookiengineer/gonano/tensors"
)

// softcap is the logit softcap used before loss/sampling.
const softcap = 15.0

// rotaryOvercompute is the factor by which the rotary embedding cache exceeds
// the training sequence length (nanochat over-computes by 10x).
const rotaryOvercompute = 10

// uniformBound returns nanochat's uniform half-width sqrt(3) * n_embd^-0.5.
func uniformBound(embeddingDimension int) float32 {
	return float32(math.Sqrt(3.0) / math.Sqrt(float64(embeddingDimension)))
}

// Transformer is the nanochat GPT model.
type Transformer struct {
	Config      Config
	paddedVocab int
	windowSizes [][2]int

	tokenEmbedding *layers.Embedding
	blocks         []*Block
	lmHead         *layers.Linear

	residLambdas  *tensors.Tensor // [n_layer]
	x0Lambdas     *tensors.Tensor // [n_layer]
	smearGate     *layers.Linear  // [24, 1]
	smearLambda   *tensors.Tensor // [1]
	backoutLambda *tensors.Tensor // [1]

	valueEmbeds map[int]*layers.Embedding // layer -> embedding (ResFormer)

	rotaryCosine, rotarySine *tensors.Tensor // [rotarySeqLen, rotaryDim/2]
}

// NewTransformer builds a Transformer with all parameters zero-initialized.
// Call InitWeights to initialize them.
func NewTransformer(config Config) *Transformer {
	config.Validate()
	paddedVocabulary := config.PaddedVocab()
	rotaryCosine, rotarySine := precomputeRotary(config.SequenceLen*rotaryOvercompute, config.RotaryDimension())

	model := &Transformer{
		Config:         config,
		paddedVocab:    paddedVocabulary,
		windowSizes:    config.WindowSizes(),
		tokenEmbedding: layers.NewEmbedding(paddedVocabulary, config.EmbedDim),
		blocks:         make([]*Block, config.NumLayer),
		lmHead:         layers.NewLinear(config.EmbedDim, paddedVocabulary),
		residLambdas:   tensors.New(config.NumLayer),
		x0Lambdas:      tensors.New(config.NumLayer),
		smearGate:      layers.NewLinear(smearGateChannels, 1),
		smearLambda:    tensors.New(1),
		backoutLambda:  tensors.New(1),
		valueEmbeds:    make(map[int]*layers.Embedding),
		rotaryCosine:   rotaryCosine,
		rotarySine:     rotarySine,
	}
	for layerIndex := 0; layerIndex < config.NumLayer; layerIndex++ {
		model.blocks[layerIndex] = NewBlock(config, hasValueEmbedding(layerIndex, config.NumLayer))
		if hasValueEmbedding(layerIndex, config.NumLayer) {
			model.valueEmbeds[layerIndex] = layers.NewEmbedding(paddedVocabulary, config.NumKVHead*config.HeadDim())
		}
	}
	return model
}

// NumLayers returns the number of transformer blocks.
func (model *Transformer) NumLayers() int { return model.Config.NumLayer }

// PaddedVocab returns the padded vocabulary size.
func (model *Transformer) PaddedVocab() int { return model.paddedVocab }

// KVHeadDim returns the KV embedding dimension (numKVHead * headDim).
func (model *Transformer) KVHeadDim() int {
	return model.Config.NumKVHead * model.Config.HeadDim()
}

// InitWeights initializes every parameter exactly as nanochat's init_weights.
func (model *Transformer) InitWeights(rng *tensors.RNG) {
	config := model.Config
	bound := uniformBound(config.EmbedDim)

	layers.InitNormal(model.tokenEmbedding.Weight, rng, 0.8)
	layers.InitNormal(model.lmHead.Weight, rng, 0.001)

	for _, block := range model.blocks {
		layers.InitUniform(block.attention.queryProjection.Weight, rng, -bound, bound)
		layers.InitUniform(block.attention.keyProjection.Weight, rng, -bound, bound)
		layers.InitUniform(block.attention.valueProjection.Weight, rng, -bound, bound)
		layers.InitZeros(block.attention.outputProjection.Weight)
		layers.InitUniform(block.mlp.inputProjection.Weight, rng, -0.4*bound, 0.4*bound)
		layers.InitZeros(block.mlp.outputProjection.Weight)
	}

	numLayers := config.NumLayer
	for layerIndex := 0; layerIndex < numLayers; layerIndex++ {
		model.residLambdas.Data[layerIndex] = 1.15 - 0.10*float32(layerIndex)/float32(max(numLayers-1, 1))
		model.x0Lambdas.Data[layerIndex] = 0.20 - 0.15*float32(layerIndex)/float32(max(numLayers-1, 1))
	}

	model.smearLambda.Data[0] = 0
	model.backoutLambda.Data[0] = 0.2
	layers.InitUniform(model.smearGate.Weight, rng, 0.0, 0.02)

	for _, valueEmbedding := range model.valueEmbeds {
		layers.InitUniform(valueEmbedding.Weight, rng, -bound, bound)
	}

	for _, block := range model.blocks {
		if block.attention.valueEmbeddingGate != nil {
			layers.InitUniform(block.attention.valueEmbeddingGate.Weight, rng, 0.0, 0.02)
		}
		if block.attention.compressor != nil {
			layers.InitUniform(block.attention.compressor.logitWeight.Weight, rng, -bound, bound)
			layers.InitZeros(block.attention.compressor.bias)
		}
		if block.attention.indexer != nil {
			layers.InitUniform(block.attention.indexer.query.Weight, rng, -bound, bound)
			layers.InitUniform(block.attention.indexer.key.Weight, rng, -bound, bound)
			layers.InitValue(block.attention.indexer.headWeights, 1)
		}
	}
}

// Forward runs the model and returns the softcapped logits of shape
// [B,T,vocab]. indexes has shape [B,T]. cache, when non-nil, enables
// autoregressive inference (KV-cache). Loss computation is intentionally kept
// out of the model; callers use tensors.CrossEntropy on the returned logits.
func (model *Transformer) Forward(indexes *tensors.Int32s, cache *KVBuffer) *tensors.Tensor {
	batchSize, sequenceLength := indexes.Shape[0], indexes.Shape[1]
	if sequenceLength > model.Config.SequenceLen {
		panic("model: sequence longer than rotary cache")
	}

	activations := model.tokenEmbedding.Forward(indexes) // [B,T,C]
	activations = normalizeLastDim(activations)

	positionOffset := 0
	if cache != nil {
		positionOffset = cache.Position()
	}

	// Smear: mix the previous token's embedding into the current token.
	switch {
	case cache == nil:
		if sequenceLength <= 1 {
			panic("model: training forward requires T > 1")
		}
		gate := model.smearGate.Forward(sliceChannels(activations, 1, sequenceLength, smearGateChannels)) // [B,T-1,1]
		gate = tensors.Scale(tensors.Sigmoid(gate), model.smearLambda.Data[0])
		smearAdd(activations, gate)
	default:
		previousEmbedding := cache.PrevEmbedding()
		cache.SetPrevEmbedding(lastTokenEmbedding(activations))
		if sequenceLength > 1 {
			gate := model.smearGate.Forward(sliceChannels(activations, 1, sequenceLength, smearGateChannels))
			gate = tensors.Scale(tensors.Sigmoid(gate), model.smearLambda.Data[0])
			smearAdd(activations, gate)
		} else if previousEmbedding != nil {
			gate := model.smearGate.Forward(sliceChannels(activations, 0, 1, smearGateChannels)) // [B,1,1]
			gate = tensors.Scale(tensors.Sigmoid(gate), model.smearLambda.Data[0])
			smearDecode(activations, gate, previousEmbedding)
		}
	}

	initialResidual := activations
	backoutLayerIndex := model.Config.NumLayer / 2
	var backoutActivation *tensors.Tensor
	for layerIndex, block := range model.blocks {
		activations = combineResidual(activations, initialResidual, model.residLambdas.Data[layerIndex], model.x0Lambdas.Data[layerIndex])
		var valueEmbedding *tensors.Tensor
		if embedding, ok := model.valueEmbeds[layerIndex]; ok {
			valueEmbedding = embedding.Forward(indexes)
		}
		activations = block.Forward(activations, valueEmbedding, model.rotaryCosine, model.rotarySine, positionOffset, model.windowSizes[layerIndex], cache, layerIndex)
		if layerIndex == backoutLayerIndex {
			backoutActivation = activations.Clone()
		}
	}

	if cache != nil {
		cache.Advance(sequenceLength)
	}

	if backoutActivation != nil {
		activations = tensors.AddScaled(activations, backoutActivation, -model.backoutLambda.Data[0])
	}
	activations = normalizeLastDim(activations)

	logits := model.lmHead.Forward(activations) // [B,T,paddedVocab]
	logits = trimVocab(logits, model.paddedVocab, model.Config.VocabSize)
	logits = tensors.Softcap(logits, softcap)
	return logits.Reshape(batchSize, sequenceLength, model.Config.VocabSize)
}

// lastTokenEmbedding copies the last token's embedding of each batch row into
// a fresh [B,1,C] tensors.
func lastTokenEmbedding(activations *tensors.Tensor) *tensors.Tensor {
	batchSize, sequenceLength, embeddingDimension := activations.Shape[0], activations.Shape[1], activations.Shape[2]
	output := tensors.New(batchSize, 1, embeddingDimension)
	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		source := activations.Data[(batchIndex*sequenceLength+(sequenceLength-1))*embeddingDimension : (batchIndex*sequenceLength+(sequenceLength-1))*embeddingDimension+embeddingDimension]
		copy(output.Data[batchIndex*embeddingDimension:(batchIndex+1)*embeddingDimension], source)
	}
	return output
}

// smearDecode adds gate * previousEmbedding to activations for the single-token
// decode case. gate has shape [B,1,1]; activations and previousEmbedding have
// shape [B,1,C].
func smearDecode(activations, gate, previousEmbedding *tensors.Tensor) {
	batchSize, embeddingDimension := activations.Shape[0], activations.Shape[2]
	activationData, gateData, previousData := activations.Data, gate.Data, previousEmbedding.Data
	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		gateValue := gateData[batchIndex]
		base := batchIndex * embeddingDimension
		for channelIndex := 0; channelIndex < embeddingDimension; channelIndex++ {
			activationData[base+channelIndex] += gateValue * previousData[base+channelIndex]
		}
	}
}

// trimVocab crops the trailing padded-vocab channels down to vocabSize, given
// a [B*T, paddedVocab] tensors.
func trimVocab(logits *tensors.Tensor, paddedVocabSize, vocabSize int) *tensors.Tensor {
	rowCount := logits.Numel() / paddedVocabSize
	output := tensors.New(rowCount, vocabSize)
	for rowIndex := 0; rowIndex < rowCount; rowIndex++ {
		copy(output.Data[rowIndex*vocabSize:(rowIndex+1)*vocabSize], logits.Data[rowIndex*paddedVocabSize:rowIndex*paddedVocabSize+vocabSize])
	}
	return output
}

// combineResidual computes residualWeight*activations + initialWeight*initialResidual element-wise.
func combineResidual(activations, initialResidual *tensors.Tensor, residualWeight, initialWeight float32) *tensors.Tensor {
	output := tensors.New(activations.Shape...)
	activationData, outputData, initialData := activations.Data, output.Data, initialResidual.Data
	for elementIndex := range activationData {
		outputData[elementIndex] = residualWeight*activationData[elementIndex] + initialWeight*initialData[elementIndex]
	}
	return output
}
