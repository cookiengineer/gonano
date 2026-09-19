package model

import (
	"github.com/cookiengineer/gonano/tensors"
)

// trainCtx holds the activations saved by TrainForward for TrainBackward.
type trainCtx struct {
	indexes *tensors.Int32s

	embeddedNormalized *tensors.Tensor // activations after token embedding + norm, before smear
	rawEmbedding       *tensors.Tensor // activations after token embedding, before norm
	smearSlice         *tensors.Tensor // [B,T-1,24] input to the smear gate
	smearSigmoid       *tensors.Tensor // [B,T-1,1] sigmoid output (pre lambda)
	initialResidual    *tensors.Tensor // activations after smear

	previousActivations []*tensors.Tensor // activations before combineResidual, per layer
	blockContexts       []*blockContext

	backoutActivations *tensors.Tensor
	finalPreNorm       *tensors.Tensor // input to the final norm
	finalNorm          *tensors.Tensor // input to lm_head
	logits             *tensors.Tensor // softcapped logits [B,T,vocab]
	valueEmbeddings    map[int]*tensors.Tensor
}

// TrainForward runs the model forward, saving activations for backprop. It
// returns the softcapped logits and the training context.
func (model *Transformer) TrainForward(indexes *tensors.Int32s) (*tensors.Tensor, *trainCtx) {
	batchSize, sequenceLength := indexes.Shape[0], indexes.Shape[1]
	context := &trainCtx{indexes: indexes, valueEmbeddings: make(map[int]*tensors.Tensor)}

	activations := model.tokenEmbedding.Forward(indexes)
	context.rawEmbedding = activations
	activations = normalizeLastDim(activations)
	context.embeddedNormalized = activations

	smearSlice := sliceChannels(activations, 1, sequenceLength, smearGateChannels)
	context.smearSlice = smearSlice
	smearSigmoid := tensors.Sigmoid(model.smearGate.Forward(smearSlice))
	context.smearSigmoid = smearSigmoid
	smearAdd(activations, tensors.Scale(smearSigmoid, model.smearLambda.Data[0]))

	initialResidual := activations
	context.initialResidual = initialResidual

	backoutLayerIndex := model.Config.NumLayer / 2
	var backoutActivations *tensors.Tensor
	for layerIndex, block := range model.blocks {
		context.previousActivations = append(context.previousActivations, activations)
		activations = combineResidual(activations, initialResidual, model.residLambdas.Data[layerIndex], model.x0Lambdas.Data[layerIndex])
		var valueEmbedding *tensors.Tensor
		if embedding, ok := model.valueEmbeds[layerIndex]; ok {
			valueEmbedding = embedding.Forward(indexes)
			context.valueEmbeddings[layerIndex] = valueEmbedding
		}
		var blockCtx *blockContext
		activations, blockCtx = block.forwardTraining(activations, valueEmbedding, model.rotaryCosine, model.rotarySine, 0, model.windowSizes[layerIndex])
		context.blockContexts = append(context.blockContexts, blockCtx)
		if layerIndex == backoutLayerIndex {
			backoutActivations = activations.Clone()
		}
	}
	context.backoutActivations = backoutActivations

	if backoutActivations != nil {
		activations = tensors.AddScaled(activations, backoutActivations, -model.backoutLambda.Data[0])
	}
	context.finalPreNorm = activations
	activations = normalizeLastDim(activations)
	context.finalNorm = activations

	logits := model.lmHead.Forward(activations)
	logits = trimVocab(logits, model.paddedVocab, model.Config.VocabSize)
	logits = tensors.Softcap(logits, softcap)
	context.logits = logits.Reshape(batchSize, sequenceLength, model.Config.VocabSize)
	return context.logits, context
}

// TrainBackward backpropagates gradLogits (the gradient of the loss with
// respect to the softcapped logits) through the model, accumulating gradients
// into every parameter's Grad buffer.
func (model *Transformer) TrainBackward(context *trainCtx, gradLogits *tensors.Tensor) {
	batchSize, sequenceLength := context.indexes.Shape[0], context.indexes.Shape[1]
	embeddingDimension := model.Config.EmbedDim

	gradSoftcap := tensors.SoftcapBackward(context.logits, gradLogits, softcap) // [B,T,vocab]
	gradPadded := tensors.New(batchSize*sequenceLength, model.paddedVocab)
	for tokenIndex := 0; tokenIndex < batchSize*sequenceLength; tokenIndex++ {
		copy(gradPadded.Data[tokenIndex*model.paddedVocab:], gradSoftcap.Data[tokenIndex*model.Config.VocabSize:(tokenIndex+1)*model.Config.VocabSize])
	}

	gradFinalNorm := model.lmHead.Backward(context.finalNorm, gradPadded).Reshape(batchSize, sequenceLength, embeddingDimension)
	gradActivations := normalizeLastDimBackward(context.finalPreNorm, gradFinalNorm)

	var gradBackoutActivations *tensors.Tensor
	if context.backoutActivations != nil {
		gradBackoutActivations = tensors.Scale(gradActivations, -model.backoutLambda.Data[0])
		model.backoutLambda.EnsureGrad()
		var backoutDot float64
		for elementIndex := 0; elementIndex < gradActivations.Numel(); elementIndex++ {
			backoutDot += float64(gradActivations.Data[elementIndex]) * float64(-context.backoutActivations.Data[elementIndex])
		}
		model.backoutLambda.Grad[0] += float32(backoutDot)
	}

	numLayers := model.Config.NumLayer
	backoutLayerIndex := numLayers / 2
	gradInitialResidual := tensors.New(batchSize, sequenceLength, embeddingDimension)
	for layerIndex := numLayers - 1; layerIndex >= 0; layerIndex-- {
		layerGradient := gradActivations
		if layerIndex == backoutLayerIndex && gradBackoutActivations != nil {
			layerGradient = tensors.Add(gradActivations, gradBackoutActivations)
		}
		gradBlockInput := model.blocks[layerIndex].backwardTraining(layerGradient, context.blockContexts[layerIndex], model.rotaryCosine, model.rotarySine, 0)

		if valueEmbeddingGradient := context.blockContexts[layerIndex].attentionContext.valueEmbeddingGradient; valueEmbeddingGradient != nil {
			keyValueDimension := model.Config.NumKVHead * model.Config.HeadDim()
			model.valueEmbeds[layerIndex].Backward(context.indexes, valueEmbeddingGradient.Reshape(batchSize, sequenceLength, keyValueDimension))
		}

		residualWeight := model.residLambdas.Data[layerIndex]
		initialWeight := model.x0Lambdas.Data[layerIndex]
		previousActivations := context.previousActivations[layerIndex]
		gradActivations = tensors.Scale(gradBlockInput, residualWeight)
		gradInitialResidual = tensors.AddScaled(gradInitialResidual, gradBlockInput, initialWeight)

		model.residLambdas.EnsureGrad()
		model.x0Lambdas.EnsureGrad()
		var residualDot, initialDot float64
		for elementIndex := 0; elementIndex < gradBlockInput.Numel(); elementIndex++ {
			residualDot += float64(gradBlockInput.Data[elementIndex]) * float64(previousActivations.Data[elementIndex])
			initialDot += float64(gradBlockInput.Data[elementIndex]) * float64(context.initialResidual.Data[elementIndex])
		}
		model.residLambdas.Grad[layerIndex] += float32(residualDot)
		model.x0Lambdas.Grad[layerIndex] += float32(initialDot)
	}
	gradActivations = tensors.Add(gradActivations, gradInitialResidual)

	// Smear backward.
	gradEmbedded := smearBackward(model, context, gradActivations, batchSize, sequenceLength, embeddingDimension)

	// Norm between the embedding and the trunk.
	gradRaw := normalizeLastDimBackward(context.rawEmbedding, gradEmbedded)
	model.tokenEmbedding.Backward(context.indexes, gradRaw)
}

// smearBackward backpropagates through the smear operation and the smear gate,
// returning the gradient with respect to the pre-smear embedding.
func smearBackward(model *Transformer, context *trainCtx, gradActivations *tensors.Tensor, batchSize, sequenceLength, embeddingDimension int) *tensors.Tensor {
	smearLambda := model.smearLambda.Data[0]
	gradEmbedded := tensors.New(batchSize, sequenceLength, embeddingDimension)
	gradGate := tensors.New(batchSize, sequenceLength-1, 1)
	var gradSmearLambda float64
	embeddedActivations := context.embeddedNormalized

	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		for rowIndex := 0; rowIndex < sequenceLength; rowIndex++ {
			base := (batchIndex*sequenceLength + rowIndex) * embeddingDimension
			if rowIndex == 0 {
				for channelIndex := 0; channelIndex < embeddingDimension; channelIndex++ {
					gradEmbedded.Data[base+channelIndex] += gradActivations.Data[base+channelIndex]
				}
				continue
			}
			gateIndex := batchIndex*(sequenceLength-1) + (rowIndex - 1)
			gateValue := smearLambda * context.smearSigmoid.Data[gateIndex]
			previousBase := (batchIndex*sequenceLength + (rowIndex - 1)) * embeddingDimension
			var gradGateSum float64
			for channelIndex := 0; channelIndex < embeddingDimension; channelIndex++ {
				gradEmbedded.Data[base+channelIndex] += gradActivations.Data[base+channelIndex]
				gradEmbedded.Data[previousBase+channelIndex] += gradActivations.Data[base+channelIndex] * gateValue
				gradGateSum += float64(gradActivations.Data[base+channelIndex]) * float64(embeddedActivations.Data[previousBase+channelIndex])
			}
			sigmoidValue := context.smearSigmoid.Data[gateIndex]
			gradSmearLambda += gradGateSum * float64(sigmoidValue)
			gradGate.Data[gateIndex] = float32(gradGateSum*float64(smearLambda)) * sigmoidValue * (1 - sigmoidValue)
		}
	}

	model.smearLambda.EnsureGrad()
	model.smearLambda.Grad[0] += float32(gradSmearLambda)
	gradSlice := model.smearGate.Backward(context.smearSlice, gradGate)
	scatterAddChannels(gradEmbedded, 1, sequenceLength, smearGateChannels, gradSlice)
	return gradEmbedded
}

// scatterAddChannels adds source into the first `channels` channels of
// destination for rows [rowStart, rowEnd), reversing sliceChannels.
func scatterAddChannels(destination *tensors.Tensor, rowStart, rowEnd, channels int, source *tensors.Tensor) {
	batchSize, sequenceLength, embeddingDimension := destination.Shape[0], destination.Shape[1], destination.Shape[2]
	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		for rowIndex := rowStart; rowIndex < rowEnd; rowIndex++ {
			from := source.Data[(batchIndex*(rowEnd-rowStart)+(rowIndex-rowStart))*channels:]
			to := destination.Data[(batchIndex*sequenceLength+rowIndex)*embeddingDimension : (batchIndex*sequenceLength+rowIndex)*embeddingDimension+channels]
			for channelIndex := 0; channelIndex < channels; channelIndex++ {
				to[channelIndex] += from[channelIndex]
			}
		}
	}
}
