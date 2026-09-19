package model

import (
	"github.com/cookiengineer/gonano/tensors"
)

// This file implements the training forward/backward passes. Unlike the
// inference Forward, the training forward saves intermediate activations so
// that TrainBackward can compute exact gradients via backpropagation. The
// analytic gradients are verified against finite differences in tests.

// negateSineTable returns a negated copy of the sine rotary table, used to
// transpose the rotary rotation during backpropagation.
func negateSineTable(sine *tensors.Tensor) *tensors.Tensor {
	negated := tensors.New(sine.Shape...)
	for index, value := range sine.Data {
		negated.Data[index] = -value
	}
	return negated
}

// attentionContext holds the activations saved by the training attention
// forward.
type attentionContext struct {
	queryRotary     *tensors.Tensor // [B,T,Hq,D] post-rotary, pre-norm
	keyRotary       *tensors.Tensor // [B,T,Hkv,D] post-rotary, pre-norm
	queryHeadMajor  *tensors.Tensor // [B,Hq,T,D] post-norm+scale
	keyHeadMajor    *tensors.Tensor // [B,Hkv,T,D] post-norm+scale
	valueHeadMajor  *tensors.Tensor // [B,Hkv,T,D] final value (post gate)
	logSumExp       *tensors.Tensor // [B,Hq,T] flash-attention statistic
	outputHeadMajor *tensors.Tensor // [B,Hq,T,D] attention output (pre outputProjection)
	window          int

	valueGateSigmoid       *tensors.Tensor // [B,T,Hkv] sigmoid gate output (pre *3), nil if no value embedding
	valueEmbedding         *tensors.Tensor // [B,T,Hkv,D] value embedding, nil if none
	valueEmbeddingGradient *tensors.Tensor // [B,T,Hkv,D] gradient wrt the value embedding (set by backward)

	compression *compressedContext // non-nil on HCA-style compressed layers
}

// forwardTraining runs the attention forward while saving activations. The
// attention itself is computed by the flash-attention kernel, which returns the
// per-query log-sum-exp instead of the full probability matrix.
func (attention *CausalSelfAttention) forwardTraining(input, valueEmbedding, cosine, sine *tensors.Tensor, positionOffset int, window [2]int) (*tensors.Tensor, *attentionContext) {
	batchSize, sequenceLength := input.Shape[0], input.Shape[1]
	query := attention.queryProjection.Forward(input).Reshape(batchSize, sequenceLength, attention.queryHeadCount, attention.headDimension)
	key := attention.keyProjection.Forward(input).Reshape(batchSize, sequenceLength, attention.keyValueHeadCount, attention.headDimension)
	value := attention.valueProjection.Forward(input).Reshape(batchSize, sequenceLength, attention.keyValueHeadCount, attention.headDimension)

	context := &attentionContext{window: window[0]}
	if valueEmbedding != nil {
		valueEmbeddingHeads := valueEmbedding.Reshape(batchSize, sequenceLength, attention.keyValueHeadCount, attention.headDimension)
		preGate := attention.valueEmbeddingGate.Forward(sliceChannels(input, 0, sequenceLength, veGateChannels)) // [B,T,Hkv]
		sigmoid := tensors.Sigmoid(preGate)
		context.valueGateSigmoid = sigmoid
		gate := tensors.Scale(sigmoid, veGateScale)
		value = addGateTimesValueEmbedding(value, gate, valueEmbeddingHeads)
		context.valueEmbedding = valueEmbeddingHeads
	}

	query = ApplyRotary(query, cosine, sine, positionOffset)
	key = ApplyRotary(key, cosine, sine, positionOffset)
	context.queryRotary = query
	context.keyRotary = key

	query = tensors.Scale(normalizeLastDim(query), qkScale)
	key = tensors.Scale(normalizeLastDim(key), qkScale)

	queryHeadMajor := toBatchHeadLayout(query)
	keyHeadMajor := toBatchHeadLayout(key)
	valueHeadMajor := toBatchHeadLayout(value)
	context.queryHeadMajor = queryHeadMajor
	context.keyHeadMajor = keyHeadMajor
	context.valueHeadMajor = valueHeadMajor

	headRatio := attention.queryHeadCount / attention.keyValueHeadCount
	headDimension := attention.headDimension
	outputHeadMajor := tensors.New(batchSize, attention.queryHeadCount, sequenceLength, headDimension)
	logSumExp := tensors.New(batchSize, attention.queryHeadCount, sequenceLength)

	if attention.compressor != nil {
		context.compression = compressForward(attention.compressor, input, keyHeadMajor, valueHeadMajor, attention.compressionRatio)
		compressedAttentionForward(queryHeadMajor, context.compression.keyCompressed, context.compression.valueCompressed, outputHeadMajor, logSumExp, headRatio, attention.compressionRatio)
	} else {
		for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
			for queryHead := 0; queryHead < attention.queryHeadCount; queryHead++ {
				keyValueHead := queryHead / headRatio
				queryHeadData := headSlice(queryHeadMajor.Data, batchIndex*attention.queryHeadCount+queryHead, sequenceLength, headDimension)
				keyHeadData := headSlice(keyHeadMajor.Data, batchIndex*attention.keyValueHeadCount+keyValueHead, sequenceLength, headDimension)
				valueHeadData := headSlice(valueHeadMajor.Data, batchIndex*attention.keyValueHeadCount+keyValueHead, sequenceLength, headDimension)
				outputHeadData := headSlice(outputHeadMajor.Data, batchIndex*attention.queryHeadCount+queryHead, sequenceLength, headDimension)
				logSumExpHead := headSlice(logSumExp.Data, batchIndex*attention.queryHeadCount+queryHead, sequenceLength, 1)
				tensors.AttentionForward(queryHeadData, keyHeadData, valueHeadData, outputHeadData, logSumExpHead, sequenceLength, sequenceLength, headDimension, positionOffset, window[0])
			}
		}
	}
	context.logSumExp = logSumExp
	context.outputHeadMajor = outputHeadMajor

	output := toBatchSequenceLayout(outputHeadMajor).Reshape(batchSize, sequenceLength, attention.queryHeadCount*attention.headDimension)
	return attention.outputProjection.Forward(output), context
}

// backwardTraining runs the attention backward given the gradient of its output
// and the saved context. It returns the gradient with respect to the attention
// input. Key and value gradients accumulate across query heads that share a
// key/value head (grouped-query attention).
func (attention *CausalSelfAttention) backwardTraining(input *tensors.Tensor, outputGradient *tensors.Tensor, context *attentionContext, cosine, sine *tensors.Tensor, positionOffset int) *tensors.Tensor {
	batchSize, sequenceLength := input.Shape[0], input.Shape[1]
	headDimension := attention.headDimension
	headRatio := attention.queryHeadCount / attention.keyValueHeadCount

	// outputProjection backward.
	outputFlat := toBatchSequenceLayout(context.outputHeadMajor).Reshape(batchSize, sequenceLength, attention.queryHeadCount*attention.headDimension)
	gradientOutputHeadMajor := attention.outputProjection.Backward(outputFlat, outputGradient).Reshape(batchSize, sequenceLength, attention.queryHeadCount, attention.headDimension)
	gradientOutputHeadMajor = toBatchHeadLayout(gradientOutputHeadMajor)

	gradientQuery := tensors.New(batchSize, attention.queryHeadCount, sequenceLength, headDimension)
	var gradientKey, gradientValue *tensors.Tensor
	var gradientInputFromCompressor *tensors.Tensor
	if attention.compressor != nil {
		blocks := context.compression.blocks
		gradientKeyCompressed := tensors.New(batchSize, attention.keyValueHeadCount, blocks, headDimension)
		gradientValueCompressed := tensors.New(batchSize, attention.keyValueHeadCount, blocks, headDimension)
		compressedAttentionBackward(context.queryHeadMajor, context.compression.keyCompressed, context.compression.valueCompressed,
			context.outputHeadMajor, gradientOutputHeadMajor, context.logSumExp,
			gradientQuery, gradientKeyCompressed, gradientValueCompressed, headRatio, attention.compressionRatio)
		gradientInputFromCompressor, gradientKey, gradientValue = compressBackward(attention.compressor, context.compression, gradientKeyCompressed, gradientValueCompressed)
	} else {
		gradientKey = tensors.New(batchSize, attention.keyValueHeadCount, sequenceLength, headDimension)
		gradientValue = tensors.New(batchSize, attention.keyValueHeadCount, sequenceLength, headDimension)
		for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
			for queryHead := 0; queryHead < attention.queryHeadCount; queryHead++ {
				keyValueHead := queryHead / headRatio
				queryHeadData := headSlice(context.queryHeadMajor.Data, batchIndex*attention.queryHeadCount+queryHead, sequenceLength, headDimension)
				keyHeadData := headSlice(context.keyHeadMajor.Data, batchIndex*attention.keyValueHeadCount+keyValueHead, sequenceLength, headDimension)
				valueHeadData := headSlice(context.valueHeadMajor.Data, batchIndex*attention.keyValueHeadCount+keyValueHead, sequenceLength, headDimension)
				outputHeadData := headSlice(context.outputHeadMajor.Data, batchIndex*attention.queryHeadCount+queryHead, sequenceLength, headDimension)
				outputGradientHeadData := headSlice(gradientOutputHeadMajor.Data, batchIndex*attention.queryHeadCount+queryHead, sequenceLength, headDimension)
				logSumExpHead := headSlice(context.logSumExp.Data, batchIndex*attention.queryHeadCount+queryHead, sequenceLength, 1)
				queryGradientHead := headSlice(gradientQuery.Data, batchIndex*attention.queryHeadCount+queryHead, sequenceLength, headDimension)
				keyGradientHead := headSlice(gradientKey.Data, batchIndex*attention.keyValueHeadCount+keyValueHead, sequenceLength, headDimension)
				valueGradientHead := headSlice(gradientValue.Data, batchIndex*attention.keyValueHeadCount+keyValueHead, sequenceLength, headDimension)

				tensors.AttentionBackward(queryHeadData, keyHeadData, valueHeadData, outputHeadData, outputGradientHeadData, logSumExpHead, queryGradientHead, keyGradientHead, valueGradientHead, sequenceLength, sequenceLength, headDimension, positionOffset, context.window)
			}
		}
	}

	// Unscale QK (qkScale) and back through QK norm.
	gradientQuery = tensors.Scale(gradientQuery, qkScale)
	gradientKey = tensors.Scale(gradientKey, qkScale)
	gradientQuerySequence := toBatchSequenceLayout(gradientQuery)
	gradientKeySequence := toBatchSequenceLayout(gradientKey)

	gradientQueryPreNorm := normalizeLastDimBackward(context.queryRotary, gradientQuerySequence)
	gradientKeyPreNorm := normalizeLastDimBackward(context.keyRotary, gradientKeySequence)

	// Rotary backward: transpose the rotation (negate the sine table).
	negatedSine := negateSineTable(sine)
	gradientQueryProjected := ApplyRotary(gradientQueryPreNorm, cosine, negatedSine, positionOffset)
	gradientKeyProjected := ApplyRotary(gradientKeyPreNorm, cosine, negatedSine, positionOffset)

	// Accumulate into the input via the projection layers.
	gradientInput := attention.queryProjection.Backward(input, gradientQueryProjected.Reshape(batchSize, sequenceLength, attention.queryHeadCount*attention.headDimension))
	gradientInput = tensors.Add(gradientInput, attention.keyProjection.Backward(input, gradientKeyProjected.Reshape(batchSize, sequenceLength, attention.keyValueHeadCount*attention.headDimension)))

	// Value embedding gate backward.
	gradientValueSequence := toBatchSequenceLayout(gradientValue)
	if context.valueEmbedding != nil {
		// gradient_valueEmbedding = gradient_value * gate ; gradient_gate =
		// sum_dim(gradient_value * valueEmbedding).
		gate := tensors.Scale(context.valueGateSigmoid, veGateScale)
		valueEmbedding := context.valueEmbedding
		gradientValueData := gradientValueSequence.Data
		gradientValueEmbedding := tensors.New(batchSize, sequenceLength, attention.keyValueHeadCount, headDimension)
		gradientGate := tensors.New(batchSize, sequenceLength, attention.keyValueHeadCount)
		rowCount := batchSize * sequenceLength * attention.keyValueHeadCount
		for row := 0; row < rowCount; row++ {
			scale := gate.Data[row]
			base := row * headDimension
			var dotProduct float64
			for dimension := 0; dimension < headDimension; dimension++ {
				gradientValueEmbedding.Data[base+dimension] = gradientValueData[base+dimension] * scale
				dotProduct += float64(gradientValueData[base+dimension]) * float64(valueEmbedding.Data[base+dimension])
			}
			gradientGate.Data[row] = float32(dotProduct)
		}
		context.valueEmbeddingGradient = gradientValueEmbedding

		// Sigmoid backward: gate = veGateScale * sigmoid(preGate).
		gradientSigmoid := tensors.Scale(gradientGate, veGateScale)
		gradientPreGate := tensors.New(batchSize, sequenceLength, attention.keyValueHeadCount)
		for index := range context.valueGateSigmoid.Data {
			sigmoid := context.valueGateSigmoid.Data[index]
			gradientPreGate.Data[index] = gradientSigmoid.Data[index] * sigmoid * (1 - sigmoid)
		}
		gradientGateSlice := attention.valueEmbeddingGate.Backward(sliceChannels(input, 0, sequenceLength, veGateChannels), gradientPreGate) // [B,T,12]
		scattered := tensors.New(batchSize, sequenceLength, attention.embeddingDimension)
		scatterAddChannels(scattered, 0, sequenceLength, veGateChannels, gradientGateSlice)
		gradientInput = tensors.Add(gradientInput, scattered)
	}

	// value projection gradient (gradientValue flows through valueProjection).
	gradientInput = tensors.Add(gradientInput, attention.valueProjection.Backward(input, gradientValueSequence.Reshape(batchSize, sequenceLength, attention.keyValueHeadCount*attention.headDimension)))

	// The compressor logit projection reads the attention input directly.
	if gradientInputFromCompressor != nil {
		gradientInput = tensors.Add(gradientInput, gradientInputFromCompressor)
	}

	return gradientInput
}

// normalizeLastDimBackward applies RMSNorm backward to an arbitrary-rank
// tensor.
func normalizeLastDimBackward(input, gradientOutput *tensors.Tensor) *tensors.Tensor {
	lastDimension := input.Shape[len(input.Shape)-1]
	rowCount := input.Numel() / lastDimension
	return tensors.RMSNormBackward(input.Reshape(rowCount, lastDimension), gradientOutput.Reshape(rowCount, lastDimension), normEps).Reshape(input.Shape...)
}

// mlpContext holds activations for the MLP backward.
type mlpContext struct {
	input     *tensors.Tensor // input to inputProjection (== norm input)
	projected *tensors.Tensor // inputProjection output (pre ReLU-squared)
	hidden    *tensors.Tensor // ReLU-squared output (input to outputProjection)
}

func (mlp *MLP) forwardTraining(input *tensors.Tensor) (*tensors.Tensor, *mlpContext) {
	projected := mlp.inputProjection.Forward(input)
	hidden := tensors.ReluSquared(projected)
	output := mlp.outputProjection.Forward(hidden)
	return output, &mlpContext{input: input, projected: projected, hidden: hidden}
}

func (mlp *MLP) backwardTraining(outputGradient *tensors.Tensor, context *mlpContext) *tensors.Tensor {
	gradientHidden := mlp.outputProjection.Backward(context.hidden, outputGradient)
	gradientProjected := tensors.ReluSquaredBackward(context.projected, gradientHidden)
	return mlp.inputProjection.Backward(context.input, gradientProjected)
}

// blockContext holds activations for the block backward.
type blockContext struct {
	input              *tensors.Tensor // block input (after combineResidual)
	attentionNormInput *tensors.Tensor // norm(input), the attention's actual input
	attentionOutput    *tensors.Tensor // attention output (pre residual)
	attentionContext   *attentionContext
	mlpNormInput       *tensors.Tensor // input after attention residual (input to mlp norm)
	mlpOutput          *tensors.Tensor
	mlpContext         *mlpContext
}

func (block *Block) forwardTraining(input, valueEmbedding, cosine, sine *tensors.Tensor, positionOffset int, window [2]int) (*tensors.Tensor, *blockContext) {
	context := &blockContext{input: input}
	attentionInput := normalizeLastDim(input)
	context.attentionNormInput = attentionInput
	attentionOutput, attentionCtx := block.attention.forwardTraining(attentionInput, valueEmbedding, cosine, sine, positionOffset, window)
	context.attentionOutput = attentionOutput
	context.attentionContext = attentionCtx
	mid := tensors.Add(input, attentionOutput)
	context.mlpNormInput = mid
	mlpOutput, mlpContext := block.mlp.forwardTraining(normalizeLastDim(mid))
	context.mlpOutput = mlpOutput
	context.mlpContext = mlpContext
	return tensors.Add(mid, mlpOutput), context
}

func (block *Block) backwardTraining(outputGradient *tensors.Tensor, context *blockContext, cosine, sine *tensors.Tensor, positionOffset int) *tensors.Tensor {
	// Forward: mid = input + attentionOutput; output = mid + mlpOutput. The
	// residual connections mean the gradient flows both through each sublayer
	// and directly (identity) across it.
	gradientMid := outputGradient

	gradientMLPNormInput := block.mlp.backwardTraining(outputGradient, context.mlpContext)
	gradientMid = tensors.Add(gradientMid, normalizeLastDimBackward(context.mlpNormInput, gradientMLPNormInput))

	gradientAttentionNormInput := block.attention.backwardTraining(context.attentionNormInput, gradientMid, context.attentionContext, cosine, sine, positionOffset)
	gradientInput := normalizeLastDimBackward(context.input, gradientAttentionNormInput)
	// Identity residual path.
	gradientInput = tensors.Add(gradientInput, gradientMid)
	return gradientInput
}
