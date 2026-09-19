package model

import (
	"math"

	"github.com/cookiengineer/gonano/tensors"
)

// compressedContext holds the compressor state for one compressed-attention
// forward pass, needed by the backward pass.
type compressedContext struct {
	input           *tensors.Tensor // [B, T, d] attention input (compressor hidden)
	keyCompressed   *tensors.Tensor // [B, Hkv, blocks, D]
	valueCompressed *tensors.Tensor
	keyContexts     []*compressorContext // len B*Hkv
	valueContexts   []*compressorContext
	blocks          int
	ratio           int
}

// compressForward compresses the head-major key and value [B, Hkv, T, D] into
// [B, Hkv, blocks, D], using the attention input as the compressor hidden
// state. One context per (batch, kv-head) is retained for backward.
func compressForward(compressor *ChannelCompressor, input, key, value *tensors.Tensor, ratio int) *compressedContext {
	batchSize := key.Shape[0]
	kvHeadCount := key.Shape[1]
	sequenceLength := key.Shape[2]
	headDimension := key.Shape[3]
	embeddingDimension := input.Shape[2]
	blocks := (sequenceLength + ratio - 1) / ratio

	keyCompressed := tensors.New(batchSize, kvHeadCount, blocks, headDimension)
	valueCompressed := tensors.New(batchSize, kvHeadCount, blocks, headDimension)
	keyContexts := make([]*compressorContext, batchSize*kvHeadCount)
	valueContexts := make([]*compressorContext, batchSize*kvHeadCount)

	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		hidden := tensors.NewWithData([]int{1, sequenceLength, embeddingDimension},
			input.Data[batchIndex*sequenceLength*embeddingDimension:(batchIndex+1)*sequenceLength*embeddingDimension])
		for head := 0; head < kvHeadCount; head++ {
			index := batchIndex*kvHeadCount + head
			keyHead := tensors.NewWithData([]int{1, sequenceLength, headDimension}, headSlice(key.Data, index, sequenceLength, headDimension))
			valueHead := tensors.NewWithData([]int{1, sequenceLength, headDimension}, headSlice(value.Data, index, sequenceLength, headDimension))
			compressedKey, keyContext := compressor.Forward(hidden, keyHead)
			compressedValue, valueContext := compressor.Forward(hidden, valueHead)
			copy(keyCompressed.Data[index*blocks*headDimension:], compressedKey.Data)
			copy(valueCompressed.Data[index*blocks*headDimension:], compressedValue.Data)
			keyContexts[index] = keyContext
			valueContexts[index] = valueContext
		}
	}

	return &compressedContext{
		input:           input,
		keyCompressed:   keyCompressed,
		valueCompressed: valueCompressed,
		keyContexts:     keyContexts,
		valueContexts:   valueContexts,
		blocks:          blocks,
		ratio:           ratio,
	}
}

// compressedAttentionForward computes attention for every (batch, query head)
// against the compressed key/value blocks that strictly precede the query's own
// compression block. Queries in block 0 have no preceding block and produce
// zero output.
func compressedAttentionForward(queryHeadMajor, keyCompressed, valueCompressed, outputHeadMajor, logSumExp *tensors.Tensor, headRatio, ratio int) {
	batchSize := queryHeadMajor.Shape[0]
	queryHeadCount := queryHeadMajor.Shape[1]
	sequenceLength := queryHeadMajor.Shape[2]
	headDimension := queryHeadMajor.Shape[3]
	kvHeadCount := keyCompressed.Shape[1]
	blocks := keyCompressed.Shape[2]

	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		for queryHead := 0; queryHead < queryHeadCount; queryHead++ {
			queryIndex := batchIndex*queryHeadCount + queryHead
			keyValueHead := queryHead / headRatio
			keyBase := (batchIndex*kvHeadCount + keyValueHead) * blocks * headDimension
			for block := 0; block < blocks; block++ {
				start := block * ratio
				end := min(start+ratio, sequenceLength)
				if end <= start {
					continue
				}
				rows := end - start
				outputSlice := outputHeadMajor.Data[queryIndex*sequenceLength*headDimension+start*headDimension : queryIndex*sequenceLength*headDimension+end*headDimension]
				if block == 0 {
					clear(outputSlice)
					if logSumExp != nil {
						for row := start; row < end; row++ {
							logSumExp.Data[queryIndex*sequenceLength+row] = float32(math.Inf(-1))
						}
					}
					continue
				}
				querySlice := queryHeadMajor.Data[queryIndex*sequenceLength*headDimension+start*headDimension : queryIndex*sequenceLength*headDimension+end*headDimension]
				keySlice := keyCompressed.Data[keyBase : keyBase+block*headDimension]
				valueSlice := valueCompressed.Data[keyBase : keyBase+block*headDimension]
				var logSumExpSlice []float32
				if logSumExp != nil {
					logSumExpSlice = logSumExp.Data[queryIndex*sequenceLength+start : queryIndex*sequenceLength+end]
				}
				// PositionOffset == block makes every compressed key index
				// (0..block-1) smaller than every query position, disabling the
				// causal mask: the block restriction is already explicit here.
				tensors.AttentionForward(querySlice, keySlice, valueSlice, outputSlice, logSumExpSlice, rows, block, headDimension, block, -1)
			}
		}
	}
}

// compressedAttentionBackward accumulates the query, compressed-key, and
// compressed-value gradients for every (batch, query head) and block.
func compressedAttentionBackward(queryHeadMajor, keyCompressed, valueCompressed, outputHeadMajor, gradientOutputHeadMajor, logSumExp, gradientQueryHeadMajor, gradientKeyCompressed, gradientValueCompressed *tensors.Tensor, headRatio, ratio int) {
	batchSize := queryHeadMajor.Shape[0]
	queryHeadCount := queryHeadMajor.Shape[1]
	sequenceLength := queryHeadMajor.Shape[2]
	headDimension := queryHeadMajor.Shape[3]
	kvHeadCount := keyCompressed.Shape[1]
	blocks := keyCompressed.Shape[2]

	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		for queryHead := 0; queryHead < queryHeadCount; queryHead++ {
			queryIndex := batchIndex*queryHeadCount + queryHead
			keyValueHead := queryHead / headRatio
			keyBase := (batchIndex*kvHeadCount + keyValueHead) * blocks * headDimension
			for block := 1; block < blocks; block++ {
				start := block * ratio
				end := min(start+ratio, sequenceLength)
				if end <= start {
					continue
				}
				rows := end - start
				querySlice := queryHeadMajor.Data[queryIndex*sequenceLength*headDimension+start*headDimension : queryIndex*sequenceLength*headDimension+end*headDimension]
				outputSlice := outputHeadMajor.Data[queryIndex*sequenceLength*headDimension+start*headDimension : queryIndex*sequenceLength*headDimension+end*headDimension]
				gradientOutputSlice := gradientOutputHeadMajor.Data[queryIndex*sequenceLength*headDimension+start*headDimension : queryIndex*sequenceLength*headDimension+end*headDimension]
				gradientQuerySlice := gradientQueryHeadMajor.Data[queryIndex*sequenceLength*headDimension+start*headDimension : queryIndex*sequenceLength*headDimension+end*headDimension]
				logSumExpSlice := logSumExp.Data[queryIndex*sequenceLength+start : queryIndex*sequenceLength+end]
				tensors.AttentionBackward(querySlice, keyCompressed.Data[keyBase:keyBase+block*headDimension], valueCompressed.Data[keyBase:keyBase+block*headDimension],
					outputSlice, gradientOutputSlice, logSumExpSlice,
					gradientQuerySlice, gradientKeyCompressed.Data[keyBase:keyBase+block*headDimension], gradientValueCompressed.Data[keyBase:keyBase+block*headDimension],
					rows, block, headDimension, block, -1)
			}
		}
	}
}

// compressTail compresses every row's completed tail block into the cache.
func (attention *CausalSelfAttention) compressTail(cache *KVBuffer, layer, batchSize, embeddingDimension, headDimension int) {
	ratio := attention.compressionRatio
	kvHeadCount := attention.keyValueHeadCount
	for batch := 0; batch < batchSize; batch++ {
		hidden := tensors.NewWithData([]int{1, ratio, embeddingDimension}, cache.TailHidden(layer, batch))
		for head := 0; head < kvHeadCount; head++ {
			index := batch*kvHeadCount + head
			keyHead := tensors.NewWithData([]int{1, ratio, headDimension}, cache.TailKey(layer, index))
			valueHead := tensors.NewWithData([]int{1, ratio, headDimension}, cache.TailValue(layer, index))
			compressedKey, _ := attention.compressor.Forward(hidden, keyHead)
			compressedValue, _ := attention.compressor.Forward(hidden, valueHead)
			cache.AppendCompressed(layer, batch, head, compressedKey.Data, compressedValue.Data)
		}
	}
	cache.AdvanceCompressedCount(layer)
	for batch := 0; batch < batchSize; batch++ {
		cache.ResetTail(layer, batch)
	}
}

// forwardCompressed is the inference path of an HCA-style compressed attention
// layer. It buffers incoming rows, compresses each completed block, and attends
// every query to the compressed blocks that strictly precede its own block.
func (attention *CausalSelfAttention) forwardCompressed(input, query, key, value *tensors.Tensor, cache *KVBuffer, layer, positionOffset int) *tensors.Tensor {
	batchSize := input.Shape[0]
	sequenceLength := input.Shape[1]
	embeddingDimension := input.Shape[2]
	queryHeadCount := attention.queryHeadCount
	kvHeadCount := attention.keyValueHeadCount
	headDimension := attention.headDimension
	ratio := attention.compressionRatio
	headRatio := queryHeadCount / kvHeadCount

	queryHeadMajor := toBatchHeadLayout(query) // [B, Hq, T, D]
	outputHeadMajor := tensors.New(batchSize, queryHeadCount, sequenceLength, headDimension)

	// Append every new row and compress blocks as they complete.
	for position := 0; position < sequenceLength; position++ {
		for batch := 0; batch < batchSize; batch++ {
			row := batch*sequenceLength + position
			hiddenRow := input.Data[row*embeddingDimension : row*embeddingDimension+embeddingDimension]
			keyRow := key.Data[row*kvHeadCount*headDimension : row*kvHeadCount*headDimension+kvHeadCount*headDimension]
			valueRow := value.Data[row*kvHeadCount*headDimension : row*kvHeadCount*headDimension+kvHeadCount*headDimension]
			cache.AppendTailRow(layer, batch, hiddenRow, keyRow, valueRow)
		}
		if cache.TailLength(layer, 0) == ratio {
			attention.compressTail(cache, layer, batchSize, embeddingDimension, headDimension)
		}
	}

	for batch := 0; batch < batchSize; batch++ {
		for queryHead := 0; queryHead < queryHeadCount; queryHead++ {
			keyValueHead := queryHead / headRatio
			queryIndex := batch*queryHeadCount + queryHead
			for position := 0; position < sequenceLength; position++ {
				absolutePosition := positionOffset + position
				count := absolutePosition / ratio
				outputRow := outputHeadMajor.Data[(queryIndex*sequenceLength+position)*headDimension : (queryIndex*sequenceLength+position+1)*headDimension]
				if count == 0 {
					clear(outputRow)
					continue
				}
				queryRow := queryHeadMajor.Data[(queryIndex*sequenceLength+position)*headDimension : (queryIndex*sequenceLength+position+1)*headDimension]
				keySlice := cache.CompressedKey(layer, batch, keyValueHead, count)
				valueSlice := cache.CompressedValue(layer, batch, keyValueHead, count)
				tensors.AttentionForward(queryRow, keySlice, valueSlice, outputRow, nil, 1, count, headDimension, count, -1)
			}
		}
	}

	output := toBatchSequenceLayout(outputHeadMajor).Reshape(batchSize, sequenceLength, queryHeadCount*headDimension)
	return attention.outputProjection.Forward(output)
}

// compressBackward backpropagates the compressed key/value gradients through
// the compressor, returning the gradient with respect to the attention input
// (the compressor hidden state) and the pre-compression key and value in
// head-major layout.
func compressBackward(compressor *ChannelCompressor, context *compressedContext, gradientKeyCompressed, gradientValueCompressed *tensors.Tensor) (gradientInput, gradientKey, gradientValue *tensors.Tensor) {
	batchSize := context.input.Shape[0]
	sequenceLength := context.input.Shape[1]
	embeddingDimension := context.input.Shape[2]
	kvHeadCount := context.keyCompressed.Shape[1]
	headDimension := context.keyCompressed.Shape[3]
	blocks := context.blocks

	gradientInput = tensors.New(batchSize, sequenceLength, embeddingDimension)
	gradientKey = tensors.New(batchSize, kvHeadCount, sequenceLength, headDimension)
	gradientValue = tensors.New(batchSize, kvHeadCount, sequenceLength, headDimension)

	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		for head := 0; head < kvHeadCount; head++ {
			index := batchIndex*kvHeadCount + head
			keyGradientSlice := tensors.NewWithData([]int{1, blocks, headDimension}, gradientKeyCompressed.Data[index*blocks*headDimension:(index+1)*blocks*headDimension])
			valueGradientSlice := tensors.NewWithData([]int{1, blocks, headDimension}, gradientValueCompressed.Data[index*blocks*headDimension:(index+1)*blocks*headDimension])
			gradientHiddenKey, preKeyGradient := compressor.Backward(keyGradientSlice, context.keyContexts[index])
			gradientHiddenValue, preValueGradient := compressor.Backward(valueGradientSlice, context.valueContexts[index])
			for element := 0; element < sequenceLength*embeddingDimension; element++ {
				gradientInput.Data[batchIndex*sequenceLength*embeddingDimension+element] += gradientHiddenKey.Data[element] + gradientHiddenValue.Data[element]
			}
			copy(gradientKey.Data[index*sequenceLength*headDimension:(index+1)*sequenceLength*headDimension], preKeyGradient.Data)
			copy(gradientValue.Data[index*sequenceLength*headDimension:(index+1)*sequenceLength*headDimension], preValueGradient.Data)
		}
	}
	return gradientInput, gradientKey, gradientValue
}
