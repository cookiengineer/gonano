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
	// selection[b*Hkv+h][token] holds the selected compressed block indices for
	// sparse layers; nil on dense layers.
	selection [][][]int
	// indexerContexts, indexerScores, and indexerTargets are the per-(batch,
	// kv-head) indexer projections, scores, and distillation targets; nil on
	// dense layers.
	indexerContexts []*indexerContext
	indexerScores   []*tensors.Tensor
	indexerTargets  []*tensors.Tensor
}

// sparseTrainingPlan scores every compressed block with the indexer, selects
// the top-k blocks per token, and computes the distillation target: for each
// query token the softmax over its allowed compressed blocks of the mean main
// attention logit across the query heads sharing that key/value head. Training
// sees the full sequence, so the position offset is zero.
func sparseTrainingPlan(attention *CausalSelfAttention, input, keyCompressed, queryHeadMajor *tensors.Tensor) ([][][]int, []*indexerContext, []*tensors.Tensor, []*tensors.Tensor) {
	batchSize := keyCompressed.Shape[0]
	kvHeadCount := keyCompressed.Shape[1]
	blockCount := keyCompressed.Shape[2]
	headDimension := keyCompressed.Shape[3]
	sequenceLength := input.Shape[1]
	embeddingDimension := input.Shape[2]
	queryHeadCount := queryHeadMajor.Shape[1]
	headRatio := queryHeadCount / kvHeadCount
	ratio := attention.compressionRatio

	selection := make([][][]int, batchSize*kvHeadCount)
	contexts := make([]*indexerContext, batchSize*kvHeadCount)
	scoresByHead := make([]*tensors.Tensor, batchSize*kvHeadCount)
	targets := make([]*tensors.Tensor, batchSize*kvHeadCount)

	for batch := 0; batch < batchSize; batch++ {
		hiddenSlice := tensors.NewWithData([]int{1, sequenceLength, embeddingDimension},
			input.Data[batch*sequenceLength*embeddingDimension:(batch+1)*sequenceLength*embeddingDimension])
		for head := 0; head < kvHeadCount; head++ {
			index := batch*kvHeadCount + head
			compressed := tensors.NewWithData([]int{1, blockCount, headDimension}, headSlice(keyCompressed.Data, index, blockCount, headDimension))
			scores, context := attention.indexer.ScoresWithContext(hiddenSlice, compressed)
			selection[index] = SelectBlocks(scores, 0, ratio, attention.sparseTopK)
			contexts[index] = context
			scoresByHead[index] = scores
			targets[index] = distillationTarget(queryHeadMajor, keyCompressed, batch, head, headRatio, ratio, sequenceLength, blockCount, headDimension)
		}
	}
	return selection, contexts, scoresByHead, targets
}

// indexerDistillationGradient backpropagates the indexer distillation loss,
// whose gradient with respect to the index scores is (softmax(I) - target). It
// returns the gradients with respect to the attention input and the compressed
// keys.
func (attention *CausalSelfAttention) indexerDistillationGradient(context *compressedContext) (gradientInput, gradientKeyCompressed *tensors.Tensor) {
	batchSize := context.keyCompressed.Shape[0]
	kvHeadCount := context.keyCompressed.Shape[1]
	blockCount := context.keyCompressed.Shape[2]
	headDimension := context.keyCompressed.Shape[3]
	sequenceLength := context.input.Shape[1]
	embeddingDimension := context.input.Shape[2]
	weight := attention.indexerLossWeight

	gradientInput = tensors.New(batchSize, sequenceLength, embeddingDimension)
	gradientKeyCompressed = tensors.New(batchSize, kvHeadCount, blockCount, headDimension)

	for batch := 0; batch < batchSize; batch++ {
		for head := 0; head < kvHeadCount; head++ {
			index := batch*kvHeadCount + head
			probabilities := maskedIndexerSoftmax(context.indexerScores[index], attention.compressionRatio)
			target := context.indexerTargets[index]
			gradientScores := tensors.New(sequenceLength, blockCount)
			for element := range gradientScores.Data {
				gradientScores.Data[element] = (probabilities[element] - target.Data[element]) * weight
			}
			gradientHidden, gradientCompressed := attention.indexer.Backward(gradientScores, context.indexerContexts[index])
			for element := range gradientHidden.Data {
				gradientInput.Data[batch*sequenceLength*embeddingDimension+element] += gradientHidden.Data[element]
			}
			copy(gradientKeyCompressed.Data[index*blockCount*headDimension:], gradientCompressed.Data)
		}
	}
	return gradientInput, gradientKeyCompressed
}

// distillationTarget builds the [T, blocks] target distribution over the
// allowed compressed blocks from the mean main attention logits.
func distillationTarget(queryHeadMajor, keyCompressed *tensors.Tensor, batch, keyValueHead, headRatio, ratio, sequenceLength, blockCount, headDimension int) *tensors.Tensor {
	target := tensors.New(sequenceLength, blockCount)
	queryHeadCount := queryHeadMajor.Shape[1]
	for token := 0; token < sequenceLength; token++ {
		allowed := token / ratio
		if allowed > blockCount {
			allowed = blockCount
		}
		if allowed == 0 {
			continue
		}
		logits := make([]float32, allowed)
		maximum := float32(math.Inf(-1))
		for block := 0; block < allowed; block++ {
			var sum float32
			for queryHead := 0; queryHead < queryHeadCount; queryHead++ {
				if queryHead/headRatio != keyValueHead {
					continue
				}
				queryRow := queryHeadMajor.Data[(batch*queryHeadCount+queryHead)*sequenceLength*headDimension+token*headDimension : (batch*queryHeadCount+queryHead)*sequenceLength*headDimension+(token+1)*headDimension]
				keyRow := keyCompressed.Data[((batch*keyCompressed.Shape[1]+keyValueHead)*blockCount+block)*headDimension : ((batch*keyCompressed.Shape[1]+keyValueHead)*blockCount+block+1)*headDimension]
				var dot float32
				for element := 0; element < headDimension; element++ {
					dot += queryRow[element] * keyRow[element]
				}
				sum += dot
			}
			logits[block] = sum / float32(headRatio)
			if logits[block] > maximum {
				maximum = logits[block]
			}
		}
		var total float64
		for block := 0; block < allowed; block++ {
			value := math.Exp(float64(logits[block] - maximum))
			target.Data[token*blockCount+block] = float32(value)
			total += value
		}
		for block := 0; block < allowed; block++ {
			target.Data[token*blockCount+block] /= float32(total)
		}
	}
	return target
}

// maskedIndexerSoftmax returns the indexer softmax restricted to each token's
// allowed blocks (others are zero).
func maskedIndexerSoftmax(scores *tensors.Tensor, ratio int) []float32 {
	tokenCount := scores.Shape[0]
	blockCount := scores.Shape[1]
	probabilities := make([]float32, len(scores.Data))
	for token := 0; token < tokenCount; token++ {
		allowed := token / ratio
		if allowed > blockCount {
			allowed = blockCount
		}
		if allowed == 0 {
			continue
		}
		maximum := scores.Data[token*blockCount]
		for block := 1; block < allowed; block++ {
			if scores.Data[token*blockCount+block] > maximum {
				maximum = scores.Data[token*blockCount+block]
			}
		}
		var total float64
		for block := 0; block < allowed; block++ {
			value := math.Exp(float64(scores.Data[token*blockCount+block] - maximum))
			probabilities[token*blockCount+block] = float32(value)
			total += value
		}
		for block := 0; block < allowed; block++ {
			probabilities[token*blockCount+block] /= float32(total)
		}
	}
	return probabilities
}

// compressedAttentionForwardSparse is the training/prefill attention over the
// selected compressed blocks for every (batch, query head).
func compressedAttentionForwardSparse(queryHeadMajor, keyCompressed, valueCompressed, outputHeadMajor, logSumExp *tensors.Tensor, selection [][][]int, headRatio, ratio int) {
	batchSize := queryHeadMajor.Shape[0]
	queryHeadCount := queryHeadMajor.Shape[1]
	sequenceLength := queryHeadMajor.Shape[2]
	headDimension := queryHeadMajor.Shape[3]
	kvHeadCount := keyCompressed.Shape[1]
	blockCount := keyCompressed.Shape[2]

	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		for queryHead := 0; queryHead < queryHeadCount; queryHead++ {
			queryIndex := batchIndex*queryHeadCount + queryHead
			keyValueHead := queryHead / headRatio
			keyBase := (batchIndex*kvHeadCount + keyValueHead) * blockCount * headDimension
			for token := 0; token < sequenceLength; token++ {
				outputSlice := outputHeadMajor.Data[queryIndex*sequenceLength*headDimension+token*headDimension : queryIndex*sequenceLength*headDimension+(token+1)*headDimension]
				selected := selection[batchIndex*kvHeadCount+keyValueHead][token]
				if len(selected) == 0 {
					clear(outputSlice)
					if logSumExp != nil {
						logSumExp.Data[queryIndex*sequenceLength+token] = float32(math.Inf(-1))
					}
					continue
				}
				// Gather the selected compressed rows into contiguous scratch.
				keySlice := make([]float32, len(selected)*headDimension)
				valueSlice := make([]float32, len(selected)*headDimension)
				for index, block := range selected {
					copy(keySlice[index*headDimension:(index+1)*headDimension], keyCompressed.Data[keyBase+block*headDimension:keyBase+(block+1)*headDimension])
					copy(valueSlice[index*headDimension:(index+1)*headDimension], valueCompressed.Data[keyBase+block*headDimension:keyBase+(block+1)*headDimension])
				}
				querySlice := queryHeadMajor.Data[queryIndex*sequenceLength*headDimension+token*headDimension : queryIndex*sequenceLength*headDimension+(token+1)*headDimension]
				var logSumExpSlice []float32
				if logSumExp != nil {
					logSumExpSlice = logSumExp.Data[queryIndex*sequenceLength+token : queryIndex*sequenceLength+token+1]
				}
				tensors.AttentionForward(querySlice, keySlice, valueSlice, outputSlice, logSumExpSlice, 1, len(selected), headDimension, len(selected), -1)
			}
		}
	}
}

// compressedAttentionBackwardSparse backpropagates through the selected blocks.
func compressedAttentionBackwardSparse(queryHeadMajor, keyCompressed, valueCompressed, outputHeadMajor, gradientOutputHeadMajor, logSumExp, gradientQueryHeadMajor, gradientKeyCompressed, gradientValueCompressed *tensors.Tensor, selection [][][]int, headRatio, ratio int) {
	batchSize := queryHeadMajor.Shape[0]
	queryHeadCount := queryHeadMajor.Shape[1]
	sequenceLength := queryHeadMajor.Shape[2]
	headDimension := queryHeadMajor.Shape[3]
	kvHeadCount := keyCompressed.Shape[1]
	blockCount := keyCompressed.Shape[2]

	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		for queryHead := 0; queryHead < queryHeadCount; queryHead++ {
			queryIndex := batchIndex*queryHeadCount + queryHead
			keyValueHead := queryHead / headRatio
			keyBase := (batchIndex*kvHeadCount + keyValueHead) * blockCount * headDimension
			for token := 0; token < sequenceLength; token++ {
				selected := selection[batchIndex*kvHeadCount+keyValueHead][token]
				if len(selected) == 0 {
					continue
				}
				keySlice := make([]float32, len(selected)*headDimension)
				valueSlice := make([]float32, len(selected)*headDimension)
				gradientKeySlice := make([]float32, len(selected)*headDimension)
				gradientValueSlice := make([]float32, len(selected)*headDimension)
				for index, block := range selected {
					copy(keySlice[index*headDimension:(index+1)*headDimension], keyCompressed.Data[keyBase+block*headDimension:keyBase+(block+1)*headDimension])
					copy(valueSlice[index*headDimension:(index+1)*headDimension], valueCompressed.Data[keyBase+block*headDimension:keyBase+(block+1)*headDimension])
				}
				querySlice := queryHeadMajor.Data[queryIndex*sequenceLength*headDimension+token*headDimension : queryIndex*sequenceLength*headDimension+(token+1)*headDimension]
				outputSlice := outputHeadMajor.Data[queryIndex*sequenceLength*headDimension+token*headDimension : queryIndex*sequenceLength*headDimension+(token+1)*headDimension]
				gradientOutputSlice := gradientOutputHeadMajor.Data[queryIndex*sequenceLength*headDimension+token*headDimension : queryIndex*sequenceLength*headDimension+(token+1)*headDimension]
				gradientQuerySlice := gradientQueryHeadMajor.Data[queryIndex*sequenceLength*headDimension+token*headDimension : queryIndex*sequenceLength*headDimension+(token+1)*headDimension]
				logSumExpSlice := logSumExp.Data[queryIndex*sequenceLength+token : queryIndex*sequenceLength+token+1]
				tensors.AttentionBackward(querySlice, keySlice, valueSlice, outputSlice, gradientOutputSlice, logSumExpSlice,
					gradientQuerySlice, gradientKeySlice, gradientValueSlice, 1, len(selected), headDimension, len(selected), -1)
				for index, block := range selected {
					for element := 0; element < headDimension; element++ {
						gradientKeyCompressed.Data[keyBase+block*headDimension+element] += gradientKeySlice[index*headDimension+element]
						gradientValueCompressed.Data[keyBase+block*headDimension+element] += gradientValueSlice[index*headDimension+element]
					}
				}
			}
		}
	}
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

	if attention.indexer != nil {
		attention.forwardCompressedSparse(input, queryHeadMajor, cache, layer, positionOffset, outputHeadMajor)
	} else {
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
	}

	output := toBatchSequenceLayout(outputHeadMajor).Reshape(batchSize, sequenceLength, queryHeadCount*headDimension)
	return attention.outputProjection.Forward(output)
}

// forwardCompressedSparse is the inference attention over the top-k compressed
// blocks selected by the lightning indexer. Selection is shared across the
// query heads that map to one key/value head.
func (attention *CausalSelfAttention) forwardCompressedSparse(input, queryHeadMajor *tensors.Tensor, cache *KVBuffer, layer, positionOffset int, outputHeadMajor *tensors.Tensor) {
	batchSize := input.Shape[0]
	sequenceLength := input.Shape[1]
	embeddingDimension := input.Shape[2]
	queryHeadCount := attention.queryHeadCount
	kvHeadCount := attention.keyValueHeadCount
	headDimension := attention.headDimension
	ratio := attention.compressionRatio
	headRatio := queryHeadCount / kvHeadCount
	topK := attention.sparseTopK

	for batch := 0; batch < batchSize; batch++ {
		available := cache.CompressedCount(layer, batch)
		hiddenSlice := tensors.NewWithData([]int{1, sequenceLength, embeddingDimension},
			input.Data[batch*sequenceLength*embeddingDimension:(batch+1)*sequenceLength*embeddingDimension])
		for keyValueHead := 0; keyValueHead < kvHeadCount; keyValueHead++ {
			var selection [][]int
			var keyData, valueData []float32
			if available > 0 {
				keyData = cache.CompressedKey(layer, batch, keyValueHead, available)
				valueData = cache.CompressedValue(layer, batch, keyValueHead, available)
				compressed := tensors.NewWithData([]int{1, available, headDimension}, keyData)
				scores := attention.indexer.Scores(hiddenSlice, compressed)
				selection = SelectBlocks(scores, positionOffset, ratio, topK)
			}
			keyBuffer := make([]float32, topK*headDimension)
			valueBuffer := make([]float32, topK*headDimension)
			for queryHead := 0; queryHead < queryHeadCount; queryHead++ {
				if queryHead/headRatio != keyValueHead {
					continue
				}
				queryIndex := batch*queryHeadCount + queryHead
				for position := 0; position < sequenceLength; position++ {
					outputRow := outputHeadMajor.Data[(queryIndex*sequenceLength+position)*headDimension : (queryIndex*sequenceLength+position+1)*headDimension]
					if available == 0 || len(selection[position]) == 0 {
						clear(outputRow)
						continue
					}
					selected := selection[position]
					for index, block := range selected {
						copy(keyBuffer[index*headDimension:(index+1)*headDimension], keyData[block*headDimension:(block+1)*headDimension])
						copy(valueBuffer[index*headDimension:(index+1)*headDimension], valueData[block*headDimension:(block+1)*headDimension])
					}
					queryRow := queryHeadMajor.Data[(queryIndex*sequenceLength+position)*headDimension : (queryIndex*sequenceLength+position+1)*headDimension]
					tensors.AttentionForward(queryRow, keyBuffer[:len(selected)*headDimension], valueBuffer[:len(selected)*headDimension], outputRow, nil, 1, len(selected), headDimension, len(selected), -1)
				}
			}
		}
	}
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
