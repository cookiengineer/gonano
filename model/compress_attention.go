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
//
// When the hierarchical indexer is enabled, a Full layer also produces the
// coarse candidate pool that matching Reindex layers consume via sharedPool, so
// training and inference restrict deeper indexers to the same search domain
// (DeepSeek-V4.1 §2.3.2). The returned producedPool is nil unless this layer is
// a Full layer with the hierarchical indexer enabled.
func sparseTrainingPlan(attention *CausalSelfAttention, input, keyCompressed, queryHeadMajor *tensors.Tensor, sharedPool [][][]int) ([][][]int, []*indexerContext, []*tensors.Tensor, []*tensors.Tensor, [][][]int) {
	batchSize := keyCompressed.Shape[0]
	kvHeadCount := keyCompressed.Shape[1]
	blockCount := keyCompressed.Shape[2]
	headDimension := keyCompressed.Shape[3]
	sequenceLength := input.Shape[1]
	embeddingDimension := input.Shape[2]
	queryHeadCount := queryHeadMajor.Shape[1]
	headRatio := queryHeadCount / kvHeadCount
	ratio := attention.compressionRatio
	usePool := attention.indexerPool > 1

	selection := make([][][]int, batchSize*kvHeadCount)
	contexts := make([]*indexerContext, batchSize*kvHeadCount)
	scoresByHead := make([]*tensors.Tensor, batchSize*kvHeadCount)
	targets := make([]*tensors.Tensor, batchSize*kvHeadCount)
	var producedPool [][][]int
	if attention.reuseMode == ReuseFull && usePool {
		producedPool = make([][][]int, batchSize*kvHeadCount)
	}

	for batch := 0; batch < batchSize; batch++ {
		hiddenSlice := tensors.NewWithData([]int{1, sequenceLength, embeddingDimension},
			input.Data[batch*sequenceLength*embeddingDimension:(batch+1)*sequenceLength*embeddingDimension])
		for head := 0; head < kvHeadCount; head++ {
			index := batch*kvHeadCount + head
			compressed := tensors.NewWithData([]int{1, blockCount, headDimension}, headSlice(keyCompressed.Data, index, blockCount, headDimension))
			scores, context := attention.indexer.ScoresWithContext(hiddenSlice, compressed)
			contexts[index] = context
			scoresByHead[index] = scores
			target := distillationTarget(queryHeadMajor, keyCompressed, batch, head, headRatio, ratio, sequenceLength, blockCount, headDimension)
			if attention.reuseMode == ReuseReindex && usePool && sharedPool != nil {
				var candidatePool [][]int
				if index < len(sharedPool) {
					candidatePool = sharedPool[index]
				}
				selection[index] = attention.indexer.selectWithinPool(context.query.Data, context.key.Data, 1, sequenceLength, blockCount, 0, ratio, attention.sparseTopK, candidatePool)
				// Deeper indexers are optimized under the same search domain
				// they use at inference (DeepSeek-V4.1 §2.3.2): restrict the
				// distillation to the shared candidate pool.
				if candidatePool != nil {
					restrictIndexerToPool(scores, target, candidatePool)
				}
			} else {
				poolSize := attention.indexerPool
				if poolSize < 1 {
					poolSize = 1
				}
				selected, candidates := attention.indexer.selectFromScoreMatrix(scores.Data, 1, sequenceLength, blockCount, 0, ratio, attention.sparseTopK, poolSize, attention.indexerCandidates)
				selection[index] = selected
				if producedPool != nil {
					producedPool[index] = candidates
				}
			}
			targets[index] = target
		}
	}
	return selection, contexts, scoresByHead, targets, producedPool
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

// restrictIndexerToPool masks an indexer's scores and distillation target to a
// shared candidate pool: entries outside the pool get a -inf score and a zero
// target, and the target is renormalized within the pool. This makes a Reindex
// layer's indexer optimize under the same search domain it uses at inference
// (DeepSeek-V4.1 §2.3.2).
func restrictIndexerToPool(scores, target *tensors.Tensor, pool [][]int) {
	blockCount := scores.Shape[1]
	rowCount := scores.Shape[0]
	for row := 0; row < rowCount; row++ {
		var allowed []int
		if row < len(pool) {
			allowed = pool[row]
		}
		inPool := make(map[int]bool, len(allowed))
		for _, entry := range allowed {
			inPool[entry] = true
		}
		base := row * blockCount
		var total float64
		for entry := 0; entry < blockCount; entry++ {
			if !inPool[entry] {
				scores.Data[base+entry] = float32(math.Inf(-1))
				target.Data[base+entry] = 0
				continue
			}
			total += float64(target.Data[base+entry])
		}
		if total > 0 {
			for entry := range inPool {
				target.Data[base+entry] = float32(float64(target.Data[base+entry]) / total)
			}
		}
	}
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
		// A row fully masked by the candidate pool has no finite score; leave
		// its probabilities at zero instead of producing NaN.
		if math.IsInf(float64(maximum), -1) {
			continue
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
func compressedAttentionBackwardSparse(queryHeadMajor, keyCompressed, valueCompressed, outputHeadMajor, gradientOutputHeadMajor, logSumExp, rowCorrection, gradientQueryHeadMajor, gradientKeyCompressed, gradientValueCompressed *tensors.Tensor, selection [][][]int, headRatio, ratio int) {
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
				var correctionSlice []float32
				if rowCorrection != nil {
					correctionSlice = rowCorrection.Data[queryIndex*sequenceLength+token : queryIndex*sequenceLength+token+1]
				}
				tensors.AttentionBackwardCorrected(querySlice, keySlice, valueSlice, outputSlice, gradientOutputSlice, logSumExpSlice, correctionSlice,
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

// mergeAttentionBranches combines two normalized attention branches that share
// one softmax over the union of their keys. Given branch outputs and
// log-sum-exps it writes the exact merged output and log-sum-exp. This is the
// training-side merge for the global compressed branch and the local
// sliding-window branch.
func mergeAttentionBranches(output, logSumExp, outputA, logSumExpA, outputB, logSumExpB *tensors.Tensor, headDim int) {
	rowCount := logSumExpA.Numel()
	for row := 0; row < rowCount; row++ {
		logA := logSumExpA.Data[row]
		logB := logSumExpB.Data[row]
		maximum := logA
		if logB > maximum {
			maximum = logB
		}
		base := row * headDim
		if maximum == float32(math.Inf(-1)) {
			clear(output.Data[base : base+headDim])
			logSumExp.Data[row] = maximum
			continue
		}
		weightA := float32(math.Exp(float64(logA - maximum)))
		weightB := float32(math.Exp(float64(logB - maximum)))
		total := weightA + weightB
		inverse := float32(1.0 / float64(total))
		for dimension := 0; dimension < headDim; dimension++ {
			output.Data[base+dimension] = (weightA*outputA.Data[base+dimension] + weightB*outputB.Data[base+dimension]) * inverse
		}
		logSumExp.Data[row] = maximum + float32(math.Log(float64(total)))
	}
}

// mergeAttentionRow combines one query row's two attention branches. It is the
// inference-side counterpart of mergeAttentionBranches.
func mergeAttentionRow(output, outputA, outputB []float32, logA, logB float32) {
	maximum := logA
	if logB > maximum {
		maximum = logB
	}
	if maximum == float32(math.Inf(-1)) {
		clear(output)
		return
	}
	weightA := float32(math.Exp(float64(logA - maximum)))
	weightB := float32(math.Exp(float64(logB - maximum)))
	total := weightA + weightB
	inverse := float32(1.0 / float64(total))
	for dimension := range output {
		output[dimension] = (weightA*outputA[dimension] + weightB*outputB[dimension]) * inverse
	}
}

// attentionForwardWindowed runs the local sliding-window branch for every
// (batch, query head) during training.
func attentionForwardWindowed(queryHeadMajor, keyHeadMajor, valueHeadMajor, outputHeadMajor, logSumExp *tensors.Tensor, batchSize, queryHeadCount, kvHeadCount, sequenceLength, headDim, positionOffset, window int) {
	headRatio := queryHeadCount / kvHeadCount
	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		for queryHead := 0; queryHead < queryHeadCount; queryHead++ {
			keyValueHead := queryHead / headRatio
			queryIndex := batchIndex*queryHeadCount + queryHead
			query := headSlice(queryHeadMajor.Data, queryIndex, sequenceLength, headDim)
			key := headSlice(keyHeadMajor.Data, batchIndex*kvHeadCount+keyValueHead, sequenceLength, headDim)
			value := headSlice(valueHeadMajor.Data, batchIndex*kvHeadCount+keyValueHead, sequenceLength, headDim)
			output := headSlice(outputHeadMajor.Data, queryIndex, sequenceLength, headDim)
			statistic := headSlice(logSumExp.Data, queryIndex, sequenceLength, 1)
			tensors.AttentionForward(query, key, value, output, statistic, sequenceLength, sequenceLength, headDim, positionOffset, window)
		}
	}
}

// attentionBackwardWindowed backpropagates the local sliding-window branch with
// the merged log-sum-exp and row correction shared with the global branch. Raw
// key/value gradients accumulate across the query heads that share a KV head.
func attentionBackwardWindowed(queryHeadMajor, keyHeadMajor, valueHeadMajor, outputHeadMajor, gradientOutputHeadMajor, logSumExp, rowCorrection, gradientQueryHeadMajor, gradientKeyHeadMajor, gradientValueHeadMajor *tensors.Tensor, batchSize, queryHeadCount, kvHeadCount, sequenceLength, headDim, positionOffset, window int) {
	headRatio := queryHeadCount / kvHeadCount
	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		for queryHead := 0; queryHead < queryHeadCount; queryHead++ {
			keyValueHead := queryHead / headRatio
			queryIndex := batchIndex*queryHeadCount + queryHead
			query := headSlice(queryHeadMajor.Data, queryIndex, sequenceLength, headDim)
			key := headSlice(keyHeadMajor.Data, batchIndex*kvHeadCount+keyValueHead, sequenceLength, headDim)
			value := headSlice(valueHeadMajor.Data, batchIndex*kvHeadCount+keyValueHead, sequenceLength, headDim)
			output := headSlice(outputHeadMajor.Data, queryIndex, sequenceLength, headDim)
			gradientOutput := headSlice(gradientOutputHeadMajor.Data, queryIndex, sequenceLength, headDim)
			statistic := headSlice(logSumExp.Data, queryIndex, sequenceLength, 1)
			var correction []float32
			if rowCorrection != nil {
				correction = headSlice(rowCorrection.Data, queryIndex, sequenceLength, 1)
			}
			gradientQuery := headSlice(gradientQueryHeadMajor.Data, queryIndex, sequenceLength, headDim)
			gradientKey := headSlice(gradientKeyHeadMajor.Data, batchIndex*kvHeadCount+keyValueHead, sequenceLength, headDim)
			gradientValue := headSlice(gradientValueHeadMajor.Data, batchIndex*kvHeadCount+keyValueHead, sequenceLength, headDim)
			tensors.AttentionBackwardCorrected(query, key, value, output, gradientOutput, statistic, correction, gradientQuery, gradientKey, gradientValue, sequenceLength, sequenceLength, headDim, positionOffset, window)
		}
	}
}

// localAttentionRow runs the local sliding-window branch for one inference query
// row against the raw key/value prefix of a (batch, KV head). floor is the
// first absolute position that may be attended; it is 0 normally and the replay
// start during CED bounded replay (where earlier cached slots are zero).
func localAttentionRow(queryRow, keyPrefix, valuePrefix []float32, absolutePosition, floor, window, headDim int, output, logSumExp []float32) {
	localStart := absolutePosition - window
	if localStart < floor {
		localStart = floor
	}
	if localStart < 0 {
		localStart = 0
	}
	localCount := absolutePosition - localStart + 1
	if localCount > absolutePosition+1 {
		localCount = absolutePosition + 1
	}
	keyLocal := keyPrefix[localStart*headDim : (localStart+localCount)*headDim]
	valueLocal := valuePrefix[localStart*headDim : (localStart+localCount)*headDim]
	// PositionOffset localCount-1 (and no window mask) makes every sliced key
	// causally valid for this query.
	tensors.AttentionForward(queryRow, keyLocal, valueLocal, output, logSumExp, 1, localCount, headDim, localCount-1, -1)
}

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
func compressedAttentionBackward(queryHeadMajor, keyCompressed, valueCompressed, outputHeadMajor, gradientOutputHeadMajor, logSumExp, rowCorrection, gradientQueryHeadMajor, gradientKeyCompressed, gradientValueCompressed *tensors.Tensor, headRatio, ratio int) {
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
				var correctionSlice []float32
				if rowCorrection != nil {
					correctionSlice = rowCorrection.Data[queryIndex*sequenceLength+start : queryIndex*sequenceLength+end]
				}
				tensors.AttentionBackwardCorrected(querySlice, keyCompressed.Data[keyBase:keyBase+block*headDimension], valueCompressed.Data[keyBase:keyBase+block*headDimension],
					outputSlice, gradientOutputSlice, logSumExpSlice, correctionSlice,
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
			if attention.indexer != nil {
				projected := attention.indexer.key.Forward(compressedKey)
				cache.AppendIndexerKey(layer, batch, head, projected.Data)
			}
		}
	}
	cache.AdvanceCompressedCount(layer)
	for batch := 0; batch < batchSize; batch++ {
		cache.ResetTail(layer, batch)
	}
}

// appendGlobalTail appends a chunk of rows to a layer's global compression
// buffer, flushing each completed block into the layer's compressed cache. It
// is shared by incremental inference and by the CED prefill global-cache fill.
func (attention *CausalSelfAttention) appendGlobalTail(cache *KVBuffer, layer int, hidden, key, value *tensors.Tensor, batchSize, sequenceLength, embeddingDimension, headDimension int) {
	ratio := attention.compressionRatio
	kvHeadCount := attention.keyValueHeadCount
	for position := 0; position < sequenceLength; position++ {
		for batch := 0; batch < batchSize; batch++ {
			row := batch*sequenceLength + position
			hiddenRow := hidden.Data[row*embeddingDimension : row*embeddingDimension+embeddingDimension]
			keyRow := key.Data[row*kvHeadCount*headDimension : row*kvHeadCount*headDimension+kvHeadCount*headDimension]
			valueRow := value.Data[row*kvHeadCount*headDimension : row*kvHeadCount*headDimension+kvHeadCount*headDimension]
			cache.AppendTailRow(layer, batch, hiddenRow, keyRow, valueRow)
		}
		if cache.TailLength(layer, 0) == ratio {
			attention.compressTail(cache, layer, batchSize, embeddingDimension, headDimension)
		}
	}
}

// forwardCompressed is the inference path of an HCA-style compressed attention
// layer. It buffers incoming rows, compresses each completed block, and attends
// every query to the compressed blocks that strictly precede its own block.
func (attention *CausalSelfAttention) forwardCompressed(input, query, key, value *tensors.Tensor, encoderHidden, cedKeySequence, cedValueSequence *tensors.Tensor, cache *KVBuffer, layer, positionOffset int, share *compressionShare) *tensors.Tensor {
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

	// The local sliding-window branch reads raw keys/values from the layer's
	// own cache prefix. SWA KV is layer-local (DeepSeek-V4.1 §2.3.1): every
	// mode computes its own, independent of the shared global compressed KV.
	var localKV []keyValueSlice
	if attention.swaWindow > 0 {
		keyHeadMajor := toBatchHeadLayout(key)
		valueHeadMajor := toBatchHeadLayout(value)
		localKV = make([]keyValueSlice, batchSize*kvHeadCount)
		for batch := 0; batch < batchSize; batch++ {
			for head := 0; head < kvHeadCount; head++ {
				slot := batch*kvHeadCount + head
				keyNew := headSlice(keyHeadMajor.Data, slot, sequenceLength, headDimension)
				valueNew := headSlice(valueHeadMajor.Data, slot, sequenceLength, headDimension)
				keyFull, valueFull := cache.writeKeyValue(layer, batch, head, keyNew, valueNew)
				localKV[slot] = keyValueSlice{key: keyFull, value: valueFull}
			}
		}
	}

	// Only a full layer buffers rows and compresses completed blocks; reuse
	// and reindex layers read the producing layer's compressed cache.
	cacheLayer := layer
	if attention.reuseMode != ReuseFull {
		cacheLayer = share.producer
	}
	replay := share != nil && share.replay
	if attention.reuseMode == ReuseFull && !replay {
		// CED decoder layers buffer the encoder hidden state together with the
		// global keys/values projected from it instead of the layer's own
		// hidden state and keys/values.
		bufferHidden := input
		bufferKey := key
		bufferValue := value
		if attention.ced && encoderHidden != nil {
			bufferHidden = encoderHidden
			bufferKey = cedKeySequence
			bufferValue = cedValueSequence
		}
		attention.appendGlobalTail(cache, layer, bufferHidden, bufferKey, bufferValue, batchSize, sequenceLength, embeddingDimension, headDimension)
	}

	if !replay && attention.reuseMode == ReuseReindex && attention.sparseTopK > 0 {
		attention.cacheReusedIndexerKeys(cache, layer, cacheLayer, batchSize, headDimension)
	}

	if attention.sparseTopK > 0 {
		attention.forwardCompressedSparse(input, queryHeadMajor, cache, layer, cacheLayer, positionOffset, outputHeadMajor, share, localKV)
	} else {
		globalOutput := make([]float32, headDimension)
		localOutput := make([]float32, headDimension)
		globalLogSumExp := make([]float32, 1)
		localLogSumExp := make([]float32, 1)
		// During bounded replay the local window is truncated to the replay
		// segment, so cached slots before positionOffset are excluded.
		localFloor := 0
		if replay {
			localFloor = positionOffset
		}
		for batch := 0; batch < batchSize; batch++ {
			for queryHead := 0; queryHead < queryHeadCount; queryHead++ {
				keyValueHead := queryHead / headRatio
				queryIndex := batch*queryHeadCount + queryHead
				for position := 0; position < sequenceLength; position++ {
					absolutePosition := positionOffset + position
					count := absolutePosition / ratio
					if available := cache.CompressedCount(cacheLayer, batch); count > available {
						count = available
					}
					rowBase := (queryIndex*sequenceLength + position) * headDimension
					outputRow := outputHeadMajor.Data[rowBase : rowBase+headDimension]
					queryRow := queryHeadMajor.Data[rowBase : rowBase+headDimension]
					if attention.swaWindow > 0 {
						if count == 0 {
							globalLogSumExp[0] = float32(math.Inf(-1))
						} else {
							keySlice := cache.CompressedKey(cacheLayer, batch, keyValueHead, count)
							valueSlice := cache.CompressedValue(cacheLayer, batch, keyValueHead, count)
							tensors.AttentionForward(queryRow, keySlice, valueSlice, globalOutput, globalLogSumExp, 1, count, headDimension, count, -1)
						}
						local := localKV[batch*kvHeadCount+keyValueHead]
						localAttentionRow(queryRow, local.key, local.value, absolutePosition, localFloor, attention.swaWindow, headDimension, localOutput, localLogSumExp)
						mergeAttentionRow(outputRow, globalOutput, localOutput, globalLogSumExp[0], localLogSumExp[0])
						continue
					}
					if count == 0 {
						clear(outputRow)
						continue
					}
					keySlice := cache.CompressedKey(cacheLayer, batch, keyValueHead, count)
					valueSlice := cache.CompressedValue(cacheLayer, batch, keyValueHead, count)
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
// query heads that map to one key/value head. localKV holds the layer's raw
// key/value prefix for the local sliding-window branch (nil when disabled).
func (attention *CausalSelfAttention) forwardCompressedSparse(input, queryHeadMajor *tensors.Tensor, cache *KVBuffer, layer, cacheLayer, positionOffset int, outputHeadMajor *tensors.Tensor, share *compressionShare, localKV []keyValueSlice) {
	batchSize := input.Shape[0]
	sequenceLength := input.Shape[1]
	embeddingDimension := input.Shape[2]
	queryHeadCount := attention.queryHeadCount
	kvHeadCount := attention.keyValueHeadCount
	headDimension := attention.headDimension
	ratio := attention.compressionRatio
	headRatio := queryHeadCount / kvHeadCount
	topK := attention.sparseTopK

	// Published selection for the group, assembled across (batch, kv-head).
	var published [][][]int
	if attention.reuseMode != ReuseReuse {
		published = make([][][]int, batchSize*kvHeadCount)
	}
	if attention.reuseMode == ReuseReuse {
		published = share.selection
	}
	// The hierarchical candidate pool is published by a Full layer and reused
	// by later Reindex layers in the group, so deeper indexers score only the
	// pool instead of the whole context (DeepSeek-V4.1 §2.3.2).
	usePool := attention.indexerPool > 1
	var candidatePools [][][]int
	if attention.reuseMode == ReuseFull && usePool {
		candidatePools = make([][][]int, batchSize*kvHeadCount)
	}

	globalScratch := make([]float32, headDimension)
	localScratch := make([]float32, headDimension)
	globalLogSumExp := make([]float32, 1)
	localLogSumExp := make([]float32, 1)
	localFloor := 0
	if share != nil && share.replay {
		localFloor = positionOffset
	}

	for batch := 0; batch < batchSize; batch++ {
		available := cache.CompressedCount(cacheLayer, batch)
		hiddenSlice := tensors.NewWithData([]int{1, sequenceLength, embeddingDimension},
			input.Data[batch*sequenceLength*embeddingDimension:(batch+1)*sequenceLength*embeddingDimension])
		for keyValueHead := 0; keyValueHead < kvHeadCount; keyValueHead++ {
			selectionIndex := batch*kvHeadCount + keyValueHead
			var selection [][]int
			var keyData, valueData []float32
			if available > 0 {
				keyData = cache.CompressedKey(cacheLayer, batch, keyValueHead, available)
				valueData = cache.CompressedValue(cacheLayer, batch, keyValueHead, available)
				switch {
				case attention.reuseMode == ReuseReuse:
					if published != nil {
						selection = published[selectionIndex]
					}
				default:
					// Full and Reindex layers cache their own indexer key
					// projections; fall back to projecting on the fly when the
					// cache is missing.
					projectedKeys := cache.IndexerKey(layer, batch, keyValueHead, available)
					if width := indexerKeyWidthOf(attention); width <= 0 || len(projectedKeys) < available*width {
						compressed := tensors.NewWithData([]int{1, available, headDimension}, keyData)
						projectedKeys = attention.indexer.key.Forward(compressed).Data
					}
					pool := attention.indexerPool
					if pool < 1 {
						pool = 1
					}
					budget := attention.indexerCandidates
					if pool == 1 {
						// Flat selection: score every available entry.
						budget = available
					}
					if attention.reuseMode == ReuseReindex && usePool && share.candidatePool != nil {
						var candidatePool [][]int
						if selectionIndex < len(share.candidatePool) {
							candidatePool = share.candidatePool[selectionIndex]
						}
						selection = attention.indexer.SelectWithinPool(hiddenSlice, projectedKeys, available, positionOffset, ratio, topK, candidatePool)
					} else {
						var candidates [][]int
						selection, candidates = attention.indexer.SelectProjectedWithCandidates(hiddenSlice, projectedKeys, available, positionOffset, ratio, topK, pool, budget)
						if candidatePools != nil {
							candidatePools[selectionIndex] = candidates
						}
					}
					if published != nil {
						published[selectionIndex] = selection
					}
				}
			}
			keyBuffer := make([]float32, topK*headDimension)
			valueBuffer := make([]float32, topK*headDimension)
			for queryHead := 0; queryHead < queryHeadCount; queryHead++ {
				if queryHead/headRatio != keyValueHead {
					continue
				}
				queryIndex := batch*queryHeadCount + queryHead
				for position := 0; position < sequenceLength; position++ {
					rowBase := (queryIndex*sequenceLength + position) * headDimension
					outputRow := outputHeadMajor.Data[rowBase : rowBase+headDimension]
					queryRow := queryHeadMajor.Data[rowBase : rowBase+headDimension]
					globalOutput := outputRow
					var branchLogSumExp []float32
					if attention.swaWindow > 0 {
						globalOutput = globalScratch
						branchLogSumExp = globalLogSumExp
						globalLogSumExp[0] = float32(math.Inf(-1))
					}
					if available != 0 && len(selection) != 0 && len(selection[position]) != 0 {
						selected := selection[position]
						for index, block := range selected {
							copy(keyBuffer[index*headDimension:(index+1)*headDimension], keyData[block*headDimension:(block+1)*headDimension])
							copy(valueBuffer[index*headDimension:(index+1)*headDimension], valueData[block*headDimension:(block+1)*headDimension])
						}
						tensors.AttentionForward(queryRow, keyBuffer[:len(selected)*headDimension], valueBuffer[:len(selected)*headDimension], globalOutput, branchLogSumExp, 1, len(selected), headDimension, len(selected), -1)
					} else if attention.swaWindow == 0 {
						clear(outputRow)
					}
					if attention.swaWindow > 0 {
						local := localKV[batch*kvHeadCount+keyValueHead]
						localAttentionRow(queryRow, local.key, local.value, positionOffset+position, localFloor, attention.swaWindow, headDimension, localScratch, localLogSumExp)
						mergeAttentionRow(outputRow, globalOutput, localScratch, globalLogSumExp[0], localLogSumExp[0])
					}
				}
			}
		}
	}
	if attention.reuseMode != ReuseReuse {
		share.selection = published
	}
	if attention.reuseMode == ReuseFull {
		// Publish the coarse candidate pool for the group's Reindex layers;
		// clear any stale pool when the hierarchical indexer is disabled.
		share.candidatePool = candidatePools
	}
}

// indexerKeyWidthOf returns the cached indexer key projection width for a
// layer's indexer (headCount*dim).
func indexerKeyWidthOf(attention *CausalSelfAttention) int {
	if attention.indexer == nil {
		return 0
	}
	return attention.indexer.HeadCount * attention.indexer.Dim
}

// cacheReusedIndexerKeys projects any newly compressed blocks of the producing
// layer with this reindex layer's own indexer key, so decode does not re-project
// every shared compressed key on each step.
func (attention *CausalSelfAttention) cacheReusedIndexerKeys(cache *KVBuffer, layer, producer, batchSize, headDimension int) {
	if attention.indexer == nil {
		return
	}
	producerCount := cache.CompressedCount(producer, 0)
	ownCount := cache.CompressedCount(layer, 0)
	for block := ownCount; block < producerCount; block++ {
		for batch := 0; batch < batchSize; batch++ {
			for head := 0; head < attention.keyValueHeadCount; head++ {
				compressed := tensors.NewWithData([]int{1, 1, headDimension}, cache.CompressedBlock(producer, batch, head, block))
				projected := attention.indexer.key.Forward(compressed)
				cache.AppendIndexerKey(layer, batch, head, projected.Data)
			}
		}
		cache.AdvanceCompressedCount(layer)
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
