package model

import (
	"sort"

	"github.com/cookiengineer/gonano/model/layers"
	"github.com/cookiengineer/gonano/tensors"
)

// SparseIndexer is the "lightning indexer" used by CSA-style sparse attention.
// It scores compressed KV blocks with a low-rank projection and selects the
// top-k blocks per query (DeepSeek-V4 eqs. 13-17). It is deliberately cheap:
// the indexer dimension is much smaller than the attention head dimension, so
// scoring every compressed block is far cheaper than full attention over those
// blocks.
type SparseIndexer struct {
	// HeadCount is the number of indexer query heads.
	HeadCount int
	// Dim is the per-head indexer dimension.
	Dim int
	// query maps the hidden state [d] to [HeadCount*Dim].
	query *layers.Linear
	// key maps a compressed KV entry [compressedWidth] to [HeadCount*Dim].
	key *layers.Linear
	// headWeights holds the learnable per-head weight w_h.
	headWeights *tensors.Tensor // [HeadCount]
}

// NewSparseIndexer builds an indexer over an embedding of width
// embeddingDimension that scores compressed entries of width compressedWidth.
func NewSparseIndexer(embeddingDimension, compressedWidth, dim, headCount int) *SparseIndexer {
	if dim < 1 || headCount < 1 {
		panic("model: indexer dim and head count must be >= 1")
	}
	return &SparseIndexer{
		HeadCount:   headCount,
		Dim:         dim,
		query:       layers.NewLinear(embeddingDimension, dim*headCount),
		key:         layers.NewLinear(compressedWidth, dim*headCount),
		headWeights: tensors.New(headCount),
	}
}

// indexerContext holds the projections needed to backpropagate the indexer.
type indexerContext struct {
	hidden         *tensors.Tensor // [B, T, d]
	compressed     *tensors.Tensor // [B, blocks, compressedWidth]
	query          *tensors.Tensor // [B*T, headCount, dim]
	key            *tensors.Tensor // [B*blocks, headCount, dim]
	batchSize      int
	sequenceLength int
	blockCount     int
}

// ScoresWithContext computes the index scores (see Scores) and retains the
// projections for Backward. The result has shape [batch*sequence, blocks].
func (indexer *SparseIndexer) ScoresWithContext(hidden, compressed *tensors.Tensor) (*tensors.Tensor, *indexerContext) {
	batchSize := hidden.Shape[0]
	sequenceLength := hidden.Shape[1]
	blockCount := compressed.Shape[1]
	dim := indexer.Dim
	headCount := indexer.HeadCount

	query := indexer.query.Forward(hidden).Reshape(batchSize*sequenceLength, headCount, dim)
	key := indexer.key.Forward(compressed).Reshape(batchSize*blockCount, headCount, dim)

	scores := tensors.New(batchSize*sequenceLength, blockCount)
	queryData, keyData, scoreData := query.Data, key.Data, scores.Data
	weightData := indexer.headWeights.Data

	for row := 0; row < batchSize*sequenceLength; row++ {
		batchIndex := row / sequenceLength
		for block := 0; block < blockCount; block++ {
			keyRow := batchIndex*blockCount + block
			var total float32
			for head := 0; head < headCount; head++ {
				queryBase := (row*headCount + head) * dim
				keyBase := (keyRow*headCount + head) * dim
				var dot float32
				for index := 0; index < dim; index++ {
					dot += queryData[queryBase+index] * keyData[keyBase+index]
				}
				if dot > 0 {
					total += weightData[head] * dot
				}
			}
			scoreData[row*blockCount+block] = total
		}
	}
	return scores, &indexerContext{
		hidden:         hidden,
		compressed:     compressed,
		query:          query,
		key:            key,
		batchSize:      batchSize,
		sequenceLength: sequenceLength,
		blockCount:     blockCount,
	}
}

// Scores computes the index score I[b, t, s] between every query row of hidden
// [batch, sequence, d] and every compressed entry of compressed
// [batch, blocks, compressedWidth]:
//
//	I[t,s] = sum_h w_h * ReLU(dot(query[t,h], key[s,h]))
//
// The result has shape [batch*sequence, blocks].
func (indexer *SparseIndexer) Scores(hidden, compressed *tensors.Tensor) *tensors.Tensor {
	scores, _ := indexer.ScoresWithContext(hidden, compressed)
	return scores
}

// Backward propagates the gradient of the index scores through the indexer,
// accumulating parameter gradients and returning gradients with respect to the
// hidden state and the compressed entries.
func (indexer *SparseIndexer) Backward(gradScores *tensors.Tensor, context *indexerContext) (gradientHidden, gradientCompressed *tensors.Tensor) {
	batchSize := context.batchSize
	sequenceLength := context.sequenceLength
	blockCount := context.blockCount
	dim := indexer.Dim
	headCount := indexer.HeadCount

	gradientQuery := tensors.New(batchSize*sequenceLength, headCount, dim)
	gradientKey := tensors.New(batchSize*blockCount, headCount, dim)
	indexer.headWeights.EnsureGrad()

	queryData, keyData := context.query.Data, context.key.Data
	gradientQueryData, gradientKeyData := gradientQuery.Data, gradientKey.Data
	gradientScoreData := gradScores.Data
	weightData := indexer.headWeights.Data

	for row := 0; row < batchSize*sequenceLength; row++ {
		batchIndex := row / sequenceLength
		for block := 0; block < blockCount; block++ {
			keyRow := batchIndex*blockCount + block
			scoreGradient := gradientScoreData[row*blockCount+block]
			for head := 0; head < headCount; head++ {
				queryBase := (row*headCount + head) * dim
				keyBase := (keyRow*headCount + head) * dim
				var dot float32
				for index := 0; index < dim; index++ {
					dot += queryData[queryBase+index] * keyData[keyBase+index]
				}
				if dot <= 0 {
					continue
				}
				indexer.headWeights.Grad[head] += scoreGradient * dot
				weighted := scoreGradient * weightData[head]
				for index := 0; index < dim; index++ {
					gradientQueryData[queryBase+index] += weighted * keyData[keyBase+index]
					gradientKeyData[keyBase+index] += weighted * queryData[queryBase+index]
				}
			}
		}
	}

	gradientHidden = indexer.query.Backward(context.hidden, gradientQuery.Reshape(batchSize*sequenceLength, headCount*dim))
	gradientCompressed = indexer.key.Backward(context.compressed, gradientKey.Reshape(batchSize*blockCount, headCount*dim))
	return gradientHidden, gradientCompressed
}

// ZeroGrad zeroes the gradients of the indexer's parameters.
func (indexer *SparseIndexer) ZeroGrad() {
	for _, parameter := range indexer.Parameters() {
		parameter.ZeroGrad()
	}
}

// TopK returns, for every query row, the indices of the k highest-scoring
// blocks in descending score order. Ties are broken toward the smaller block
// index. If k exceeds the block count the full set is returned.
func TopK(scores *tensors.Tensor, k int) [][]int {
	rowCount := scores.Shape[0]
	blockCount := scores.Shape[1]
	if k > blockCount {
		k = blockCount
	}
	if k < 0 {
		k = 0
	}
	selected := make([][]int, rowCount)
	for row := 0; row < rowCount; row++ {
		order := make([]int, blockCount)
		for block := 0; block < blockCount; block++ {
			order[block] = block
		}
		rowScores := scores.Data[row*blockCount : (row+1)*blockCount]
		sort.SliceStable(order, func(left, right int) bool {
			if rowScores[order[left]] != rowScores[order[right]] {
				return rowScores[order[left]] > rowScores[order[right]]
			}
			return order[left] < order[right]
		})
		selected[row] = append([]int(nil), order[:k]...)
	}
	return selected
}

// SelectBlocks returns, for every query token, the indices of the at most
// topK highest-scoring compressed blocks that strictly precede the query's own
// compression block. Token t has absolute position positionOffset+t, so its
// allowed blocks are [0, min((positionOffset+t)/ratio, blockCount)). The
// selection is ordered by descending index score with ties broken toward the
// smaller block index.
func SelectBlocks(scores *tensors.Tensor, positionOffset, ratio, topK int) [][]int {
	tokenCount := scores.Shape[0]
	blockCount := scores.Shape[1]
	selection := make([][]int, tokenCount)
	for token := 0; token < tokenCount; token++ {
		allowed := (positionOffset + token) / ratio
		if allowed > blockCount {
			allowed = blockCount
		}
		if allowed <= 0 || topK <= 0 {
			selection[token] = []int{}
			continue
		}
		order := make([]int, allowed)
		for block := 0; block < allowed; block++ {
			order[block] = block
		}
		rowScores := scores.Data[token*blockCount : token*blockCount+allowed]
		sort.SliceStable(order, func(left, right int) bool {
			if rowScores[order[left]] != rowScores[order[right]] {
				return rowScores[order[left]] > rowScores[order[right]]
			}
			return order[left] < order[right]
		})
		count := topK
		if count > allowed {
			count = allowed
		}
		selection[token] = append([]int(nil), order[:count]...)
	}
	return selection
}

// argsortDescending returns indices ordered by descending value, ties broken
// toward the smaller index.
func argsortDescending(values []float32) []int {
	order := make([]int, len(values))
	for index := range order {
		order[index] = index
	}
	sort.SliceStable(order, func(left, right int) bool {
		if values[order[left]] != values[order[right]] {
			return values[order[left]] > values[order[right]]
		}
		return order[left] < order[right]
	})
	return order
}

// HierarchicalSelect is the two-level indexer of DeepSeek-V4.1 (§2.3.2).
// Compressed entries are grouped into super-blocks of `pool` entries; the query
// is scored against pooled key representations, the best super-blocks are
// chosen, and only the entries inside those super-blocks are scored in full.
// The number of fully scored entries per token is bounded by candidateBudget,
// which makes deeper indexers constant-cost instead of linear in context
// length.
func (indexer *SparseIndexer) HierarchicalSelect(hidden, compressed *tensors.Tensor, positionOffset, ratio, topK, pool, candidateBudget int) [][]int {
	key := indexer.key.Forward(compressed) // [B*blocks, H, dim]
	return indexer.SelectProjected(hidden, key.Data, compressed.Shape[1], positionOffset, ratio, topK, pool, candidateBudget)
}

// SelectProjected runs the hierarchical selection against precomputed key
// projections [batch*blocks*H*dim]. Caching these projections lets decode skip
// re-projecting every compressed key on each step.
func (indexer *SparseIndexer) SelectProjected(hidden *tensors.Tensor, keyProjections []float32, blockCount, positionOffset, ratio, topK, pool, candidateBudget int) [][]int {
	if pool < 1 {
		pool = 1
	}
	batchSize := hidden.Shape[0]
	sequenceLength := hidden.Shape[1]
	dim := indexer.Dim
	headCount := indexer.HeadCount

	query := indexer.query.Forward(hidden) // [B*T, H, dim]
	queryData := query.Data
	keyData := keyProjections
	weightData := indexer.headWeights.Data
	headWidth := headCount * dim

	groupsPerToken := (candidateBudget + pool - 1) / pool
	if groupsPerToken < 1 {
		groupsPerToken = 1
	}

	selection := make([][]int, batchSize*sequenceLength)
	for batch := 0; batch < batchSize; batch++ {
		batchKeyBase := batch * blockCount * headWidth
		groups := (blockCount + pool - 1) / pool
		pooled := make([]float32, groups*headWidth)
		for group := 0; group < groups; group++ {
			start := group * pool
			end := min(start+pool, blockCount)
			count := end - start
			if count <= 0 {
				continue
			}
			for element := 0; element < headWidth; element++ {
				var sum float32
				for entry := start; entry < end; entry++ {
					sum += keyData[batchKeyBase+entry*headWidth+element]
				}
				pooled[group*headWidth+element] = sum / float32(count)
			}
		}

		for token := 0; token < sequenceLength; token++ {
			row := batch*sequenceLength + token
			allowed := (positionOffset + token) / ratio
			if allowed > blockCount {
				allowed = blockCount
			}
			if allowed <= 0 || topK <= 0 {
				selection[row] = []int{}
				continue
			}
			queryBase := row * headWidth

			allowedGroups := (allowed + pool - 1) / pool
			coarse := make([]float32, allowedGroups)
			for group := 0; group < allowedGroups; group++ {
				var total float32
				for head := 0; head < headCount; head++ {
					var dot float32
					for dimension := 0; dimension < dim; dimension++ {
						dot += queryData[queryBase+head*dim+dimension] * pooled[group*headWidth+head*dim+dimension]
					}
					if dot > 0 {
						total += weightData[head] * dot
					}
				}
				coarse[group] = total
			}

			chosenGroups := argsortDescending(coarse)
			if len(chosenGroups) > groupsPerToken {
				chosenGroups = chosenGroups[:groupsPerToken]
			}
			candidates := make([]int, 0, groupsPerToken*pool)
			for _, group := range chosenGroups {
				start := group * pool
				end := min(start+pool, allowed)
				for entry := start; entry < end; entry++ {
					candidates = append(candidates, entry)
				}
			}
			if len(candidates) == 0 {
				selection[row] = []int{}
				continue
			}

			fine := make([]float32, len(candidates))
			for index, entry := range candidates {
				var total float32
				for head := 0; head < headCount; head++ {
					var dot float32
					entryBase := batchKeyBase + entry*headWidth + head*dim
					for dimension := 0; dimension < dim; dimension++ {
						dot += queryData[queryBase+head*dim+dimension] * keyData[entryBase+dimension]
					}
					if dot > 0 {
						total += weightData[head] * dot
					}
				}
				fine[index] = total
			}
			order := argsortDescending(fine)
			count := min(topK, len(order))
			selected := make([]int, count)
			for index := 0; index < count; index++ {
				selected[index] = candidates[order[index]]
			}
			selection[row] = selected
		}
	}
	return selection
}

// Parameters returns the indexer's trainable tensors.
func (indexer *SparseIndexer) Parameters() []*tensors.Tensor {
	return []*tensors.Tensor{indexer.query.Weight, indexer.key.Weight, indexer.headWeights}
}
