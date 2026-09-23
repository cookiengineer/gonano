package model

import (
	"math"
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

	// Score each batch's key set with the fused, vectorized indexer kernel.
	queryWidth := headCount * dim
	backend := tensors.KernelBackend()
	for batch := 0; batch < batchSize; batch++ {
		queryBatch := queryData[batch*sequenceLength*queryWidth : (batch+1)*sequenceLength*queryWidth]
		keyBatch := keyData[batch*blockCount*queryWidth : (batch+1)*blockCount*queryWidth]
		scoreBatch := scoreData[batch*sequenceLength*blockCount : (batch+1)*sequenceLength*blockCount]
		backend.IndexerScores(scoreBatch, queryBatch, keyBatch, weightData, sequenceLength, blockCount, headCount, dim)
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
		rowScores := scores.Data[row*blockCount : (row+1)*blockCount]
		selected[row] = topKIndices(rowScores, k)
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
		rowScores := scores.Data[token*blockCount : token*blockCount+allowed]
		count := min(topK, allowed)
		selection[token] = topKIndices(rowScores, count)
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

// topKIndices returns the indices of the k largest values in descending order,
// with ties broken toward the smaller index. It uses a bounded size-k selection
// when k is much smaller than the input and falls back to a full sort otherwise.
// The result is identical to argsortDescending(values)[:k].
func topKIndices(values []float32, k int) []int {
	if k <= 0 {
		return []int{}
	}
	if k >= len(values) {
		return argsortDescending(values)
	}
	// For large k the bounded heap's constant factors lose to the optimized
	// sort, so only use it when it meaningfully prunes the comparison count.
	if k*4 >= len(values) {
		return argsortDescending(values)[:k]
	}
	heap := make([]int, 0, k)
	for index := range values {
		if len(heap) < k {
			heap = append(heap, index)
			siftUpIndex(values, heap, len(heap)-1)
			continue
		}
		if indexBetter(values, index, heap[0]) {
			heap[0] = index
			siftDownIndex(values, heap, 0)
		}
	}
	// Pop the worst (heap root) repeatedly to obtain ascending order, then
	// reverse into descending order.
	ordered := make([]int, len(heap))
	for count := len(heap) - 1; count >= 0; count-- {
		ordered[count] = heap[0]
		heap[0] = heap[len(heap)-1]
		heap = heap[:len(heap)-1]
		siftDownIndex(values, heap, 0)
	}
	return ordered
}

// indexBetter reports whether candidate index a outranks b: larger value, or
// equal value with a smaller index.
func indexBetter(values []float32, a, b int) bool {
	if values[a] != values[b] {
		return values[a] > values[b]
	}
	return a < b
}

// siftUpIndex restores the min-heap invariant where the root is the worst
// retained element.
func siftUpIndex(values []float32, heap []int, position int) {
	for position > 0 {
		parent := (position - 1) / 2
		if !indexBetter(values, heap[parent], heap[position]) {
			break
		}
		heap[parent], heap[position] = heap[position], heap[parent]
		position = parent
	}
}

// siftDownIndex restores the min-heap invariant from the root.
func siftDownIndex(values []float32, heap []int, position int) {
	size := len(heap)
	for {
		left := 2*position + 1
		if left >= size {
			return
		}
		worst := position
		if indexBetter(values, heap[position], heap[left]) {
			worst = left
		}
		if right := left + 1; right < size && indexBetter(values, heap[worst], heap[right]) {
			worst = right
		}
		if worst == position {
			return
		}
		heap[position], heap[worst] = heap[worst], heap[position]
		position = worst
	}
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
	selection, _ := indexer.SelectProjectedWithCandidates(hidden, keyProjections, blockCount, positionOffset, ratio, topK, pool, candidateBudget)
	return selection
}

// SelectProjectedWithCandidates is SelectProjected and additionally returns the
// candidate block pool that was fully scored after the coarse stage, one slice
// per query row. Publishing this pool lets later Reindex layers skip the coarse
// scoring pass and score only the pool (DeepSeek-V4.1 §2.3.2).
//
// The coarse stage assigns every super-block of `pool` compressed entries the
// maximum index score among its entries, selects the best super-blocks, and
// collects their entries into the candidate pool (DeepSeek-V4.1 §2.3.2). The
// final top-k is then taken over those entries' scores.
func (indexer *SparseIndexer) SelectProjectedWithCandidates(hidden *tensors.Tensor, keyProjections []float32, blockCount, positionOffset, ratio, topK, pool, candidateBudget int) (selection, candidatePool [][]int) {
	query := indexer.query.Forward(hidden) // [B*T, H, dim]
	return indexer.selectProjectedScores(query.Data, keyProjections, hidden.Shape[0], hidden.Shape[1], blockCount, positionOffset, ratio, topK, pool, candidateBudget)
}

// SelectWithinPool selects the top-k blocks for every query row from a
// caller-provided candidate pool. It is the Reindex-layer path: it computes the
// indexer query but skips the coarse pooled scoring entirely, so its per-query
// cost is bounded by the pool size rather than the context length.
func (indexer *SparseIndexer) SelectWithinPool(hidden *tensors.Tensor, keyProjections []float32, blockCount, positionOffset, ratio, topK int, candidatePool [][]int) [][]int {
	query := indexer.query.Forward(hidden)
	return indexer.selectWithinPool(query.Data, keyProjections, hidden.Shape[0], hidden.Shape[1], blockCount, positionOffset, ratio, topK, candidatePool)
}

// indexerScoreChunkRows bounds how many query rows are scored at once when the
// hierarchical selector materializes per-entry scores. It keeps peak scratch
// memory independent of sequence length.
const indexerScoreChunkRows = 256

// selectFromScoreMatrix runs the block-max hierarchical selection against a
// precomputed score matrix [batch*sequence, blockCount]. It is used by training,
// which already materializes the full index scores for the distillation loss.
func (indexer *SparseIndexer) selectFromScoreMatrix(scores []float32, batchSize, sequenceLength, blockCount, positionOffset, ratio, topK, pool, candidateBudget int) (selection, candidatePool [][]int) {
	if pool < 1 {
		pool = 1
	}
	rowCount := batchSize * sequenceLength
	selection = make([][]int, rowCount)
	candidatePool = make([][]int, rowCount)
	if blockCount == 0 {
		for row := 0; row < rowCount; row++ {
			selection[row] = []int{}
			candidatePool[row] = []int{}
		}
		return selection, candidatePool
	}
	groups := (blockCount + pool - 1) / pool
	blockMax := make([]float32, rowCount*groups)
	tensors.KernelBackend().IndexerBlockMax(blockMax, scores, rowCount, blockCount, pool)
	indexer.selectRows(scores, blockMax, selection, candidatePool, batchSize, sequenceLength, blockCount, groups, positionOffset, ratio, topK, pool, candidateBudget)
	return selection, candidatePool
}

// selectProjectedScores computes the full per-entry index scores from the query
// and key projections, in row chunks, and runs the block-max selection. The
// chunking bounds peak scratch memory while the fused IndexerScores kernel still
// handles the arithmetic.
func (indexer *SparseIndexer) selectProjectedScores(queryData, keyData []float32, batchSize, sequenceLength, blockCount, positionOffset, ratio, topK, pool, candidateBudget int) (selection, candidatePool [][]int) {
	if pool < 1 {
		pool = 1
	}
	rowCount := batchSize * sequenceLength
	selection = make([][]int, rowCount)
	candidatePool = make([][]int, rowCount)
	if blockCount == 0 {
		for row := 0; row < rowCount; row++ {
			selection[row] = []int{}
			candidatePool[row] = []int{}
		}
		return selection, candidatePool
	}
	groups := (blockCount + pool - 1) / pool
	headWidth := indexer.HeadCount * indexer.Dim
	weight := indexer.headWeights.Data
	backend := tensors.KernelBackend()
	useKernel := sequenceLength >= indexerKernelCoarseMinRows

	chunkRows := min(indexerScoreChunkRows, sequenceLength)
	scoreChunk := make([]float32, chunkRows*blockCount)
	blockMaxChunk := make([]float32, chunkRows*groups)

	for batch := 0; batch < batchSize; batch++ {
		keyBatch := keyData[batch*blockCount*headWidth : (batch+1)*blockCount*headWidth]
		queryBatch := queryData[batch*sequenceLength*headWidth : (batch+1)*sequenceLength*headWidth]
		for chunkStart := 0; chunkStart < sequenceLength; chunkStart += chunkRows {
			chunkEnd := min(chunkStart+chunkRows, sequenceLength)
			rows := chunkEnd - chunkStart
			scores := scoreChunk[:rows*blockCount]
			queryChunk := queryBatch[chunkStart*headWidth : chunkEnd*headWidth]
			if useKernel {
				backend.IndexerScores(scores, queryChunk, keyBatch, weight, rows, blockCount, indexer.HeadCount, indexer.Dim)
			} else {
				indexerScoresScalar(scores, queryChunk, keyBatch, weight, rows, blockCount, indexer.HeadCount, indexer.Dim)
			}
			blockMax := blockMaxChunk[:rows*groups]
			backend.IndexerBlockMax(blockMax, scores, rows, blockCount, pool)

			// The rows within a chunk are contiguous positions, so their
			// causality limits advance together; selectRows recomputes each
			// row's allowed range and patches the partial trailing group.
			selectBatch := make([][]int, rows)
			poolBatch := make([][]int, rows)
			indexer.selectRows(scores, blockMax, selectBatch, poolBatch, 1, rows, blockCount, groups, positionOffset+chunkStart, ratio, topK, pool, candidateBudget)
			for row := 0; row < rows; row++ {
				absolute := batch*sequenceLength + chunkStart + row
				selection[absolute] = selectBatch[row]
				candidatePool[absolute] = poolBatch[row]
			}
		}
	}
	return selection, candidatePool
}

// selectRows applies the block-max selection to each query row of a score
// matrix whose rows start at positionOffset. It patches the causal boundary of
// the partial trailing super-block before choosing the top groups.
func (indexer *SparseIndexer) selectRows(scores, blockMax []float32, selection, candidatePool [][]int, batchSize, sequenceLength, blockCount, groups, positionOffset, ratio, topK, pool, candidateBudget int) {
	for batch := 0; batch < batchSize; batch++ {
		for token := 0; token < sequenceLength; token++ {
			row := batch*sequenceLength + token
			allowed := (positionOffset + token) / ratio
			if allowed > blockCount {
				allowed = blockCount
			}
			scoreRow := scores[row*blockCount : (row+1)*blockCount]
			blockMaxRow := blockMax[row*groups : (row+1)*groups]
			selection[row], candidatePool[row] = selectRowBlockMax(scoreRow, blockMaxRow, allowed, topK, pool, candidateBudget)
		}
	}
}

// selectRowBlockMax returns the top-k selection and candidate pool for one query
// row. blockMaxRow holds the maximum score of every super-block; its partial
// trailing group is recomputed over the causally allowed entries so that future
// entries cannot influence the choice.
func selectRowBlockMax(scoreRow, blockMaxRow []float32, allowed, topK, pool, candidateBudget int) (selection, candidatePool []int) {
	if allowed <= 0 || topK <= 0 {
		return []int{}, []int{}
	}
	if pool < 1 {
		pool = 1
	}
	groupsPerToken := (candidateBudget + pool - 1) / pool
	if groupsPerToken < 1 {
		groupsPerToken = 1
	}
	allowedGroups := (allowed + pool - 1) / pool
	if allowed%pool != 0 && allowedGroups > 0 {
		group := allowedGroups - 1
		maximum := float32(math.Inf(-1))
		for entry := group * pool; entry < allowed; entry++ {
			if scoreRow[entry] > maximum {
				maximum = scoreRow[entry]
			}
		}
		blockMaxRow[group] = maximum
	}
	chosen := topKIndices(blockMaxRow[:allowedGroups], groupsPerToken)
	candidates := make([]int, 0, groupsPerToken*pool)
	for _, group := range chosen {
		start := group * pool
		end := min(start+pool, allowed)
		for entry := start; entry < end; entry++ {
			candidates = append(candidates, entry)
		}
	}
	return topKFromScoreRow(scoreRow, candidates, topK), candidates
}

// topKFromScoreRow returns the top-k candidate entries by their precomputed
// scores, in descending score order with ties broken toward the smaller entry.
func topKFromScoreRow(scoreRow []float32, candidates []int, topK int) []int {
	if len(candidates) == 0 || topK <= 0 {
		return []int{}
	}
	fine := make([]float32, len(candidates))
	for index, entry := range candidates {
		fine[index] = scoreRow[entry]
	}
	order := topKIndices(fine, topK)
	selected := make([]int, len(order))
	for index, position := range order {
		selected[index] = candidates[position]
	}
	return selected
}

// indexerScoresScalar is the non-vectorized fallback for single-token decode,
// where the fused kernel's per-row goroutine overhead dominates.
func indexerScoresScalar(destination, query, key, weight []float32, rowCount, blockCount, headCount, dim int) {
	headWidth := headCount * dim
	for row := 0; row < rowCount; row++ {
		queryRow := query[row*headWidth:]
		outRow := destination[row*blockCount:]
		for block := 0; block < blockCount; block++ {
			keyBlock := key[block*headWidth:]
			var total float32
			for head := 0; head < headCount; head++ {
				queryHead := queryRow[head*dim:]
				keyHead := keyBlock[head*dim:]
				var dot float32
				for element := 0; element < dim; element++ {
					dot += queryHead[element] * keyHead[element]
				}
				if dot > 0 {
					total += weight[head] * dot
				}
			}
			outRow[block] = total
		}
	}
}

// indexerKernelCoarseMinRows is the query-row count at which the batched
// scoring kernel beats the per-row scalar loop.
const indexerKernelCoarseMinRows = 16

// selectWithinPool scores only the candidate blocks of each query row and
// selects its top-k. The caller must guarantee the pool contains only causally
// allowed blocks (as published by the producing Full layer for the same
// positions).
func (indexer *SparseIndexer) selectWithinPool(queryData, keyData []float32, batchSize, sequenceLength, blockCount, positionOffset, ratio, topK int, candidatePool [][]int) [][]int {
	dim := indexer.Dim
	headCount := indexer.HeadCount
	headWidth := headCount * dim
	selection := make([][]int, batchSize*sequenceLength)
	for batch := 0; batch < batchSize; batch++ {
		batchKeyBase := batch * blockCount * headWidth
		for token := 0; token < sequenceLength; token++ {
			row := batch*sequenceLength + token
			if topK <= 0 || candidatePool == nil || row >= len(candidatePool) {
				selection[row] = []int{}
				continue
			}
			allowed := (positionOffset + token) / ratio
			if allowed > blockCount {
				allowed = blockCount
			}
			selection[row] = indexer.fineSelect(queryData, keyData, row, batchKeyBase, headWidth, candidatePool[row], topK, allowed)
		}
	}
	return selection
}

// fineSelect scores the candidate blocks for one query row and returns the
// top-k in descending score order with ties broken toward the smaller block
// index. Candidates with a block index >= maxBlock are ignored, which enforces
// causality when the pool originated from a different query.
func (indexer *SparseIndexer) fineSelect(queryData, keyData []float32, row, batchKeyBase, headWidth int, candidates []int, topK, maxBlock int) []int {
	if len(candidates) == 0 || topK <= 0 {
		return []int{}
	}
	dim := indexer.Dim
	headCount := indexer.HeadCount
	weightData := indexer.headWeights.Data
	queryBase := row * headWidth
	kept := make([]int, 0, len(candidates))
	fine := make([]float32, 0, len(candidates))
	for _, entry := range candidates {
		if entry < 0 || entry >= maxBlock {
			continue
		}
		var total float32
		entryBase := batchKeyBase + entry*headWidth
		for head := 0; head < headCount; head++ {
			var dot float32
			queryHead := queryBase + head*dim
			keyHead := entryBase + head*dim
			for dimension := 0; dimension < dim; dimension++ {
				dot += queryData[queryHead+dimension] * keyData[keyHead+dimension]
			}
			if dot > 0 {
				total += weightData[head] * dot
			}
		}
		kept = append(kept, entry)
		fine = append(fine, total)
	}
	if len(kept) == 0 {
		return []int{}
	}
	order := topKIndices(fine, topK)
	selected := make([]int, len(order))
	for index := range order {
		selected[index] = kept[order[index]]
	}
	return selected
}

// Parameters returns the indexer's trainable tensors.
func (indexer *SparseIndexer) Parameters() []*tensors.Tensor {
	return []*tensors.Tensor{indexer.query.Weight, indexer.key.Weight, indexer.headWeights}
}
