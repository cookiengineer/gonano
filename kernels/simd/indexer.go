package simdbackend

import (
	"math"
	"simd"

	"github.com/cookiengineer/gonano/internal/parallel"
)

// IndexerScores computes the lightning indexer's per-(row, block) scores with a
// vectorized per-head dot product, parallelized across query rows:
//
//	destination[row, block] = sum_h weight[h] * max(0, dot(query[row,h], key[block,h]))
//
// query is [rowCount, headCount, dim], key is [blockCount, headCount, dim],
// weight is [headCount], and destination is [rowCount, blockCount].
func (backend *Backend) IndexerScores(destination, query, key, weight []float32, rowCount, blockCount, headCount, dim int) {
	parallel.KernelPool().For(0, rowCount, func(row int) {
		queryRow := query[row*headCount*dim:]
		outRow := destination[row*blockCount:]
		for block := 0; block < blockCount; block++ {
			keyBlock := key[block*headCount*dim:]
			var total float32
			for head := 0; head < headCount; head++ {
				dot := dotProduct(queryRow[head*dim:], keyBlock[head*dim:], dim)
				if dot > 0 {
					total += weight[head] * dot
				}
			}
			outRow[block] = total
		}
	})
}

// IndexerBlockMax reduces consecutive non-overlapping groups of `groupSize`
// score columns of source [rowCount, blockCount] into destination
// [rowCount, ceil(blockCount/groupSize)]: destination[row, g] is the maximum of
// source[row, g*groupSize:(g+1)*groupSize]. The trailing group keeps its actual
// column count. It implements the hierarchical indexer's block score, where a
// block's score is the maximum index score among its entries (DeepSeek-V4.1
// §2.3.2).
func (backend *Backend) IndexerBlockMax(destination, scores []float32, rowCount, blockCount, groupSize int) {
	indexerBlockMaxCore(destination, scores, rowCount, blockCount, groupSize)
}

// indexerBlockMaxCore holds the vectorized body of IndexerBlockMax, for the
// same compiler reason as addCore in elementwise.go.
func indexerBlockMaxCore(destination, scores []float32, rowCount, blockCount, groupSize int) {
	if groupSize < 1 {
		groupSize = 1
	}
	groups := (blockCount + groupSize - 1) / groupSize
	parallel.KernelPool().For(0, rowCount, func(row int) {
		sourceRow := scores[row*blockCount : (row+1)*blockCount]
		destinationRow := destination[row*groups : (row+1)*groups]
		for group := 0; group < groups; group++ {
			start := group * groupSize
			end := min(start+groupSize, blockCount)
			maximum := float32(math.Inf(-1))
			column := start
			if end-start >= float32LaneCount {
				accumulator := simd.LoadFloat32s(sourceRow[column:])
				column += float32LaneCount
				for ; column+float32LaneCount <= end; column += float32LaneCount {
					accumulator = accumulator.Max(simd.LoadFloat32s(sourceRow[column:]))
				}
				maximum = horizontalMax(accumulator)
			}
			for ; column < end; column++ {
				if sourceRow[column] > maximum {
					maximum = sourceRow[column]
				}
			}
			destinationRow[group] = maximum
		}
	})
}
