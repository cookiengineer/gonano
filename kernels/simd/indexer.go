package simdbackend

import (
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

// PooledMean averages consecutive non-overlapping groups of `groupSize` rows of
// source [blockCount, width] into destination [ceil(blockCount/groupSize),
// width]. The trailing group keeps its actual row count.
func (backend *Backend) PooledMean(destination, source []float32, blockCount, groupSize, width int) {
	pooledMeanCore(destination, source, blockCount, groupSize, width)
}

// pooledMeanCore holds the vectorized body of PooledMean.
func pooledMeanCore(destination, source []float32, blockCount, groupSize, width int) {
	if groupSize < 1 {
		groupSize = 1
	}
	groups := (blockCount + groupSize - 1) / groupSize
	for group := 0; group < groups; group++ {
		start := group * groupSize
		end := min(start+groupSize, blockCount)
		count := end - start
		if count <= 0 {
			continue
		}
		destinationRow := destination[group*width : (group+1)*width]
		clear(destinationRow)
		for entry := start; entry < end; entry++ {
			sourceRow := source[entry*width : (entry+1)*width]
			column := 0
			for ; column+float32LaneCount <= width; column += float32LaneCount {
				accumulator := simd.LoadFloat32s(destinationRow[column:])
				accumulator = accumulator.Add(simd.LoadFloat32s(sourceRow[column:]))
				accumulator.Store(destinationRow[column:])
			}
			for ; column < width; column++ {
				destinationRow[column] += sourceRow[column]
			}
		}
		inverse := float32(1) / float32(count)
		inverseVector := simd.BroadcastFloat32s(inverse)
		column := 0
		for ; column+float32LaneCount <= width; column += float32LaneCount {
			simd.LoadFloat32s(destinationRow[column:]).Mul(inverseVector).Store(destinationRow[column:])
		}
		for ; column < width; column++ {
			destinationRow[column] *= inverse
		}
	}
}
