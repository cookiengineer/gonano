package scalar

import "math"

// IndexerScores computes the lightning indexer's per-(row, block) scores:
//
//	destination[row, block] = sum_h weight[h] * max(0, dot(query[row,h], key[block,h]))
//
// query is [rowCount, headCount, dim], key is [blockCount, headCount, dim],
// weight is [headCount], and destination is [rowCount, blockCount].
func (backend *Backend) IndexerScores(destination, query, key, weight []float32, rowCount, blockCount, headCount, dim int) {
	for row := 0; row < rowCount; row++ {
		queryRow := query[row*headCount*dim:]
		outRow := destination[row*blockCount:]
		for block := 0; block < blockCount; block++ {
			keyBlock := key[block*headCount*dim:]
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

// IndexerBlockMax reduces consecutive non-overlapping groups of `groupSize`
// score columns of source [rowCount, blockCount] into destination
// [rowCount, ceil(blockCount/groupSize)]: destination[row, g] is the maximum of
// source[row, g*groupSize:(g+1)*groupSize]. The trailing group keeps its actual
// column count. It implements the hierarchical indexer's block score, where a
// block's score is the maximum index score among its entries (DeepSeek-V4.1
// §2.3.2).
func (backend *Backend) IndexerBlockMax(destination, scores []float32, rowCount, blockCount, groupSize int) {
	if groupSize < 1 {
		groupSize = 1
	}
	groups := (blockCount + groupSize - 1) / groupSize
	for row := 0; row < rowCount; row++ {
		sourceRow := scores[row*blockCount : (row+1)*blockCount]
		destinationRow := destination[row*groups : (row+1)*groups]
		for group := 0; group < groups; group++ {
			start := group * groupSize
			end := min(start+groupSize, blockCount)
			maximum := float32(math.Inf(-1))
			for entry := start; entry < end; entry++ {
				if sourceRow[entry] > maximum {
					maximum = sourceRow[entry]
				}
			}
			destinationRow[group] = maximum
		}
	}
}
