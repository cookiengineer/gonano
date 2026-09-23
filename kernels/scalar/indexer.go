package scalar

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

// PooledMean averages consecutive non-overlapping groups of `groupSize` rows of
// source [blockCount, width] into destination [ceil(blockCount/groupSize),
// width]. The trailing group keeps its actual row count.
func (backend *Backend) PooledMean(destination, source []float32, blockCount, groupSize, width int) {
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
			for element := 0; element < width; element++ {
				destinationRow[element] += sourceRow[element]
			}
		}
		inverse := float32(1) / float32(count)
		for element := 0; element < width; element++ {
			destinationRow[element] *= inverse
		}
	}
}
