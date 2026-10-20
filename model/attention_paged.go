package model

import (
	"math"

	"github.com/cookiengineer/gonano/internal/parallel"
	"github.com/cookiengineer/gonano/tensors"
)

// pagedAttentionParallelThreshold is the page count above which decode
// attention distributes blocks across workers.
const pagedAttentionParallelThreshold = 4

type pagedPageResult struct {
	output []float32
	lse    float32
}

// mergePagedResults combines per-page attention outputs with an online
// log-sum-exp. Each merge is associative, so a pairwise tree gives the same
// result as a serial scan while letting independent pairs merge in parallel.
// Empty pages (for example, a short final block) are dropped first.
func mergePagedResults(output, logSumExp []float32, results []pagedPageResult, count int) {
	items := make([]pagedPageResult, 0, count)
	for index := 0; index < count; index++ {
		if results[index].output == nil || math.IsInf(float64(results[index].lse), -1) {
			continue
		}
		items = append(items, results[index])
	}
	if len(items) == 0 {
		clear(output)
		logSumExp[0] = float32(math.Inf(-1))
		return
	}
	if len(items) == 1 || len(items) < pagedAttentionParallelThreshold || parallel.Default().Workers() < 2 {
		mergePagedResultsSerial(items, output, logSumExp)
		return
	}
	for len(items) > 1 {
		next := make([]pagedPageResult, (len(items)+1)/2)
		parallel.KernelPool().For(0, len(next), func(index int) {
			left := &items[2*index]
			if 2*index+1 >= len(items) {
				next[index] = *left
				return
			}
			right := &items[2*index+1]
			mergeAttentionRow(left.output, left.output, right.output, left.lse, right.lse)
			left.lse = logAddExp(left.lse, right.lse)
			next[index] = *left
		})
		items = next
	}
	copy(output, items[0].output)
	logSumExp[0] = items[0].lse
}

func mergePagedResultsSerial(items []pagedPageResult, output, logSumExp []float32) {
	clear(output)
	logSumExp[0] = float32(math.Inf(-1))
	for index := range items {
		if math.IsInf(float64(logSumExp[0]), -1) {
			copy(output, items[index].output)
			logSumExp[0] = items[index].lse
			continue
		}
		mergeAttentionRow(output, output, items[index].output, logSumExp[0], items[index].lse)
		logSumExp[0] = logAddExp(logSumExp[0], items[index].lse)
	}
}

// pagedRawAttention computes one decode query's attention directly over the
// physical raw K/V blocks, merging each block's normalized result with an
// online log-sum-exp. limit is the number of valid token positions.
func pagedRawAttention(cache *KVBuffer, layer, row, head, limit int, query, output, logSumExp []float32) {
	headDimension := len(query)
	clear(output)
	logSumExp[0] = float32(math.Inf(-1))
	pageOutput := make([]float32, headDimension)
	pageLogSumExp := make([]float32, 1)
	blocks := cache.BlockTableSize(row)
	pageCount := (limit + KVBlockSize - 1) / KVBlockSize
	if pageCount > blocks {
		pageCount = blocks
	}
	if pageCount >= pagedAttentionParallelThreshold && parallel.Default().Workers() > 1 {
		results := make([]pagedPageResult, pageCount)
		parallel.KernelPool().For(0, pageCount, func(block int) {
			start := block * KVBlockSize
			rows := limit - start
			if rows > KVBlockSize {
				rows = KVBlockSize
			}
			key, value := cache.RawBlock(layer, row, head, block)
			if key == nil || value == nil || len(key) < rows*headDimension || len(value) < rows*headDimension {
				return
			}
			result := &results[block]
			result.output = make([]float32, headDimension)
			statistic := make([]float32, 1)
			tensors.AttentionForward(query, key[:rows*headDimension], value[:rows*headDimension], result.output, statistic, 1, rows, headDimension, rows, -1)
			result.lse = statistic[0]
		})
		mergePagedResults(output, logSumExp, results, pageCount)
		return
	}
	for block := 0; block < blocks; block++ {
		start := block * KVBlockSize
		if start >= limit {
			break
		}
		rows := limit - start
		if rows > KVBlockSize {
			rows = KVBlockSize
		}
		key, value := cache.RawBlock(layer, row, head, block)
		if key == nil || len(key) < rows*headDimension {
			continue
		}
		if value == nil || len(value) < rows*headDimension {
			continue
		}
		tensors.AttentionForward(query, key[:rows*headDimension], value[:rows*headDimension], pageOutput, pageLogSumExp, 1, rows, headDimension, rows, -1)
		mergeAttentionRow(output, output, pageOutput, logSumExp[0], pageLogSumExp[0])
		logSumExp[0] = logAddExp(logSumExp[0], pageLogSumExp[0])
	}
}

// pagedCompressedAttention computes one decode query's attention directly over
// the physical compressed blocks.
func pagedCompressedAttention(cache *KVBuffer, layer, row, head, count int, query, output, logSumExp []float32) {
	headDimension := len(query)
	clear(output)
	logSumExp[0] = float32(math.Inf(-1))
	entries := cache.CompressedEntriesPerBlock(layer)
	if entries <= 0 {
		return
	}
	pageOutput := make([]float32, headDimension)
	pageLogSumExp := make([]float32, 1)
	consumed := 0
	blocks := cache.BlockTableSize(row)
	pageCount := (count + entries - 1) / entries
	if pageCount > blocks {
		pageCount = blocks
	}
	if pageCount >= pagedAttentionParallelThreshold && parallel.Default().Workers() > 1 {
		results := make([]pagedPageResult, pageCount)
		parallel.KernelPool().For(0, pageCount, func(block int) {
			pageStart := block * entries
			rows := count - pageStart
			if rows > entries {
				rows = entries
			}
			key, value := cache.CompressedBlockData(layer, row, head, block)
			if key == nil || value == nil || len(key) < rows*headDimension || len(value) < rows*headDimension {
				return
			}
			result := &results[block]
			result.output = make([]float32, headDimension)
			statistic := make([]float32, 1)
			tensors.AttentionForward(query, key[:rows*headDimension], value[:rows*headDimension], result.output, statistic, 1, rows, headDimension, rows, -1)
			result.lse = statistic[0]
		})
		mergePagedResults(output, logSumExp, results, pageCount)
		return
	}
	for block := 0; block < blocks && consumed < count; block++ {
		rows := count - consumed
		if rows > entries {
			rows = entries
		}
		key, value := cache.CompressedBlockData(layer, row, head, block)
		if key == nil || len(key) < rows*headDimension || value == nil || len(value) < rows*headDimension {
			consumed += rows
			continue
		}
		tensors.AttentionForward(query, key[:rows*headDimension], value[:rows*headDimension], pageOutput, pageLogSumExp, 1, rows, headDimension, rows, -1)
		mergeAttentionRow(output, output, pageOutput, logSumExp[0], pageLogSumExp[0])
		logSumExp[0] = logAddExp(logSumExp[0], pageLogSumExp[0])
		consumed += rows
	}
}

// pagedMLAAttention computes one decode query's absorbed-MLA attention directly
// over latent/rope blocks. The page scratch is bounded by one KV block.
func pagedMLAAttention(mla *MLAttention, cache *KVBuffer, layer, row, queryHead, kvHead, count int, query, output, logSumExp []float32) {
	headDimension := len(query)
	latentDimension := mla.latentDim
	ropeDimension := mla.ropeDim
	kvHeads := mla.kvHeadCount
	clear(output)
	logSumExp[0] = float32(math.Inf(-1))

	blocks := cache.BlockTableSize(row)
	pageCount := (count + KVBlockSize - 1) / KVBlockSize
	if pageCount > blocks {
		pageCount = blocks
	}
	if pageCount >= pagedAttentionParallelThreshold && parallel.Default().Workers() > 1 {
		results := make([]pagedPageResult, pageCount)
		parallel.KernelPool().For(0, pageCount, func(block int) {
			pageStart := block * KVBlockSize
			rows := count - pageStart
			if rows > KVBlockSize {
				rows = KVBlockSize
			}
			latent, rope := cache.MLABlock(layer, row, block)
			if latent == nil || rope == nil {
				return
			}
			pageKey := make([]float32, rows*headDimension)
			pageValue := make([]float32, rows*headDimension)
			for token := 0; token < rows; token++ {
				keyBase := token * headDimension
				copy(pageKey[keyBase:keyBase+latentDimension], latent[token*latentDimension:(token+1)*latentDimension])
				ropeBase := token*kvHeads*ropeDimension + kvHead*ropeDimension
				copy(pageKey[keyBase+latentDimension:keyBase+headDimension], rope[ropeBase:ropeBase+ropeDimension])
				copy(pageValue[keyBase:keyBase+latentDimension], latent[token*latentDimension:(token+1)*latentDimension])
				clear(pageValue[keyBase+latentDimension : keyBase+headDimension])
			}
			result := &results[block]
			result.output = make([]float32, headDimension)
			statistic := make([]float32, 1)
			tensors.AttentionForward(query, pageKey, pageValue, result.output, statistic, 1, rows, headDimension, rows, -1)
			result.lse = statistic[0]
		})
		mergePagedResults(output, logSumExp, results, pageCount)
		return
	}

	pageKey := make([]float32, KVBlockSize*headDimension)
	pageValue := make([]float32, KVBlockSize*headDimension)
	pageOutput := make([]float32, headDimension)
	pageLogSumExp := make([]float32, 1)
	consumed := 0
	for block := 0; block < blocks && consumed < count; block++ {
		rows := count - consumed
		if rows > KVBlockSize {
			rows = KVBlockSize
		}
		latent, rope := cache.MLABlock(layer, row, block)
		if latent == nil || rope == nil {
			consumed += rows
			continue
		}
		for token := 0; token < rows; token++ {
			keyBase := token * headDimension
			copy(pageKey[keyBase:keyBase+latentDimension], latent[token*latentDimension:(token+1)*latentDimension])
			ropeBase := token*kvHeads*ropeDimension + kvHead*ropeDimension
			copy(pageKey[keyBase+latentDimension:keyBase+headDimension], rope[ropeBase:ropeBase+ropeDimension])
			copy(pageValue[keyBase:keyBase+latentDimension], latent[token*latentDimension:(token+1)*latentDimension])
			clear(pageValue[keyBase+latentDimension : keyBase+headDimension])
		}
		tensors.AttentionForward(query, pageKey[:rows*headDimension], pageValue[:rows*headDimension], pageOutput, pageLogSumExp, 1, rows, headDimension, rows, -1)
		mergeAttentionRow(output, output, pageOutput, logSumExp[0], pageLogSumExp[0])
		logSumExp[0] = logAddExp(logSumExp[0], pageLogSumExp[0])
		consumed += rows
	}
}
