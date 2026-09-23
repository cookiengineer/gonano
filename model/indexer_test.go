package model

import (
	"math"
	"sort"
	"testing"

	"github.com/cookiengineer/gonano/tensors"
)

func testIndexer() (*SparseIndexer, *tensors.Tensor, *tensors.Tensor) {
	indexer := NewSparseIndexer(4, 5, 2, 2) // d=4, compressedWidth=5, dim=2, heads=2
	rng := tensors.NewRNG(3)
	for _, parameter := range indexer.Parameters() {
		for index := range parameter.Data {
			parameter.Data[index] = rng.NormFloat32()
		}
	}
	hidden := tensors.New(1, 2, 4)
	compressed := tensors.New(1, 3, 5)
	tensors.FillNormal(hidden, rng, 1)
	tensors.FillNormal(compressed, rng, 1)
	return indexer, hidden, compressed
}

func TestSparseIndexerScoresMatchNaive(t *testing.T) {
	indexer, hidden, compressed := testIndexer()
	scores := indexer.Scores(hidden, compressed)

	dim := indexer.Dim
	headCount := indexer.HeadCount
	sequenceLength := hidden.Shape[1]
	blockCount := compressed.Shape[1]

	for token := 0; token < sequenceLength; token++ {
		for block := 0; block < blockCount; block++ {
			var want float32
			for head := 0; head < headCount; head++ {
				var dot float32
				for k := 0; k < dim; k++ {
					queryElement := float32(0)
					for i := 0; i < hidden.Shape[2]; i++ {
						queryElement += hidden.Data[token*hidden.Shape[2]+i] * indexer.query.Weight.Data[(head*dim+k)*hidden.Shape[2]+i]
					}
					keyElement := float32(0)
					for j := 0; j < compressed.Shape[2]; j++ {
						keyElement += compressed.Data[block*compressed.Shape[2]+j] * indexer.key.Weight.Data[(head*dim+k)*compressed.Shape[2]+j]
					}
					dot += queryElement * keyElement
				}
				if dot > 0 {
					want += indexer.headWeights.Data[head] * dot
				}
			}
			got := scores.Data[token*blockCount+block]
			if math.Abs(float64(got-want)) > 1e-4 {
				t.Fatalf("score[%d][%d] = %v, want %v", token, block, got, want)
			}
		}
	}
}

func indexerSoftmax(scores []float32, blockCount int) []float32 {
	probabilities := make([]float32, len(scores))
	for row := 0; row*blockCount < len(scores); row++ {
		base := row * blockCount
		maximum := scores[base]
		for block := 1; block < blockCount; block++ {
			if scores[base+block] > maximum {
				maximum = scores[base+block]
			}
		}
		var total float64
		for block := 0; block < blockCount; block++ {
			exponent := math.Exp(float64(scores[base+block] - maximum))
			probabilities[base+block] = float32(exponent)
			total += exponent
		}
		for block := 0; block < blockCount; block++ {
			probabilities[base+block] /= float32(total)
		}
	}
	return probabilities
}

func TestSparseIndexerGradient(t *testing.T) {
	indexer, hidden, compressed := testIndexer()
	rng := tensors.NewRNG(17)
	blockCount := compressed.Shape[1]
	target := tensors.New(hidden.Shape[0]*hidden.Shape[1], blockCount)
	for row := 0; row < target.Shape[0]; row++ {
		var total float32
		for block := 0; block < blockCount; block++ {
			target.Data[row*blockCount+block] = rng.NormFloat32()
			if target.Data[row*blockCount+block] < 0 {
				target.Data[row*blockCount+block] = -target.Data[row*blockCount+block]
			}
			total += target.Data[row*blockCount+block]
		}
		for block := 0; block < blockCount; block++ {
			target.Data[row*blockCount+block] /= total
		}
	}

	scores, context := indexer.ScoresWithContext(hidden, compressed)
	probabilities := indexerSoftmax(scores.Data, blockCount)
	gradScores := tensors.New(scores.Shape...)
	for index := range gradScores.Data {
		gradScores.Data[index] = probabilities[index] - target.Data[index]
	}
	indexer.ZeroGrad()
	gradientHidden, gradientCompressed := indexer.Backward(gradScores, context)

	// Random directions for the inputs and parameters.
	dirHidden := tensors.New(hidden.Shape...)
	dirCompressed := tensors.New(compressed.Shape...)
	tensors.FillNormal(dirHidden, rng, 1)
	tensors.FillNormal(dirCompressed, rng, 1)
	parameters := indexer.Parameters()
	directions := make([][]float32, len(parameters))
	for index, parameter := range parameters {
		directions[index] = make([]float32, parameter.Numel())
		for element := range directions[index] {
			directions[index][element] = rng.NormFloat32()
		}
	}

	analytic := float64(0)
	for index, value := range gradientHidden.Data {
		analytic += float64(value) * float64(dirHidden.Data[index])
	}
	for index, value := range gradientCompressed.Data {
		analytic += float64(value) * float64(dirCompressed.Data[index])
	}
	for parameterIndex, parameter := range parameters {
		for index, value := range parameter.Grad {
			analytic += float64(value) * float64(directions[parameterIndex][index])
		}
	}

	loss := func() float32 {
		currentScores, _ := indexer.ScoresWithContext(hidden, compressed)
		currentProbabilities := indexerSoftmax(currentScores.Data, blockCount)
		var total float64
		for index, probability := range currentProbabilities {
			if probability > 0 {
				total -= float64(target.Data[index]) * math.Log(float64(probability))
			}
		}
		return float32(total)
	}

	epsilon := float32(1e-3)
	perturb := func(sign float32) {
		for index := range hidden.Data {
			hidden.Data[index] += sign * epsilon * dirHidden.Data[index]
		}
		for index := range compressed.Data {
			compressed.Data[index] += sign * epsilon * dirCompressed.Data[index]
		}
		for parameterIndex, parameter := range parameters {
			for index := range parameter.Data {
				parameter.Data[index] += sign * epsilon * directions[parameterIndex][index]
			}
		}
	}
	perturb(1)
	lossPlus := loss()
	perturb(-2)
	lossMinus := loss()
	perturb(1)
	numeric := float64(lossPlus-lossMinus) / (2 * float64(epsilon))

	scale := math.Abs(analytic) + 1e-4
	if math.Abs(numeric-analytic) > 5e-2*scale {
		t.Fatalf("indexer gradient mismatch: analytic=%v numeric=%v", analytic, numeric)
	}
}

// TestSelectWithinPoolMatchesFineStage verifies that scoring only the published
// candidate pool reproduces the fine stage of the hierarchical selection, and
// that every selected block comes from the pool.
func TestSelectWithinPoolMatchesFineStage(t *testing.T) {
	indexer, hidden, compressed := testIndexer()
	key := indexer.key.Forward(compressed)
	blockCount := compressed.Shape[1]
	selection, pool := indexer.SelectProjectedWithCandidates(hidden, key.Data, blockCount, 0, 2, 2, 2, 3)
	withinPool := indexer.SelectWithinPool(hidden, key.Data, blockCount, 0, 2, 2, pool)
	for row := range selection {
		if len(selection[row]) != len(withinPool[row]) {
			t.Fatalf("row %d: within-pool length %d, want %d", row, len(withinPool[row]), len(selection[row]))
		}
		for index := range selection[row] {
			if selection[row][index] != withinPool[row][index] {
				t.Fatalf("row %d: within-pool %v, want %v", row, withinPool[row], selection[row])
			}
		}
		inPool := map[int]bool{}
		for _, block := range pool[row] {
			inPool[block] = true
		}
		for _, block := range selection[row] {
			if !inPool[block] {
				t.Fatalf("row %d: selected block %d not in candidate pool %v", row, block, pool[row])
			}
		}
	}
}

// TestHierarchicalPoolBoundedByBudget checks that the published pool respects
// the candidate budget and the causality mask.
func TestHierarchicalPoolBoundedByBudget(t *testing.T) {
	indexer := NewSparseIndexer(4, 5, 2, 2)
	rng := tensors.NewRNG(9)
	for _, parameter := range indexer.Parameters() {
		for index := range parameter.Data {
			parameter.Data[index] = rng.NormFloat32()
		}
	}
	hidden := tensors.New(1, 6, 4)
	compressed := tensors.New(1, 64, 5)
	tensors.FillNormal(hidden, rng, 1)
	tensors.FillNormal(compressed, rng, 1)

	key := indexer.key.Forward(compressed)
	poolSize, budget := 4, 8
	_, pool := indexer.SelectProjectedWithCandidates(hidden, key.Data, compressed.Shape[1], 0, 2, 4, poolSize, budget)
	groupsPerToken := (budget + poolSize - 1) / poolSize
	bound := groupsPerToken * poolSize
	for row := range pool {
		if len(pool[row]) > bound {
			t.Fatalf("row %d: pool size %d exceeds bound %d", row, len(pool[row]), bound)
		}
		allowed := row / 2 // positionOffset 0, ratio 2
		for _, block := range pool[row] {
			if block < 0 || block >= allowed {
				t.Fatalf("row %d: candidate block %d outside causal range [0,%d)", row, block, allowed)
			}
		}
	}
}

// TestHierarchicalPoolCoversAllWhenUnbounded verifies that when the candidate
// budget covers every block the pool-based selection equals the full selection,
// so sharing the pool is lossless.
func TestHierarchicalPoolCoversAllWhenUnbounded(t *testing.T) {
	indexer, hidden, compressed := testIndexer()
	key := indexer.key.Forward(compressed)
	blockCount := compressed.Shape[1]
	full := indexer.SelectProjected(hidden, key.Data, blockCount, 0, 2, 2, 2, 1000)
	selection, pool := indexer.SelectProjectedWithCandidates(hidden, key.Data, blockCount, 0, 2, 2, 2, 1000)
	withinPool := indexer.SelectWithinPool(hidden, key.Data, blockCount, 0, 2, 2, pool)
	for row := range full {
		for _, candidate := range []struct {
			name string
			got  []int
		}{
			{"selection", selection[row]},
			{"withinPool", withinPool[row]},
		} {
			if len(candidate.got) != len(full[row]) {
				t.Fatalf("row %d %s: length %d, want %d", row, candidate.name, len(candidate.got), len(full[row]))
			}
			for index := range full[row] {
				if candidate.got[index] != full[row][index] {
					t.Fatalf("row %d %s: %v, want %v", row, candidate.name, candidate.got, full[row])
				}
			}
		}
	}
}

// referenceTopK is an independent full-sort implementation used to verify the
// bounded top-k selection.
func referenceTopK(values []float32, k int) []int {
	if k <= 0 {
		return []int{}
	}
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
	if k < len(order) {
		order = order[:k]
	}
	return order
}

func TestTopKIndicesMatchesFullSort(t *testing.T) {
	rng := tensors.NewRNG(2024)
	for _, length := range []int{0, 1, 2, 5, 16, 64, 257, 1000} {
		values := make([]float32, length)
		for index := range values {
			// Quantize so the input contains many ties.
			values[index] = float32(int(rng.NormFloat32() * 4))
		}
		for _, k := range []int{-1, 0, 1, 3, length / 2, length, length + 5} {
			got := topKIndices(values, k)
			want := referenceTopK(values, k)
			if len(got) != len(want) {
				t.Fatalf("length %d k %d: got %d indices, want %d", length, k, len(got), len(want))
			}
			for index := range want {
				if got[index] != want[index] {
					t.Fatalf("length %d k %d: got %v, want %v", length, k, got, want)
				}
			}
		}
	}
}

func TestTopKIndicesTieBreak(t *testing.T) {
	// Three equal maxima: the smallest indices win, in order.
	values := []float32{5, 5, 5, 1, 2}
	got := topKIndices(values, 2)
	if len(got) != 2 || got[0] != 0 || got[1] != 1 {
		t.Fatalf("topKIndices = %v, want [0 1]", got)
	}
}

func TestHierarchicalSelectMatchesFullWhenUnbounded(t *testing.T) {
	indexer, hidden, compressed := testIndexer()
	scores := indexer.Scores(hidden, compressed)
	full := SelectBlocks(scores, 0, 2, 2)
	for _, pool := range []int{1, 2, 3} {
		hierarchical := indexer.HierarchicalSelect(hidden, compressed, 0, 2, 2, pool, 1000)
		for row := range full {
			if len(hierarchical[row]) != len(full[row]) {
				t.Fatalf("pool %d row %d: length %d, want %d", pool, row, len(hierarchical[row]), len(full[row]))
			}
			for index := range full[row] {
				if hierarchical[row][index] != full[row][index] {
					t.Fatalf("pool %d row %d: selection %v, want %v", pool, row, hierarchical[row], full[row])
				}
			}
		}
	}
}

// TestHierarchicalPoolUsesBlockMax verifies the coarse stage scores a
// super-block by the maximum index score among its entries, not by a pooled
// mean. Row {4,0,3,3} with super-blocks of two entries gives block maxima
// [4,3], so the first block is selected even though the second has a higher
// mean (3 vs 2).
func TestHierarchicalPoolUsesBlockMax(t *testing.T) {
	indexer := NewSparseIndexer(4, 5, 2, 2)
	scores := tensors.NewWithData([]int{1, 4}, []float32{4, 0, 3, 3})
	// sequenceLength 1, positionOffset 4, ratio 1 -> allowed = 4.
	selection, pool := indexer.selectFromScoreMatrix(scores.Data, 1, 1, 4, 4, 1, 1, 2, 2)
	if len(pool[0]) != 2 || pool[0][0] != 0 || pool[0][1] != 1 {
		t.Fatalf("pool = %v, want [0 1] (block-max, not mean)", pool[0])
	}
	if len(selection[0]) != 1 || selection[0][0] != 0 {
		t.Fatalf("selection = %v, want [0]", selection[0])
	}
}

// TestReindexDistillationRestrictedToPool verifies that masking an indexer's
// scores and targets to a candidate pool zeroes everything outside the pool and
// renormalizes the target inside it.
func TestReindexDistillationRestrictedToPool(t *testing.T) {
	scores := tensors.NewWithData([]int{2, 4}, []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	})
	target := tensors.NewWithData([]int{2, 4}, []float32{
		0.1, 0.2, 0.3, 0.4,
		0.25, 0.25, 0.25, 0.25,
	})
	pool := [][]int{{1, 3}, {0, 2}}
	restrictIndexerToPool(scores, target, pool)

	for _, entry := range []int{0, 2} {
		if !math.IsInf(float64(scores.Data[entry]), -1) {
			t.Fatalf("row 0 entry %d score = %v, want -inf", entry, scores.Data[entry])
		}
		if target.Data[entry] != 0 {
			t.Fatalf("row 0 entry %d target = %v, want 0", entry, target.Data[entry])
		}
	}
	if math.Abs(float64(target.Data[1])-1.0/3.0) > 1e-6 || math.Abs(float64(target.Data[3])-2.0/3.0) > 1e-6 {
		t.Fatalf("row 0 in-pool target = [%v %v], want [1/3 2/3]", target.Data[1], target.Data[3])
	}
	if scores.Data[1] != 2 || scores.Data[3] != 4 {
		t.Fatalf("row 0 in-pool scores changed: %v %v", scores.Data[1], scores.Data[3])
	}
	if !math.IsInf(float64(scores.Data[5]), -1) || scores.Data[4] != 5 || scores.Data[6] != 7 {
		t.Fatalf("row 1 scores = %v, want [5 -inf 7 -inf]", scores.Data[4:8])
	}
}

func TestHierarchicalSelectRespectsCausality(t *testing.T) {
	indexer, hidden, compressed := testIndexer()
	selection := indexer.HierarchicalSelect(hidden, compressed, 0, 2, 2, 2, 2)
	if len(selection[0]) != 0 {
		t.Fatalf("token 0 must have no preceding block, got %v", selection[0])
	}
	for _, row := range selection {
		for _, block := range row {
			if block < 0 || block >= compressed.Shape[1] {
				t.Fatalf("selected out-of-range block %d", block)
			}
		}
	}
}

func TestSelectBlocksRespectsCausalityAndTopK(t *testing.T) {
	// 5 tokens, 4 blocks, ratio 2: token t may use blocks < t/2.
	// Scores.put the highest scores at later blocks so causality matters.
	scores := tensors.NewWithData([]int{5, 4}, []float32{
		1, 2, 3, 4,
		5, 1, 2, 3,
		1, 1, 9, 1,
		1, 1, 1, 8,
		7, 7, 7, 7,
	})
	selection := SelectBlocks(scores, 0, 2, 2)
	if len(selection[0]) != 0 {
		t.Fatalf("token 0 must have no preceding block, got %v", selection[0])
	}
	// Token 2: allowed blocks 0..0 -> only block 0.
	if len(selection[2]) != 1 || selection[2][0] != 0 {
		t.Fatalf("token 2 selection = %v, want [0]", selection[2])
	}
	// Token 4: allowed blocks 0..1, scores [7,7] tie -> smaller index first.
	if len(selection[4]) != 2 || selection[4][0] != 0 || selection[4][1] != 1 {
		t.Fatalf("token 4 selection = %v, want [0 1]", selection[4])
	}
}

func TestTopKOrdersAndBreaksTies(t *testing.T) {
	scores := tensors.NewWithData([]int{2, 5}, []float32{
		1, 9, 3, 9, 2,
		4, 5, 6, 7, 8,
	})
	selected := TopK(scores, 2)
	// Row 0: 9 at indices 1 and 3 (tie -> smaller index first), then 3.
	want := []int{1, 3}
	for index := range want {
		if selected[0][index] != want[index] {
			t.Fatalf("top-k row 0 = %v, want %v", selected[0], want)
		}
	}
	if len(selected[0]) != 2 {
		t.Fatalf("top-k length = %d, want 2", len(selected[0]))
	}
}

func TestTopKClampsToBlockCount(t *testing.T) {
	scores := tensors.NewWithData([]int{1, 3}, []float32{1, 2, 3})
	selected := TopK(scores, 10)
	if len(selected[0]) != 3 {
		t.Fatalf("top-k length = %d, want 3", len(selected[0]))
	}
	for index, block := range selected[0] {
		if block != 2-index {
			t.Fatalf("descending order broken at %d: %v", index, selected[0])
		}
	}
}
