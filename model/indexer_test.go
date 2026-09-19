package model

import (
	"math"
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
