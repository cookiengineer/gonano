package simdbackend_test

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/kernels"
	"github.com/cookiengineer/gonano/kernels/kerneltest"
	"github.com/cookiengineer/gonano/kernels/scalar"
	simdbackend "github.com/cookiengineer/gonano/kernels/simd"
)

// TestElementwiseParity verifies every element-wise method against the scalar
// reference across lengths that exercise aligned vectors and scalar tails.
func TestElementwiseParity(t *testing.T) {
	simdBackend := simdbackend.New()
	scalarBackend := scalar.New()

	binaryCases := []struct {
		name     string
		simdOp   func(dst, left, right []float32)
		scalarOp func(dst, left, right []float32)
	}{
		{"Add", simdBackend.Add, scalarBackend.Add},
		{"Subtract", simdBackend.Subtract, scalarBackend.Subtract},
		{"Multiply", simdBackend.Multiply, scalarBackend.Multiply},
		{"Divide", simdBackend.Divide, scalarBackend.Divide},
	}
	for _, testCase := range binaryCases {
		for _, length := range []int{1, 7, 31, 64, 513, 1031} {
			left := kerneltest.Data(length)
			right := kerneltest.DataB(length)
			got := make([]float32, length)
			want := make([]float32, length)
			testCase.simdOp(got, left, right)
			testCase.scalarOp(want, left, right)
			kerneltest.AssertSlicesClose(t, got, want, 1e-5, 1e-6)
		}
	}

	unaryCases := []struct {
		name     string
		simdOp   func(dst, source []float32)
		scalarOp func(dst, source []float32)
	}{
		{"Negate", simdBackend.Negate, scalarBackend.Negate},
		{"Abs", simdBackend.Abs, scalarBackend.Abs},
		{"Square", simdBackend.Square, scalarBackend.Square},
		{"ReluSquared", simdBackend.ReluSquared, scalarBackend.ReluSquared},
		{"Exp", simdBackend.Exp, scalarBackend.Exp},
		{"Sigmoid", simdBackend.Sigmoid, scalarBackend.Sigmoid},
		{"Tanh", simdBackend.Tanh, scalarBackend.Tanh},
	}
	for _, testCase := range unaryCases {
		for _, length := range []int{1, 7, 31, 64, 513, 1031} {
			source := kerneltest.Data(length)
			got := make([]float32, length)
			want := make([]float32, length)
			testCase.simdOp(got, source)
			testCase.scalarOp(want, source)
			kerneltest.AssertSlicesClose(t, got, want, 1e-5, 1e-6)
		}
	}

	// SwiGLU (binary + clamp) and its backward.
	for _, length := range []int{1, 7, 31, 64, 513, 1031} {
		gate := kerneltest.Data(length)
		up := kerneltest.DataB(length)
		outputGradient := kerneltest.DataB(length)
		for _, clamp := range []float32{0, 2} {
			got := make([]float32, length)
			want := make([]float32, length)
			simdBackend.SwiGLU(got, gate, up, clamp)
			scalarBackend.SwiGLU(want, gate, up, clamp)
			kerneltest.AssertSlicesClose(t, got, want, 1e-5, 1e-6)

			gotGate := make([]float32, length)
			gotUp := make([]float32, length)
			wantGate := make([]float32, length)
			wantUp := make([]float32, length)
			simdBackend.SwiGLUBackward(gotGate, gotUp, gate, up, outputGradient, clamp)
			scalarBackend.SwiGLUBackward(wantGate, wantUp, gate, up, outputGradient, clamp)
			kerneltest.AssertSlicesClose(t, gotGate, wantGate, 1e-4, 1e-5)
			kerneltest.AssertSlicesClose(t, gotUp, wantUp, 1e-4, 1e-5)
		}
	}

	// Scale, AddScaled, and Rsqrt need nonzero sources.
	for _, length := range []int{1, 7, 31, 64, 513, 1031} {
		source := kerneltest.DataB(length)
		scaled := kerneltest.Data(length)

		gotScale := make([]float32, length)
		wantScale := make([]float32, length)
		simdBackend.Scale(gotScale, source, 2.5)
		scalarBackend.Scale(wantScale, source, 2.5)
		kerneltest.AssertSlicesClose(t, gotScale, wantScale, 1e-5, 1e-6)

		gotAddScaled := make([]float32, length)
		wantAddScaled := make([]float32, length)
		simdBackend.AddScaled(gotAddScaled, source, scaled, 1.5)
		scalarBackend.AddScaled(wantAddScaled, source, scaled, 1.5)
		kerneltest.AssertSlicesClose(t, gotAddScaled, wantAddScaled, 1e-5, 1e-6)

		gotRsqrt := make([]float32, length)
		wantRsqrt := make([]float32, length)
		simdBackend.Rsqrt(gotRsqrt, source)
		scalarBackend.Rsqrt(wantRsqrt, source)
		kerneltest.AssertSlicesClose(t, gotRsqrt, wantRsqrt, 1e-5, 1e-6)
	}
}

// TestReductionParity verifies Sum, Max, and ArgMax against the scalar
// reference.
func TestReductionParity(t *testing.T) {
	simdBackend := simdbackend.New()
	scalarBackend := scalar.New()
	for _, length := range []int{1, 7, 31, 64, 513, 1031} {
		values := kerneltest.Data(length)
		if !kerneltest.Close(simdBackend.Sum(values), scalarBackend.Sum(values), 1e-4, 1e-4) {
			t.Fatalf("Sum mismatch at length %d: got %v, want %v", length, simdBackend.Sum(values), scalarBackend.Sum(values))
		}
		if simdBackend.Max(values) != scalarBackend.Max(values) {
			t.Fatalf("Max mismatch at length %d: got %v, want %v", length, simdBackend.Max(values), scalarBackend.Max(values))
		}
		if simdBackend.ArgMax(values) != scalarBackend.ArgMax(values) {
			t.Fatalf("ArgMax mismatch at length %d: got %d, want %d", length, simdBackend.ArgMax(values), scalarBackend.ArgMax(values))
		}
	}
}

// TestLinearAlgebraParity verifies MatMul, MatMulTransposed, and DotProduct
// against the scalar reference, including non-vector-multiple dimensions.
func TestLinearAlgebraParity(t *testing.T) {
	simdBackend := simdbackend.New()
	scalarBackend := scalar.New()

	shapes := []struct{ rows, columns, inner int }{
		{3, 5, 17},
		{4, 6, 13},
		{64, 129, 200},
		{97, 131, 256},
		{400, 128, 512},
	}
	for _, shape := range shapes {
		left := kerneltest.Data(shape.rows * shape.inner)
		right := kerneltest.DataB(shape.inner * shape.columns)

		gotProduct := make([]float32, shape.rows*shape.columns)
		wantProduct := make([]float32, shape.rows*shape.columns)
		simdBackend.MatMul(gotProduct, left, right, shape.rows, shape.columns, shape.inner)
		scalarBackend.MatMul(wantProduct, left, right, shape.rows, shape.columns, shape.inner)
		kerneltest.AssertSlicesClose(t, gotProduct, wantProduct, 2e-3, 1e-4)

		transposed := kerneltest.DataB(shape.columns * shape.inner)
		gotTransposed := make([]float32, shape.rows*shape.columns)
		wantTransposed := make([]float32, shape.rows*shape.columns)
		simdBackend.MatMulTransposed(gotTransposed, left, transposed, shape.rows, shape.columns, shape.inner)
		scalarBackend.MatMulTransposed(wantTransposed, left, transposed, shape.rows, shape.columns, shape.inner)
		kerneltest.AssertSlicesClose(t, gotTransposed, wantTransposed, 2e-3, 1e-4)
	}

	left := kerneltest.Data(257)
	right := kerneltest.DataB(257)
	gotDot := simdBackend.DotProduct(left, right)
	wantDot := scalarBackend.DotProduct(left, right)
	if !kerneltest.Close(gotDot, wantDot, 1e-4, 1e-4) {
		t.Fatalf("DotProduct mismatch: got %v, want %v", gotDot, wantDot)
	}
}

// TestRowParity verifies the fused SoftmaxLastDim and RMSNormLastDim against
// the scalar reference.
func TestRowParity(t *testing.T) {
	simdBackend := simdbackend.New()
	scalarBackend := scalar.New()
	cases := []struct{ rows, columns int }{
		{1, 3},
		{2, 4},
		{7, 13},
		{16, 64},
		{5, 129},
	}
	for _, testCase := range cases {
		source := kerneltest.Data(testCase.rows * testCase.columns)

		gotSoftmax := make([]float32, len(source))
		wantSoftmax := make([]float32, len(source))
		simdBackend.SoftmaxLastDim(gotSoftmax, source, testCase.rows, testCase.columns)
		scalarBackend.SoftmaxLastDim(wantSoftmax, source, testCase.rows, testCase.columns)
		kerneltest.AssertSlicesClose(t, gotSoftmax, wantSoftmax, 1e-4, 1e-5)

		gotNorm := make([]float32, len(source))
		wantNorm := make([]float32, len(source))
		simdBackend.RMSNormLastDim(gotNorm, source, testCase.rows, testCase.columns, 1e-6)
		scalarBackend.RMSNormLastDim(wantNorm, source, testCase.rows, testCase.columns, 1e-6)
		kerneltest.AssertSlicesClose(t, gotNorm, wantNorm, 1e-4, 1e-5)
	}
}

// TestAttentionParityLarge exercises multiple query and key blocks at once, so
// the online-softmax rescaling path is covered at block boundaries.
func TestAttentionParityLarge(t *testing.T) {
	simdBackend := simdbackend.New()
	scalarBackend := scalar.New()
	queryLength, keyLength, headDim := 200, 200, 32
	query := kerneltest.Data(queryLength * headDim)
	key := kerneltest.Data(keyLength * headDim)
	value := kerneltest.DataB(keyLength * headDim)
	parameters := kernels.AttentionForwardParameters{
		Query: query, Key: key, Value: value,
		QueryLength: queryLength, KeyLength: keyLength, HeadDim: headDim,
		PositionOffset: 0, Window: -1,
	}
	gotOutput := make([]float32, queryLength*headDim)
	gotLogSumExp := make([]float32, queryLength)
	wantOutput := make([]float32, queryLength*headDim)
	wantLogSumExp := make([]float32, queryLength)
	simdBackend.AttentionForward(parameters, kernels.AttentionForwardResult{Output: gotOutput, LogSumExp: gotLogSumExp})
	scalarBackend.AttentionForward(parameters, kernels.AttentionForwardResult{Output: wantOutput, LogSumExp: wantLogSumExp})
	kerneltest.AssertSlicesClose(t, gotOutput, wantOutput, 5e-3, 1e-5)
	kerneltest.AssertSlicesClose(t, gotLogSumExp, wantLogSumExp, 5e-3, 1e-5)
}

// splitAttention runs split-K forward + combine through a backend.
func splitAttention(backend kernels.Backend, parameters kernels.AttentionForwardParameters, splits int) ([]float32, []float32) {
	output := make([]float32, parameters.QueryLength*parameters.HeadDim)
	logSumExp := make([]float32, parameters.QueryLength)
	partials := make([]kernels.AttentionSplitResult, splits)
	keysPerSplit := (parameters.KeyLength + splits - 1) / splits
	for split := 0; split < splits; split++ {
		keyStart := split * keysPerSplit
		keyEnd := min(keyStart+keysPerSplit, parameters.KeyLength)
		partial := kernels.AttentionSplitResult{
			Accumulator: make([]float32, parameters.QueryLength*parameters.HeadDim),
			Maximum:     make([]float32, parameters.QueryLength),
			Sum:         make([]float32, parameters.QueryLength),
		}
		backend.AttentionForwardSplit(kernels.AttentionSplitParameters{
			Query:          parameters.Query,
			Key:            parameters.Key,
			Value:          parameters.Value,
			QueryLength:    parameters.QueryLength,
			KeyLength:      parameters.KeyLength,
			HeadDim:        parameters.HeadDim,
			PositionOffset: parameters.PositionOffset,
			Window:         parameters.Window,
			KeyStart:       keyStart,
			KeyEnd:         keyEnd,
		}, partial)
		partials[split] = partial
	}
	backend.AttentionCombine(
		kernels.AttentionCombineParameters{Partials: partials, QueryLength: parameters.QueryLength, HeadDim: parameters.HeadDim},
		kernels.AttentionForwardResult{Output: output, LogSumExp: logSumExp},
	)
	return output, logSumExp
}

// TestAttentionSplitParity verifies that split-K shards recombine to the same
// result as the single-pass flash attention, for both the SIMD and scalar
// backends, across causal, decode, and sliding-window shapes.
func TestAttentionSplitParity(t *testing.T) {
	simdBackend := simdbackend.New()
	scalarBackend := scalar.New()
	cases := []struct {
		name           string
		queryLength    int
		keyLength      int
		headDim        int
		positionOffset int
		window         int
	}{
		{"causal", 6, 6, 8, 0, -1},
		{"decode", 1, 10, 8, 9, -1},
		{"sliding", 6, 6, 8, 0, 2},
		{"sliding-decode", 1, 10, 4, 9, 3},
		{"wide-head", 5, 5, 16, 0, -1},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			query := kerneltest.Data(testCase.queryLength * testCase.headDim)
			key := kerneltest.Data(testCase.keyLength * testCase.headDim)
			value := kerneltest.DataB(testCase.keyLength * testCase.headDim)
			parameters := kernels.AttentionForwardParameters{
				Query:          query,
				Key:            key,
				Value:          value,
				QueryLength:    testCase.queryLength,
				KeyLength:      testCase.keyLength,
				HeadDim:        testCase.headDim,
				PositionOffset: testCase.positionOffset,
				Window:         testCase.window,
			}
			reference := make([]float32, testCase.queryLength*testCase.headDim)
			referenceLogSumExp := make([]float32, testCase.queryLength)
			scalarBackend.AttentionForward(parameters, kernels.AttentionForwardResult{Output: reference, LogSumExp: referenceLogSumExp})

			for _, splits := range []int{1, 2, 3, 5, testCase.keyLength} {
				simdOutput, simdLogSumExp := splitAttention(simdBackend, parameters, splits)
				kerneltest.AssertSlicesClose(t, simdOutput, reference, 5e-3, 1e-5)
				kerneltest.AssertSlicesClose(t, simdLogSumExp, referenceLogSumExp, 5e-3, 1e-5)

				scalarOutput, scalarLogSumExp := splitAttention(scalarBackend, parameters, splits)
				kerneltest.AssertSlicesClose(t, scalarOutput, reference, 2e-4, 1e-5)
				kerneltest.AssertSlicesClose(t, scalarLogSumExp, referenceLogSumExp, 2e-4, 1e-5)
			}
		})
	}
}

// TestAttentionBackwardCorrectionParity verifies that the optional row
// correction used to merge several attention branches is implemented
// identically by the SIMD and scalar backends.
func TestAttentionBackwardCorrectionParity(t *testing.T) {
	simdBackend := simdbackend.New()
	scalarBackend := scalar.New()
	queryLength, keyLength, headDim := 7, 9, 8
	query := kerneltest.Data(queryLength * headDim)
	key := kerneltest.Data(keyLength * headDim)
	value := kerneltest.DataB(keyLength * headDim)
	outputGradient := kerneltest.Data(queryLength * headDim)
	logSumExp := make([]float32, queryLength)
	correction := kerneltest.Data(queryLength)
	for index := range logSumExp {
		logSumExp[index] = -float32(index) - 0.5
	}

	parameters := kernels.AttentionBackwardParameters{
		Query: query, Key: key, Value: value, OutputGradient: outputGradient,
		LogSumExp: logSumExp, RowCorrection: correction,
		QueryLength: queryLength, KeyLength: keyLength, HeadDim: headDim,
		PositionOffset: queryLength + keyLength, Window: -1,
	}
	gotQuery := make([]float32, queryLength*headDim)
	gotKey := make([]float32, keyLength*headDim)
	gotValue := make([]float32, keyLength*headDim)
	wantQuery := make([]float32, queryLength*headDim)
	wantKey := make([]float32, keyLength*headDim)
	wantValue := make([]float32, keyLength*headDim)
	simdBackend.AttentionBackward(parameters, kernels.AttentionBackwardResult{QueryGradient: gotQuery, KeyGradient: gotKey, ValueGradient: gotValue})
	scalarBackend.AttentionBackward(parameters, kernels.AttentionBackwardResult{QueryGradient: wantQuery, KeyGradient: wantKey, ValueGradient: wantValue})
	kerneltest.AssertSlicesClose(t, gotQuery, wantQuery, 5e-3, 1e-5)
	kerneltest.AssertSlicesClose(t, gotKey, wantKey, 5e-3, 1e-5)
	kerneltest.AssertSlicesClose(t, gotValue, wantValue, 5e-3, 1e-5)
}

// TestAttentionParity verifies the attention forward and backward against the
// scalar reference across decode, sliding-window, and full-causal shapes.
func TestAttentionParity(t *testing.T) {
	simdBackend := simdbackend.New()
	scalarBackend := scalar.New()
	cases := []struct {
		name           string
		queryLength    int
		keyLength      int
		headDim        int
		positionOffset int
		window         int
	}{
		{"causal", 6, 6, 8, 0, -1},
		{"decode", 1, 10, 8, 9, -1},
		{"sliding", 6, 6, 8, 0, 2},
		{"sliding-decode", 1, 10, 4, 9, 3},
		{"wide-head", 5, 5, 16, 0, -1},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			query := kerneltest.Data(testCase.queryLength * testCase.headDim)
			key := kerneltest.Data(testCase.keyLength * testCase.headDim)
			value := kerneltest.DataB(testCase.keyLength * testCase.headDim)

			parameters := kernels.AttentionForwardParameters{
				Query:          query,
				Key:            key,
				Value:          value,
				QueryLength:    testCase.queryLength,
				KeyLength:      testCase.keyLength,
				HeadDim:        testCase.headDim,
				PositionOffset: testCase.positionOffset,
				Window:         testCase.window,
			}

			gotOutput := make([]float32, testCase.queryLength*testCase.headDim)
			gotLogSumExp := make([]float32, testCase.queryLength)
			wantOutput := make([]float32, testCase.queryLength*testCase.headDim)
			wantLogSumExp := make([]float32, testCase.queryLength)
			simdBackend.AttentionForward(parameters, kernels.AttentionForwardResult{Output: gotOutput, LogSumExp: gotLogSumExp})
			scalarBackend.AttentionForward(parameters, kernels.AttentionForwardResult{Output: wantOutput, LogSumExp: wantLogSumExp})
			kerneltest.AssertSlicesClose(t, gotOutput, wantOutput, 2e-3, 1e-5)
			kerneltest.AssertSlicesClose(t, gotLogSumExp, wantLogSumExp, 2e-3, 1e-5)

			outputGradient := kerneltest.Data(testCase.queryLength * testCase.headDim)
			gotQueryGradient := make([]float32, testCase.queryLength*testCase.headDim)
			gotKeyGradient := make([]float32, testCase.keyLength*testCase.headDim)
			gotValueGradient := make([]float32, testCase.keyLength*testCase.headDim)
			wantQueryGradient := make([]float32, testCase.queryLength*testCase.headDim)
			wantKeyGradient := make([]float32, testCase.keyLength*testCase.headDim)
			wantValueGradient := make([]float32, testCase.keyLength*testCase.headDim)

			backwardParameters := kernels.AttentionBackwardParameters{
				Query:          query,
				Key:            key,
				Value:          value,
				Output:         wantOutput,
				OutputGradient: outputGradient,
				LogSumExp:      wantLogSumExp,
				QueryLength:    testCase.queryLength,
				KeyLength:      testCase.keyLength,
				HeadDim:        testCase.headDim,
				PositionOffset: testCase.positionOffset,
				Window:         testCase.window,
			}
			simdBackend.AttentionBackward(backwardParameters, kernels.AttentionBackwardResult{
				QueryGradient: gotQueryGradient,
				KeyGradient:   gotKeyGradient,
				ValueGradient: gotValueGradient,
			})
			scalarBackend.AttentionBackward(backwardParameters, kernels.AttentionBackwardResult{
				QueryGradient: wantQueryGradient,
				KeyGradient:   wantKeyGradient,
				ValueGradient: wantValueGradient,
			})
			kerneltest.AssertSlicesClose(t, gotQueryGradient, wantQueryGradient, 5e-3, 1e-5)
			kerneltest.AssertSlicesClose(t, gotKeyGradient, wantKeyGradient, 5e-3, 1e-5)
			kerneltest.AssertSlicesClose(t, gotValueGradient, wantValueGradient, 5e-3, 1e-5)
		})
	}
}

// TestIndexerScoresParity verifies the lightning-indexer scoring kernel against
// the scalar reference across non-vector-multiple shapes.
func TestIndexerScoresParity(t *testing.T) {
	simdBackend := simdbackend.New()
	scalarBackend := scalar.New()

	shapes := []struct{ rows, blocks, heads, dim int }{
		{1, 1, 1, 1},
		{3, 7, 2, 5},
		{8, 33, 4, 13},
		{17, 129, 3, 64},
		{40, 200, 5, 128},
	}
	for _, shape := range shapes {
		query := kerneltest.Data(shape.rows * shape.heads * shape.dim)
		key := kerneltest.DataB(shape.blocks * shape.heads * shape.dim)
		weight := kerneltest.Data(shape.heads)
		got := make([]float32, shape.rows*shape.blocks)
		want := make([]float32, shape.rows*shape.blocks)
		simdBackend.IndexerScores(got, query, key, weight, shape.rows, shape.blocks, shape.heads, shape.dim)
		scalarBackend.IndexerScores(want, query, key, weight, shape.rows, shape.blocks, shape.heads, shape.dim)
		kerneltest.AssertSlicesClose(t, got, want, 2e-3, 1e-5)
	}
}

// TestIndexerBlockMaxParity verifies the indexer's block-max reduction kernel
// against the scalar reference, including a partial trailing group.
func TestIndexerBlockMaxParity(t *testing.T) {
	simdBackend := simdbackend.New()
	scalarBackend := scalar.New()

	cases := []struct{ rows, blocks, groupSize int }{
		{1, 1, 1},
		{3, 5, 2},
		{8, 64, 8},
		{17, 129, 7},
		{40, 200, 16},
	}
	for _, testCase := range cases {
		groups := (testCase.blocks + testCase.groupSize - 1) / testCase.groupSize
		source := kerneltest.Data(testCase.rows * testCase.blocks)
		got := make([]float32, testCase.rows*groups)
		want := make([]float32, testCase.rows*groups)
		simdBackend.IndexerBlockMax(got, source, testCase.rows, testCase.blocks, testCase.groupSize)
		scalarBackend.IndexerBlockMax(want, source, testCase.rows, testCase.blocks, testCase.groupSize)
		kerneltest.AssertSlicesClose(t, got, want, 1e-5, 1e-6)
	}
}

// TestMoEGateTopKParity verifies the fused router softmax + top-k kernel
// against the scalar reference, including the exact selection order.
func TestMoEGateTopKParity(t *testing.T) {
	simdBackend := simdbackend.New()
	scalarBackend := scalar.New()

	cases := []struct{ rows, experts, k int }{
		{1, 2, 1},
		{3, 4, 2},
		{5, 8, 8},
		{7, 33, 3},
		{16, 64, 6},
	}
	for _, testCase := range cases {
		logits := make([]float32, testCase.rows*testCase.experts)
		for index := range logits {
			logits[index] = float32(math.Sin(float64(index)*1.3)) * 2
		}
		bias := make([]float32, testCase.experts)
		for index := range bias {
			bias[index] = float32(index%5)*0.1 - 0.2
		}

		gotProbs := make([]float32, len(logits))
		wantProbs := make([]float32, len(logits))
		gotSelected := make([]int32, testCase.rows*testCase.k)
		wantSelected := make([]int32, testCase.rows*testCase.k)
		gotWeights := make([]float32, testCase.rows*testCase.k)
		wantWeights := make([]float32, testCase.rows*testCase.k)

		simdBackend.MoEGateTopK(logits, bias, gotProbs, gotSelected, gotWeights, testCase.rows, testCase.experts, testCase.k)
		scalarBackend.MoEGateTopK(logits, bias, wantProbs, wantSelected, wantWeights, testCase.rows, testCase.experts, testCase.k)

		kerneltest.AssertSlicesClose(t, gotProbs, wantProbs, 1e-5, 1e-6)
		kerneltest.AssertSlicesClose(t, gotWeights, wantWeights, 1e-5, 1e-6)
		for index := range gotSelected {
			if gotSelected[index] != wantSelected[index] {
				t.Fatalf("selection mismatch at %d: got %d, want %d (rows=%d experts=%d k=%d)",
					index, gotSelected[index], wantSelected[index], testCase.rows, testCase.experts, testCase.k)
			}
		}
	}
}

// TestGroupedMatMulTransposedParity verifies the per-expert grouped GEMM
// against the scalar reference, including empty experts and uneven counts.
func TestGroupedMatMulTransposedParity(t *testing.T) {
	simdBackend := simdbackend.New()
	scalarBackend := scalar.New()

	cases := []struct {
		counts  []int32
		columns int
		inner   int
	}{
		{[]int32{3, 0, 2, 5}, 6, 13},
		{[]int32{1}, 4, 1},
		{[]int32{0, 4, 0, 1}, 7, 64},
		{[]int32{5, 5, 5}, 33, 17},
	}
	for _, testCase := range cases {
		experts := len(testCase.counts)
		totalRows := 0
		for _, count := range testCase.counts {
			totalRows += int(count)
		}
		input := kerneltest.Data(totalRows * testCase.inner)
		weight := kerneltest.DataB(experts * testCase.columns * testCase.inner)

		got := make([]float32, totalRows*testCase.columns)
		want := make([]float32, totalRows*testCase.columns)
		simdBackend.GroupedMatMulTransposed(got, input, weight, testCase.counts, experts, testCase.columns, testCase.inner)
		scalarBackend.GroupedMatMulTransposed(want, input, weight, testCase.counts, experts, testCase.columns, testCase.inner)
		kerneltest.AssertSlicesClose(t, got, want, 2e-3, 1e-4)
	}
}

// TestTopKIndicesParity verifies the vectorized top-k selection against the
// scalar reference, including tied scores and k equal to the row width, so the
// exact tie-break must agree.
func TestTopKIndicesParity(t *testing.T) {
	simdBackend := simdbackend.New()
	scalarBackend := scalar.New()

	cases := []struct{ rows, count, k int }{
		{1, 1, 1},
		{3, 4, 2},
		{4, 8, 8},
		{7, 33, 3},
		{16, 64, 6},
	}
	for _, testCase := range cases {
		// Scores deliberately repeat across experts so ties exercise the
		// smaller-index tie-break, and repeated across rows so lane handling is
		// covered at several offsets.
		scores := make([]float32, testCase.rows*testCase.count)
		for index := range scores {
			scores[index] = float32(index%5) * 0.25
		}
		got := make([]int32, testCase.rows*testCase.k)
		want := make([]int32, testCase.rows*testCase.k)
		simdBackend.TopKIndices(got, scores, testCase.rows, testCase.count, testCase.k)
		scalarBackend.TopKIndices(want, scores, testCase.rows, testCase.count, testCase.k)
		for index := range want {
			if got[index] != want[index] {
				t.Fatalf("top-k mismatch at %d: got %d, want %d (rows=%d count=%d k=%d)",
					index, got[index], want[index], testCase.rows, testCase.count, testCase.k)
			}
		}
	}
}
