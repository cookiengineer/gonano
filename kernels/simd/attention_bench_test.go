package simdbackend_test

import (
	"fmt"
	"testing"

	"github.com/cookiengineer/gonano/kernels"
	"github.com/cookiengineer/gonano/kernels/kerneltest"
	"github.com/cookiengineer/gonano/kernels/scalar"
	simdbackend "github.com/cookiengineer/gonano/kernels/simd"
)

// BenchmarkFlashAttentionForward measures the SIMD flash-attention forward
// against the scalar reference at typical prefill sequence lengths. The scalar
// backend materializes the score matrix; the SIMD backend streams it.
func BenchmarkFlashAttentionForward(b *testing.B) {
	for _, sequenceLength := range []int{128, 512, 1024} {
		for _, headDim := range []int{64, 128} {
			query := kerneltest.Data(sequenceLength * headDim)
			key := kerneltest.Data(sequenceLength * headDim)
			value := kerneltest.DataB(sequenceLength * headDim)
			parameters := kernels.AttentionForwardParameters{
				Query: query, Key: key, Value: value,
				QueryLength: sequenceLength, KeyLength: sequenceLength, HeadDim: headDim,
				PositionOffset: 0, Window: -1,
			}
			output := make([]float32, sequenceLength*headDim)
			logSumExp := make([]float32, sequenceLength)

			b.Run(fmt.Sprintf("simd/seq%d/head%d", sequenceLength, headDim), func(b *testing.B) {
				backend := simdbackend.New()
				b.ReportAllocs()
				for iteration := 0; iteration < b.N; iteration++ {
					backend.AttentionForward(parameters, kernels.AttentionForwardResult{Output: output, LogSumExp: logSumExp})
				}
			})
			b.Run(fmt.Sprintf("scalar/seq%d/head%d", sequenceLength, headDim), func(b *testing.B) {
				backend := scalar.New()
				b.ReportAllocs()
				for iteration := 0; iteration < b.N; iteration++ {
					backend.AttentionForward(parameters, kernels.AttentionForwardResult{Output: output, LogSumExp: logSumExp})
				}
			})
		}
	}
}
