package simdbackend_test

import (
	"fmt"
	"testing"

	"github.com/cookiengineer/gonano/kernels/kerneltest"
	"github.com/cookiengineer/gonano/kernels/scalar"
	simdbackend "github.com/cookiengineer/gonano/kernels/simd"
)

// BenchmarkTopKIndices measures the fused SIMD top-k selection against the
// scalar reference at expert counts typical of a MoE router.
func BenchmarkTopKIndices(b *testing.B) {
	rows := 256
	k := 6
	for _, experts := range []int{64, 256, 512} {
		scores := kerneltest.Data(rows * experts)
		destination := make([]int32, rows*k)
		b.Run(fmt.Sprintf("simd/experts%d", experts), func(b *testing.B) {
			backend := simdbackend.New()
			b.ReportAllocs()
			for iteration := 0; iteration < b.N; iteration++ {
				backend.TopKIndices(destination, scores, rows, experts, k)
			}
		})
		b.Run(fmt.Sprintf("scalar/experts%d", experts), func(b *testing.B) {
			backend := scalar.New()
			b.ReportAllocs()
			for iteration := 0; iteration < b.N; iteration++ {
				backend.TopKIndices(destination, scores, rows, experts, k)
			}
		})
	}
}

// BenchmarkMoEGateTopK measures the full router softmax plus selection.
func BenchmarkMoEGateTopK(b *testing.B) {
	rows := 256
	experts := 256
	k := 6
	logits := kerneltest.Data(rows * experts)
	bias := make([]float32, experts)
	probs := make([]float32, rows*experts)
	selected := make([]int32, rows*k)
	weights := make([]float32, rows*k)

	b.Run("simd", func(b *testing.B) {
		backend := simdbackend.New()
		b.ReportAllocs()
		for iteration := 0; iteration < b.N; iteration++ {
			backend.MoEGateTopK(logits, bias, probs, selected, weights, rows, experts, k)
		}
	})
	b.Run("scalar", func(b *testing.B) {
		backend := scalar.New()
		b.ReportAllocs()
		for iteration := 0; iteration < b.N; iteration++ {
			backend.MoEGateTopK(logits, bias, probs, selected, weights, rows, experts, k)
		}
	})
}
