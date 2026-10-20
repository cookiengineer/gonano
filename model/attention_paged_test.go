package model

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/internal/parallel"
)

func TestMergePagedResultsParallelMatchesSerial(t *testing.T) {
	previous := parallel.Default()
	parallel.SetDefault(parallel.NewPool(4))
	defer parallel.SetDefault(previous)

	dimension := 8
	for _, count := range []int{4, 5, 7, 11} {
		results := make([]pagedPageResult, count)
		for index := range results {
			if index == 2 || index == count-1 {
				results[index] = pagedPageResult{lse: float32(math.Inf(-1))}
				continue
			}
			output := make([]float32, dimension)
			for channel := range output {
				output[channel] = float32(index*10+channel) * 0.01
			}
			results[index] = pagedPageResult{output: output, lse: -float32(index+1) * 0.5}
		}
		valid := make([]pagedPageResult, 0, count)
		parallelResults := make([]pagedPageResult, count)
		for index := range results {
			if results[index].output != nil {
				parallelResults[index].output = append([]float32(nil), results[index].output...)
				valid = append(valid, pagedPageResult{
					output: append([]float32(nil), results[index].output...),
					lse:    results[index].lse,
				})
			}
			parallelResults[index].lse = results[index].lse
		}
		serialResults := valid
		serialOutput := make([]float32, dimension)
		parallelOutput := make([]float32, dimension)
		serialLogSumExp := make([]float32, 1)
		parallelLogSumExp := make([]float32, 1)
		mergePagedResultsSerial(serialResults, serialOutput, serialLogSumExp)
		mergePagedResults(parallelOutput, parallelLogSumExp, parallelResults, count)
		if math.Abs(float64(serialLogSumExp[0]-parallelLogSumExp[0])) > 1e-5 {
			t.Fatalf("count=%d: log-sum-exp = %v, want %v", count, parallelLogSumExp[0], serialLogSumExp[0])
		}
		for channel := range serialOutput {
			if math.Abs(float64(serialOutput[channel]-parallelOutput[channel])) > 1e-5 {
				t.Fatalf("count=%d channel=%d: output = %v, want %v", count, channel, parallelOutput[channel], serialOutput[channel])
			}
		}
	}
}
