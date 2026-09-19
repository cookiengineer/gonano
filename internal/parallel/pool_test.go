package parallel

import (
	"runtime"
	"sync/atomic"
	"testing"
	"time"
)

// peakConcurrency runs fn over count iterations through the pool and returns
// the maximum number of iterations observed executing at the same time. Each
// iteration sleeps briefly so overlapping workers are observable.
func peakConcurrency(pool *Pool, count int) int {
	var active, peak atomic.Int64
	pool.For(0, count, func(int) {
		current := active.Add(1)
		for {
			previous := peak.Load()
			if current <= previous || peak.CompareAndSwap(previous, current) {
				break
			}
		}
		time.Sleep(time.Millisecond)
		active.Add(-1)
	})
	return int(peak.Load())
}

// TestPoolForGateBelowMinChunkRunsInline pins the current behavior that an
// index range smaller than minChunk executes inline. Attention and matmul call
// For with ranges of this size, so this is why they are not parallel today.
func TestPoolForGateBelowMinChunkRunsInline(t *testing.T) {
	pool := NewPool(4)
	if peak := peakConcurrency(pool, 10); peak != 1 {
		t.Fatalf("small range peak concurrency = %d, want 1 (inline)", peak)
	}
}

// TestPoolForGateAtMinChunkParallelizes documents the inverse: once a range
// reaches minChunk, the pool actually fans out across workers.
func TestPoolForGateAtMinChunkParallelizes(t *testing.T) {
	if runtime.GOMAXPROCS(0) < 2 {
		t.Skip("needs at least two CPUs")
	}
	pool := NewPool(4)
	if peak := peakConcurrency(pool, 4*pool.minChunk); peak < 2 {
		t.Fatalf("large range peak concurrency = %d, want >= 2", peak)
	}
}

// TestPoolWithMinChunkParallelizesSmall documents the fix direction used by
// M1c: lowering minChunk makes small block-index loops run on multiple workers.
func TestPoolWithMinChunkParallelizesSmall(t *testing.T) {
	if runtime.GOMAXPROCS(0) < 2 {
		t.Skip("needs at least two CPUs")
	}
	pool := NewPool(4).WithMinChunk(1)
	if peak := peakConcurrency(pool, 8); peak < 2 {
		t.Fatalf("WithMinChunk(1) peak concurrency = %d, want >= 2", peak)
	}
}

func TestPoolForCoversAll(test *testing.T) {
	pool := NewPool(runtime.GOMAXPROCS(0))
	count := 1_000_000
	var seen atomic.Int64
	pool.For(0, count, func(index int) {
		seen.Add(int64(index))
	})
	want := int64(count) * int64(count-1) / 2
	if seen.Load() != want {
		test.Fatalf("For sum = %d, want %d", seen.Load(), want)
	}
}

func TestPoolForInlineSmall(test *testing.T) {
	pool := NewPool(1)
	count := 0
	pool.For(0, 10, func(index int) { count++ })
	if count != 10 {
		test.Fatalf("count = %d, want 10", count)
	}
}

func TestPoolChunksCoversAll(test *testing.T) {
	pool := NewPool(4)
	count := 100_000
	var sum atomic.Int64
	pool.Chunks(0, count, func(chunkStart, chunkEnd int) {
		var localSum int64
		for index := chunkStart; index < chunkEnd; index++ {
			localSum += int64(index)
		}
		sum.Add(localSum)
	})
	want := int64(count) * int64(count-1) / 2
	if sum.Load() != want {
		test.Fatalf("Chunks sum = %d, want %d", sum.Load(), want)
	}
}

func TestReduceSum(test *testing.T) {
	pool := NewPool(4)
	count := 1_000_000
	total := Reduce(pool, 0, count,
		func(chunkStart, chunkEnd int) int64 {
			var accumulator int64
			for index := chunkStart; index < chunkEnd; index++ {
				accumulator += int64(index)
			}
			return accumulator
		},
		func(left, right int64) int64 { return left + right },
	)
	want := int64(count) * int64(count-1) / 2
	if total != want {
		test.Fatalf("Reduce sum = %d, want %d", total, want)
	}
}

func TestReduceEmpty(test *testing.T) {
	pool := NewPool(4)
	total := Reduce(pool, 5, 5,
		func(chunkStart, chunkEnd int) int64 { return 42 },
		func(left, right int64) int64 { return left + right },
	)
	if total != 0 {
		test.Fatalf("empty reduce = %d, want 0", total)
	}
}

func TestWorkersDefault(test *testing.T) {
	if Default().Workers() != runtime.GOMAXPROCS(0) {
		test.Fatalf("Default workers = %d, want %d", Default().Workers(), runtime.GOMAXPROCS(0))
	}
}

func TestWorkersCustom(test *testing.T) {
	if got := NewPool(3).Workers(); got != 3 {
		test.Fatalf("Workers = %d, want 3", got)
	}
}
