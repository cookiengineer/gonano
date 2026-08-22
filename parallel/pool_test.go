package parallel

import (
	"runtime"
	"sync/atomic"
	"testing"
)

func TestPoolForCoversAll(t *testing.T) {
	p := NewPool(runtime.GOMAXPROCS(0))
	n := 1_000_000
	var seen atomic.Int64
	p.For(0, n, func(i int) {
		seen.Add(int64(i))
	})
	want := int64(n) * int64(n-1) / 2
	if seen.Load() != want {
		t.Fatalf("For sum = %d, want %d", seen.Load(), want)
	}
}

func TestPoolForInlineSmall(t *testing.T) {
	p := NewPool(1)
	count := 0
	p.For(0, 10, func(i int) { count++ })
	if count != 10 {
		t.Fatalf("count = %d, want 10", count)
	}
}

func TestPoolChunksCoversAll(t *testing.T) {
	p := NewPool(4)
	n := 100_000
	var sum atomic.Int64
	p.Chunks(0, n, func(s, e int) {
		var local int64
		for i := s; i < e; i++ {
			local += int64(i)
		}
		sum.Add(local)
	})
	want := int64(n) * int64(n-1) / 2
	if sum.Load() != want {
		t.Fatalf("Chunks sum = %d, want %d", sum.Load(), want)
	}
}

func TestReduceSum(t *testing.T) {
	p := NewPool(4)
	n := 1_000_000
	total := Reduce(p, 0, n,
		func(s, e int) int64 {
			var acc int64
			for i := s; i < e; i++ {
				acc += int64(i)
			}
			return acc
		},
		func(a, b int64) int64 { return a + b },
	)
	want := int64(n) * int64(n-1) / 2
	if total != want {
		t.Fatalf("Reduce sum = %d, want %d", total, want)
	}
}

func TestReduceEmpty(t *testing.T) {
	p := NewPool(4)
	total := Reduce(p, 5, 5,
		func(s, e int) int64 { return 42 },
		func(a, b int64) int64 { return a + b },
	)
	if total != 0 {
		t.Fatalf("empty reduce = %d, want 0", total)
	}
}

func TestWorkersDefault(t *testing.T) {
	if Default().Workers() != runtime.GOMAXPROCS(0) {
		t.Fatalf("Default workers = %d, want %d", Default().Workers(), runtime.GOMAXPROCS(0))
	}
}

func TestWorkersCustom(t *testing.T) {
	if got := NewPool(3).Workers(); got != 3 {
		t.Fatalf("Workers = %d, want 3", got)
	}
}
