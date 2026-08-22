// Package parallel provides a shared goroutine worker abstraction used
// throughout gonano to parallelize tensor operations, reductions, and data
// loading across all available CPUs.
//
// The model is "goroutine-per-op data parallel": a single process owns the
// model and fans out individual operations (matmul tiles, reductions,
// tokenization) across a fixed-size pool of worker goroutines.
package parallel

import (
	"runtime"
	"sync"
)

// Pool executes work items in parallel across a fixed number of workers.
// It is safe for concurrent use. The zero value runs everything inline.
type Pool struct {
	workers int
	// minChunk is the smallest range size that is parallelized; smaller
	// ranges run inline to avoid goroutine overhead.
	minChunk int
}

// NewPool returns a Pool with the given number of workers. A non-positive
// workers value defaults to runtime.GOMAXPROCS(0).
func NewPool(workers int) *Pool {
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	return &Pool{workers: workers, minChunk: 4096}
}

// WithMinChunk returns a copy of the pool with a custom parallelization
// threshold (ranges smaller than minChunk run inline).
func (p *Pool) WithMinChunk(minChunk int) *Pool {
	workers := p.workers
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	return &Pool{workers: workers, minChunk: minChunk}
}

// Workers reports the number of worker goroutines the pool uses.
func (p *Pool) Workers() int {
	if p == nil || p.workers <= 0 {
		return runtime.GOMAXPROCS(0)
	}
	return p.workers
}

// For invokes fn(i) for every i in [start, end), distributing the iterations
// across the pool's workers. It blocks until all iterations complete.
func (p *Pool) For(start, end int, fn func(i int)) {
	if end <= start {
		return
	}
	if end-start < p.minChunk {
		for i := start; i < end; i++ {
			fn(i)
		}
		return
	}
	workers := p.Workers()
	if workers > end-start {
		workers = end - start
	}
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		// Static partition: contiguous ranges give better cache locality than
		// a shared atomic counter.
		lo := start + (end-start)*w/workers
		hi := start + (end-start)*(w+1)/workers
		go func(lo, hi int) {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				fn(i)
			}
		}(lo, hi)
	}
	wg.Wait()
}

// Chunks partitions [start, end) into at most workers contiguous ranges and
// invokes fn(start, end) for each. This is the primitive used by matmul and
// elementwise kernels, which hand each worker a contiguous slice of work.
func (p *Pool) Chunks(start, end int, fn func(s, e int)) {
	if end <= start {
		return
	}
	if end-start < p.minChunk {
		fn(start, end)
		return
	}
	workers := p.Workers()
	if workers > end-start {
		workers = end - start
	}
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		lo := start + (end-start)*w/workers
		hi := start + (end-start)*(w+1)/workers
		go func(lo, hi int) {
			defer wg.Done()
			fn(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

// Reduce splits [start, end) into contiguous chunks, reduces each chunk with
// reduce, and folds the partial results with combine (invoked sequentially).
// combine must be associative and commutative. It blocks until complete.
func Reduce[T any](p *Pool, start, end int, reduce func(s, e int) T, combine func(a, b T) T) T {
	if end <= start {
		var zero T
		return zero
	}
	if end-start < p.minChunk {
		return reduce(start, end)
	}
	workers := p.Workers()
	if workers > end-start {
		workers = end - start
	}
	partials := make([]T, workers)
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		lo := start + (end-start)*w/workers
		hi := start + (end-start)*(w+1)/workers
		idx := w
		go func(lo, hi, idx int) {
			defer wg.Done()
			partials[idx] = reduce(lo, hi)
		}(lo, hi, idx)
	}
	wg.Wait()
	total := partials[0]
	for i := 1; i < workers; i++ {
		total = combine(total, partials[i])
	}
	return total
}

var defaultPool = NewPool(0)

// Default returns a process-wide pool sized to GOMAXPROCS.
func Default() *Pool { return defaultPool }

// SetDefault replaces the process-wide pool (used by tests and by commands
// that want a custom worker count).
func SetDefault(p *Pool) { defaultPool = p }
