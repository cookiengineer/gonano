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
func (pool *Pool) WithMinChunk(minChunk int) *Pool {
	workers := pool.workers
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	return &Pool{workers: workers, minChunk: minChunk}
}

// Workers reports the number of worker goroutines the pool uses.
func (pool *Pool) Workers() int {
	if pool == nil || pool.workers <= 0 {
		return runtime.GOMAXPROCS(0)
	}
	return pool.workers
}

// For invokes fn(index) for every index in [start, end), distributing the
// iterations across the pool's workers. It blocks until all iterations
// complete.
func (pool *Pool) For(start, end int, fn func(index int)) {
	if end <= start {
		return
	}
	if end-start < pool.minChunk {
		for index := start; index < end; index++ {
			fn(index)
		}
		return
	}
	workers := pool.Workers()
	if workers > end-start {
		workers = end - start
	}
	var waitGroup sync.WaitGroup
	waitGroup.Add(workers)
	for worker := 0; worker < workers; worker++ {
		// Static partition: contiguous ranges give better cache locality than
		// a shared atomic counter.
		lo := start + (end-start)*worker/workers
		hi := start + (end-start)*(worker+1)/workers
		go func(lo, hi int) {
			defer waitGroup.Done()
			for index := lo; index < hi; index++ {
				fn(index)
			}
		}(lo, hi)
	}
	waitGroup.Wait()
}

// Chunks partitions [start, end) into at most workers contiguous ranges and
// invokes fn(chunkStart, chunkEnd) for each. This is the primitive used by
// matmul and elementwise kernels, which hand each worker a contiguous slice of
// work.
func (pool *Pool) Chunks(start, end int, fn func(chunkStart, chunkEnd int)) {
	if end <= start {
		return
	}
	if end-start < pool.minChunk {
		fn(start, end)
		return
	}
	workers := pool.Workers()
	if workers > end-start {
		workers = end - start
	}
	var waitGroup sync.WaitGroup
	waitGroup.Add(workers)
	for worker := 0; worker < workers; worker++ {
		lo := start + (end-start)*worker/workers
		hi := start + (end-start)*(worker+1)/workers
		go func(lo, hi int) {
			defer waitGroup.Done()
			fn(lo, hi)
		}(lo, hi)
	}
	waitGroup.Wait()
}

// Reduce splits [start, end) into contiguous chunks, reduces each chunk with
// reduce, and folds the partial results with combine (invoked sequentially).
// combine must be associative and commutative. It blocks until complete.
func Reduce[Result any](pool *Pool, start, end int, reduce func(chunkStart, chunkEnd int) Result, combine func(left, right Result) Result) Result {
	if end <= start {
		var zero Result
		return zero
	}
	if end-start < pool.minChunk {
		return reduce(start, end)
	}
	workers := pool.Workers()
	if workers > end-start {
		workers = end - start
	}
	partials := make([]Result, workers)
	var waitGroup sync.WaitGroup
	waitGroup.Add(workers)
	for worker := 0; worker < workers; worker++ {
		lo := start + (end-start)*worker/workers
		hi := start + (end-start)*(worker+1)/workers
		slot := worker
		go func(lo, hi, slot int) {
			defer waitGroup.Done()
			partials[slot] = reduce(lo, hi)
		}(lo, hi, slot)
	}
	waitGroup.Wait()
	total := partials[0]
	for index := 1; index < workers; index++ {
		total = combine(total, partials[index])
	}
	return total
}

var defaultPool = NewPool(0)

// Default returns a process-wide pool sized to GOMAXPROCS.
func Default() *Pool { return defaultPool }

// KernelPool returns the default pool with a small parallelization threshold.
// It is meant for kernels whose loop indices are substantial work items (matmul
// row/column blocks, attention (batch, head) pairs) rather than individual
// elements, so short index ranges should still fan out across cores. The
// default pool's element-count heuristic would run these inline.
func KernelPool() *Pool { return defaultPool.WithMinChunk(1) }

// SetDefault replaces the process-wide pool (used by tests and by commands
// that want a custom worker count).
func SetDefault(pool *Pool) { defaultPool = pool }
