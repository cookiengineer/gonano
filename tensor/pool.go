package tensor

import "github.com/cookiengineer/gonano/parallel"

// parallelChunks partitions [0, n) across the process pool, executing fn on
// each contiguous range. Small ranges fall back to inline execution inside the
// pool.
func parallelChunks(n int, fn func(s, e int)) {
	parallel.Default().Chunks(0, n, fn)
}

// parallelFor runs fn(i) for i in [0, n) across the process pool.
func parallelFor(n int, fn func(i int)) {
	parallel.Default().For(0, n, fn)
}
