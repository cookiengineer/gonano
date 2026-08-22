// gonano: a pure-Go reimplementation of the nanochat LLM training/inference
// harness, built on the experimental Go 1.27 `simd` standard-library package.
//
// Build requirement: the `simd` package is gated behind the goexperiment.simd
// build tag. Every build, test, and run must set:
//
//	GOEXPERIMENT=simd go build ./...
//	GOEXPERIMENT=simd go test ./...
//
// Numeric precision is float32 everywhere; parallelism is goroutine-per-op via
// the `parallel` package.
module github.com/cookiengineer/gonano

go 1.27.0
