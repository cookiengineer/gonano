package infer

import (
	"time"
)

// Measurement is the result of one timed generation run.
type Measurement struct {
	// TTFT is the time to the first token (prompt prefill + KV replication +
	// sampling the first token).
	TTFT time.Duration
	// StepTimes are the per-decode-step durations (one per token after the
	// first).
	StepTimes []time.Duration
	// NumTokens is the total number of tokens produced.
	NumTokens int
}

// Measure runs one timed generation and reports TTFT and per-step timings.
// The first step includes the batch-1 prefill; subsequent steps are decode.
func Measure(engine *Engine, tokens []int, numSamples, decodeTokens int, temperature float32, topK int, seed uint64) Measurement {
	gen := engine.Generate(tokens, numSamples, decodeTokens, temperature, topK, seed)

	var m Measurement
	start := time.Now()
	first := true
	stepStart := time.Now()

	gen(func(column, mask []int) bool {
		m.NumTokens++
		if first {
			m.TTFT = time.Since(start)
			first = false
			stepStart = time.Now()
			return true
		}
		m.StepTimes = append(m.StepTimes, time.Since(stepStart))
		stepStart = time.Now()
		return true
	})
	return m
}
