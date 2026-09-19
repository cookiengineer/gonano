package inference

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
	// DecodeSteps is the number of token-by-token decode forwards measured,
	// equal to len(StepTimes).
	DecodeSteps int
	// WeightBytes is the total bytes of matmul weights read across the measured
	// decode steps. Weights are shared across batch rows, so this does not
	// scale with the batch size.
	WeightBytes int64
	// KVBytes is the total bytes of KV cache read across the measured decode
	// steps. Each batch row reads its own KV prefix, so this scales with both
	// the batch size and the growing context length.
	KVBytes int64
}

// Measure runs one timed generation and reports TTFT and per-step timings.
// The first step includes the batch-1 prefill; subsequent steps are decode.
func Measure(engine *Engine, tokens []int, numSamples, decodeTokens int, temperature float32, topK int, seed uint64) Measurement {
	generate := engine.Generate(tokens, numSamples, decodeTokens, temperature, topK, seed)

	var measurement Measurement
	start := time.Now()
	first := true
	stepStart := time.Now()

	// Between the first and second yielded tokens the engine performed the
	// first token-by-token decode forward, at context length prompt+1. Each
	// subsequent step advances the context by one.
	decodeContext := len(tokens) + 1

	generate(func(column, mask []int) bool {
		measurement.NumTokens++
		if first {
			measurement.TTFT = time.Since(start)
			first = false
			stepStart = time.Now()
			return true
		}
		measurement.StepTimes = append(measurement.StepTimes, time.Since(stepStart))
		measurement.DecodeSteps++
		measurement.WeightBytes += int64(engine.Model.WeightReadBytes())
		measurement.KVBytes += int64(numSamples) * int64(engine.Model.KVReadBytes(decodeContext))
		decodeContext++
		stepStart = time.Now()
		return true
	})
	return measurement
}
