// Package kerneltest provides shared test fixtures and comparisons for kernel
// backend parity tests. It is a normal package (not _test.go) so both the SIMD
// and scalar backends can import it from their test files.
package kerneltest

import (
	"math"
	"testing"
)

// Data returns deterministic test values in roughly [-3.5, 3.5).
func Data(count int) []float32 {
	values := make([]float32, count)
	for index := range values {
		values[index] = float32(index%7) - 3.5 + 0.1*float32(index%3)
	}
	return values
}

// DataB returns deterministic positive test values.
func DataB(count int) []float32 {
	values := make([]float32, count)
	for index := range values {
		values[index] = float32(index%5) + 0.5
	}
	return values
}

// Close reports whether got and want agree within a relative tolerance and an
// absolute floor. The absolute floor keeps comparisons meaningful near zero.
func Close(got, want, relativeTolerance, absoluteFloor float32) bool {
	difference := float32(math.Abs(float64(got - want)))
	scale := float32(math.Abs(float64(want)))
	if scale < 1 {
		scale = 1
	}
	return difference <= absoluteFloor+relativeTolerance*scale
}

// AssertSlicesClose fails the test if any element differs beyond tolerance.
func AssertSlicesClose(t *testing.T, got, want []float32, relativeTolerance, absoluteFloor float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}
	for index := range got {
		if !Close(got[index], want[index], relativeTolerance, absoluteFloor) {
			t.Fatalf("value mismatch at %d: got %v, want %v", index, got[index], want[index])
		}
	}
}
