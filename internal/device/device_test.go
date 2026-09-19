package device

import (
	"runtime"
	"testing"
)

func TestDetect(test *testing.T) {
	cpu := Detect()
	if cpu.NumCPUs < 1 {
		test.Fatalf("NumCPUs = %d, want >= 1", cpu.NumCPUs)
	}
	if cpu.NumCPUs != runtime.GOMAXPROCS(0) {
		test.Fatalf("NumCPUs = %d, want GOMAXPROCS %d", cpu.NumCPUs, runtime.GOMAXPROCS(0))
	}
	switch cpu.VectorBitSize {
	case 128, 256, 512:
	default:
		test.Fatalf("unexpected VectorBitSize = %d", cpu.VectorBitSize)
	}
}

func TestLaneCountMatchesLen(test *testing.T) {
	cpu := Detect()
	if cpu.LaneCount() != Float32sLen() {
		test.Fatalf("LaneCount %d != Float32sLen %d", cpu.LaneCount(), Float32sLen())
	}
	if Float32sLen() < 4 {
		test.Fatalf("Float32sLen = %d, want >= 4", Float32sLen())
	}
	if Float64sLen() < 2 {
		test.Fatalf("Float64sLen = %d, want >= 2", Float64sLen())
	}
}

func TestPeakFlopsPositive(test *testing.T) {
	cpu := Detect()
	if PeakFlopsPerSecond(cpu) <= 0 {
		test.Fatalf("PeakFlopsPerSecond = %f, want > 0", PeakFlopsPerSecond(cpu))
	}
}
