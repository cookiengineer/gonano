package device

import (
	"runtime"
	"testing"
)

func TestDetect(t *testing.T) {
	c := Detect()
	if c.NumCPUs < 1 {
		t.Fatalf("NumCPUs = %d, want >= 1", c.NumCPUs)
	}
	if c.NumCPUs != runtime.GOMAXPROCS(0) {
		t.Fatalf("NumCPUs = %d, want GOMAXPROCS %d", c.NumCPUs, runtime.GOMAXPROCS(0))
	}
	switch c.VectorBitSize {
	case 128, 256, 512:
	default:
		t.Fatalf("unexpected VectorBitSize = %d", c.VectorBitSize)
	}
}

func TestLaneCountMatchesLen(t *testing.T) {
	c := Detect()
	if c.LaneCount() != Float32sLen() {
		t.Fatalf("LaneCount %d != Float32sLen %d", c.LaneCount(), Float32sLen())
	}
	if Float32sLen() < 4 {
		t.Fatalf("Float32sLen = %d, want >= 4", Float32sLen())
	}
	if Float64sLen() < 2 {
		t.Fatalf("Float64sLen = %d, want >= 2", Float64sLen())
	}
}

func TestPeakFlopsPositive(t *testing.T) {
	c := Detect()
	if PeakFlopsPerSecond(c) <= 0 {
		t.Fatalf("PeakFlopsPerSecond = %f, want > 0", PeakFlopsPerSecond(c))
	}
}
