// Package device reports the host CPU capabilities used to size parallel
// pools. Roofline estimates live in roofline.go.
package device

// PeakFlopsPerSecond returns an approximate peak float32 FMA throughput for
// the host CPU in flops/sec. It is a rough estimate used only for MFU-style
// reporting; the value is derived from the SIMD vector width, the number of
// CPUs, and an assumed base clock.
//
// Each SIMD lane performs one fused multiply-add (2 flops) per cycle. We do
// not read the actual CPU frequency (it varies with turbo/P-states), so this
// estimate is intentionally conservative and documented as approximate.
func PeakFlopsPerSecond(c CPU) float64 {
	if c.Emulated {
		// Emulated SIMD has no vectorized throughput advantage; report the
		// scalar FMA rate as a rough bound.
		return float64(c.NumCPUs) * 2 * assumedGHz * 1e9
	}
	lanes := c.LaneCount()
	return float64(c.NumCPUs) * float64(lanes) * 2 * assumedGHz * 1e9
}

// assumedGHz is a conservative base frequency used for roofline estimates.
// It is a documented approximation, not a measurement.
const assumedGHz = 3.0
