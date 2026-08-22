// Package device reports the hardware capabilities that gonano uses to size
// its parallel pools and estimate roofline performance.
//
// The only supported device is the host CPU. All floating point compute is
// performed in float32 using the simd standard-library package.
package device

import (
	"runtime"
	"simd"
)

// CPU describes the host CPU device.
type CPU struct {
	// NumCPUs is the number of logical CPUs available to the process.
	NumCPUs int
	// VectorBitSize is the SIMD vector width in bits (128, 256, or 512).
	VectorBitSize int
	// Emulated reports whether SIMD operations are emulated in pure Go.
	Emulated bool
}

// Detect returns the current host CPU capabilities.
func Detect() CPU {
	return CPU{
		NumCPUs:       runtime.GOMAXPROCS(0),
		VectorBitSize: simd.VectorBitSize(),
		Emulated:      simd.Emulated(),
	}
}

// LaneCount returns the number of float32 lanes per SIMD vector on this CPU.
func (c CPU) LaneCount() int { return c.VectorBitSize / 32 }

// Float32sLen returns the number of float32 lanes per SIMD vector.
func Float32sLen() int { return simd.VectorBitSize() / 32 }

// Float64sLen returns the number of float64 lanes per SIMD vector.
func Float64sLen() int { return simd.VectorBitSize() / 64 }
