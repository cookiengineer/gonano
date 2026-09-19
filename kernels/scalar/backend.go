// Package scalar implements the kernels.Backend contract in portable,
// dependency-free Go. It is the reference implementation used to verify the
// SIMD backend element by element and to run on hosts without SIMD support.
package scalar

// Backend is the pure-Go implementation of kernels.Backend.
type Backend struct{}

// New returns a new scalar backend.
func New() *Backend { return &Backend{} }
