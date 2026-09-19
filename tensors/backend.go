// Package tensors provides gonano's numeric substrate: dense, row-major,
// contiguous float32 tensors (and int32 tensors for token ids), together with
// the high-level operations that allocate results and gradient buffers.
//
// The arithmetic itself is delegated to a kernels.Backend, so the same tensor
// code runs on the SIMD backend, the portable scalar backend, or any future
// implementation. Call KernelBackend to inspect the active backend and
// UseKernelBackend to replace it (primarily in tests).
package tensors

import (
	"github.com/cookiengineer/gonano/kernels"
	simdbackend "github.com/cookiengineer/gonano/kernels/simd"
)

// activeKernelBackend is the numeric backend used by every operation in this
// package. It defaults to the SIMD implementation.
var activeKernelBackend kernels.Backend = simdbackend.New()

// KernelBackend returns the numeric backend currently in use.
func KernelBackend() kernels.Backend { return activeKernelBackend }

// UseKernelBackend replaces the numeric backend. It exists so tests and
// portable builds can select the scalar reference implementation, or any other
// kernels.Backend.
func UseKernelBackend(backend kernels.Backend) { activeKernelBackend = backend }
