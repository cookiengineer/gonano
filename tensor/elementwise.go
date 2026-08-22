package tensor

import "simd"

// This file implements SIMD-accelerated elementwise kernels over contiguous
// float32 slices. Each kernel processes full SIMD vectors and a scalar tail.
// All kernels are in-place-capable: dst may alias a or b.

// addKernel computes dst[i] = a[i] + b[i].
func addKernel(dst, a, b []float32) {
	n := len(a)
	i := 0
	for ; i+float32Lanes <= n; i += float32Lanes {
		av := simd.LoadFloat32s(a[i:])
		bv := simd.LoadFloat32s(b[i:])
		av.Add(bv).Store(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] + b[i]
	}
}

// subKernel computes dst[i] = a[i] - b[i].
func subKernel(dst, a, b []float32) {
	n := len(a)
	i := 0
	for ; i+float32Lanes <= n; i += float32Lanes {
		av := simd.LoadFloat32s(a[i:])
		bv := simd.LoadFloat32s(b[i:])
		av.Sub(bv).Store(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] - b[i]
	}
}

// mulKernel computes dst[i] = a[i] * b[i].
func mulKernel(dst, a, b []float32) {
	n := len(a)
	i := 0
	for ; i+float32Lanes <= n; i += float32Lanes {
		av := simd.LoadFloat32s(a[i:])
		bv := simd.LoadFloat32s(b[i:])
		av.Mul(bv).Store(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] * b[i]
	}
}

// divKernel computes dst[i] = a[i] / b[i].
func divKernel(dst, a, b []float32) {
	n := len(a)
	i := 0
	for ; i+float32Lanes <= n; i += float32Lanes {
		av := simd.LoadFloat32s(a[i:])
		bv := simd.LoadFloat32s(b[i:])
		av.Div(bv).Store(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] / b[i]
	}
}

// scaleKernel computes dst[i] = a[i] * s.
func scaleKernel(dst, a []float32, s float32) {
	n := len(a)
	sv := simd.BroadcastFloat32s(s)
	i := 0
	for ; i+float32Lanes <= n; i += float32Lanes {
		av := simd.LoadFloat32s(a[i:])
		av.Mul(sv).Store(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] * s
	}
}

// addScaledKernel computes dst[i] = a[i] + s*b[i]. It is the residual-connection
// primitive (x = x + s*out) used throughout the transformer.
func addScaledKernel(dst, a, b []float32, s float32) {
	n := len(a)
	sv := simd.BroadcastFloat32s(s)
	i := 0
	for ; i+float32Lanes <= n; i += float32Lanes {
		av := simd.LoadFloat32s(a[i:])
		bv := simd.LoadFloat32s(b[i:])
		// MulAdd(x, y, z) = x*y + z, so bv.MulAdd(sv, av) = s*b + a.
		bv.MulAdd(sv, av).Store(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] + s*b[i]
	}
}

// negKernel computes dst[i] = -a[i].
func negKernel(dst, a []float32) {
	n := len(a)
	i := 0
	for ; i+float32Lanes <= n; i += float32Lanes {
		av := simd.LoadFloat32s(a[i:])
		av.Neg().Store(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = -a[i]
	}
}

// absKernel computes dst[i] = |a[i]|.
func absKernel(dst, a []float32) {
	n := len(a)
	i := 0
	for ; i+float32Lanes <= n; i += float32Lanes {
		av := simd.LoadFloat32s(a[i:])
		av.Abs().Store(dst[i:])
	}
	for ; i < n; i++ {
		if a[i] < 0 {
			dst[i] = -a[i]
		} else {
			dst[i] = a[i]
		}
	}
}

// squaredKernel computes dst[i] = a[i] * a[i].
func squaredKernel(dst, a []float32) {
	n := len(a)
	i := 0
	for ; i+float32Lanes <= n; i += float32Lanes {
		av := simd.LoadFloat32s(a[i:])
		av.Mul(av).Store(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] * a[i]
	}
}

// relu2Kernel computes dst[i] = relu(a[i])^2, the MLP activation of nanochat.
func relu2Kernel(dst, a []float32) {
	n := len(a)
	zero := simd.BroadcastFloat32s(0)
	i := 0
	for ; i+float32Lanes <= n; i += float32Lanes {
		av := simd.LoadFloat32s(a[i:])
		r := av.Max(zero)
		r.Mul(r).Store(dst[i:])
	}
	for ; i < n; i++ {
		v := a[i]
		if v < 0 {
			v = 0
		}
		dst[i] = v * v
	}
}
