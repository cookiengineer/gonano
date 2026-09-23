package tensors

// Public element-wise operations. Each allocates a fresh result tensor and
// delegates the arithmetic to the active kernels.Backend. In-place slice
// kernels live in the backend itself.

// requireSameShape panics unless left and right have identical shapes.
func requireSameShape(left, right *Tensor) {
	if len(left.Shape) != len(right.Shape) {
		panic("tensors: shape rank mismatch")
	}
	for dimension := range left.Shape {
		if left.Shape[dimension] != right.Shape[dimension] {
			panic("tensors: shape mismatch")
		}
	}
}

// Add returns left + right element-wise.
func Add(left, right *Tensor) *Tensor {
	requireSameShape(left, right)
	result := New(left.Shape...)
	KernelBackend().Add(result.Data, left.Data, right.Data)
	return result
}

// Subtract returns left - right element-wise.
func Subtract(left, right *Tensor) *Tensor {
	requireSameShape(left, right)
	result := New(left.Shape...)
	KernelBackend().Subtract(result.Data, left.Data, right.Data)
	return result
}

// Multiply returns left * right element-wise.
func Multiply(left, right *Tensor) *Tensor {
	requireSameShape(left, right)
	result := New(left.Shape...)
	KernelBackend().Multiply(result.Data, left.Data, right.Data)
	return result
}

// Divide returns left / right element-wise.
func Divide(left, right *Tensor) *Tensor {
	requireSameShape(left, right)
	result := New(left.Shape...)
	KernelBackend().Divide(result.Data, left.Data, right.Data)
	return result
}

// Scale returns source * factor.
func Scale(source *Tensor, factor float32) *Tensor {
	result := New(source.Shape...)
	KernelBackend().Scale(result.Data, source.Data, factor)
	return result
}

// AddScaled returns base + factor*scaled element-wise, the residual-connection
// primitive used throughout the transformer.
func AddScaled(base, scaled *Tensor, factor float32) *Tensor {
	requireSameShape(base, scaled)
	result := New(base.Shape...)
	KernelBackend().AddScaled(result.Data, base.Data, scaled.Data, factor)
	return result
}

// Negate returns -source.
func Negate(source *Tensor) *Tensor {
	result := New(source.Shape...)
	KernelBackend().Negate(result.Data, source.Data)
	return result
}

// Abs returns |source|.
func Abs(source *Tensor) *Tensor {
	result := New(source.Shape...)
	KernelBackend().Abs(result.Data, source.Data)
	return result
}

// Square returns source*source element-wise.
func Square(source *Tensor) *Tensor {
	result := New(source.Shape...)
	KernelBackend().Square(result.Data, source.Data)
	return result
}

// ReluSquared returns max(source, 0)^2 element-wise.
func ReluSquared(source *Tensor) *Tensor {
	result := New(source.Shape...)
	KernelBackend().ReluSquared(result.Data, source.Data)
	return result
}

// SwiGLU returns the clamped SwiGLU activation silu(min(gate,clamp)) *
// clamp(up,-clamp,clamp) element-wise. A clamp <= 0 disables clamping.
func SwiGLU(gate, up *Tensor, clamp float32) *Tensor {
	requireSameShape(gate, up)
	result := New(gate.Shape...)
	KernelBackend().SwiGLU(result.Data, gate.Data, up.Data, clamp)
	return result
}

// Copy returns a deep copy of source.
func Copy(source *Tensor) *Tensor { return source.Clone() }
