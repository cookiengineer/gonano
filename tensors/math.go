package tensors

// Transcendental and scalar activation operations. The SIMD backend computes
// exp/tanh/sigmoid element-wise with math.Exp because the simd package exposes
// no such functions; these tensors are small relative to the matmuls that
// dominate runtime.

// Sigmoid returns the element-wise sigmoid of source.
func Sigmoid(source *Tensor) *Tensor {
	result := New(source.Shape...)
	KernelBackend().Sigmoid(result.Data, source.Data)
	return result
}

// Tanh returns the element-wise hyperbolic tangent of source.
func Tanh(source *Tensor) *Tensor {
	result := New(source.Shape...)
	KernelBackend().Tanh(result.Data, source.Data)
	return result
}

// Exp returns the element-wise exponential of source.
func Exp(source *Tensor) *Tensor {
	result := New(source.Shape...)
	KernelBackend().Exp(result.Data, source.Data)
	return result
}

// Rsqrt returns the element-wise reciprocal square root of source.
func Rsqrt(source *Tensor) *Tensor {
	result := New(source.Shape...)
	KernelBackend().Rsqrt(result.Data, source.Data)
	return result
}

// Softcap applies the logit softcap used by nanochat:
// result = softcap * tanh(source / softcap).
func Softcap(source *Tensor, softcapValue float32) *Tensor {
	result := New(source.Shape...)
	inverseSoftcap := 1.0 / softcapValue
	KernelBackend().Scale(result.Data, source.Data, inverseSoftcap)
	KernelBackend().Tanh(result.Data, result.Data)
	KernelBackend().Scale(result.Data, result.Data, softcapValue)
	return result
}
