package tensor

import "math"

// Scalar transcendental kernels. The simd package exposes no exp/tanh/sigmoid,
// so these are computed element-wise with math.Exp. They are applied to
// small tensors (logits, gates) relative to the matmuls that dominate runtime,
// so the scalar cost is negligible.

// expKernel computes dst[i] = exp(a[i]).
func expKernel(dst, a []float32) {
	for i, v := range a {
		dst[i] = float32(math.Exp(float64(v)))
	}
}

// sigmoidKernel computes dst[i] = sigmoid(a[i]).
func sigmoidKernel(dst, a []float32) {
	for i, v := range a {
		dst[i] = float32(1.0 / (1.0 + math.Exp(float64(-v))))
	}
}

// tanhKernel computes dst[i] = tanh(a[i]).
func tanhKernel(dst, a []float32) {
	for i, v := range a {
		dst[i] = float32(math.Tanh(float64(v)))
	}
}

// rsqrtKernel computes dst[i] = 1/sqrt(a[i]).
func rsqrtKernel(dst, a []float32) {
	for i, v := range a {
		dst[i] = float32(1.0 / math.Sqrt(float64(v)))
	}
}

// Sigmoid returns element-wise sigmoid.
func Sigmoid(x *Tensor) *Tensor {
	out := New(x.Shape...)
	sigmoidKernel(out.Data, x.Data)
	return out
}

// Tanh returns element-wise tanh.
func Tanh(x *Tensor) *Tensor {
	out := New(x.Shape...)
	tanhKernel(out.Data, x.Data)
	return out
}

// Exp returns element-wise exp.
func Exp(x *Tensor) *Tensor {
	out := New(x.Shape...)
	expKernel(out.Data, x.Data)
	return out
}

// Rsqrt returns element-wise reciprocal square root.
func Rsqrt(x *Tensor) *Tensor {
	out := New(x.Shape...)
	rsqrtKernel(out.Data, x.Data)
	return out
}

// Softcap applies the logit softcap used by nanochat:
// out = softcap * tanh(x / softcap).
func Softcap(x *Tensor, softcap float32) *Tensor {
	out := New(x.Shape...)
	inv := 1.0 / softcap
	for i, v := range x.Data {
		out.Data[i] = softcap * float32(math.Tanh(float64(v)*float64(inv)))
	}
	return out
}
