package tensor

// Public elementwise operations over Tensors. Each returns a newly allocated
// tensor. Slices-level kernels (in-place capable) are in elementwise.go.

// requireSameShape panics unless a and b have identical shapes.
func requireSameShape(a, b *Tensor) {
	if len(a.Shape) != len(b.Shape) {
		panic("tensor: shape rank mismatch")
	}
	for i := range a.Shape {
		if a.Shape[i] != b.Shape[i] {
			panic("tensor: shape mismatch")
		}
	}
}

func binaryOp(a, b *Tensor, kernel func(dst, a, b []float32)) *Tensor {
	requireSameShape(a, b)
	out := New(a.Shape...)
	kernel(out.Data, a.Data, b.Data)
	return out
}

// Add returns a + b element-wise.
func Add(a, b *Tensor) *Tensor { return binaryOp(a, b, addKernel) }

// Sub returns a - b element-wise.
func Sub(a, b *Tensor) *Tensor { return binaryOp(a, b, subKernel) }

// Mul returns a * b element-wise.
func Mul(a, b *Tensor) *Tensor { return binaryOp(a, b, mulKernel) }

// Div returns a / b element-wise.
func Div(a, b *Tensor) *Tensor { return binaryOp(a, b, divKernel) }

// Scale returns a * s.
func Scale(a *Tensor, s float32) *Tensor {
	out := New(a.Shape...)
	scaleKernel(out.Data, a.Data, s)
	return out
}

// AddScaled returns a + s*b element-wise (the residual-connection primitive).
func AddScaled(a, b *Tensor, s float32) *Tensor {
	requireSameShape(a, b)
	out := New(a.Shape...)
	addScaledKernel(out.Data, a.Data, b.Data, s)
	return out
}

// Neg returns -a.
func Neg(a *Tensor) *Tensor {
	out := New(a.Shape...)
	negKernel(out.Data, a.Data)
	return out
}

// Abs returns |a|.
func Abs(a *Tensor) *Tensor {
	out := New(a.Shape...)
	absKernel(out.Data, a.Data)
	return out
}

// Squared returns a*a element-wise.
func Squared(a *Tensor) *Tensor {
	out := New(a.Shape...)
	squaredKernel(out.Data, a.Data)
	return out
}

// Relu2 returns relu(a)^2 element-wise.
func Relu2(a *Tensor) *Tensor {
	out := New(a.Shape...)
	relu2Kernel(out.Data, a.Data)
	return out
}

// Copy returns a deep copy of a.
func Copy(a *Tensor) *Tensor { return a.Clone() }
