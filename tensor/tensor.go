// Package tensor implements the numeric substrate of gonano: dense
// float32 tensors, integer tensors for token ids, and SIMD-accelerated,
// goroutine-parallel kernels (matmul, elementwise, reductions, softmax, norms).
//
// Tensors are row-major and contiguous. Every SIMD kernel has a scalar twin
// used as a reference in tests, so correctness is verified independently of
// the vectorized implementation.
package tensor

import (
	"fmt"
	"strings"
)

// Tensor is a dense, row-major, contiguous multi-dimensional array of
// float32. It is the common currency of all gonano computation.
type Tensor struct {
	Shape []int
	Data  []float32
	// Grad holds the accumulated gradient of Data. It is nil until needed
	// (EnsureGrad). The training loop backpropagates into Grad and the
	// optimizer reads it.
	Grad []float32
}

// New returns a zeroed Tensor with the given shape. It panics if the shape
// has zero or negative dimensions.
func New(shape ...int) *Tensor {
	if len(shape) == 0 {
		panic("tensor: empty shape")
	}
	for _, d := range shape {
		if d <= 0 {
			panic(fmt.Sprintf("tensor: invalid dimension %d", d))
		}
	}
	n := 1
	for _, d := range shape {
		n *= d
	}
	return &Tensor{Shape: append([]int(nil), shape...), Data: make([]float32, n)}
}

// NewWithData wraps existing data as a Tensor of the given shape. The caller
// keeps ownership of data. It panics if len(data) != numel(shape).
func NewWithData(shape []int, data []float32) *Tensor {
	if len(shape) == 0 {
		panic("tensor: empty shape")
	}
	n := 1
	for _, d := range shape {
		if d <= 0 {
			panic(fmt.Sprintf("tensor: invalid dimension %d", d))
		}
		n *= d
	}
	if len(data) != n {
		panic(fmt.Sprintf("tensor: data length %d != numel %d", len(data), n))
	}
	return &Tensor{Shape: append([]int(nil), shape...), Data: data}
}

// NewFull returns a Tensor of the given shape filled with fill.
func NewFull(shape []int, fill float32) *Tensor {
	t := New(shape...)
	for i := range t.Data {
		t.Data[i] = fill
	}
	return t
}

// Numel returns the total number of elements.
func (t *Tensor) Numel() int { return len(t.Data) }

// Rank returns the number of dimensions.
func (t *Tensor) Rank() int { return len(t.Shape) }

// Strides returns the row-major strides for the tensor's shape.
func Strides(shape []int) []int {
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}

// Offset converts a multi-index into a flat offset.
func (t *Tensor) Offset(indexes ...int) int {
	if len(indexes) != len(t.Shape) {
		panic(fmt.Sprintf("tensor: index rank %d != shape rank %d", len(indexes), len(t.Shape)))
	}
	off := 0
	for i, idx := range indexes {
		if idx < 0 || idx >= t.Shape[i] {
			panic(fmt.Sprintf("tensor: index %d out of range in dim %d (shape %v)", idx, i, t.Shape))
		}
		off = off*t.Shape[i] + idx
	}
	return off
}

// Reshape returns a view of the tensor with a new shape of equal numel. The
// returned Tensor shares the underlying Data slice.
func (t *Tensor) Reshape(shape ...int) *Tensor {
	n := 1
	for _, d := range shape {
		if d <= 0 {
			panic(fmt.Sprintf("tensor: invalid reshape dim %d", d))
		}
		n *= d
	}
	if n != t.Numel() {
		panic(fmt.Sprintf("tensor: reshape numel %d != %d", n, t.Numel()))
	}
	return &Tensor{Shape: append([]int(nil), shape...), Data: t.Data}
}

// View returns a Tensor that shares Data with the receiver, described by a
// new shape. It is a convenience for reinterpreting flat buffers.
func (t *Tensor) View(shape ...int) *Tensor { return t.Reshape(shape...) }

// Clone returns a deep copy.
func (t *Tensor) Clone() *Tensor {
	out := New(t.Shape...)
	copy(out.Data, t.Data)
	return out
}

// Set fills the tensor with v.
func (t *Tensor) Set(v float32) {
	for i := range t.Data {
		t.Data[i] = v
	}
}

// Get2 indexes a rank-2 tensor.
func (t *Tensor) Get2(i, j int) float32 { return t.Data[i*t.Shape[1]+j] }

// Set2 indexes a rank-2 tensor.
func (t *Tensor) Set2(i, j int, v float32) { t.Data[i*t.Shape[1]+j] = v }

// Get3 indexes a rank-3 tensor.
func (t *Tensor) Get3(i, j, k int) float32 {
	return t.Data[(i*t.Shape[1]+j)*t.Shape[2]+k]
}

// Set3 indexes a rank-3 tensor.
func (t *Tensor) Set3(i, j, k int, v float32) { t.Data[(i*t.Shape[1]+j)*t.Shape[2]+k] = v }

// Get4 indexes a rank-4 tensor.
func (t *Tensor) Get4(i, j, k, l int) float32 {
	return t.Data[((i*t.Shape[1]+j)*t.Shape[2]+k)*t.Shape[3]+l]
}

// Set4 indexes a rank-4 tensor.
func (t *Tensor) Set4(i, j, k, l int, v float32) {
	t.Data[((i*t.Shape[1]+j)*t.Shape[2]+k)*t.Shape[3]+l] = v
}

// String renders the shape of the tensor.
func (t *Tensor) String() string {
	parts := make([]string, len(t.Shape))
	for i, d := range t.Shape {
		parts[i] = fmt.Sprint(d)
	}
	return "tensor[" + strings.Join(parts, "×") + "]"
}

// EnsureGrad allocates the gradient buffer if it does not exist yet.
func (t *Tensor) EnsureGrad() {
	if t.Grad == nil {
		t.Grad = make([]float32, len(t.Data))
	}
}

// ZeroGrad zeroes the gradient buffer (allocating it if needed).
func (t *Tensor) ZeroGrad() {
	if t.Grad == nil {
		t.Grad = make([]float32, len(t.Data))
		return
	}
	for i := range t.Grad {
		t.Grad[i] = 0
	}
}

// AddGrad adds g into the gradient buffer, allocating it if needed.
func (t *Tensor) AddGrad(g []float32) {
	t.EnsureGrad()
	for i := range g {
		t.Grad[i] += g[i]
	}
}
