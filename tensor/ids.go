package tensor

import "fmt"

// Int32s is a dense, row-major, contiguous multi-dimensional array of int32,
// used for token ids and loss masks. Integer tensors are never used for SIMD
// arithmetic (there is no integer matmul in gonano); they exist to carry
// indices and masks into float kernels.
type Int32s struct {
	Shape []int
	Data  []int32
}

// NewInt32s returns a zeroed integer tensor of the given shape.
func NewInt32s(shape ...int) *Int32s {
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
	return &Int32s{Shape: append([]int(nil), shape...), Data: make([]int32, n)}
}

// NewInt32sWithData wraps existing data as an Int32s of the given shape.
func NewInt32sWithData(shape []int, data []int32) *Int32s {
	n := 1
	for _, d := range shape {
		n *= d
	}
	if len(data) != n {
		panic(fmt.Sprintf("tensor: data length %d != numel %d", len(data), n))
	}
	return &Int32s{Shape: append([]int(nil), shape...), Data: data}
}

// Numel returns the total number of elements.
func (t *Int32s) Numel() int { return len(t.Data) }

// Reshape returns a view sharing Data with the receiver.
func (t *Int32s) Reshape(shape ...int) *Int32s {
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
	return &Int32s{Shape: append([]int(nil), shape...), Data: t.Data}
}

// Clone returns a deep copy.
func (t *Int32s) Clone() *Int32s {
	out := NewInt32s(t.Shape...)
	copy(out.Data, t.Data)
	return out
}

// Get2 indexes a rank-2 tensor.
func (t *Int32s) Get2(i, j int) int32 { return t.Data[i*t.Shape[1]+j] }

// Set2 indexes a rank-2 tensor.
func (t *Int32s) Set2(i, j int, v int32) { t.Data[i*t.Shape[1]+j] = v }
