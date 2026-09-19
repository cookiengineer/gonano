package tensors

import "fmt"

// Int32s is a dense, row-major, contiguous multi-dimensional array of int32,
// used for token ids and loss masks. Integer tensors carry no arithmetic
// kernels; they exist to carry indices and masks into float kernels.
type Int32s struct {
	Shape []int
	Data  []int32
}

// NewInt32s returns a zeroed integer tensor of the given shape.
func NewInt32s(shape ...int) *Int32s {
	if len(shape) == 0 {
		panic("tensors: empty shape")
	}
	elementCount := 1
	for _, dimension := range shape {
		if dimension <= 0 {
			panic(fmt.Sprintf("tensors: invalid dimension %d", dimension))
		}
		elementCount *= dimension
	}
	return &Int32s{Shape: append([]int(nil), shape...), Data: make([]int32, elementCount)}
}

// NewInt32sWithData wraps existing data as an Int32s of the given shape.
func NewInt32sWithData(shape []int, data []int32) *Int32s {
	elementCount := 1
	for _, dimension := range shape {
		elementCount *= dimension
	}
	if len(data) != elementCount {
		panic(fmt.Sprintf("tensors: data length %d != numel %d", len(data), elementCount))
	}
	return &Int32s{Shape: append([]int(nil), shape...), Data: data}
}

// Numel returns the total number of elements.
func (tensor *Int32s) Numel() int { return len(tensor.Data) }

// Reshape returns a view sharing Data with the receiver.
func (tensor *Int32s) Reshape(shape ...int) *Int32s {
	elementCount := 1
	for _, dimension := range shape {
		if dimension <= 0 {
			panic(fmt.Sprintf("tensors: invalid reshape dim %d", dimension))
		}
		elementCount *= dimension
	}
	if elementCount != tensor.Numel() {
		panic(fmt.Sprintf("tensors: reshape numel %d != %d", elementCount, tensor.Numel()))
	}
	return &Int32s{Shape: append([]int(nil), shape...), Data: tensor.Data}
}

// Clone returns a deep copy.
func (tensor *Int32s) Clone() *Int32s {
	clone := NewInt32s(tensor.Shape...)
	copy(clone.Data, tensor.Data)
	return clone
}

// Get2 indexes a rank-2 tensor.
func (tensor *Int32s) Get2(row, column int) int32 {
	return tensor.Data[row*tensor.Shape[1]+column]
}

// Set2 indexes a rank-2 tensor.
func (tensor *Int32s) Set2(row, column int, value int32) {
	tensor.Data[row*tensor.Shape[1]+column] = value
}
