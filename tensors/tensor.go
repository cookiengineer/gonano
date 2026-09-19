package tensors

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

// New returns a zeroed Tensor with the given shape. It panics if the shape has
// zero or negative dimensions.
func New(shape ...int) *Tensor {
	if len(shape) == 0 {
		panic("tensors: empty shape")
	}
	for _, dimension := range shape {
		if dimension <= 0 {
			panic(fmt.Sprintf("tensors: invalid dimension %d", dimension))
		}
	}
	elementCount := 1
	for _, dimension := range shape {
		elementCount *= dimension
	}
	return &Tensor{Shape: append([]int(nil), shape...), Data: make([]float32, elementCount)}
}

// NewWithData wraps existing data as a Tensor of the given shape. The caller
// keeps ownership of data. It panics if len(data) != numel(shape).
func NewWithData(shape []int, data []float32) *Tensor {
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
	if len(data) != elementCount {
		panic(fmt.Sprintf("tensors: data length %d != numel %d", len(data), elementCount))
	}
	return &Tensor{Shape: append([]int(nil), shape...), Data: data}
}

// NewFull returns a Tensor of the given shape filled with fillValue.
func NewFull(shape []int, fillValue float32) *Tensor {
	tensor := New(shape...)
	for index := range tensor.Data {
		tensor.Data[index] = fillValue
	}
	return tensor
}

// Numel returns the total number of elements.
func (tensor *Tensor) Numel() int { return len(tensor.Data) }

// Rank returns the number of dimensions.
func (tensor *Tensor) Rank() int { return len(tensor.Shape) }

// Strides returns the row-major strides for the tensor's shape.
func Strides(shape []int) []int {
	strides := make([]int, len(shape))
	stride := 1
	for dimension := len(shape) - 1; dimension >= 0; dimension-- {
		strides[dimension] = stride
		stride *= shape[dimension]
	}
	return strides
}

// Offset converts a multi-index into a flat offset.
func (tensor *Tensor) Offset(indexes ...int) int {
	if len(indexes) != len(tensor.Shape) {
		panic(fmt.Sprintf("tensors: index rank %d != shape rank %d", len(indexes), len(tensor.Shape)))
	}
	offset := 0
	for dimension, index := range indexes {
		if index < 0 || index >= tensor.Shape[dimension] {
			panic(fmt.Sprintf("tensors: index %d out of range in dim %d (shape %v)", index, dimension, tensor.Shape))
		}
		offset = offset*tensor.Shape[dimension] + index
	}
	return offset
}

// Reshape returns a view of the tensor with a new shape of equal numel. The
// returned Tensor shares the underlying Data slice.
func (tensor *Tensor) Reshape(shape ...int) *Tensor {
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
	return &Tensor{Shape: append([]int(nil), shape...), Data: tensor.Data}
}

// View returns a Tensor that shares Data with the receiver, described by a new
// shape. It is a convenience for reinterpreting flat buffers.
func (tensor *Tensor) View(shape ...int) *Tensor { return tensor.Reshape(shape...) }

// Clone returns a deep copy.
func (tensor *Tensor) Clone() *Tensor {
	clone := New(tensor.Shape...)
	copy(clone.Data, tensor.Data)
	return clone
}

// Set fills the tensor with value.
func (tensor *Tensor) Set(value float32) {
	for index := range tensor.Data {
		tensor.Data[index] = value
	}
}

// Get2 indexes a rank-2 tensor.
func (tensor *Tensor) Get2(row, column int) float32 {
	return tensor.Data[row*tensor.Shape[1]+column]
}

// Set2 indexes a rank-2 tensor.
func (tensor *Tensor) Set2(row, column int, value float32) {
	tensor.Data[row*tensor.Shape[1]+column] = value
}

// Get3 indexes a rank-3 tensor.
func (tensor *Tensor) Get3(first, second, third int) float32 {
	return tensor.Data[(first*tensor.Shape[1]+second)*tensor.Shape[2]+third]
}

// Set3 indexes a rank-3 tensor.
func (tensor *Tensor) Set3(first, second, third int, value float32) {
	tensor.Data[(first*tensor.Shape[1]+second)*tensor.Shape[2]+third] = value
}

// Get4 indexes a rank-4 tensor.
func (tensor *Tensor) Get4(first, second, third, fourth int) float32 {
	return tensor.Data[((first*tensor.Shape[1]+second)*tensor.Shape[2]+third)*tensor.Shape[3]+fourth]
}

// Set4 indexes a rank-4 tensor.
func (tensor *Tensor) Set4(first, second, third, fourth int, value float32) {
	tensor.Data[((first*tensor.Shape[1]+second)*tensor.Shape[2]+third)*tensor.Shape[3]+fourth] = value
}

// String renders the shape of the tensor.
func (tensor *Tensor) String() string {
	parts := make([]string, len(tensor.Shape))
	for index, dimension := range tensor.Shape {
		parts[index] = fmt.Sprint(dimension)
	}
	return "tensor[" + strings.Join(parts, "×") + "]"
}

// EnsureGrad allocates the gradient buffer if it does not exist yet.
func (tensor *Tensor) EnsureGrad() {
	if tensor.Grad == nil {
		tensor.Grad = make([]float32, len(tensor.Data))
	}
}

// ZeroGrad zeroes the gradient buffer (allocating it if needed).
func (tensor *Tensor) ZeroGrad() {
	if tensor.Grad == nil {
		tensor.Grad = make([]float32, len(tensor.Data))
		return
	}
	for index := range tensor.Grad {
		tensor.Grad[index] = 0
	}
}

// AddGrad adds gradient into the gradient buffer, allocating it if needed.
func (tensor *Tensor) AddGrad(gradient []float32) {
	tensor.EnsureGrad()
	for index := range gradient {
		tensor.Grad[index] += gradient[index]
	}
}
