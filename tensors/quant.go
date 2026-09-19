package tensors

import "math"

// QuantizedInt8Rows is a symmetric per-row int8 quantization of a rank-2
// weight matrix. Each row has its own scale so that the largest magnitude in
// the row maps to 127.
type QuantizedInt8Rows struct {
	Packed  []int8
	Scales  []float32
	Rows    int
	Columns int
}

// QuantizeInt8Rows quantizes each row of a [rows, columns] weight matrix to
// symmetric int8.
func QuantizeInt8Rows(weight *Tensor) QuantizedInt8Rows {
	rows := weight.Shape[0]
	columns := weight.Shape[1]
	packed := make([]int8, rows*columns)
	scales := make([]float32, rows)
	for row := 0; row < rows; row++ {
		rowData := weight.Data[row*columns : (row+1)*columns]
		maximum := float32(0)
		for _, value := range rowData {
			if magnitude := float32(math.Abs(float64(value))); magnitude > maximum {
				maximum = magnitude
			}
		}
		scale := float32(1)
		if maximum > 0 {
			scale = maximum / 127
		}
		scales[row] = scale
		inverse := float32(1) / scale
		for column, value := range rowData {
			quantized := math.Round(float64(value * inverse))
			if quantized > 127 {
				quantized = 127
			}
			if quantized < -127 {
				quantized = -127
			}
			packed[row*columns+column] = int8(quantized)
		}
	}
	return QuantizedInt8Rows{Packed: packed, Scales: scales, Rows: rows, Columns: columns}
}

// DequantizeInt8Rows reconstructs a float32 tensor from a per-row int8
// quantization.
func DequantizeInt8Rows(quantized QuantizedInt8Rows) *Tensor {
	output := New(quantized.Rows, quantized.Columns)
	for row := 0; row < quantized.Rows; row++ {
		scale := quantized.Scales[row]
		for column := 0; column < quantized.Columns; column++ {
			output.Data[row*quantized.Columns+column] = float32(quantized.Packed[row*quantized.Columns+column]) * scale
		}
	}
	return output
}

// FakeQuantizeInt8Rows returns a float32 copy of weight with each row quantized
// to int8 and immediately dequantized. It is the quantization-aware-training
// simulation used in the forward pass; gradients flow through unchanged
// (straight-through estimator).
func FakeQuantizeInt8Rows(weight *Tensor) *Tensor {
	return DequantizeInt8Rows(QuantizeInt8Rows(weight))
}
