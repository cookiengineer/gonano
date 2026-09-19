package tensors

import (
	"math"
	"testing"
)

func TestQuantizeInt8RowsRoundTrip(t *testing.T) {
	weight := NewWithData([]int{2, 4}, []float32{
		0.1, -0.2, 0.3, -0.4,
		1.0, 2.0, -3.0, 4.0,
	})
	quantized := QuantizeInt8Rows(weight)
	dequantized := DequantizeInt8Rows(quantized)
	for row := 0; row < 2; row++ {
		scale := quantized.Scales[row]
		for column := 0; column < 4; column++ {
			index := row*4 + column
			if errorValue := float32(math.Abs(float64(weight.Data[index] - dequantized.Data[index]))); errorValue > scale*0.5+1e-6 {
				t.Fatalf("row %d col %d: error %v exceeds half a quantization step %v", row, column, errorValue, scale*0.5)
			}
		}
	}
}

func TestQuantizeInt8RowsZero(t *testing.T) {
	weight := New(1, 3)
	quantized := QuantizeInt8Rows(weight)
	if quantized.Scales[0] != 1 {
		t.Fatalf("zero row scale = %v, want 1", quantized.Scales[0])
	}
	for _, value := range DequantizeInt8Rows(quantized).Data {
		if value != 0 {
			t.Fatalf("zero row dequantized to %v", value)
		}
	}
}

func TestQuantizeInt8RowsPerRowScale(t *testing.T) {
	weight := NewWithData([]int{2, 2}, []float32{
		1, 1,
		4, 4,
	})
	quantized := QuantizeInt8Rows(weight)
	if quantized.Scales[0] == quantized.Scales[1] {
		t.Fatalf("per-row scales should differ, both %v", quantized.Scales[0])
	}
	// The largest magnitude in each row must be preserved exactly.
	dequantized := DequantizeInt8Rows(quantized)
	if dequantized.Data[0] != 1 || dequantized.Data[2] != 4 {
		t.Fatalf("maxima not preserved: %v", dequantized.Data)
	}
}
