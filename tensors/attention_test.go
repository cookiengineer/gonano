package tensors

import "testing"

// TestAttentionForwardWrapper verifies the tensors-level attention wrapper
// delegates correctly and produces a probability-weighted combination of the
// value vectors.
func TestAttentionForwardWrapper(t *testing.T) {
	useScalarBackend(t)
	query := NewWithData([]int{2, 2}, []float32{1, 0, 0, 1})
	key := NewWithData([]int{2, 2}, []float32{1, 0, 0, 1})
	value := NewWithData([]int{2, 2}, []float32{1, 0, 0, 1})
	output := make([]float32, 4)
	logSumExp := make([]float32, 2)

	AttentionForward(query.Data, key.Data, value.Data, output, logSumExp, 2, 2, 2, 0, -1)

	// Row 0 can only attend to key 0, so the output row must equal value row 0.
	if !close(output[0], 1, 1e-5, 1e-6) || !close(output[1], 0, 1e-5, 1e-6) {
		t.Fatalf("causal output row 0 = %v, want [1 0]", output[0:2])
	}
	for _, value := range logSumExp {
		if value != value {
			t.Fatal("logSumExp contains NaN")
		}
	}
}

// TestAttentionBackwardWrapper verifies the tensors-level attention backward
// wrapper populates all three gradients.
func TestAttentionBackwardWrapper(t *testing.T) {
	useScalarBackend(t)
	query := NewWithData([]int{2, 2}, []float32{1, 0, 0, 1})
	key := NewWithData([]int{2, 2}, []float32{1, 0, 0, 1})
	value := NewWithData([]int{2, 2}, []float32{1, 0, 0, 1})
	output := make([]float32, 4)
	logSumExp := make([]float32, 2)
	AttentionForward(query.Data, key.Data, value.Data, output, logSumExp, 2, 2, 2, 0, -1)

	outputGradient := []float32{1, 1, 1, 1}
	queryGradient := make([]float32, 4)
	keyGradient := make([]float32, 4)
	valueGradient := make([]float32, 4)
	AttentionBackward(query.Data, key.Data, value.Data, output, outputGradient, logSumExp, queryGradient, keyGradient, valueGradient, 2, 2, 2, 0, -1)

	for _, gradient := range [][]float32{queryGradient, keyGradient, valueGradient} {
		for _, element := range gradient {
			if element != element {
				t.Fatal("attention backward produced NaN")
			}
		}
	}
}

func TestSoftmaxAliasAndCopy(t *testing.T) {
	useScalarBackend(t)
	source := NewWithData([]int{1, 2}, []float32{1, 2})
	if Softmax(source).Numel() != 2 {
		t.Fatal("Softmax alias failed")
	}
	copied := Copy(source)
	copied.Data[0] = 99
	if source.Data[0] == 99 {
		t.Fatal("Copy must be a deep copy")
	}
}
