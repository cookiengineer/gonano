package tensors

import (
	"testing"
)

func TestCrossEntropyGradMatchesSoftmaxMinusOnehot(t *testing.T) {
	useScalarBackend(t)
	logits := NewWithData([]int{1, 3}, []float32{0, 0, 0})
	targets := NewInt32sWithData([]int{1}, []int32{1})
	gradient := CrossEntropyGrad(logits, targets, -1, 1)
	want := []float32{1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0}
	assertTensorClose(t, gradient, NewWithData([]int{1, 3}, want), 1e-5, 1e-6)
}

func TestCrossEntropyGradIgnoresMaskedPositions(t *testing.T) {
	useScalarBackend(t)
	logits := NewWithData([]int{2, 3}, []float32{0, 0, 0, 0, 0, 0})
	targets := NewInt32sWithData([]int{2}, []int32{-1, 0})
	gradient := CrossEntropyGrad(logits, targets, -1, 1)
	for column := 0; column < 3; column++ {
		if gradient.Data[column] != 0 {
			t.Fatalf("masked row must have zero gradient, got %v", gradient.Data[column])
		}
	}
}

func TestMatMulBackwardShapes(t *testing.T) {
	useScalarBackend(t)

	// MatMul: left [2,3] @ right [3,4] -> gradientOutput [2,4].
	left := NewWithData([]int{2, 3}, testData(6))
	right := NewWithData([]int{3, 4}, testDataB(12))
	gradientOutput := NewWithData([]int{2, 4}, testDataB(8))
	gradientLeft, gradientRight := MatMulBackward(gradientOutput, left, right)
	if gradientLeft.Shape[0] != 2 || gradientLeft.Shape[1] != 3 {
		t.Fatalf("gradientLeft shape = %v", gradientLeft.Shape)
	}
	if gradientRight.Shape[0] != 3 || gradientRight.Shape[1] != 4 {
		t.Fatalf("gradientRight shape = %v", gradientRight.Shape)
	}

	// MatMulTransposed: left [2,3] @ right [4,3]^T -> gradientOutput [2,4].
	transposedLeft := NewWithData([]int{2, 3}, testData(6))
	transposedRight := NewWithData([]int{4, 3}, testDataB(12))
	gradientLeftTransposed, gradientRightTransposed := MatMulTransposedBackward(gradientOutput, transposedLeft, transposedRight)
	if gradientLeftTransposed.Shape[0] != 2 || gradientLeftTransposed.Shape[1] != 3 {
		t.Fatalf("transposed gradientLeft shape = %v", gradientLeftTransposed.Shape)
	}
	if gradientRightTransposed.Shape[0] != 4 || gradientRightTransposed.Shape[1] != 3 {
		t.Fatalf("transposed gradientRight shape = %v", gradientRightTransposed.Shape)
	}
}

func TestReluSquaredBackward(t *testing.T) {
	useScalarBackend(t)
	input := NewWithData([]int{4}, []float32{-1, 2, -3, 4})
	gradientOutput := NewWithData([]int{4}, []float32{1, 1, 1, 1})
	gradient := ReluSquaredBackward(input, gradientOutput)
	assertTensorClose(t, gradient, NewWithData([]int{4}, []float32{0, 4, 0, 8}), 1e-5, 1e-6)
}

func TestRMSNormBackwardFinite(t *testing.T) {
	useScalarBackend(t)
	input := NewWithData([]int{1, 4}, []float32{1, 2, 3, 4})
	gradientOutput := NewWithData([]int{1, 4}, []float32{1, 1, 1, 1})
	gradient := RMSNormBackward(input, gradientOutput, 1e-6)
	if gradient.Numel() != 4 {
		t.Fatalf("gradient size = %d", gradient.Numel())
	}
	for _, value := range gradient.Data {
		if value != value {
			t.Fatal("RMSNormBackward produced NaN")
		}
	}
}

func TestSoftcapSigmoidScaleBackward(t *testing.T) {
	useScalarBackend(t)
	output := NewWithData([]int{2}, []float32{0.5, -0.5})
	gradientOutput := NewWithData([]int{2}, []float32{1, 1})

	softcapGradient := SoftcapBackward(output, gradientOutput, 15)
	for _, value := range softcapGradient.Data {
		if value != value {
			t.Fatal("SoftcapBackward produced NaN")
		}
	}
	sigmoidGradient := SigmoidBackward(output, gradientOutput)
	if !close(sigmoidGradient.Data[0], 0.25, 1e-5, 1e-6) {
		t.Fatalf("SigmoidBackward = %v, want 0.25", sigmoidGradient.Data[0])
	}
	scaleGradient := ScaleBackward(gradientOutput, 2)
	assertTensorClose(t, scaleGradient, NewWithData([]int{2}, []float32{2, 2}), 1e-6, 1e-6)
}

func TestInt32sReshapeAndClone(t *testing.T) {
	source := NewInt32sWithData([]int{2, 3}, []int32{1, 2, 3, 4, 5, 6})
	reshaped := source.Reshape(3, 2)
	if reshaped.Shape[0] != 3 || reshaped.Shape[1] != 2 {
		t.Fatalf("reshape shape = %v", reshaped.Shape)
	}
	reshaped.Data[0] = 99
	if source.Data[0] != 99 {
		t.Fatal("reshape must share data")
	}
	clone := source.Clone()
	clone.Data[0] = 1
	if source.Data[0] != 99 {
		t.Fatal("clone must not share data")
	}
}
