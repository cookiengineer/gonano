package layers

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/tensors"
)

func TestLinearBackward(test *testing.T) {
	layer := NewLinear(3, 2)
	layer.Weight.Data = []float32{
		1, 0, 0,
		0, 1, 0,
	}
	input := tensors.NewWithData([]int{1, 3}, []float32{2, 3, 4})
	gradientOutput := tensors.NewWithData([]int{1, 2}, []float32{5, 6})

	gradientInput := layer.Backward(input, gradientOutput)

	// gradientInput = gradientOutput @ Weight = [5, 6, 0].
	wantInput := []float32{5, 6, 0}
	for index := range wantInput {
		if math.Abs(float64(gradientInput.Data[index]-wantInput[index])) > 1e-5 {
			test.Fatalf("gradientInput[%d] = %v, want %v", index, gradientInput.Data[index], wantInput[index])
		}
	}

	// Weight.Grad = gradientOutput^T @ input.
	wantWeight := []float32{10, 15, 20, 12, 18, 24}
	for index := range wantWeight {
		if math.Abs(float64(layer.Weight.Grad[index]-wantWeight[index])) > 1e-5 {
			test.Fatalf("Weight.Grad[%d] = %v, want %v", index, layer.Weight.Grad[index], wantWeight[index])
		}
	}
}

func TestEmbeddingBackwardAccumulatesRepeatedIDs(test *testing.T) {
	embedding := NewEmbedding(3, 2)
	indices := tensors.NewInt32sWithData([]int{3}, []int32{1, 1, 2})
	gradientOutput := tensors.NewWithData([]int{3, 2}, []float32{
		1, 1,
		2, 2,
		3, 3,
	})
	embedding.Backward(indices, gradientOutput)

	// Row 1 receives the two gradients for the repeated id; row 2 receives one.
	want := []float32{0, 0, 3, 3, 3, 3}
	for index := range want {
		if math.Abs(float64(embedding.Weight.Grad[index]-want[index])) > 1e-5 {
			test.Fatalf("Weight.Grad[%d] = %v, want %v", index, embedding.Weight.Grad[index], want[index])
		}
	}
}

func TestInitNormalStatistics(test *testing.T) {
	rng := tensors.NewRNG(11)
	weight := tensors.New(20000)
	InitNormal(weight, rng, 0.5)
	var mean float64
	for _, value := range weight.Data {
		mean += float64(value)
	}
	mean /= float64(len(weight.Data))
	if math.Abs(mean) > 0.02 {
		test.Fatalf("normal mean = %v, want ~0", mean)
	}
}

func TestInitValue(test *testing.T) {
	weight := tensors.New(3)
	InitValue(weight, 7.5)
	for _, value := range weight.Data {
		if value != 7.5 {
			test.Fatalf("InitValue = %v, want 7.5", value)
		}
	}
}
