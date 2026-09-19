package layers

import (
	"testing"

	"github.com/cookiengineer/gonano/tensors"
)

func TestLinearForward2D(test *testing.T) {
	layer := NewLinear(3, 2)
	// Set weight rows to known values.
	layer.Weight.Data = []float32{
		1, 0, 0,
		0, 1, 0,
	}
	input := tensors.NewWithData([]int{4, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	})
	output := layer.Forward(input)
	if output.Shape[0] != 4 || output.Shape[1] != 2 {
		test.Fatalf("shape = %v, want [4 2]", output.Shape)
	}
	// output = input @ W^T with W = [[1,0,0],[0,1,0]] -> output[:,0] = input[:,0], output[:,1] = input[:,1].
	for row := 0; row < 4; row++ {
		if output.Get2(row, 0) != input.Get2(row, 0) || output.Get2(row, 1) != input.Get2(row, 1) {
			test.Fatalf("row %d = %v, want %v", row, []float32{output.Get2(row, 0), output.Get2(row, 1)}, []float32{input.Get2(row, 0), input.Get2(row, 1)})
		}
	}
}

func TestLinearForward3D(test *testing.T) {
	layer := NewLinear(2, 2)
	layer.Weight.Data = []float32{1, 1, 1, 1} // both outputs sum both inputs
	input := tensors.NewWithData([]int{2, 3, 2}, []float32{
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12,
	})
	output := layer.Forward(input)
	if output.Shape[0] != 2 || output.Shape[1] != 3 || output.Shape[2] != 2 {
		test.Fatalf("shape = %v, want [2 3 2]", output.Shape)
	}
	// Each output = sum of the two inputs.
	for batch := 0; batch < 2; batch++ {
		for row := 0; row < 3; row++ {
			want := input.Get3(batch, row, 0) + input.Get3(batch, row, 1)
			if output.Get3(batch, row, 0) != want || output.Get3(batch, row, 1) != want {
				test.Fatalf("(%d,%d) = %v, want %v", batch, row, []float32{output.Get3(batch, row, 0), output.Get3(batch, row, 1)}, want)
			}
		}
	}
}

func TestLinearForwardMatchesMatMulTransB(test *testing.T) {
	rng := tensors.NewRNG(1)
	layer := NewLinear(7, 5)
	tensors.FillNormal(layer.Weight, rng, 1)
	input := tensors.New(11, 7)
	tensors.FillNormal(input, rng, 1)
	got := layer.Forward(input)
	want := tensors.MatMulTransposed(input, layer.Weight)
	for index := range got.Data {
		if got.Data[index] != want.Data[index] {
			test.Fatalf("mismatch at %d", index)
		}
	}
}

func TestEmbeddingForward(test *testing.T) {
	embedding := NewEmbedding(5, 3)
	embedding.Weight.Data = []float32{
		0, 0, 0,
		1, 1, 1,
		2, 2, 2,
		3, 3, 3,
		4, 4, 4,
	}
	indices := tensors.NewInt32sWithData([]int{2, 2}, []int32{3, 1, 0, 4})
	output := embedding.Forward(indices)
	if output.Shape[0] != 2 || output.Shape[1] != 2 || output.Shape[2] != 3 {
		test.Fatalf("shape = %v, want [2 2 3]", output.Shape)
	}
	wantIDs := []int32{3, 1, 0, 4}
	for index, tokenID := range wantIDs {
		for column := 0; column < 3; column++ {
			if output.Data[index*3+column] != float32(tokenID) {
				test.Fatalf("embed[%d][%d] = %v, want %d", index, column, output.Data[index*3+column], tokenID)
			}
		}
	}
}

func TestInitZeros(test *testing.T) {
	weight := tensors.New(4, 4)
	InitZeros(weight)
	for _, value := range weight.Data {
		if value != 0 {
			test.Fatal("InitZeros left nonzero element")
		}
	}
}

func TestInitUniformBounds(test *testing.T) {
	rng := tensors.NewRNG(2)
	weight := tensors.New(1000)
	InitUniform(weight, rng, -1.5, 2.5)
	for _, value := range weight.Data {
		if value < -1.5 || value >= 2.5 {
			test.Fatalf("uniform sample %v out of bounds", value)
		}
	}
}

func TestStdToUniformBound(test *testing.T) {
	// std = bound/sqrt(3) -> bound = sqrt(3)*std.
	if stdToUniformBound(1.0) != 1.7320508075688772 {
		test.Fatalf("sqrt(3) = %v", stdToUniformBound(1.0))
	}
}
