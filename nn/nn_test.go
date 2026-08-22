package nn

import (
	"testing"

	"github.com/cookiengineer/gonano/tensor"
)

func TestLinearForward2D(t *testing.T) {
	l := NewLinear(3, 2)
	// Set weight rows to known values.
	l.Weight.Data = []float32{
		1, 0, 0,
		0, 1, 0,
	}
	x := tensor.NewWithData([]int{4, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	})
	y := l.Forward(x)
	if y.Shape[0] != 4 || y.Shape[1] != 2 {
		t.Fatalf("shape = %v, want [4 2]", y.Shape)
	}
	// y = x @ W^T with W = [[1,0,0],[0,1,0]] -> y[:,0] = x[:,0], y[:,1] = x[:,1].
	for i := 0; i < 4; i++ {
		if y.Get2(i, 0) != x.Get2(i, 0) || y.Get2(i, 1) != x.Get2(i, 1) {
			t.Fatalf("row %d = %v, want %v", i, []float32{y.Get2(i, 0), y.Get2(i, 1)}, []float32{x.Get2(i, 0), x.Get2(i, 1)})
		}
	}
}

func TestLinearForward3D(t *testing.T) {
	l := NewLinear(2, 2)
	l.Weight.Data = []float32{1, 1, 1, 1} // both outputs sum both inputs
	x := tensor.NewWithData([]int{2, 3, 2}, []float32{
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12,
	})
	y := l.Forward(x)
	if y.Shape[0] != 2 || y.Shape[1] != 3 || y.Shape[2] != 2 {
		t.Fatalf("shape = %v, want [2 3 2]", y.Shape)
	}
	// Each output = sum of the two inputs.
	for b := 0; b < 2; b++ {
		for i := 0; i < 3; i++ {
			want := x.Get3(b, i, 0) + x.Get3(b, i, 1)
			if y.Get3(b, i, 0) != want || y.Get3(b, i, 1) != want {
				t.Fatalf("(%d,%d) = %v, want %v", b, i, []float32{y.Get3(b, i, 0), y.Get3(b, i, 1)}, want)
			}
		}
	}
}

func TestLinearForwardMatchesMatMulTransB(t *testing.T) {
	rng := tensor.NewRNG(1)
	l := NewLinear(7, 5)
	tensor.FillNormal(l.Weight, rng, 1)
	x := tensor.New(11, 7)
	tensor.FillNormal(x, rng, 1)
	got := l.Forward(x)
	want := tensor.MatMulTransB(x, l.Weight)
	for i := range got.Data {
		if got.Data[i] != want.Data[i] {
			t.Fatalf("mismatch at %d", i)
		}
	}
}

func TestEmbeddingForward(t *testing.T) {
	e := NewEmbedding(5, 3)
	e.Weight.Data = []float32{
		0, 0, 0,
		1, 1, 1,
		2, 2, 2,
		3, 3, 3,
		4, 4, 4,
	}
	idx := tensor.NewInt32sWithData([]int{2, 2}, []int32{3, 1, 0, 4})
	out := e.Forward(idx)
	if out.Shape[0] != 2 || out.Shape[1] != 2 || out.Shape[2] != 3 {
		t.Fatalf("shape = %v, want [2 2 3]", out.Shape)
	}
	wantIDs := []int32{3, 1, 0, 4}
	for i, id := range wantIDs {
		for d := 0; d < 3; d++ {
			if out.Data[i*3+d] != float32(id) {
				t.Fatalf("embed[%d][%d] = %v, want %d", i, d, out.Data[i*3+d], id)
			}
		}
	}
}

func TestInitZeros(t *testing.T) {
	w := tensor.New(4, 4)
	InitZeros(w)
	for _, v := range w.Data {
		if v != 0 {
			t.Fatal("InitZeros left nonzero element")
		}
	}
}

func TestInitUniformBounds(t *testing.T) {
	rng := tensor.NewRNG(2)
	w := tensor.New(1000)
	InitUniform(w, rng, -1.5, 2.5)
	for _, v := range w.Data {
		if v < -1.5 || v >= 2.5 {
			t.Fatalf("uniform sample %v out of bounds", v)
		}
	}
}

func TestStdToUniformBound(t *testing.T) {
	// std = b/sqrt(3) -> b = sqrt(3)*std.
	if stdToUniformBound(1.0) != 1.7320508075688772 {
		t.Fatalf("sqrt(3) = %v", stdToUniformBound(1.0))
	}
}
