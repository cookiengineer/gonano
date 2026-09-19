package tensors

import (
	"math"
	"testing"
)

// close reports whether got and want are within rel relative tolerance with an
// absolute floor of absFloor.
func close(got, want, rel, absFloor float32) bool {
	d := float32(math.Abs(float64(got - want)))
	bound := rel * (1 + float32(math.Abs(float64(want))))
	if bound < absFloor {
		bound = absFloor
	}
	return d <= bound
}

func assertTensorClose(t *testing.T, got, want *Tensor, rel, absFloor float32) {
	t.Helper()
	if len(got.Shape) != len(want.Shape) {
		t.Fatalf("shape rank %v != %v", got.Shape, want.Shape)
	}
	for i := range got.Shape {
		if got.Shape[i] != want.Shape[i] {
			t.Fatalf("shape %v != %v", got.Shape, want.Shape)
		}
	}
	for i := range got.Data {
		if !close(got.Data[i], want.Data[i], rel, absFloor) {
			t.Fatalf("mismatch at %d: got %v want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestNewAndNumel(t *testing.T) {
	x := New(2, 3, 4)
	if x.Numel() != 24 {
		t.Fatalf("Numel = %d, want 24", x.Numel())
	}
	if x.Rank() != 3 {
		t.Fatalf("Rank = %d, want 3", x.Rank())
	}
}

func TestNewInvalidShape(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for invalid shape")
		}
	}()
	New(2, 0)
}

func TestReshape(t *testing.T) {
	x := New(2, 6)
	for i := range x.Data {
		x.Data[i] = float32(i)
	}
	y := x.Reshape(3, 4)
	if y.Get2(0, 3) != 3 {
		t.Fatalf("reshape value = %v, want 3", y.Get2(0, 3))
	}
	// Shares data.
	y.Set2(0, 0, 99)
	if x.Data[0] != 99 {
		t.Fatal("Reshape must share underlying data")
	}
}

func TestIndexing4D(t *testing.T) {
	x := New(2, 3, 4, 5)
	x.Set4(1, 2, 3, 4, 42)
	if x.Get4(1, 2, 3, 4) != 42 {
		t.Fatal("Set4/Get4 roundtrip failed")
	}
}

func TestStrides(t *testing.T) {
	got := Strides([]int{2, 3, 4})
	want := []int{12, 4, 1}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("strides = %v, want %v", got, want)
		}
	}
}

func TestOffset(t *testing.T) {
	x := New(2, 3, 4)
	if x.Offset(1, 2, 3) != 1*12+2*4+3 {
		t.Fatalf("Offset = %d, want %d", x.Offset(1, 2, 3), 1*12+2*4+3)
	}
}

func TestInt32sBasic(t *testing.T) {
	x := NewInt32s(2, 3)
	x.Set2(1, 2, 7)
	if x.Get2(1, 2) != 7 {
		t.Fatal("Int32s Set2/Get2 failed")
	}
	if x.Numel() != 6 {
		t.Fatalf("Numel = %d, want 6", x.Numel())
	}
}
