package tensors

import (
	"math"
	"testing"
)

func TestSumAll(t *testing.T) {
	x := NewWithData([]int{100}, testData(100))
	var want float64
	for _, v := range x.Data {
		want += float64(v)
	}
	if !close(SumAll(x), float32(want), 1e-5, 1e-6) {
		t.Fatalf("SumAll = %v want %v", SumAll(x), want)
	}
}

func TestMaxAll(t *testing.T) {
	x := NewWithData([]int{100}, testData(100))
	var want float32 = float32(math.Inf(-1))
	for _, v := range x.Data {
		if v > want {
			want = v
		}
	}
	if MaxAll(x) != want {
		t.Fatalf("MaxAll = %v want %v", MaxAll(x), want)
	}
}

func TestArgMax(t *testing.T) {
	x := NewWithData([]int{5}, []float32{1, 5, 3, 9, 2})
	if ArgMax(x) != 3 {
		t.Fatalf("ArgMax = %d, want 3", ArgMax(x))
	}
}

func TestSumLastDim(t *testing.T) {
	x := NewWithData([]int{3, 4}, []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	})
	got := SumLastDim(x)
	want := []float32{10, 26, 42}
	for i := range want {
		if got.Data[i] != want[i] {
			t.Fatalf("row %d sum = %v want %v", i, got.Data[i], want[i])
		}
	}
}

func TestArgMaxLastDim(t *testing.T) {
	x := NewWithData([]int{3, 4}, []float32{
		1, 9, 3, 4,
		5, 6, 7, 8,
		12, 10, 11, 2,
	})
	got := ArgMaxLastDim(x)
	want := []int32{1, 3, 0}
	for i := range want {
		if got.Data[i] != want[i] {
			t.Fatalf("row %d argmax = %d want %d", i, got.Data[i], want[i])
		}
	}
}

func TestSoftmaxLastDim(t *testing.T) {
	x := NewWithData([]int{2, 4}, []float32{1, 2, 3, 4, 1, 1, 1, 1})
	got := SoftmaxLastDim(x)
	for r := 0; r < 2; r++ {
		var sum float64
		for c := 0; c < 4; c++ {
			sum += float64(got.Get2(r, c))
		}
		if math.Abs(sum-1.0) > 1e-5 {
			t.Fatalf("row %d softmax sums to %v, want 1.0", r, sum)
		}
	}
	// Row 0: increasing logits -> increasing probabilities.
	if got.Get2(0, 0) >= got.Get2(0, 3) {
		t.Fatal("softmax should be monotonically increasing for increasing logits")
	}
	// Row 1: equal logits -> uniform.
	for c := 0; c < 4; c++ {
		if !close(got.Get2(1, c), 0.25, 1e-4, 1e-6) {
			t.Fatalf("uniform softmax[%d] = %v want 0.25", c, got.Get2(1, c))
		}
	}
}

func TestSoftmaxStableWithLargeValues(t *testing.T) {
	// Values that would overflow naive exp.
	x := NewWithData([]int{1, 3}, []float32{1000, 1000, 1000})
	got := SoftmaxLastDim(x)
	for c := 0; c < 3; c++ {
		if !close(got.Data[c], 1.0/3.0, 1e-4, 1e-6) {
			t.Fatalf("softmax[%d] = %v want 1/3", c, got.Data[c])
		}
	}
}

func TestRMSNormLastDim(t *testing.T) {
	// A row of [1,2,2] has rms = sqrt((1+4+4)/3) = sqrt(3).
	x := NewWithData([]int{1, 3}, []float32{1, 2, 2})
	got := RMSNormLastDim(x, 1e-6)
	want := 1.0 / math.Sqrt(3.0)
	for c := 0; c < 3; c++ {
		expected := float32(float64(x.Data[c]) * want)
		if !close(got.Data[c], expected, 1e-4, 1e-6) {
			t.Fatalf("rmsnorm[%d] = %v want %v", c, got.Data[c], expected)
		}
	}
}

func TestRMSNormUnitNorm(t *testing.T) {
	x := NewWithData([]int{1, 64}, testData(64))
	got := RMSNormLastDim(x, 1e-6)
	var sumSq float64
	for _, v := range got.Data {
		sumSq += float64(v) * float64(v)
	}
	rms := math.Sqrt(sumSq / 64)
	if math.Abs(rms-1.0) > 1e-4 {
		t.Fatalf("rms = %v want 1.0", rms)
	}
}

func TestSigmoidTanhSoftcap(t *testing.T) {
	x := NewWithData([]int{3}, []float32{0, 1, -1})
	s := Sigmoid(x)
	if !close(s.Data[0], 0.5, 1e-5, 1e-6) || !close(s.Data[1], 0.731058, 1e-5, 1e-6) {
		t.Fatalf("sigmoid = %v", s.Data)
	}
	th := Tanh(x)
	if !close(th.Data[0], 0, 1e-5, 1e-6) || !close(th.Data[1], 0.761594, 1e-5, 1e-6) {
		t.Fatalf("tanh = %v", th.Data)
	}
	sc := Softcap(NewWithData([]int{1}, []float32{7.5}), 15)
	if !close(sc.Data[0], 15*float32(math.Tanh(0.5)), 1e-4, 1e-5) {
		t.Fatalf("softcap = %v", sc.Data[0])
	}
}
