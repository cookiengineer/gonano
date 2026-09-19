package tensors

import (
	"math"
	"testing"
)

func TestRNGDeterminism(t *testing.T) {
	a := NewRNG(42)
	b := NewRNG(42)
	for i := 0; i < 100; i++ {
		if a.Float32() != b.Float32() {
			t.Fatal("same seed must produce identical sequences")
		}
	}
}

func TestRNGDifferentSeeds(t *testing.T) {
	a := NewRNG(1)
	b := NewRNG(2)
	same := true
	for i := 0; i < 100; i++ {
		if a.Float32() != b.Float32() {
			same = false
			break
		}
	}
	if same {
		t.Fatal("different seeds should diverge")
	}
}

func TestFillNormalStatistics(t *testing.T) {
	rng := NewRNG(7)
	n := 100_000
	x := New(n)
	FillNormal(x, rng, 1.0)
	var mean, varSum float64
	for _, v := range x.Data {
		mean += float64(v)
	}
	mean /= float64(n)
	for _, v := range x.Data {
		d := float64(v) - mean
		varSum += d * d
	}
	std := math.Sqrt(varSum / float64(n))
	if math.Abs(mean) > 0.02 {
		t.Fatalf("normal mean = %v, want ~0", mean)
	}
	if math.Abs(std-1.0) > 0.02 {
		t.Fatalf("normal std = %v, want ~1", std)
	}
}

func TestFillUniformBounds(t *testing.T) {
	rng := NewRNG(3)
	x := New(10000)
	FillUniform(x, rng, -2, 3)
	for _, v := range x.Data {
		if v < -2 || v >= 3 {
			t.Fatalf("uniform sample %v out of bounds", v)
		}
	}
}

func TestSampleMultinomialArgmax(t *testing.T) {
	rng := NewRNG(1)
	logits := []float32{1, 9, 3, 5}
	if got := SampleMultinomial(rng, logits, 0); got != 1 {
		t.Fatalf("temperature 0 should argmax: got %d want 1", got)
	}
}

func TestSampleMultinomialRespectsDistribution(t *testing.T) {
	rng := NewRNG(42)
	logits := []float32{0, 10, 0} // middle token dominates
	counts := make([]int, 3)
	const trials = 10000
	for i := 0; i < trials; i++ {
		counts[SampleMultinomial(rng, logits, 1.0)]++
	}
	if counts[1] < trials/2 {
		t.Fatalf("dominant token sampled %d/%d times, want majority", counts[1], trials)
	}
}

func TestSumAllMatchesScalar(t *testing.T) {
	values := testData(5000)
	var want float64
	for _, value := range values {
		want += float64(value)
	}
	source := NewWithData([]int{5000}, values)
	if !close(SumAll(source), float32(want), 1e-4, 1e-5) {
		t.Fatalf("SumAll = %v want %v", SumAll(source), want)
	}
}

func TestMaxAllMatchesScalar(t *testing.T) {
	values := testData(5000)
	want := float32(math.Inf(-1))
	for _, value := range values {
		if value > want {
			want = value
		}
	}
	source := NewWithData([]int{5000}, values)
	if MaxAll(source) != want {
		t.Fatalf("MaxAll = %v want %v", MaxAll(source), want)
	}
}
