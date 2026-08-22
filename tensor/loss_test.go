package tensor

import (
	"math"
	"testing"
)

func TestCrossEntropyPerPosition(t *testing.T) {
	// Two valid positions (targets 2 and 0) and one ignored (-1).
	logits := NewWithData([]int{3, 4}, []float32{
		0, 0, 5, 0, // position 0: target 2
		1, 1, 1, 1, // position 1: ignored
		2, 0, 0, 0, // position 2: target 0
	})
	targets := NewInt32sWithData([]int{3}, []int32{2, -1, 0})
	losses, valid := CrossEntropyPerPosition(logits, targets, -1)
	if valid != 2 {
		t.Fatalf("valid = %d, want 2", valid)
	}
	if losses[1] != 0 {
		t.Fatalf("ignored loss = %v, want 0", losses[1])
	}
	// loss = log(sum(exp(row))) - row[target], computed independently.
	want0 := float32(math.Log(math.Exp(5)+3) - 5)
	want2 := float32(math.Log(math.Exp(2)+3) - 2)
	if math.Abs(float64(losses[0]-want0)) > 1e-5 {
		t.Fatalf("loss[0] = %v, want %v", losses[0], want0)
	}
	if math.Abs(float64(losses[2]-want2)) > 1e-5 {
		t.Fatalf("loss[2] = %v, want %v", losses[2], want2)
	}
}

func TestCrossEntropyKnownValue(t *testing.T) {
	// Single position, uniform logits over vocab 4 -> loss = ln(4).
	logits := NewWithData([]int{1, 4}, []float32{1, 1, 1, 1})
	targets := NewInt32sWithData([]int{1}, []int32{0})
	loss := CrossEntropy(logits, targets, -1)
	want := float32(math.Log(4))
	if math.Abs(float64(loss-want)) > 1e-5 {
		t.Fatalf("loss = %v, want %v", loss, want)
	}
}

func TestCrossEntropySum(t *testing.T) {
	logits := NewWithData([]int{2, 4}, []float32{
		1, 1, 1, 1,
		1, 1, 1, 1,
	})
	targets := NewInt32sWithData([]int{2}, []int32{0, 1})
	sum := CrossEntropySum(logits, targets, -1)
	want := 2 * float32(math.Log(4))
	if math.Abs(float64(sum-want)) > 1e-4 {
		t.Fatalf("sum = %v, want %v", sum, want)
	}
}

func TestCrossEntropyAllIgnored(t *testing.T) {
	logits := NewWithData([]int{2, 4}, []float32{1, 1, 1, 1, 1, 1, 1, 1})
	targets := NewInt32sWithData([]int{2}, []int32{-1, -1})
	if got := CrossEntropy(logits, targets, -1); got != 0 {
		t.Fatalf("all-ignored mean = %v, want 0", got)
	}
}
