package tensors

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

func TestDistillationMatchesCrossEntropyForOneHotTeacher(t *testing.T) {
	logits := NewWithData([]int{2, 4}, []float32{
		0.5, -1, 2, 0.3,
		-0.2, 1.5, -0.5, 0.9,
	})
	// A large target logit makes the teacher distribution effectively one-hot.
	teacher := NewWithData([]int{2, 4}, []float32{
		0, 0, 100, 0,
		0, 100, 0, 0,
	})
	targets := NewInt32sWithData([]int{2}, []int32{2, 1})

	got := DistillationLoss(logits, teacher, 1, nil)
	want := CrossEntropy(logits, targets, -1)
	if math.Abs(float64(got-want)) > 1e-3 {
		t.Fatalf("distillation loss = %v, want cross-entropy %v", got, want)
	}

	gradGot := DistillationGrad(logits, teacher, 1, nil, 1)
	gradWant := CrossEntropyGrad(logits, targets, -1, 1)
	for index := range gradWant.Data {
		if math.Abs(float64(gradGot.Data[index]-gradWant.Data[index])) > 1e-3 {
			t.Fatalf("distillation grad[%d] = %v, want %v", index, gradGot.Data[index], gradWant.Data[index])
		}
	}
}

func TestDistillationGradMatchesNumeric(t *testing.T) {
	student := NewWithData([]int{1, 3}, []float32{0.3, -0.7, 1.1})
	teacher := NewWithData([]int{1, 3}, []float32{0.8, 0.2, -0.4})
	gradient := DistillationGrad(student, teacher, 1, nil, 1)

	const epsilon = 1e-3
	for index := range student.Data {
		original := student.Data[index]
		student.Data[index] = original + epsilon
		plus, _ := DistillationLossPerPosition(student, teacher, 1, nil)
		student.Data[index] = original - epsilon
		minus, _ := DistillationLossPerPosition(student, teacher, 1, nil)
		student.Data[index] = original
		numeric := float64(plus[0]-minus[0]) / (2 * epsilon)
		if math.Abs(numeric-float64(gradient.Data[index])) > 1e-3 {
			t.Fatalf("grad[%d] = %v, numeric = %v", index, gradient.Data[index], numeric)
		}
	}
}

func TestDistillationSelfIsZeroGrad(t *testing.T) {
	logits := NewWithData([]int{2, 4}, testData(8))
	gradient := DistillationGrad(logits, logits.Clone(), 1, nil, 1)
	for index, value := range gradient.Data {
		if math.Abs(float64(value)) > 1e-6 {
			t.Fatalf("self-distillation grad[%d] = %v, want 0", index, value)
		}
	}
}

func TestDistillationMaskZeroesPositions(t *testing.T) {
	student := NewWithData([]int{2, 3}, []float32{0.1, 0.2, 0.3, 1, 2, 3})
	teacher := NewWithData([]int{2, 3}, []float32{3, 2, 1, -1, -2, -3})
	mask := NewInt32sWithData([]int{2}, []int32{0, 1})

	losses, valid := DistillationLossPerPosition(student, teacher, 1, mask)
	if valid != 1 {
		t.Fatalf("valid = %d, want 1", valid)
	}
	if losses[0] != 0 {
		t.Fatalf("masked loss = %v, want 0", losses[0])
	}
	if losses[1] == 0 {
		t.Fatal("unmasked loss must be non-zero")
	}
	gradient := DistillationGrad(student, teacher, 1, mask, 1)
	for index := 0; index < 3; index++ {
		if gradient.Data[index] != 0 {
			t.Fatalf("masked gradient[%d] = %v, want 0", index, gradient.Data[index])
		}
	}
}

func TestDistillationTemperatureSoftensGrad(t *testing.T) {
	student := NewWithData([]int{1, 3}, []float32{2, -1, 0.5})
	teacher := NewWithData([]int{1, 3}, []float32{-0.5, 1, 2})
	sharp := DistillationGrad(student, teacher, 1, nil, 1)
	soft := DistillationGrad(student, teacher, 4, nil, 1)
	sharpNorm, softNorm := 0.0, 0.0
	for index := range sharp.Data {
		sharpNorm += math.Abs(float64(sharp.Data[index]))
		softNorm += math.Abs(float64(soft.Data[index]))
	}
	if softNorm >= sharpNorm {
		t.Fatalf("higher temperature should shrink the gradient: sharp=%v soft=%v", sharpNorm, softNorm)
	}
}
