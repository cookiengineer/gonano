package trainer

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
)

// newDistillPair builds a student and an independently initialized teacher with
// non-zero projections so their logits differ at the start.
func newDistillPair() (*model.Transformer, *model.Transformer) {
	config := model.Config{
		SequenceLen: 16, VocabSize: 16, NumLayer: 1, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	student := model.NewTransformer(config)
	student.InitWeights(tensors.NewRNG(1))
	teacher := model.NewTransformer(config)
	teacher.InitWeights(tensors.NewRNG(2))
	breakZeroProjections(student, 7)
	breakZeroProjections(teacher, 9)
	return student, teacher
}

func breakZeroProjections(model *model.Transformer, seed uint64) {
	perturb := tensors.NewRNG(seed)
	for _, parameter := range model.Parameters() {
		for index := range parameter.Data {
			parameter.Data[index] += perturb.NormFloat32() * 0.2
		}
	}
}

func distillBatch() (*tensors.Int32s, *tensors.Int32s) {
	inputs := tensors.NewInt32sWithData([]int{1, 8}, []int32{1, 2, 3, 4, 5, 6, 7, 8})
	mask := tensors.NewInt32sWithData([]int{1, 8}, []int32{1, 1, 1, 1, 1, 1, 1, 1})
	return inputs, mask
}

func TestDistillStepReducesLoss(t *testing.T) {
	student, teacher := newDistillPair()
	groups := student.SetupOptimizer(0.01, 0.1, 0.01, 0.0, 0.1, false)
	trainer := NewTrainer(student, groups, 1)
	trainer.WarmupSteps = 0
	trainer.WarmdownRatio = 0.0

	inputs, mask := distillBatch()
	first := trainer.DistillStep(teacher, inputs, mask, 1)
	trainer.StepOptimizer(0, 100)
	var last float32
	for step := 1; step < 100; step++ {
		last = trainer.DistillStep(teacher, inputs, mask, 1)
		trainer.StepOptimizer(step, 100)
	}
	if math.IsNaN(float64(last)) || math.IsInf(float64(last), 0) {
		t.Fatalf("loss became non-finite: %v", last)
	}
	if last >= first {
		t.Fatalf("distillation loss did not decrease: %v -> %v", first, last)
	}
}

func TestDistillSelfZeroGradient(t *testing.T) {
	student, _ := newDistillPair()
	groups := student.SetupOptimizer(0.01, 0.1, 0.01, 0.0, 0.1, false)
	trainer := NewTrainer(student, groups, 1)
	student.ZeroGrad()

	inputs, mask := distillBatch()
	trainer.DistillStep(student, inputs, mask, 1)

	for _, parameter := range student.Parameters() {
		for index, value := range parameter.Grad {
			if math.Abs(float64(value)) > 1e-4 {
				t.Fatalf("self-distillation gradient[%d] = %v, want 0", index, value)
			}
		}
	}
}

func TestDistillMaskRestrictsGradient(t *testing.T) {
	student, teacher := newDistillPair()
	groups := student.SetupOptimizer(0.01, 0.1, 0.01, 0.0, 0.1, false)
	trainer := NewTrainer(student, groups, 1)
	student.ZeroGrad()

	inputs := tensors.NewInt32sWithData([]int{1, 8}, []int32{1, 2, 3, 4, 5, 6, 7, 8})
	zeroMask := tensors.NewInt32sWithData([]int{1, 8}, []int32{0, 0, 0, 0, 0, 0, 0, 0})
	if loss := trainer.DistillStep(teacher, inputs, zeroMask, 1); loss != 0 {
		t.Fatalf("all-masked loss = %v, want 0", loss)
	}
	for _, parameter := range student.Parameters() {
		for index, value := range parameter.Grad {
			if value != 0 {
				t.Fatalf("all-masked gradient[%d] = %v, want 0", index, value)
			}
		}
	}
}

func TestDistillLoop(t *testing.T) {
	student, teacher := newDistillPair()
	groups := student.SetupOptimizer(0.01, 0.1, 0.01, 0.0, 0.1, false)
	inputs, mask := distillBatch()
	provider := func() (*tensors.Int32s, *tensors.Int32s, bool) { return inputs, mask, true }

	losses := Distill(student, teacher, groups, provider, 5, 1, nil)
	if len(losses) != 5 {
		t.Fatalf("losses = %d, want 5", len(losses))
	}
	for _, loss := range losses {
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("non-finite loss %v", loss)
		}
	}
}
