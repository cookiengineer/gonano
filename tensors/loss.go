package tensors

import (
	"math"

	"github.com/cookiengineer/gonano/internal/parallel"
)

// CrossEntropyPerPosition computes the per-position cross-entropy loss between
// logits (shape [rowCount, vocabularySize]) and targets (shape [rowCount]).
// Positions where target == ignoreIndex are skipped and receive a loss of zero.
// It returns the per-position losses and the number of valid positions.
//
// The computation uses the numerically stable log-sum-exp formulation:
// loss = logsumexp(row) - row[target].
func CrossEntropyPerPosition(logits *Tensor, targets *Int32s, ignoreIndex int32) ([]float32, int) {
	rowCount, vocabularySize := logits.Shape[0], logits.Shape[1]
	if targets.Numel() != rowCount {
		panic("tensors: CrossEntropy logits/targets row mismatch")
	}
	losses := make([]float32, rowCount)
	validCount := 0
	for _, target := range targets.Data {
		if target != ignoreIndex {
			validCount++
		}
	}

	backend := KernelBackend()
	parallel.Default().Chunks(0, rowCount, func(rowStart, rowEnd int) {
		for row := rowStart; row < rowEnd; row++ {
			target := targets.Data[row]
			if target == ignoreIndex {
				continue
			}
			rowLogits := logits.Data[row*vocabularySize : (row+1)*vocabularySize]
			maximum := backend.Max(rowLogits)
			var sum float64
			for _, value := range rowLogits {
				sum += math.Exp(float64(value - maximum))
			}
			losses[row] = maximum + float32(math.Log(sum)) - rowLogits[int(target)]
		}
	})
	return losses, validCount
}

// CrossEntropy returns the mean cross-entropy loss over valid positions. It
// returns zero when there are no valid positions.
func CrossEntropy(logits *Tensor, targets *Int32s, ignoreIndex int32) float32 {
	losses, validCount := CrossEntropyPerPosition(logits, targets, ignoreIndex)
	if validCount == 0 {
		return 0
	}
	var sum float64
	for _, loss := range losses {
		sum += float64(loss)
	}
	return float32(sum / float64(validCount))
}

// CrossEntropySum returns the summed cross-entropy loss over valid positions.
func CrossEntropySum(logits *Tensor, targets *Int32s, ignoreIndex int32) float32 {
	losses, _ := CrossEntropyPerPosition(logits, targets, ignoreIndex)
	var sum float64
	for _, loss := range losses {
		sum += float64(loss)
	}
	return float32(sum)
}

// DistillationLossPerPosition computes the per-position forward KL from the
// student to the teacher's soft distribution — the cross-entropy of the teacher
// distribution under the student:
//
//	loss = -sum_v softmax(teacher/temperature)_v * log softmax(student/temperature)_v
//
// The teacher entropy term is constant with respect to the student, so it is
// dropped. mask, when non-nil, marks valid positions with a non-zero value;
// masked positions receive a loss of zero. It returns the per-position losses
// and the number of valid positions.
//
// This is the on-policy distillation objective of DeepSeek-V4.1 §5.2.4.
func DistillationLossPerPosition(student, teacher *Tensor, temperature float32, mask *Int32s) ([]float32, int) {
	rowCount, vocabularySize := student.Shape[0], student.Shape[1]
	if teacher.Numel() != student.Numel() {
		panic("tensors: distillation student/teacher shape mismatch")
	}
	if mask != nil && mask.Numel() != rowCount {
		panic("tensors: distillation mask row mismatch")
	}
	tau := distillationTemperature(temperature)
	losses := make([]float32, rowCount)
	validCount := 0
	for row := 0; row < rowCount; row++ {
		if mask != nil && mask.Data[row] == 0 {
			continue
		}
		validCount++
		studentRow := student.Data[row*vocabularySize : (row+1)*vocabularySize]
		teacherRow := teacher.Data[row*vocabularySize : (row+1)*vocabularySize]
		losses[row] = distillationRowLoss(studentRow, teacherRow, tau)
	}
	return losses, validCount
}

// DistillationLoss returns the mean forward-KL distillation loss over valid
// positions. It returns zero when there are no valid positions.
func DistillationLoss(student, teacher *Tensor, temperature float32, mask *Int32s) float32 {
	losses, validCount := DistillationLossPerPosition(student, teacher, temperature, mask)
	if validCount == 0 {
		return 0
	}
	var sum float64
	for _, loss := range losses {
		sum += float64(loss)
	}
	return float32(sum / float64(validCount))
}

// DistillationGrad returns the gradient of the distillation loss with respect
// to the student logits, scaled by scale:
//
//	d(loss)/d(student) = (softmax(student/temperature) -
//	                      softmax(teacher/temperature)) / temperature
//
// Masked positions receive a zero gradient.
func DistillationGrad(student, teacher *Tensor, temperature float32, mask *Int32s, scale float32) *Tensor {
	rowCount, vocabularySize := student.Shape[0], student.Shape[1]
	if teacher.Numel() != student.Numel() {
		panic("tensors: distillation student/teacher shape mismatch")
	}
	if mask != nil && mask.Numel() != rowCount {
		panic("tensors: distillation mask row mismatch")
	}
	tau := distillationTemperature(temperature)
	gradient := New(rowCount, vocabularySize)
	parallel.Default().Chunks(0, rowCount, func(rowStart, rowEnd int) {
		for row := rowStart; row < rowEnd; row++ {
			if mask != nil && mask.Data[row] == 0 {
				continue
			}
			studentRow := student.Data[row*vocabularySize : (row+1)*vocabularySize]
			teacherRow := teacher.Data[row*vocabularySize : (row+1)*vocabularySize]
			gradientRow := gradient.Data[row*vocabularySize : (row+1)*vocabularySize]
			distillationRowGrad(studentRow, teacherRow, tau, scale, gradientRow)
		}
	})
	return gradient
}

// distillationTemperature clamps the temperature to a positive value.
func distillationTemperature(temperature float32) float64 {
	if temperature <= 0 {
		return 1
	}
	return float64(temperature)
}

// distillationRowLoss computes one row of the forward-KL loss from the student
// to the teacher in a numerically stable two-pass form.
func distillationRowLoss(studentRow, teacherRow []float32, tau float64) float32 {
	studentMax := math.Inf(-1)
	teacherMax := math.Inf(-1)
	for index := range studentRow {
		studentScaled := float64(studentRow[index]) / tau
		teacherScaled := float64(teacherRow[index]) / tau
		if studentScaled > studentMax {
			studentMax = studentScaled
		}
		if teacherScaled > teacherMax {
			teacherMax = teacherScaled
		}
	}
	var studentSum, teacherSum, weighted float64
	for index := range studentRow {
		studentScaled := float64(studentRow[index])/tau - studentMax
		teacherScaled := float64(teacherRow[index])/tau - teacherMax
		studentWeight := math.Exp(studentScaled)
		teacherWeight := math.Exp(teacherScaled)
		studentSum += studentWeight
		teacherSum += teacherWeight
		weighted += teacherWeight * (float64(studentRow[index]) / tau)
	}
	return float32(studentMax + math.Log(studentSum) - weighted/teacherSum)
}

// distillationRowGrad writes (softmax(student/tau) - softmax(teacher/tau))/tau
// scaled by scale into gradientRow.
func distillationRowGrad(studentRow, teacherRow []float32, tau float64, scale float32, gradientRow []float32) {
	studentMax := math.Inf(-1)
	teacherMax := math.Inf(-1)
	for index := range studentRow {
		if value := float64(studentRow[index]) / tau; value > studentMax {
			studentMax = value
		}
		if value := float64(teacherRow[index]) / tau; value > teacherMax {
			teacherMax = value
		}
	}
	var studentSum, teacherSum float64
	for index := range studentRow {
		studentSum += math.Exp(float64(studentRow[index])/tau - studentMax)
		teacherSum += math.Exp(float64(teacherRow[index])/tau - teacherMax)
	}
	inverseStudent := 1.0 / studentSum
	inverseTeacher := 1.0 / teacherSum
	gradientScale := float64(scale) / tau
	for index := range studentRow {
		studentProbability := math.Exp(float64(studentRow[index])/tau-studentMax) * inverseStudent
		teacherProbability := math.Exp(float64(teacherRow[index])/tau-teacherMax) * inverseTeacher
		gradientRow[index] = float32((studentProbability - teacherProbability) * gradientScale)
	}
}
