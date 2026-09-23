package tensors

import "math"

// Backward primitives for the operations used by the transformer. These are
// the analytic gradients, verified against finite differences in tests.

// CrossEntropyGrad returns the gradient of the (scaled) mean cross-entropy
// loss with respect to the logits: (softmax(logits) - onehot(target)) * scale,
// with ignored positions zeroed.
func CrossEntropyGrad(logits *Tensor, targets *Int32s, ignoreIndex int32, scale float32) *Tensor {
	rowCount, vocabularySize := logits.Shape[0], logits.Shape[1]
	gradient := New(rowCount, vocabularySize)
	gradientData := gradient.Data
	backend := KernelBackend()
	for row := 0; row < rowCount; row++ {
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
		inverseSum := 1.0 / sum
		base := row * vocabularySize
		for column := 0; column < vocabularySize; column++ {
			probability := float32(math.Exp(float64(rowLogits[column]-maximum)) * inverseSum)
			if int32(column) == target {
				probability -= 1
			}
			gradientData[base+column] = probability * scale
		}
	}
	return gradient
}

// MatMulTransposedBackward returns the gradients of left@right^T:
// gradLeft = gradOut @ right and gradRight = gradOut^T @ left.
func MatMulTransposedBackward(gradOut, left, right *Tensor) (gradLeft, gradRight *Tensor) {
	gradLeft = MatMul(gradOut, right)
	gradRight = MatMul(Transpose(gradOut), left)
	return gradLeft, gradRight
}

// MatMulBackward returns the gradients of left@right:
// gradLeft = gradOut @ right^T and gradRight = left^T @ gradOut.
func MatMulBackward(gradOut, left, right *Tensor) (gradLeft, gradRight *Tensor) {
	gradLeft = MatMulTransposed(gradOut, right)
	gradRight = MatMul(Transpose(left), gradOut)
	return gradLeft, gradRight
}

// ReluSquaredBackward returns gradOut * 2*relu(input).
func ReluSquaredBackward(input, gradOut *Tensor) *Tensor {
	result := New(input.Shape...)
	for index, value := range input.Data {
		if value > 0 {
			result.Data[index] = gradOut.Data[index] * 2 * value
		}
	}
	return result
}

// SwiGLUBackward returns the gradients of the clamped SwiGLU with respect to
// gate and up, given the forward operands and the output gradient.
func SwiGLUBackward(gate, up, gradOut *Tensor, clamp float32) (gradGate, gradUp *Tensor) {
	requireSameShape(gate, up)
	requireSameShape(gate, gradOut)
	gradGate = New(gate.Shape...)
	gradUp = New(gate.Shape...)
	KernelBackend().SwiGLUBackward(gradGate.Data, gradUp.Data, gate.Data, up.Data, gradOut.Data, clamp)
	return gradGate, gradUp
}

// RMSNormBackward returns the gradient of RMSNorm applied over the last
// dimension. With y = x / r and r = sqrt(mean(x^2)+epsilon):
// grad_x = grad_y/r - x * (dot(grad_y, x) / (D * r^3)).
func RMSNormBackward(input, gradOut *Tensor, epsilon float32) *Tensor {
	rowCount := input.Numel() / input.Shape[len(input.Shape)-1]
	rowWidth := input.Shape[len(input.Shape)-1]
	result := New(input.Shape...)
	for row := 0; row < rowCount; row++ {
		inputRow := input.Data[row*rowWidth : (row+1)*rowWidth]
		gradientRow := gradOut.Data[row*rowWidth : (row+1)*rowWidth]
		resultRow := result.Data[row*rowWidth : (row+1)*rowWidth]
		var sumSquares float64
		var dotProduct float64
		for column := 0; column < rowWidth; column++ {
			sumSquares += float64(inputRow[column]) * float64(inputRow[column])
			dotProduct += float64(gradientRow[column]) * float64(inputRow[column])
		}
		meanSquare := sumSquares / float64(rowWidth)
		rootMeanSquare := math.Sqrt(meanSquare + float64(epsilon))
		inverseRootMeanSquare := 1.0 / rootMeanSquare
		coefficient := dotProduct / (float64(rowWidth) * rootMeanSquare * rootMeanSquare * rootMeanSquare)
		for column := 0; column < rowWidth; column++ {
			resultRow[column] = float32(float64(gradientRow[column])*inverseRootMeanSquare - coefficient*float64(inputRow[column]))
		}
	}
	return result
}

// SoftcapBackward returns gradOut * (1 - (output/softcapValue)^2), where output
// is the softcapped result output = softcapValue*tanh(input/softcapValue).
func SoftcapBackward(output, gradOut *Tensor, softcapValue float32) *Tensor {
	result := New(output.Shape...)
	inverseSoftcap := 1.0 / softcapValue
	for index, value := range output.Data {
		scaled := float64(value) * float64(inverseSoftcap)
		result.Data[index] = gradOut.Data[index] * float32(1-scaled*scaled)
	}
	return result
}

// SigmoidBackward returns gradOut * output * (1 - output), where output is the
// sigmoid of the forward input.
func SigmoidBackward(output, gradOut *Tensor) *Tensor {
	result := New(output.Shape...)
	for index, value := range output.Data {
		result.Data[index] = gradOut.Data[index] * value * (1 - value)
	}
	return result
}

// ScaleBackward returns gradOut * factor.
func ScaleBackward(gradOut *Tensor, factor float32) *Tensor {
	return Scale(gradOut, factor)
}
