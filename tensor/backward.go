package tensor

import "math"

// Backward primitives for the operations used by the transformer. These are
// the analytic gradients, verified against finite differences in tests.

// CrossEntropyGrad returns the gradient of the (scaled) mean cross-entropy
// loss with respect to logits: (softmax(logits) - onehot(target)) * scale,
// with ignored positions zeroed.
func CrossEntropyGrad(logits *Tensor, targets *Int32s, ignore int32, scale float32) *Tensor {
	rows, vocab := logits.Shape[0], logits.Shape[1]
	grad := New(rows, vocab)
	gd := grad.Data
	for i := 0; i < rows; i++ {
		y := targets.Data[i]
		if y == ignore {
			continue
		}
		row := logits.Data[i*vocab : (i+1)*vocab]
		m := sliceMax(row)
		var sum float64
		for _, v := range row {
			sum += math.Exp(float64(v - m))
		}
		inv := 1.0 / sum
		base := i * vocab
		for j := 0; j < vocab; j++ {
			p := float32(math.Exp(float64(row[j]-m)) * inv)
			if int32(j) == y {
				p -= 1
			}
			gd[base+j] = p * scale
		}
	}
	return grad
}

// MatMulTransBBackward returns the gradients of a = b@b^T: grad_a = grad_out@b
// and grad_b = grad_out^T@a.
func MatMulTransBBackward(gradOut, a, b *Tensor) (gradA, gradB *Tensor) {
	gradA = MatMul(gradOut, b)                 // [M,K] = [M,N] @ [N,K]
	gradB = MatMul(Transpose(gradOut), a)      // [N,K] = [N,M] @ [M,K]
	return gradA, gradB
}

// MatMulBackward returns the gradients of a@b: grad_a = grad_out@b^T and
// grad_b = a^T@grad_out.
func MatMulBackward(gradOut, a, b *Tensor) (gradA, gradB *Tensor) {
	gradA = MatMulTransB(gradOut, b)       // [M,K] = [M,N] @ [N,K]
	gradB = MatMul(Transpose(a), gradOut)  // [K,N] = [K,M] @ [M,N]
	return gradA, gradB
}

// Relu2Backward returns grad_out * 2*relu(x).
func Relu2Backward(x, gradOut *Tensor) *Tensor {
	out := New(x.Shape...)
	for i, v := range x.Data {
		if v > 0 {
			out.Data[i] = gradOut.Data[i] * 2 * v
		}
	}
	return out
}

// RMSNormBackward returns the gradient of RMSNorm applied over the last dim.
// y = x / r with r = sqrt(mean(x^2)+eps):
// grad_x = grad_y/r - x * (dot(grad_y,x) / (D * r^3)).
func RMSNormBackward(x, gradOut *Tensor, eps float32) *Tensor {
	rows := x.Numel() / x.Shape[len(x.Shape)-1]
	d := x.Shape[len(x.Shape)-1]
	out := New(x.Shape...)
	for i := 0; i < rows; i++ {
		xr := x.Data[i*d : (i+1)*d]
		gr := gradOut.Data[i*d : (i+1)*d]
		or := out.Data[i*d : (i+1)*d]
		var sumSq float64
		var dot float64
		for j := 0; j < d; j++ {
			sumSq += float64(xr[j]) * float64(xr[j])
			dot += float64(gr[j]) * float64(xr[j])
		}
		meanSq := sumSq / float64(d)
		r := math.Sqrt(meanSq + float64(eps))
		invR := 1.0 / r
		coeff := dot / (float64(d) * r * r * r)
		for j := 0; j < d; j++ {
			or[j] = float32(float64(gr[j])*invR - coeff*float64(xr[j]))
		}
	}
	return out
}

// SoftcapBackward returns grad_out * (1 - (y/softcap)^2), where y is the
// softcapped output y = softcap*tanh(x/softcap).
func SoftcapBackward(y, gradOut *Tensor, softcap float32) *Tensor {
	out := New(y.Shape...)
	inv := 1.0 / softcap
	for i, v := range y.Data {
		t := float64(v) * float64(inv)
		out.Data[i] = gradOut.Data[i] * float32(1-t*t)
	}
	return out
}

// SigmoidBackward returns grad_out * y * (1 - y), where y = sigmoid(x).
func SigmoidBackward(y, gradOut *Tensor) *Tensor {
	out := New(y.Shape...)
	for i, v := range y.Data {
		out.Data[i] = gradOut.Data[i] * v * (1 - v)
	}
	return out
}

// ScaleBackward returns grad_out * s.
func ScaleBackward(gradOut *Tensor, s float32) *Tensor {
	return Scale(gradOut, s)
}
