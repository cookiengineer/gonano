package simdbackend

import (
	"math"

	"simd"
)

// Every method below that touches a simd intrinsic delegates to a package-level
// core function. The Go compiler cannot type-check a receiver method that
// contains an intrinsic (it reports "missing Types entry"), so keeping the
// intrinsic bodies in plain functions is required.

// Add computes destination[i] = left[i] + right[i].
func (backend *Backend) Add(destination, left, right []float32) { addCore(destination, left, right) }

// Subtract computes destination[i] = left[i] - right[i].
func (backend *Backend) Subtract(destination, left, right []float32) {
	subtractCore(destination, left, right)
}

// Multiply computes destination[i] = left[i] * right[i].
func (backend *Backend) Multiply(destination, left, right []float32) {
	multiplyCore(destination, left, right)
}

// Divide computes destination[i] = left[i] / right[i].
func (backend *Backend) Divide(destination, left, right []float32) {
	divideCore(destination, left, right)
}

// Scale computes destination[i] = source[i] * factor.
func (backend *Backend) Scale(destination, source []float32, factor float32) {
	scaleCore(destination, source, factor)
}

// AddScaled computes destination[i] = base[i] + factor*scaled[i].
func (backend *Backend) AddScaled(destination, base, scaled []float32, factor float32) {
	addScaledCore(destination, base, scaled, factor)
}

// Negate computes destination[i] = -source[i].
func (backend *Backend) Negate(destination, source []float32) { negateCore(destination, source) }

// Abs computes destination[i] = |source[i]|.
func (backend *Backend) Abs(destination, source []float32) { absCore(destination, source) }

// Square computes destination[i] = source[i] * source[i].
func (backend *Backend) Square(destination, source []float32) { squareCore(destination, source) }

// ReluSquared computes destination[i] = max(source[i], 0)^2.
func (backend *Backend) ReluSquared(destination, source []float32) {
	reluSquaredCore(destination, source)
}

// SwiGLU computes the clamped SwiGLU activation with SIMD vectorization.
func (backend *Backend) SwiGLU(destination, gate, up []float32, clamp float32) {
	swigluCore(destination, gate, up, clamp)
}

// SwiGLUBackward computes the clamped SwiGLU gradients with SIMD vectorization.
func (backend *Backend) SwiGLUBackward(gateGradient, upGradient, gate, up, outputGradient []float32, clamp float32) {
	swigluBackwardCore(gateGradient, upGradient, gate, up, outputGradient, clamp)
}

// Exp computes destination[i] = exp(source[i]) using the vectorized exp
// approximation.
func (backend *Backend) Exp(destination, source []float32) { expCore(destination, source) }

// Sigmoid computes destination[i] = 1 / (1 + exp(-source[i])) using the
// vectorized exp approximation.
func (backend *Backend) Sigmoid(destination, source []float32) { sigmoidCore(destination, source) }

// Tanh computes destination[i] = tanh(source[i]). The simd package exposes no
// tanh, so this is computed element-wise.
func (backend *Backend) Tanh(destination, source []float32) {
	for index, value := range source {
		destination[index] = float32(math.Tanh(float64(value)))
	}
}

// Rsqrt computes destination[i] = 1 / sqrt(source[i]). The simd package
// exposes no reciprocal square root, so this is computed element-wise.
func (backend *Backend) Rsqrt(destination, source []float32) {
	for index, value := range source {
		destination[index] = float32(1.0 / math.Sqrt(float64(value)))
	}
}

func addCore(destination, left, right []float32) {
	length := len(left)
	index := 0
	for ; index+float32LaneCount <= length; index += float32LaneCount {
		leftVector := simd.LoadFloat32s(left[index:])
		rightVector := simd.LoadFloat32s(right[index:])
		leftVector.Add(rightVector).Store(destination[index:])
	}
	for ; index < length; index++ {
		destination[index] = left[index] + right[index]
	}
}

func subtractCore(destination, left, right []float32) {
	length := len(left)
	index := 0
	for ; index+float32LaneCount <= length; index += float32LaneCount {
		leftVector := simd.LoadFloat32s(left[index:])
		rightVector := simd.LoadFloat32s(right[index:])
		leftVector.Sub(rightVector).Store(destination[index:])
	}
	for ; index < length; index++ {
		destination[index] = left[index] - right[index]
	}
}

func multiplyCore(destination, left, right []float32) {
	length := len(left)
	index := 0
	for ; index+float32LaneCount <= length; index += float32LaneCount {
		leftVector := simd.LoadFloat32s(left[index:])
		rightVector := simd.LoadFloat32s(right[index:])
		leftVector.Mul(rightVector).Store(destination[index:])
	}
	for ; index < length; index++ {
		destination[index] = left[index] * right[index]
	}
}

func divideCore(destination, left, right []float32) {
	length := len(left)
	index := 0
	for ; index+float32LaneCount <= length; index += float32LaneCount {
		leftVector := simd.LoadFloat32s(left[index:])
		rightVector := simd.LoadFloat32s(right[index:])
		leftVector.Div(rightVector).Store(destination[index:])
	}
	for ; index < length; index++ {
		destination[index] = left[index] / right[index]
	}
}

func scaleCore(destination, source []float32, factor float32) {
	length := len(source)
	factorVector := simd.BroadcastFloat32s(factor)
	index := 0
	for ; index+float32LaneCount <= length; index += float32LaneCount {
		sourceVector := simd.LoadFloat32s(source[index:])
		sourceVector.Mul(factorVector).Store(destination[index:])
	}
	for ; index < length; index++ {
		destination[index] = source[index] * factor
	}
}

func addScaledCore(destination, base, scaled []float32, factor float32) {
	length := len(base)
	factorVector := simd.BroadcastFloat32s(factor)
	index := 0
	for ; index+float32LaneCount <= length; index += float32LaneCount {
		baseVector := simd.LoadFloat32s(base[index:])
		scaledVector := simd.LoadFloat32s(scaled[index:])
		// MulAdd(x, y, z) = x*y + z, so this is scaled*factor + base.
		scaledVector.MulAdd(factorVector, baseVector).Store(destination[index:])
	}
	for ; index < length; index++ {
		destination[index] = base[index] + factor*scaled[index]
	}
}

func negateCore(destination, source []float32) {
	length := len(source)
	index := 0
	for ; index+float32LaneCount <= length; index += float32LaneCount {
		simd.LoadFloat32s(source[index:]).Neg().Store(destination[index:])
	}
	for ; index < length; index++ {
		destination[index] = -source[index]
	}
}

func absCore(destination, source []float32) {
	length := len(source)
	index := 0
	for ; index+float32LaneCount <= length; index += float32LaneCount {
		simd.LoadFloat32s(source[index:]).Abs().Store(destination[index:])
	}
	for ; index < length; index++ {
		if source[index] < 0 {
			destination[index] = -source[index]
		} else {
			destination[index] = source[index]
		}
	}
}

func squareCore(destination, source []float32) {
	length := len(source)
	index := 0
	for ; index+float32LaneCount <= length; index += float32LaneCount {
		sourceVector := simd.LoadFloat32s(source[index:])
		sourceVector.Mul(sourceVector).Store(destination[index:])
	}
	for ; index < length; index++ {
		destination[index] = source[index] * source[index]
	}
}

func reluSquaredCore(destination, source []float32) {
	length := len(source)
	zeroVector := simd.BroadcastFloat32s(0)
	index := 0
	for ; index+float32LaneCount <= length; index += float32LaneCount {
		sourceVector := simd.LoadFloat32s(source[index:])
		reluVector := sourceVector.Max(zeroVector)
		reluVector.Mul(reluVector).Store(destination[index:])
	}
	for ; index < length; index++ {
		value := source[index]
		if value < 0 {
			value = 0
		}
		destination[index] = value * value
	}
}

func swigluCore(destination, gate, up []float32, clamp float32) {
	length := min(len(destination), min(len(gate), len(up)))
	index := 0
	if clamp > 0 {
		clampVector := simd.BroadcastFloat32s(clamp)
		negativeClamp := simd.BroadcastFloat32s(-clamp)
		for ; index+float32LaneCount <= length; index += float32LaneCount {
			gateVector := simd.LoadFloat32s(gate[index:]).Min(clampVector)
			upVector := simd.LoadFloat32s(up[index:]).Min(clampVector).Max(negativeClamp)
			sigmoid := sigmoidVector(gateVector)
			gateVector.Mul(sigmoid).Mul(upVector).Store(destination[index:])
		}
		for ; index < length; index++ {
			destination[index] = swigluScalar(gate[index], up[index], clamp)
		}
		return
	}
	for ; index+float32LaneCount <= length; index += float32LaneCount {
		gateVector := simd.LoadFloat32s(gate[index:])
		upVector := simd.LoadFloat32s(up[index:])
		sigmoid := sigmoidVector(gateVector)
		gateVector.Mul(sigmoid).Mul(upVector).Store(destination[index:])
	}
	for ; index < length; index++ {
		destination[index] = swigluScalar(gate[index], up[index], clamp)
	}
}

func swigluBackwardCore(gateGradient, upGradient, gate, up, outputGradient []float32, clamp float32) {
	length := min(len(gate), min(len(up), min(len(gateGradient),
		min(len(upGradient), len(outputGradient)))))
	index := 0
	one := simd.BroadcastFloat32s(1)
	if clamp > 0 {
		clampVector := simd.BroadcastFloat32s(clamp)
		negativeClamp := simd.BroadcastFloat32s(-clamp)
		for ; index+float32LaneCount <= length; index += float32LaneCount {
			rawGate := simd.LoadFloat32s(gate[index:])
			rawUp := simd.LoadFloat32s(up[index:])
			gateVector := rawGate.Min(clampVector)
			upVector := rawUp.Min(clampVector).Max(negativeClamp)
			gateMask := one.Masked(rawGate.Less(clampVector))
			upMask := one.Masked(rawUp.LessEqual(clampVector).And(rawUp.GreaterEqual(negativeClamp)))
			sigmoid := sigmoidVector(gateVector)
			silu := gateVector.Mul(sigmoid)
			// d(silu)/dg = sigmoid*(1 + g*(1-sigmoid)).
			siluGradient := sigmoid.Mul(one.Add(gateVector.Mul(one.Sub(sigmoid))))
			outputVector := simd.LoadFloat32s(outputGradient[index:])
			outputVector.Mul(upVector).Mul(siluGradient).Mul(gateMask).Store(gateGradient[index:])
			outputVector.Mul(silu).Mul(upMask).Store(upGradient[index:])
		}
		for ; index < length; index++ {
			gateGradient[index], upGradient[index] = swigluGradientScalar(gate[index], up[index], outputGradient[index], clamp)
		}
		return
	}
	for ; index+float32LaneCount <= length; index += float32LaneCount {
		gateVector := simd.LoadFloat32s(gate[index:])
		upVector := simd.LoadFloat32s(up[index:])
		sigmoid := sigmoidVector(gateVector)
		silu := gateVector.Mul(sigmoid)
		siluGradient := sigmoid.Mul(one.Add(gateVector.Mul(one.Sub(sigmoid))))
		outputVector := simd.LoadFloat32s(outputGradient[index:])
		outputVector.Mul(upVector).Mul(siluGradient).Store(gateGradient[index:])
		outputVector.Mul(silu).Store(upGradient[index:])
	}
	for ; index < length; index++ {
		gateGradient[index], upGradient[index] = swigluGradientScalar(gate[index], up[index], outputGradient[index], clamp)
	}
}

// sigmoidVector computes 1/(1+exp(-x)) for every lane, reusing the vectorized
// exp approximation.
func sigmoidVector(x simd.Float32s) simd.Float32s {
	one := simd.BroadcastFloat32s(1)
	return one.Div(one.Add(expVector(x.Neg())))
}

// swigluScalar is the scalar tail of swigluCore.
func swigluScalar(gate, up, clamp float32) float32 {
	gateValue, _ := clampSwiGLUGate(gate, clamp)
	upValue, _ := clampSwiGLUUp(up, clamp)
	sigmoid := float32(1.0 / (1.0 + math.Exp(float64(-gateValue))))
	return gateValue * sigmoid * upValue
}

// swigluGradientScalar is the scalar tail of swigluBackwardCore.
func swigluGradientScalar(gate, up, outputGradient, clamp float32) (float32, float32) {
	gateValue, gateMask := clampSwiGLUGate(gate, clamp)
	upValue, upMask := clampSwiGLUUp(up, clamp)
	sigmoid := float32(1.0 / (1.0 + math.Exp(float64(-gateValue))))
	silu := gateValue * sigmoid
	siluGradient := sigmoid * (1 + gateValue*(1-sigmoid))
	return outputGradient * upValue * siluGradient * gateMask, outputGradient * silu * upMask
}

func clampSwiGLUGate(value, clamp float32) (float32, float32) {
	if clamp > 0 && value >= clamp {
		return clamp, 0
	}
	return value, 1
}

func clampSwiGLUUp(value, clamp float32) (float32, float32) {
	if clamp > 0 {
		if value > clamp {
			return clamp, 0
		}
		if value < -clamp {
			return -clamp, 0
		}
	}
	return value, 1
}
