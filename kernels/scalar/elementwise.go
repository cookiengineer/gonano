package scalar

import "math"

// Add computes destination[i] = left[i] + right[i].
func (backend *Backend) Add(destination, left, right []float32) {
	for index := range left {
		destination[index] = left[index] + right[index]
	}
}

// Subtract computes destination[i] = left[i] - right[i].
func (backend *Backend) Subtract(destination, left, right []float32) {
	for index := range left {
		destination[index] = left[index] - right[index]
	}
}

// Multiply computes destination[i] = left[i] * right[i].
func (backend *Backend) Multiply(destination, left, right []float32) {
	for index := range left {
		destination[index] = left[index] * right[index]
	}
}

// Divide computes destination[i] = left[i] / right[i].
func (backend *Backend) Divide(destination, left, right []float32) {
	for index := range left {
		destination[index] = left[index] / right[index]
	}
}

// Scale computes destination[i] = source[i] * factor.
func (backend *Backend) Scale(destination, source []float32, factor float32) {
	for index, value := range source {
		destination[index] = value * factor
	}
}

// AddScaled computes destination[i] = base[i] + factor*scaled[i].
func (backend *Backend) AddScaled(destination, base, scaled []float32, factor float32) {
	for index, value := range base {
		destination[index] = value + factor*scaled[index]
	}
}

// Negate computes destination[i] = -source[i].
func (backend *Backend) Negate(destination, source []float32) {
	for index, value := range source {
		destination[index] = -value
	}
}

// Abs computes destination[i] = |source[i]|.
func (backend *Backend) Abs(destination, source []float32) {
	for index, value := range source {
		if value < 0 {
			destination[index] = -value
		} else {
			destination[index] = value
		}
	}
}

// Square computes destination[i] = source[i] * source[i].
func (backend *Backend) Square(destination, source []float32) {
	for index, value := range source {
		destination[index] = value * value
	}
}

// ReluSquared computes destination[i] = max(source[i], 0)^2.
func (backend *Backend) ReluSquared(destination, source []float32) {
	for index, value := range source {
		if value < 0 {
			value = 0
		}
		destination[index] = value * value
	}
}

// Exp computes destination[i] = exp(source[i]).
func (backend *Backend) Exp(destination, source []float32) {
	for index, value := range source {
		destination[index] = float32(math.Exp(float64(value)))
	}
}

// Sigmoid computes destination[i] = 1 / (1 + exp(-source[i])).
func (backend *Backend) Sigmoid(destination, source []float32) {
	for index, value := range source {
		destination[index] = float32(1.0 / (1.0 + math.Exp(float64(-value))))
	}
}

// Tanh computes destination[i] = tanh(source[i]).
func (backend *Backend) Tanh(destination, source []float32) {
	for index, value := range source {
		destination[index] = float32(math.Tanh(float64(value)))
	}
}

// Rsqrt computes destination[i] = 1 / sqrt(source[i]).
func (backend *Backend) Rsqrt(destination, source []float32) {
	for index, value := range source {
		destination[index] = float32(1.0 / math.Sqrt(float64(value)))
	}
}
