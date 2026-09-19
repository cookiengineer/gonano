package tensors

// testData returns deterministic test values in roughly [-3.5, 3.5).
func testData(count int) []float32 {
	values := make([]float32, count)
	for index := range values {
		values[index] = float32(index%7) - 3.5 + 0.1*float32(index%3)
	}
	return values
}

// testDataB returns deterministic positive test values.
func testDataB(count int) []float32 {
	values := make([]float32, count)
	for index := range values {
		values[index] = float32(index%5) + 0.5
	}
	return values
}
