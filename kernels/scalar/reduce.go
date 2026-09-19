package scalar

// Sum returns the sum of every element.
func (backend *Backend) Sum(values []float32) float32 {
	var total float32
	for _, value := range values {
		total += value
	}
	return total
}

// Max returns the largest element, or zero for an empty slice.
func (backend *Backend) Max(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}
	maximum := values[0]
	for _, value := range values[1:] {
		if value > maximum {
			maximum = value
		}
	}
	return maximum
}

// ArgMax returns the index of the largest element, resolving ties to the
// smallest index. An empty slice returns zero.
func (backend *Backend) ArgMax(values []float32) int {
	bestIndex := 0
	for index := 1; index < len(values); index++ {
		if values[index] > values[bestIndex] {
			bestIndex = index
		}
	}
	return bestIndex
}
