package tensors

// Transpose returns the transpose of a rank-2 tensor ([rows, columns] ->
// [columns, rows]).
func Transpose(source *Tensor) *Tensor {
	if source.Rank() != 2 {
		panic("tensors: Transpose requires a rank-2 tensor")
	}
	rows, columns := source.Shape[0], source.Shape[1]
	result := New(columns, rows)
	for row := 0; row < rows; row++ {
		for column := 0; column < columns; column++ {
			result.Data[column*rows+row] = source.Data[row*columns+column]
		}
	}
	return result
}
