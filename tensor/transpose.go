package tensor

// Transpose returns the transpose of a 2D tensor ([m,n] -> [n,m]).
func Transpose(x *Tensor) *Tensor {
	if x.Rank() != 2 {
		panic("tensor: Transpose requires a rank-2 tensor")
	}
	m, n := x.Shape[0], x.Shape[1]
	out := New(n, m)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			out.Data[j*m+i] = x.Data[i*n+j]
		}
	}
	return out
}
