// Package kernels defines the numeric backend contract for gonano. Every
// tensor operation in the tensors package is ultimately expressed as a call to
// a Backend, so alternative implementations (SIMD, scalar, accelerators) can be
// swapped without touching model code.
//
// A Backend operates on contiguous, row-major float32 slices rather than on
// tensor objects. That keeps the contract dependency-free and makes backends
// easy to test and reuse. Unless a method documents otherwise, destination
// slices may alias their inputs, and callers must size them correctly.
//
// Implementations live in kernels/simd (the production backend) and
// kernels/scalar (a portable reference backend used for parity tests and
// environments without SIMD support).
package kernels

// Elementwise is the contract for element-wise vector operations.
//
// Every method writes one output value per input element. Destination and
// sources must have the same length. Implementations must tolerate the
// destination aliasing any source.
type Elementwise interface {
	// Add computes destination[i] = left[i] + right[i].
	Add(destination, left, right []float32)
	// Subtract computes destination[i] = left[i] - right[i].
	Subtract(destination, left, right []float32)
	// Multiply computes destination[i] = left[i] * right[i].
	Multiply(destination, left, right []float32)
	// Divide computes destination[i] = left[i] / right[i].
	Divide(destination, left, right []float32)
	// Scale computes destination[i] = source[i] * factor.
	Scale(destination, source []float32, factor float32)
	// AddScaled computes destination[i] = base[i] + factor*scaled[i]. It is the
	// residual-connection primitive used throughout the transformer.
	AddScaled(destination, base, scaled []float32, factor float32)
	// Negate computes destination[i] = -source[i].
	Negate(destination, source []float32)
	// Abs computes destination[i] = |source[i]|.
	Abs(destination, source []float32)
	// Square computes destination[i] = source[i] * source[i].
	Square(destination, source []float32)
	// ReluSquared computes destination[i] = max(source[i], 0)^2, the nanochat
	// MLP activation.
	ReluSquared(destination, source []float32)
	// Exp computes destination[i] = exp(source[i]).
	Exp(destination, source []float32)
	// Sigmoid computes destination[i] = 1 / (1 + exp(-source[i])).
	Sigmoid(destination, source []float32)
	// Tanh computes destination[i] = tanh(source[i]).
	Tanh(destination, source []float32)
	// Rsqrt computes destination[i] = 1 / sqrt(source[i]).
	Rsqrt(destination, source []float32)
}

// Reductions is the contract for turning a vector into a scalar.
type Reductions interface {
	// Sum returns the sum of every element.
	Sum(values []float32) float32
	// Max returns the largest element, or zero for an empty slice.
	Max(values []float32) float32
	// ArgMax returns the index of the largest element. Ties resolve to the
	// smallest index, and an empty slice returns zero.
	ArgMax(values []float32) int
}

// LinearAlgebra is the contract for matrix multiplication and dot products.
// Matrices are row-major and contiguous.
type LinearAlgebra interface {
	// MatMul computes destination = left @ right, where left is
	// [rowCount, innerCount] and right is [innerCount, columnCount]. The
	// destination is [rowCount, columnCount].
	MatMul(destination, left, right []float32, rowCount, columnCount, innerCount int)
	// MatMulTransposed computes destination = left @ right^T, where left is
	// [rowCount, innerCount] and right is [columnCount, innerCount] (a weight
	// matrix stored as [out, in]). The destination is [rowCount, columnCount].
	MatMulTransposed(destination, left, right []float32, rowCount, columnCount, innerCount int)
	// DotProduct returns the sum of left[i]*right[i] over the shorter input.
	DotProduct(left, right []float32) float32
}

// Rows is the contract for fused operations that normalize each row of a
// matrix independently.
type Rows interface {
	// SoftmaxLastDim computes a numerically stable softmax over the columns of
	// every row. source and destination are [rowCount, columnCount].
	SoftmaxLastDim(destination, source []float32, rowCount, columnCount int)
	// RMSNormLastDim normalizes every row to unit root-mean-square without
	// affine parameters, matching nanochat's stateless norm. source and
	// destination are [rowCount, columnCount].
	RMSNormLastDim(destination, source []float32, rowCount, columnCount int, epsilon float32)
}

// AttentionForwardParameters describes a single (batch, head) attention
// forward pass. Query is [queryLength, headDim]; Key and Value are
// [keyLength, headDim]. PositionOffset is the absolute position of the first
// query row (nonzero during KV-cache decoding). Window below zero means full
// causal context; otherwise it is the left sliding-window width.
type AttentionForwardParameters struct {
	Query          []float32
	Key            []float32
	Value          []float32
	QueryLength    int
	KeyLength      int
	HeadDim        int
	PositionOffset int
	Window         int
}

// AttentionForwardResult holds the outputs of an attention forward pass.
// Output is [queryLength, headDim]. LogSumExp is optional; when supplied it
// receives the per-query log-sum-exp (length queryLength) needed by
// AttentionBackward.
type AttentionForwardResult struct {
	Output    []float32
	LogSumExp []float32
}

// AttentionBackwardParameters describes a single (batch, head) attention
// backward pass. Output and OutputGradient are [queryLength, headDim];
// LogSumExp is the forward statistic (length queryLength).
type AttentionBackwardParameters struct {
	Query          []float32
	Key            []float32
	Value          []float32
	Output         []float32
	OutputGradient []float32
	LogSumExp      []float32
	QueryLength    int
	KeyLength      int
	HeadDim        int
	PositionOffset int
	Window         int
}

// AttentionBackwardResult holds the gradients of an attention backward pass.
// The gradients are accumulated into the supplied slices (not overwritten),
// because grouped-query attention shares Key and Value across several query
// heads.
type AttentionBackwardResult struct {
	QueryGradient []float32
	KeyGradient   []float32
	ValueGradient []float32
}

// Attention is the contract for masked, causal attention over contiguous
// slices. Forward receives the softmax statistics; Backward consumes them.
type Attention interface {
	// AttentionForward computes
	// Output = softmax(mask(Query @ Key^T)) @ Value without requiring callers
	// to allocate the score matrix.
	AttentionForward(parameters AttentionForwardParameters, result AttentionForwardResult)
	// AttentionBackward computes the query/key/value gradients from the saved
	// forward statistics and accumulates them into the result slices.
	AttentionBackward(parameters AttentionBackwardParameters, result AttentionBackwardResult)
}

// Backend is the complete numeric contract. A backend must implement every
// group; the fused groups (Rows, Attention) exist so implementations can fuse
// operations for speed rather than composing them from Elementwise.
type Backend interface {
	Elementwise
	Reductions
	LinearAlgebra
	Rows
	Attention
}
