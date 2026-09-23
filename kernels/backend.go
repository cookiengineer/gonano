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

// Indexer is the contract for the lightning indexer's sparse scoring. It
// computes, for every query row and key block, the ReLU-weighted multi-head dot
// product that selects which compressed blocks each query attends to
// (DeepSeek-V4.1 §2.3). query is [rowCount, headCount, dim], key is
// [blockCount, headCount, dim], weight is [headCount], and destination is
// [rowCount, blockCount]:
//
//	destination[row, block] = sum_h weight[h] * max(0, dot(query[row,h], key[block,h]))
type Indexer interface {
	// IndexerScores computes the per-(row, block) index scores. weight must
	// have headCount elements.
	IndexerScores(destination, query, key, weight []float32, rowCount, blockCount, headCount, dim int)
	// PooledMean averages consecutive non-overlapping groups of `groupSize`
	// rows of source [blockCount, width] into destination [ceil(blockCount/
	// groupSize), width]. The trailing group keeps its actual row count.
	PooledMean(destination, source []float32, blockCount, groupSize, width int)
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
//
// RowCorrection is optional (length queryLength). When supplied, it replaces
// the row correction term sum_j P_ij (dO_i . V_ij) with the provided value.
// This lets several attention branches (for example a global compressed branch
// and a local sliding-window branch) share one merged softmax: each branch is
// backpropagated with the merged log-sum-exp and the merged correction
// dO_i . O_i, which is exactly the gradient of the union of their keys.
type AttentionBackwardParameters struct {
	Query          []float32
	Key            []float32
	Value          []float32
	Output         []float32
	OutputGradient []float32
	LogSumExp      []float32
	RowCorrection  []float32
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

// AttentionSplitParameters describes one contiguous key range
// [KeyStart, KeyEnd) of a (batch, head) attention pass. It is used by split-K
// (flash-decoding) parallelism, where the key dimension is partitioned across
// workers and the per-shard statistics are merged by AttentionCombine. Key and
// Value are the full [KeyLength, HeadDim] arrays; the shard selects a sub-range
// of keys, and masking still uses absolute key indices.
type AttentionSplitParameters struct {
	Query          []float32
	Key            []float32
	Value          []float32
	QueryLength    int
	KeyLength      int
	HeadDim        int
	PositionOffset int
	Window         int
	KeyStart       int
	KeyEnd         int
}

// AttentionSplitResult is one key shard's unnormalized attention output and its
// running softmax statistics. All fields are indexed per query row:
// Accumulator is [QueryLength, HeadDim] (the unnormalized sum of p*Value),
// Maximum and Sum are [QueryLength].
type AttentionSplitResult struct {
	Accumulator []float32
	Maximum     []float32
	Sum         []float32
}

// AttentionCombineParameters merges the shards of one (batch, head) pass into a
// normalized output.
type AttentionCombineParameters struct {
	Partials    []AttentionSplitResult
	QueryLength int
	HeadDim     int
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
	// AttentionForwardSplit computes the unnormalized attention output and
	// softmax statistics for one contiguous key shard, enabling split-K
	// parallelism. Result slices must be pre-sized to the query length (and
	// query length times head dim for the accumulator).
	AttentionForwardSplit(parameters AttentionSplitParameters, result AttentionSplitResult)
	// AttentionCombine merges per-shard statistics into the normalized output
	// and optional log-sum-exp.
	AttentionCombine(parameters AttentionCombineParameters, result AttentionForwardResult)
}

// Backend is the complete numeric contract. A backend must implement every
// group; the fused groups (Rows, Attention) exist so implementations can fuse
// operations for speed rather than composing them from Elementwise.
type Backend interface {
	Elementwise
	Reductions
	LinearAlgebra
	Indexer
	Rows
	Attention
}
