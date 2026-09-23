package tensors

import "github.com/cookiengineer/gonano/kernels"

// AttentionForward computes Output = softmax(mask(Query @ Key^T)) @ Value for a
// single (batch, head), delegating to the active backend. Query and Output are
// [queryLength, headDim]; Key and Value are [keyLength, headDim]. PositionOffset
// is the absolute position of the first query row (nonzero during KV-cache
// decoding) and a Window below zero means full causal context. LogSumExp is
// optional; when supplied it receives the per-query log-sum-exp for backward.
func AttentionForward(query, key, value, output, logSumExp []float32, queryLength, keyLength, headDim, positionOffset, window int) {
	AttentionForwardSegment(query, key, value, output, logSumExp, nil, queryLength, keyLength, headDim, positionOffset, window)
}

// AttentionForwardSegment is AttentionForward with sample-level attention
// masking: segmentIDs has one entry per key position and a query may attend
// only to keys in its own segment (DeepSeek-V4.1 §4.2.2). A nil slice disables
// the mask.
func AttentionForwardSegment(query, key, value, output, logSumExp []float32, segmentIDs []int32, queryLength, keyLength, headDim, positionOffset, window int) {
	KernelBackend().AttentionForward(
		kernels.AttentionForwardParameters{
			Query:          query,
			Key:            key,
			Value:          value,
			QueryLength:    queryLength,
			KeyLength:      keyLength,
			HeadDim:        headDim,
			PositionOffset: positionOffset,
			Window:         window,
			SegmentIDs:     segmentIDs,
		},
		kernels.AttentionForwardResult{Output: output, LogSumExp: logSumExp},
	)
}

// AttentionForwardSplit computes the unnormalized attention output and softmax
// statistics for the key shard [keyStart, keyEnd) of a single (batch, head),
// delegating to the active backend. Query is [queryLength, headDim]; Key and
// Value are the full [keyLength, headDim] arrays. Accumulator is
// [queryLength, headDim], Maximum and Sum are [queryLength].
func AttentionForwardSplit(query, key, value, accumulator, maximum, sum []float32, queryLength, keyLength, headDim, positionOffset, window, keyStart, keyEnd int) {
	AttentionForwardSplitSegment(query, key, value, accumulator, maximum, sum, nil, queryLength, keyLength, headDim, positionOffset, window, keyStart, keyEnd)
}

// AttentionForwardSplitSegment is AttentionForwardSplit with sample-level
// attention masking (see AttentionForwardSegment).
func AttentionForwardSplitSegment(query, key, value, accumulator, maximum, sum []float32, segmentIDs []int32, queryLength, keyLength, headDim, positionOffset, window, keyStart, keyEnd int) {
	KernelBackend().AttentionForwardSplit(
		kernels.AttentionSplitParameters{
			Query:          query,
			Key:            key,
			Value:          value,
			QueryLength:    queryLength,
			KeyLength:      keyLength,
			HeadDim:        headDim,
			PositionOffset: positionOffset,
			Window:         window,
			KeyStart:       keyStart,
			KeyEnd:         keyEnd,
			SegmentIDs:     segmentIDs,
		},
		kernels.AttentionSplitResult{Accumulator: accumulator, Maximum: maximum, Sum: sum},
	)
}

// AttentionCombine merges split-K partials into the normalized output and
// optional per-query log-sum-exp, delegating to the active backend.
func AttentionCombine(partials []kernels.AttentionSplitResult, output, logSumExp []float32, queryLength, headDim int) {
	KernelBackend().AttentionCombine(
		kernels.AttentionCombineParameters{Partials: partials, QueryLength: queryLength, HeadDim: headDim},
		kernels.AttentionForwardResult{Output: output, LogSumExp: logSumExp},
	)
}

// AttentionBackward accumulates the query, key, and value gradients from the
// saved forward statistics, delegating to the active backend. The gradients are
// accumulated (not overwritten), so grouped-query attention may invoke this
// once per query head while sharing one key/value gradient buffer.
func AttentionBackward(query, key, value, output, outputGradient, logSumExp, queryGradient, keyGradient, valueGradient []float32, queryLength, keyLength, headDim, positionOffset, window int) {
	AttentionBackwardSegment(query, key, value, output, outputGradient, logSumExp, nil, queryGradient, keyGradient, valueGradient, queryLength, keyLength, headDim, positionOffset, window)
}

// AttentionBackwardSegment is AttentionBackward with the sample-level attention
// mask (see AttentionForwardSegment). segmentIDs must be the same slice used in
// the forward pass.
func AttentionBackwardSegment(query, key, value, output, outputGradient, logSumExp []float32, segmentIDs []int32, queryGradient, keyGradient, valueGradient []float32, queryLength, keyLength, headDim, positionOffset, window int) {
	KernelBackend().AttentionBackward(
		kernels.AttentionBackwardParameters{
			Query:          query,
			Key:            key,
			Value:          value,
			Output:         output,
			OutputGradient: outputGradient,
			LogSumExp:      logSumExp,
			QueryLength:    queryLength,
			KeyLength:      keyLength,
			HeadDim:        headDim,
			PositionOffset: positionOffset,
			Window:         window,
			SegmentIDs:     segmentIDs,
		},
		kernels.AttentionBackwardResult{
			QueryGradient: queryGradient,
			KeyGradient:   keyGradient,
			ValueGradient: valueGradient,
		},
	)
}

// AttentionBackwardCorrected is AttentionBackward with an explicit row
// correction term (length queryLength). It is used to merge several attention
// branches (e.g. a global compressed branch and a local sliding-window branch)
// that share one softmax: every branch is backpropagated with the merged
// log-sum-exp and the merged correction dO_i . O_i.
func AttentionBackwardCorrected(query, key, value, output, outputGradient, logSumExp, rowCorrection, queryGradient, keyGradient, valueGradient []float32, queryLength, keyLength, headDim, positionOffset, window int) {
	AttentionBackwardCorrectedSegment(query, key, value, output, outputGradient, logSumExp, rowCorrection, nil, queryGradient, keyGradient, valueGradient, queryLength, keyLength, headDim, positionOffset, window)
}

// AttentionBackwardCorrectedSegment is AttentionBackwardCorrected with the
// sample-level attention mask (see AttentionForwardSegment).
func AttentionBackwardCorrectedSegment(query, key, value, output, outputGradient, logSumExp, rowCorrection []float32, segmentIDs []int32, queryGradient, keyGradient, valueGradient []float32, queryLength, keyLength, headDim, positionOffset, window int) {
	KernelBackend().AttentionBackward(
		kernels.AttentionBackwardParameters{
			Query:          query,
			Key:            key,
			Value:          value,
			Output:         output,
			OutputGradient: outputGradient,
			LogSumExp:      logSumExp,
			RowCorrection:  rowCorrection,
			QueryLength:    queryLength,
			KeyLength:      keyLength,
			HeadDim:        headDim,
			PositionOffset: positionOffset,
			Window:         window,
			SegmentIDs:     segmentIDs,
		},
		kernels.AttentionBackwardResult{
			QueryGradient: queryGradient,
			KeyGradient:   keyGradient,
			ValueGradient: valueGradient,
		},
	)
}
