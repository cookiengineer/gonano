package tensors

// MoEGateTopK computes the router softmax probabilities and the top-k expert
// selection for logits of shape [rows, experts]. It returns probs
// [rows, experts], the selected expert indices [rows, k], and the
// corresponding softmax affinities [rows, k]. bias is [experts] and
// participates only in selection, not in the returned weighting.
func MoEGateTopK(logits, bias *Tensor, rows, experts, k int) (probs *Tensor, selected *Int32s, selectedWeights *Tensor) {
	probs = New(rows, experts)
	selected = NewInt32s(rows, k)
	selectedWeights = New(rows, k)
	KernelBackend().MoEGateTopK(logits.Data, bias.Data, probs.Data, selected.Data, selectedWeights.Data, rows, experts, k)
	return probs, selected, selectedWeights
}

// GroupedMatMulTransposed computes one matmul per expert: with a packed weight
// [experts, columns, inner], expert e computes destination_e = input_e @
// weight_e^T. The input rows are grouped by expert in ascending expert order,
// with tokenCounts[e] rows for expert e; the result has the same row grouping.
func GroupedMatMulTransposed(input, weight *Tensor, tokenCounts []int32, experts, columns, inner int) *Tensor {
	totalRows := 0
	for _, count := range tokenCounts {
		totalRows += int(count)
	}
	result := New(totalRows, columns)
	KernelBackend().GroupedMatMulTransposed(result.Data, input.Data, weight.Data, tokenCounts, experts, columns, inner)
	return result
}
