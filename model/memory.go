package model

// TrainingBytesPerParameter is the float32 training footprint per parameter:
// weights (4 bytes) + gradients (4 bytes) + optimizer state (4 bytes for the
// flash preset's Muon/Sinkhorn momentum buffers).
const TrainingBytesPerParameter = 12

// runtimeReserveBytes is a fixed allowance for the Go runtime, the tokenizer,
// data buffers, and per-step scratch space when estimating training memory.
const runtimeReserveBytes = int64(2) << 30

// activationTensorsPerLayer is a rough upper bound on the number of
// [batch, sequence, embed] float32 activation tensors TrainForward keeps alive
// per layer for the backward pass.
const activationTensorsPerLayer = 10

// TotalParamsForConfig returns the number of parameters a config allocates
// (token and value embeddings, transformer matrices, the language-model head,
// and scalar parameters) without building the model. It omits a handful of
// small tensors (the compressor bias, indexer heads, and router biases), so it
// is an estimate accurate to within a couple of percent; it is intended for
// memory preflight checks, not exact accounting.
func TotalParamsForConfig(config Config) int64 {
	// ScalingParamsForConfig covers the transformer matrices plus lm_head.
	scaling := ScalingParamsForConfig(config)
	vocabulary := int64(config.PaddedVocab())
	embeddingDimension := int64(config.EmbedDim)
	kvWidth := int64(config.NumKVHead) * int64(config.HeadDim())
	valueEmbeddings := int64((config.NumLayer+1)/2) * vocabulary * kvWidth
	tokenEmbedding := vocabulary * embeddingDimension
	// resid_lambdas, x0_lambdas, smear_gate, smear_lambda, backout_lambda.
	scalars := int64(2*config.NumLayer + 24 + 1 + 1)
	return scaling + tokenEmbedding + valueEmbeddings + scalars
}

// EstimatedTrainingMemoryBytes estimates the peak resident memory a training
// run needs for the given config and per-step batch shape: the training-state
// footprint (TrainingBytesPerParameter per parameter) plus saved activations
// and a fixed runtime reserve. The result is a conservative planning estimate,
// not an exact measurement.
func EstimatedTrainingMemoryBytes(config Config, deviceBatch, sequenceLength int) int64 {
	if deviceBatch < 1 {
		deviceBatch = 1
	}
	if sequenceLength < 1 {
		sequenceLength = 1
	}
	parameters := TotalParamsForConfig(config)
	state := int64(TrainingBytesPerParameter) * parameters
	activations := int64(deviceBatch) * int64(sequenceLength) * int64(config.EmbedDim) * 4 *
		int64(config.NumLayer) * activationTensorsPerLayer
	return state + activations + runtimeReserveBytes
}
