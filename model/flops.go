package model

// FLOPs and parameter accounting, ported from nanochat's gpt.py estimate_flops
// and num_scaling_params.

// MatmulParams returns the number of parameters participating in matrix
// multiplications with the token stream (every layers.Linear weight). Embeddings
// and scalar parameters are excluded.
func (model *Transformer) MatmulParams() int {
	total := 0
	for _, block := range model.blocks {
		if block.attention.mla != nil {
			total += block.attention.mla.numParameters()
			total += block.attention.outputProjection.Weight.Numel()
			if block.attention.valueEmbeddingGate != nil {
				total += block.attention.valueEmbeddingGate.Weight.Numel()
			}
			total += block.mlp.inputProjection.Weight.Numel()
			total += block.mlp.outputProjection.Weight.Numel()
			continue
		}
		total += block.attention.queryProjection.Weight.Numel()
		total += block.attention.keyProjection.Weight.Numel()
		total += block.attention.valueProjection.Weight.Numel()
		total += block.attention.outputProjection.Weight.Numel()
		if block.attention.queryDown != nil {
			total += block.attention.queryDown.Weight.Numel()
		}
		if block.attention.kvDown != nil {
			total += block.attention.kvDown.Weight.Numel()
		}
		if block.attention.valueEmbeddingGate != nil {
			total += block.attention.valueEmbeddingGate.Weight.Numel()
		}
		total += block.mlp.inputProjection.Weight.Numel()
		total += block.mlp.outputProjection.Weight.Numel()
	}
	total += model.lmHead.Weight.Numel()
	total += model.smearGate.Weight.Numel()
	return total
}

// attentionLengths returns the global and local (sliding-window) attention
// context lengths for one layer at the given context length. With compression
// the global branch attends to ceil(context/ratio) compressed entries and, when
// SWAWindow is set, the local branch attends to at most SWAWindow recent
// tokens. Without compression the historical per-layer window applies.
func (model *Transformer) attentionLengths(layer, contextLength int) (global, local int) {
	if model.Config.Compression() > 1 {
		ratio := model.Config.Compression()
		global = (contextLength + ratio - 1) / ratio
		if global < 1 {
			global = 1
		}
		local = model.Config.SWAWindowSize()
		if local > contextLength {
			local = contextLength
		}
		return global, local
	}
	global = contextLength
	if window := model.windowSizes[layer][0]; window >= 0 && window < global {
		global = window
	}
	return global, 0
}

// EstimateFlopsPerToken returns the FLOPs per token for a full forward+backward
// pass, following nanochat's formula: 6 flops per matmul param plus attention
// flops (12*h*q*effective_seq per layer, capped by the sliding window).
func (model *Transformer) EstimateFlopsPerToken() float64 {
	headCount := model.Config.NumHead
	headDimension := model.Config.HeadDim()
	sequenceLength := model.Config.SequenceLen
	attentionFlops := 0.0
	for layer := 0; layer < model.Config.NumLayer; layer++ {
		global, local := model.attentionLengths(layer, sequenceLength)
		attentionFlops += 12 * float64(headCount) * float64(headDimension) * float64(global+local)
	}
	return 6*float64(model.MatmulParams()) + attentionFlops
}

// EstimateDecodeFlops returns the forward FLOPs to decode one token at the
// given context length.
func (model *Transformer) EstimateDecodeFlops(contextLength int) float64 {
	headCount := model.Config.NumHead
	headDimension := model.Config.HeadDim()
	attentionFlops := 0.0
	for layer := 0; layer < model.Config.NumLayer; layer++ {
		global, local := model.attentionLengths(layer, contextLength)
		attentionFlops += 4 * float64(headCount) * float64(headDimension) * float64(global+local)
	}
	return 2*float64(model.MatmulParams()) + attentionFlops
}

// EstimatePrefillFlops returns the forward FLOPs to prefill numTokens tokens.
func (model *Transformer) EstimatePrefillFlops(numTokens int) float64 {
	if model.Config.CEDEnabled() {
		return model.estimatePrefillFlopsCED(numTokens)
	}
	headCount := model.Config.NumHead
	headDimension := model.Config.HeadDim()
	attentionFlops := 0.0
	ratio := model.Config.Compression()
	for layer := 0; layer < model.Config.NumLayer; layer++ {
		global, local := model.attentionLengths(layer, numTokens)
		var globalPairs, localPairs float64
		if ratio > 1 {
			// Each compressed block holds `ratio` queries that attend to the
			// strictly preceding compressed keys.
			blockCount := float64(global)
			globalPairs = float64(ratio) * blockCount * (blockCount - 1) / 2
		} else {
			effective := float64(global)
			globalPairs = effective*(effective+1)/2 + float64(numTokens-int(effective))*effective
		}
		if local > 0 {
			effective := float64(local)
			localPairs = effective*(effective+1)/2 + float64(numTokens-int(effective))*effective
		}
		attentionFlops += 4 * float64(headCount) * float64(headDimension) * (globalPairs + localPairs)
	}
	return 2*float64(model.MatmulParams())*float64(numTokens) + attentionFlops
}

// estimatePrefillFlopsCED accounts for the causal encoder-decoder split: the
// encoder runs over the full prompt, while the decoder is replayed only over
// the last SWAWindow tokens. Decoder global keys/values are still projected
// from the full encoder hidden state.
func (model *Transformer) estimatePrefillFlopsCED(numTokens int) float64 {
	config := model.Config
	headCount := config.NumHead
	headDimension := config.HeadDim()
	replayTokens := numTokens
	if window := config.SWAWindowSize(); window < replayTokens {
		replayTokens = window
	}
	ratio := config.Compression()

	matmulFlops := 0.0
	for layer := 0; layer < config.NumLayer; layer++ {
		block := model.blocks[layer]
		isDecoder := config.IsDecoderLayer(layer)
		tokens := float64(numTokens)
		if isDecoder {
			tokens = float64(replayTokens)
		}
		blockParams := block.attention.queryProjection.Weight.Numel() +
			block.attention.outputProjection.Weight.Numel() +
			block.mlp.inputProjection.Weight.Numel() +
			block.mlp.outputProjection.Weight.Numel()
		if block.attention.queryDown != nil {
			blockParams += block.attention.queryDown.Weight.Numel()
		}
		if block.attention.kvDown != nil {
			blockParams += block.attention.kvDown.Weight.Numel()
		}
		if block.attention.valueEmbeddingGate != nil {
			blockParams += block.attention.valueEmbeddingGate.Weight.Numel()
		}
		kvParams := block.attention.keyProjection.Weight.Numel() + block.attention.valueProjection.Weight.Numel()
		if isDecoder {
			// Global K/V over the full encoder hidden state plus local K/V over
			// the replayed tokens.
			matmulFlops += 2 * float64(kvParams) * (float64(numTokens) + tokens)
		} else {
			matmulFlops += 2 * float64(kvParams) * tokens
		}
		matmulFlops += 2 * float64(blockParams) * tokens
	}
	matmulFlops += 2 * float64(model.lmHead.Weight.Numel()) * float64(replayTokens)
	matmulFlops += 2 * float64(model.smearGate.Weight.Numel()) * float64(numTokens)

	attentionFlops := 0.0
	for layer := 0; layer < config.NumLayer; layer++ {
		global, local := model.attentionLengths(layer, numTokens)
		if config.IsDecoderLayer(layer) {
			// Only the replayed queries are computed, each still attending to
			// the full global compressed key set and the local window.
			attentionFlops += 4 * float64(headCount) * float64(headDimension) * float64(replayTokens) * (float64(global) + float64(local))
			continue
		}
		var globalPairs, localPairs float64
		if ratio > 1 {
			blockCount := float64(global)
			globalPairs = float64(ratio) * blockCount * (blockCount - 1) / 2
		}
		if local > 0 {
			effective := float64(local)
			localPairs = effective*(effective+1)/2 + float64(numTokens-int(effective))*effective
		}
		attentionFlops += 4 * float64(headCount) * float64(headDimension) * (globalPairs + localPairs)
	}
	return matmulFlops + attentionFlops
}

// WeightReadBytes returns the bytes of matmul weights read by one decode step.
// Decode re-reads every matmul parameter once per step; embeddings and scalar
// parameters are negligible and excluded, matching MatmulParams.
func (model *Transformer) WeightReadBytes() int {
	return model.MatmulParams() * 4
}

// KVBytesPerToken returns the bytes to store one token of KV cache across all
// layers (float32).
func (model *Transformer) KVBytesPerToken() int {
	headDimension := model.Config.HeadDim()
	return model.Config.NumLayer * 2 * model.Config.NumKVHead * headDimension * 4
}

// KVReadBytes returns the bytes of KV cache read by one decode step at the
// given context length (sliding-window layers read only the recent window).
func (model *Transformer) KVReadBytes(contextLength int) int {
	headDimension := model.Config.HeadDim()
	total := 0
	for _, windowSize := range model.windowSizes {
		window := windowSize[0]
		effectiveLength := contextLength
		if window >= 0 && window < effectiveLength {
			effectiveLength = window
		}
		total += 2 * model.Config.NumKVHead * headDimension * 4 * effectiveLength
	}
	return total
}

// ScalingParams reports parameter counts per group, mirroring nanochat's
// num_scaling_params. Kaplan/Chinchilla differ on which groups to include;
// callers (scaling laws) pick the combination that fits best.
type ScalingParams struct {
	WTE                 int
	ValueEmbeds         int
	LMHead              int
	TransformerMatrices int
	Scalars             int
	Total               int
}

// ScalingParamsForConfig returns the number of scaling parameters
// (transformer matrices + lm_head) for a config, computed without building a
// model. This is the count nanochat uses for its scaling-law fits.
func ScalingParamsForConfig(config Config) int64 {
	config.Validate()
	embeddingDimension := config.EmbedDim
	// cq, ck, cv, cproj (4*E*E) + c_fc, c_proj (8*E*E), with the query and KV
	// blocks replaced by their low-rank factorizations when configured.
	queryParams := embeddingDimension * embeddingDimension
	if rank := config.QueryRank(); rank > 0 {
		queryParams = 2 * embeddingDimension * rank
	}
	kvParams := 2 * embeddingDimension * embeddingDimension
	if rank := config.KVRank(); rank > 0 {
		kvParams = 3 * embeddingDimension * rank
	}
	perBlock := queryParams + kvParams + embeddingDimension*embeddingDimension + 8*embeddingDimension*embeddingDimension
	if config.MLAEnabled() {
		perBlock = mlaParamsForConfig(config) + embeddingDimension*embeddingDimension + 8*embeddingDimension*embeddingDimension
	}
	numValueEmbeddings := (config.NumLayer + 1) / 2
	transformerMatrices := int64(config.NumLayer)*int64(perBlock) + int64(numValueEmbeddings)*int64(12*config.NumKVHead)
	lmHead := int64(config.EmbedDim) * int64(config.PaddedVocab())
	return transformerMatrices + lmHead
}

// EstimateFlopsPerTokenForConfig returns the FLOPs per token for a config,
// computed formulaically without building a model.
func EstimateFlopsPerTokenForConfig(config Config) float64 {
	config.Validate()
	embeddingDimension := config.EmbedDim
	queryParams := embeddingDimension * embeddingDimension
	if rank := config.QueryRank(); rank > 0 {
		queryParams = 2 * embeddingDimension * rank
	}
	kvParams := 2 * embeddingDimension * embeddingDimension
	if rank := config.KVRank(); rank > 0 {
		kvParams = 3 * embeddingDimension * rank
	}
	perBlock := queryParams + kvParams + embeddingDimension*embeddingDimension + 8*embeddingDimension*embeddingDimension
	if config.MLAEnabled() {
		perBlock = mlaParamsForConfig(config) + embeddingDimension*embeddingDimension + 8*embeddingDimension*embeddingDimension
	}
	numValueEmbeddings := (config.NumLayer + 1) / 2
	matmulParameters := int64(config.NumLayer)*int64(perBlock) +
		int64(numValueEmbeddings)*int64(12*config.NumKVHead) +
		int64(embeddingDimension)*int64(config.PaddedVocab()) + 24 // lm_head + smear_gate

	headCount := config.NumHead
	headDimension := config.HeadDim()
	sequenceLength := config.SequenceLen
	attentionFlops := 0.0
	for _, windowSize := range config.WindowSizes() {
		window := windowSize[0]
		effectiveLength := sequenceLength
		if window >= 0 && window < effectiveLength {
			effectiveLength = window
		}
		attentionFlops += 12 * float64(headCount) * float64(headDimension) * float64(effectiveLength)
	}
	return 6*float64(matmulParameters) + attentionFlops
}

// NumScalingParams returns the per-group parameter counts.
func (model *Transformer) NumScalingParams() ScalingParams {
	var scalingParams ScalingParams
	scalingParams.WTE = model.tokenEmbedding.Weight.Numel()
	for _, valueEmbedding := range model.valueEmbeds {
		scalingParams.ValueEmbeds += valueEmbedding.Weight.Numel()
	}
	scalingParams.LMHead = model.lmHead.Weight.Numel()
	for _, block := range model.blocks {
		if block.attention.mla != nil {
			scalingParams.TransformerMatrices += block.attention.mla.numParameters()
			scalingParams.TransformerMatrices += block.attention.outputProjection.Weight.Numel()
			if block.attention.valueEmbeddingGate != nil {
				scalingParams.TransformerMatrices += block.attention.valueEmbeddingGate.Weight.Numel()
			}
			scalingParams.TransformerMatrices += block.mlp.inputProjection.Weight.Numel()
			scalingParams.TransformerMatrices += block.mlp.outputProjection.Weight.Numel()
			continue
		}
		scalingParams.TransformerMatrices += block.attention.queryProjection.Weight.Numel()
		scalingParams.TransformerMatrices += block.attention.keyProjection.Weight.Numel()
		scalingParams.TransformerMatrices += block.attention.valueProjection.Weight.Numel()
		scalingParams.TransformerMatrices += block.attention.outputProjection.Weight.Numel()
		if block.attention.queryDown != nil {
			scalingParams.TransformerMatrices += block.attention.queryDown.Weight.Numel()
		}
		if block.attention.kvDown != nil {
			scalingParams.TransformerMatrices += block.attention.kvDown.Weight.Numel()
		}
		if block.attention.valueEmbeddingGate != nil {
			scalingParams.TransformerMatrices += block.attention.valueEmbeddingGate.Weight.Numel()
		}
		scalingParams.TransformerMatrices += block.mlp.inputProjection.Weight.Numel()
		scalingParams.TransformerMatrices += block.mlp.outputProjection.Weight.Numel()
	}
	scalingParams.Scalars = model.residLambdas.Numel() + model.x0Lambdas.Numel() +
		model.smearGate.Weight.Numel() + model.smearLambda.Numel() + model.backoutLambda.Numel()
	scalingParams.Total = scalingParams.WTE + scalingParams.ValueEmbeds + scalingParams.LMHead + scalingParams.TransformerMatrices + scalingParams.Scalars
	return scalingParams
}
