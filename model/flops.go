package model

// FLOPs and parameter accounting, ported from nanochat's gpt.py estimate_flops
// and num_scaling_params.

// MatmulParams returns the number of parameters participating in matrix
// multiplications with the token stream (every nn.Linear weight). Embeddings
// and scalar parameters are excluded.
func (m *Transformer) MatmulParams() int {
	total := 0
	for _, block := range m.h {
		total += block.Attn.cq.Weight.Numel()
		total += block.Attn.ck.Weight.Numel()
		total += block.Attn.cv.Weight.Numel()
		total += block.Attn.cproj.Weight.Numel()
		if block.Attn.veGate != nil {
			total += block.Attn.veGate.Weight.Numel()
		}
		total += block.MLP.CFc.Weight.Numel()
		total += block.MLP.CProj.Weight.Numel()
	}
	total += m.lm.Weight.Numel()
	total += m.smearGate.Weight.Numel()
	return total
}

// EstimateFlopsPerToken returns the FLOPs per token for a full forward+backward
// pass, following nanochat's formula: 6 flops per matmul param plus attention
// flops (12*h*q*effective_seq per layer, capped by the sliding window).
func (m *Transformer) EstimateFlopsPerToken() float64 {
	h := m.Config.NumHead
	q := m.Config.HeadDim()
	t := m.Config.SequenceLen
	attn := 0.0
	for _, w := range m.windowSizes {
		window := w[0]
		eff := t
		if window >= 0 && window < eff {
			eff = window
		}
		attn += 12 * float64(h) * float64(q) * float64(eff)
	}
	return 6*float64(m.MatmulParams()) + attn
}

// EstimateDecodeFlops returns the forward FLOPs to decode one token at the
// given context length.
func (m *Transformer) EstimateDecodeFlops(contextLen int) float64 {
	h := m.Config.NumHead
	q := m.Config.HeadDim()
	attn := 0.0
	for _, w := range m.windowSizes {
		window := w[0]
		eff := contextLen
		if window >= 0 && window < eff {
			eff = window
		}
		attn += 4 * float64(h) * float64(q) * float64(eff)
	}
	return 2*float64(m.MatmulParams()) + attn
}

// EstimatePrefillFlops returns the forward FLOPs to prefill numTokens tokens.
func (m *Transformer) EstimatePrefillFlops(numTokens int) float64 {
	h := m.Config.NumHead
	q := m.Config.HeadDim()
	attn := 0.0
	for _, w := range m.windowSizes {
		window := w[0]
		weff := window
		if window < 0 || window > numTokens {
			weff = numTokens
		}
		attended := float64(weff)*(float64(weff)+1)/2 + float64(numTokens-weff)*float64(weff)
		attn += 4 * float64(h) * float64(q) * attended
	}
	return 2*float64(m.MatmulParams())*float64(numTokens) + attn
}

// KVBytesPerToken returns the bytes to store one token of KV cache across all
// layers (float32).
func (m *Transformer) KVBytesPerToken() int {
	headDim := m.Config.HeadDim()
	return m.Config.NumLayer * 2 * m.Config.NumKVHead * headDim * 4
}

// KVReadBytes returns the bytes of KV cache read by one decode step at the
// given context length (sliding-window layers read only the recent window).
func (m *Transformer) KVReadBytes(contextLen int) int {
	headDim := m.Config.HeadDim()
	total := 0
	for _, w := range m.windowSizes {
		window := w[0]
		eff := contextLen
		if window >= 0 && window < eff {
			eff = window
		}
		total += 2 * m.Config.NumKVHead * headDim * 4 * eff
	}
	return total
}

// ScalingParams reports parameter counts per group, mirroring nanochat's
// num_scaling_params. Kaplan/Chinchilla differ on which groups to include;
// callers (scaling laws) pick the combination that fits best.
type ScalingParams struct {
	WTE                   int
	ValueEmbeds           int
	LMHead                int
	TransformerMatrices   int
	Scalars               int
	Total                 int
}

// ScalingParamsForConfig returns the number of scaling parameters
// (transformer matrices + lm_head) for a config, computed without building a
// model. This is the count nanochat uses for its scaling-law fits.
func ScalingParamsForConfig(cfg Config) int64 {
	cfg.Validate()
	e := cfg.EmbedDim
	perBlock := 12 * e * e // cq, ck, cv, cproj (4*E*E) + c_fc, c_proj (8*E*E)
	numVE := (cfg.NumLayer + 1) / 2
	transformerMatrices := int64(cfg.NumLayer)*int64(perBlock) + int64(numVE)*int64(12*cfg.NumKVHead)
	lmHead := int64(cfg.EmbedDim) * int64(cfg.PaddedVocab())
	return transformerMatrices + lmHead
}

// EstimateFlopsPerTokenForConfig returns the FLOPs per token for a config,
// computed formulaically without building a model.
func EstimateFlopsPerTokenForConfig(cfg Config) float64 {
	cfg.Validate()
	e := cfg.EmbedDim
	perBlock := 12 * e * e
	numVE := (cfg.NumLayer + 1) / 2
	matmulParams := int64(cfg.NumLayer)*int64(perBlock) +
		int64(numVE)*int64(12*cfg.NumKVHead) +
		int64(e)*int64(cfg.PaddedVocab()) + 24 // lm_head + smear_gate

	h := cfg.NumHead
	q := cfg.HeadDim()
	t := cfg.SequenceLen
	attn := 0.0
	for _, w := range cfg.WindowSizes() {
		window := w[0]
		eff := t
		if window >= 0 && window < eff {
			eff = window
		}
		attn += 12 * float64(h) * float64(q) * float64(eff)
	}
	return 6*float64(matmulParams) + attn
}

// NumScalingParams returns the per-group parameter counts.
func (m *Transformer) NumScalingParams() ScalingParams {
	var p ScalingParams
	p.WTE = m.wte.Weight.Numel()
	for _, ve := range m.valueEmbeds {
		p.ValueEmbeds += ve.Weight.Numel()
	}
	p.LMHead = m.lm.Weight.Numel()
	for _, block := range m.h {
		p.TransformerMatrices += block.Attn.cq.Weight.Numel()
		p.TransformerMatrices += block.Attn.ck.Weight.Numel()
		p.TransformerMatrices += block.Attn.cv.Weight.Numel()
		p.TransformerMatrices += block.Attn.cproj.Weight.Numel()
		if block.Attn.veGate != nil {
			p.TransformerMatrices += block.Attn.veGate.Weight.Numel()
		}
		p.TransformerMatrices += block.MLP.CFc.Weight.Numel()
		p.TransformerMatrices += block.MLP.CProj.Weight.Numel()
	}
	p.Scalars = m.residLambdas.Numel() + m.x0Lambdas.Numel() +
		m.smearGate.Weight.Numel() + m.smearLambda.Numel() + m.backoutLambda.Numel()
	p.Total = p.WTE + p.ValueEmbeds + p.LMHead + p.TransformerMatrices + p.Scalars
	return p
}
