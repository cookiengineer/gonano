package model

import (
	"math"

	"github.com/cookiengineer/gonano/nn"
	"github.com/cookiengineer/gonano/tensor"
)

// softcap is the logit softcap used before loss/sampling.
const softcap = 15.0

// rotaryOvercompute is the factor by which the rotary embedding cache exceeds
// the training sequence length (nanochat over-computes by 10x).
const rotaryOvercompute = 10

// uniformBound returns nanochat's uniform half-width sqrt(3) * n_embd^-0.5.
func uniformBound(embedDim int) float32 {
	return float32(math.Sqrt(3.0) / math.Sqrt(float64(embedDim)))
}

// Transformer is the nanochat GPT model.
type Transformer struct {
	Config      Config
	paddedVocab int
	windowSizes [][2]int

	wte *nn.Embedding
	h   []*Block
	lm  *nn.Linear

	residLambdas  *tensor.Tensor // [n_layer]
	x0Lambdas     *tensor.Tensor // [n_layer]
	smearGate     *nn.Linear    // [24, 1]
	smearLambda   *tensor.Tensor // [1]
	backoutLambda *tensor.Tensor // [1]

	valueEmbeds map[int]*nn.Embedding // layer -> embedding (ResFormer)

	cos, sin *tensor.Tensor // [rotarySeqLen, headDim/2]
}

// NewTransformer builds a Transformer with all parameters zero-initialized.
// Call InitWeights to initialize them.
func NewTransformer(cfg Config) *Transformer {
	cfg.Validate()
	padded := cfg.PaddedVocab()
	headDim := cfg.HeadDim()
	cos, sin := precomputeRotary(cfg.SequenceLen*rotaryOvercompute, headDim)

	m := &Transformer{
		Config:        cfg,
		paddedVocab:   padded,
		windowSizes:   cfg.WindowSizes(),
		wte:           nn.NewEmbedding(padded, cfg.EmbedDim),
		h:             make([]*Block, cfg.NumLayer),
		lm:            nn.NewLinear(cfg.EmbedDim, padded),
		residLambdas:  tensor.New(cfg.NumLayer),
		x0Lambdas:     tensor.New(cfg.NumLayer),
		smearGate:     nn.NewLinear(smearGateChannels, 1),
		smearLambda:   tensor.New(1),
		backoutLambda: tensor.New(1),
		valueEmbeds:   make(map[int]*nn.Embedding),
		cos:           cos,
		sin:           sin,
	}
	for i := 0; i < cfg.NumLayer; i++ {
		m.h[i] = NewBlock(cfg, hasVE(i, cfg.NumLayer))
		if hasVE(i, cfg.NumLayer) {
			m.valueEmbeds[i] = nn.NewEmbedding(padded, cfg.NumKVHead*headDim)
		}
	}
	return m
}

// NumLayers returns the number of transformer blocks.
func (m *Transformer) NumLayers() int { return m.Config.NumLayer }

// PaddedVocab returns the padded vocabulary size.
func (m *Transformer) PaddedVocab() int { return m.paddedVocab }

// KVHeadDim returns the KV embedding dimension (numKVHead * headDim).
func (m *Transformer) KVHeadDim() int { return m.Config.NumKVHead * m.Config.HeadDim() }

// InitWeights initializes every parameter exactly as nanochat's init_weights.
func (m *Transformer) InitWeights(rng *tensor.RNG) {
	cfg := m.Config
	s := uniformBound(cfg.EmbedDim)

	nn.InitNormal(m.wte.Weight, rng, 0.8)
	nn.InitNormal(m.lm.Weight, rng, 0.001)

	for _, block := range m.h {
		nn.InitUniform(block.Attn.cq.Weight, rng, -s, s)
		nn.InitUniform(block.Attn.ck.Weight, rng, -s, s)
		nn.InitUniform(block.Attn.cv.Weight, rng, -s, s)
		nn.InitZeros(block.Attn.cproj.Weight)
		nn.InitUniform(block.MLP.CFc.Weight, rng, -0.4*s, 0.4*s)
		nn.InitZeros(block.MLP.CProj.Weight)
	}

	nLayer := cfg.NumLayer
	for i := 0; i < nLayer; i++ {
		m.residLambdas.Data[i] = 1.15 - 0.10*float32(i)/float32(max(nLayer-1, 1))
		m.x0Lambdas.Data[i] = 0.20 - 0.15*float32(i)/float32(max(nLayer-1, 1))
	}

	m.smearLambda.Data[0] = 0
	m.backoutLambda.Data[0] = 0.2
	nn.InitUniform(m.smearGate.Weight, rng, 0.0, 0.02)

	for _, ve := range m.valueEmbeds {
		nn.InitUniform(ve.Weight, rng, -s, s)
	}

	for _, block := range m.h {
		if block.Attn.veGate != nil {
			nn.InitUniform(block.Attn.veGate.Weight, rng, 0.0, 0.02)
		}
	}
}

// Forward runs the model and returns the softcapped logits of shape
// [B,T,vocab]. idx has shape [B,T]. cache, when non-nil, enables
// autoregressive inference (KV-cache). Loss computation is intentionally kept
// out of the model; callers use tensor.CrossEntropy on the returned logits.
func (m *Transformer) Forward(idx *tensor.Int32s, cache *KVBuffer) *tensor.Tensor {
	b, t := idx.Shape[0], idx.Shape[1]
	if t > m.Config.SequenceLen {
		panic("model: sequence longer than rotary cache")
	}

	x := m.wte.Forward(idx) // [B,T,C]
	x = normLastDim(x)

	t0 := 0
	if cache != nil {
		t0 = cache.Position()
	}

	// Smear: mix the previous token's embedding into the current token.
	switch {
	case cache == nil:
		if t <= 1 {
			panic("model: training forward requires T > 1")
		}
		gate := m.smearGate.Forward(sliceChannels(x, 1, t, smearGateChannels)) // [B,T-1,1]
		gate = tensor.Scale(tensor.Sigmoid(gate), m.smearLambda.Data[0])
		smearAdd(x, gate)
	default:
		xPre := cache.PrevEmbedding()
		cache.SetPrevEmbedding(lastTokenEmbedding(x))
		if t > 1 {
			gate := m.smearGate.Forward(sliceChannels(x, 1, t, smearGateChannels))
			gate = tensor.Scale(tensor.Sigmoid(gate), m.smearLambda.Data[0])
			smearAdd(x, gate)
		} else if xPre != nil {
			gate := m.smearGate.Forward(sliceChannels(x, 0, 1, smearGateChannels)) // [B,1,1]
			gate = tensor.Scale(tensor.Sigmoid(gate), m.smearLambda.Data[0])
			smearDecode(x, gate, xPre)
		}
	}

	x0 := x
	backoutLayer := m.Config.NumLayer / 2
	var xBackout *tensor.Tensor
	for i, block := range m.h {
		x = combineResidual(x, x0, m.residLambdas.Data[i], m.x0Lambdas.Data[i])
		var ve *tensor.Tensor
		if emb, ok := m.valueEmbeds[i]; ok {
			ve = emb.Forward(idx)
		}
		x = block.Forward(x, ve, m.cos, m.sin, t0, m.windowSizes[i], cache, i)
		if i == backoutLayer {
			xBackout = x.Clone()
		}
	}

	if cache != nil {
		cache.Advance(t)
	}

	if xBackout != nil {
		x = tensor.AddScaled(x, xBackout, -m.backoutLambda.Data[0])
	}
	x = normLastDim(x)

	logits := m.lm.Forward(x) // [B,T,paddedVocab]
	logits = trimVocab(logits, m.paddedVocab, m.Config.VocabSize)
	logits = tensor.Softcap(logits, softcap)
	return logits.Reshape(b, t, m.Config.VocabSize)
}

// lastTokenEmbedding copies the last token's embedding of each batch row into
// a fresh [B,1,C] tensor.
func lastTokenEmbedding(x *tensor.Tensor) *tensor.Tensor {
	b, t, c := x.Shape[0], x.Shape[1], x.Shape[2]
	out := tensor.New(b, 1, c)
	for bb := 0; bb < b; bb++ {
		src := x.Data[(bb*t+(t-1))*c : (bb*t+(t-1))*c+c]
		copy(out.Data[bb*c:(bb+1)*c], src)
	}
	return out
}

// smearDecode adds gate * xPre to x for the single-token decode case. gate has
// shape [B,1,1]; x and xPre have shape [B,1,C].
func smearDecode(x, gate, xPre *tensor.Tensor) {
	b, c := x.Shape[0], x.Shape[2]
	xd, gd, pd := x.Data, gate.Data, xPre.Data
	for bb := 0; bb < b; bb++ {
		g := gd[bb]
		base := bb * c
		for j := 0; j < c; j++ {
			xd[base+j] += g * pd[base+j]
		}
	}
}

// trimVocab crops the trailing padded-vocab channels down to vocabSize, given
// a [B*T, paddedVocab] tensor.
func trimVocab(logits *tensor.Tensor, padded, vocab int) *tensor.Tensor {
	rows := logits.Numel() / padded
	out := tensor.New(rows, vocab)
	for i := 0; i < rows; i++ {
		copy(out.Data[i*vocab:(i+1)*vocab], logits.Data[i*padded:i*padded+vocab])
	}
	return out
}

// combineResidual computes resid*x + x0lamb*x0 element-wise.
func combineResidual(x, x0 *tensor.Tensor, resid, x0lamb float32) *tensor.Tensor {
	out := tensor.New(x.Shape...)
	xd, od, zd := x.Data, out.Data, x0.Data
	for i := range xd {
		od[i] = resid*xd[i] + x0lamb*zd[i]
	}
	return out
}
