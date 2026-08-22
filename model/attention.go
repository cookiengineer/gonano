package model

import (
	"github.com/cookiengineer/gonano/nn"
	"github.com/cookiengineer/gonano/parallel"
	"github.com/cookiengineer/gonano/tensor"
)

// normEps is the epsilon used in RMSNorm, matching torch's RMSNorm default.
const normEps = 1e-6

// qkScale is nanochat's "sharper attention" scaling, split across Q and K.
const qkScale = 1.2

// veGateChannels is the number of leading embedding channels the value-embedding
// gate reads.
const veGateChannels = 12

// veGateScale multiplies the sigmoid gate so it ranges over (0, 3).
const veGateScale = 3.0

// maskedScore replaces disallowed attention positions before softmax. It is a
// large negative finite value so that exp() underflows to zero.
const maskedScore = -1e30

// normLastDim applies stateless RMSNorm over the last dimension of an
// arbitrary-rank tensor.
func normLastDim(x *tensor.Tensor) *tensor.Tensor {
	last := x.Shape[len(x.Shape)-1]
	rows := x.Numel() / last
	return tensor.RMSNormLastDim(x.Reshape(rows, last), normEps).Reshape(x.Shape...)
}

// toBH transposes [B,T,H,D] to [B,H,T,D], making per-(batch,head) slices
// contiguous for the attention kernels.
func toBH(x *tensor.Tensor) *tensor.Tensor {
	b, t, h, d := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	out := tensor.New(b, h, t, d)
	src, dst := x.Data, out.Data
	parallel.Default().For(0, b*h, func(i int) {
		bb := i / h
		hh := i % h
		for tt := 0; tt < t; tt++ {
			s := ((bb*t+tt)*h + hh) * d
			copy(dst[((bb*h+hh)*t+tt)*d:], src[s:s+d])
		}
	})
	return out
}

// toBT transposes [B,H,T,D] back to [B,T,H,D].
func toBT(x *tensor.Tensor) *tensor.Tensor {
	b, h, t, d := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	out := tensor.New(b, t, h, d)
	src, dst := x.Data, out.Data
	parallel.Default().For(0, b*h, func(i int) {
		bb := i / h
		hh := i % h
		for tt := 0; tt < t; tt++ {
			s := ((bb*h+hh)*t + tt) * d
			copy(dst[((bb*t+tt)*h+hh)*d:], src[s:s+d])
		}
	})
	return out
}

// attentionHead computes softmax(q @ k^T, masked) @ v for a single head. q is
// [tq,D], k and v are [tk,D], and t0 is the global position of the first
// query token (nonzero during KV-cache decode). window < 0 means full context.
func attentionHead(q, k, v []float32, tq, tk, d, t0, window int) []float32 {
	q2 := tensor.NewWithData([]int{tq, d}, q)
	k2 := tensor.NewWithData([]int{tk, d}, k)
	scores := tensor.MatMulTransB(q2, k2)
	sd := scores.Data
	for i := 0; i < tq; i++ {
		qpos := t0 + i
		for j := 0; j < tk; j++ {
			if j > qpos || (window >= 0 && qpos-j > window) {
				sd[i*tk+j] = maskedScore
			}
		}
	}
	probs := tensor.SoftmaxLastDim(scores)
	v2 := tensor.NewWithData([]int{tk, d}, v)
	return tensor.MatMul(probs, v2).Data
}

// CausalSelfAttention is a multi-head (grouped-query) causal attention layer
// with QK-norm, rotary embeddings, and an optional value-embedding gate.
type CausalSelfAttention struct {
	numHead   int
	numKVHead int
	embedDim  int
	headDim   int

	cq    *nn.Linear
	ck    *nn.Linear
	cv    *nn.Linear
	cproj *nn.Linear

	veGate *nn.Linear // nil on layers without value embeddings
}

// NewCausalSelfAttention builds an attention layer. hasVE selects whether the
// ResFormer value-embedding gate is present on this layer.
func NewCausalSelfAttention(cfg Config, hasVE bool) *CausalSelfAttention {
	headDim := cfg.HeadDim()
	a := &CausalSelfAttention{
		numHead:   cfg.NumHead,
		numKVHead: cfg.NumKVHead,
		embedDim:  cfg.EmbedDim,
		headDim:   headDim,
		cq:        nn.NewLinear(cfg.EmbedDim, cfg.NumHead*headDim),
		ck:        nn.NewLinear(cfg.EmbedDim, cfg.NumKVHead*headDim),
		cv:        nn.NewLinear(cfg.EmbedDim, cfg.NumKVHead*headDim),
		cproj:     nn.NewLinear(cfg.EmbedDim, cfg.EmbedDim),
	}
	if hasVE {
		a.veGate = nn.NewLinear(veGateChannels, cfg.NumKVHead)
	}
	return a
}

// Forward computes attention for x of shape [B,T,C]. ve (value embeddings) is
// non-nil on ResFormer layers. cos/sin are the rotary tables; t0 is the
// position offset; window is the (left,right) sliding window; cache, when
// non-nil, stores/reads KV. The result has shape [B,T,C].
func (a *CausalSelfAttention) Forward(x, ve, cos, sin *tensor.Tensor, t0 int, window [2]int, cache *KVBuffer, layer int) *tensor.Tensor {
	b, t, c := x.Shape[0], x.Shape[1], x.Shape[2]
	_ = c
	q := a.cq.Forward(x).Reshape(b, t, a.numHead, a.headDim)
	k := a.ck.Forward(x).Reshape(b, t, a.numKVHead, a.headDim)
	v := a.cv.Forward(x).Reshape(b, t, a.numKVHead, a.headDim)

	if ve != nil {
		ve4 := ve.Reshape(b, t, a.numKVHead, a.headDim)
		gate := a.veGate.Forward(sliceChannels(x, 0, t, veGateChannels)) // [B,T,Hkv]
		gate = tensor.Scale(tensor.Sigmoid(gate), veGateScale)
		v = addGateTimesVE(v, gate, ve4)
	}

	q = ApplyRotary(q, cos, sin, t0)
	k = ApplyRotary(k, cos, sin, t0)

	q = tensor.Scale(normLastDim(q), qkScale)
	k = tensor.Scale(normLastDim(k), qkScale)

	// Transpose to [B,H,T,D] so per-head slices are contiguous.
	qBH := toBH(q)     // [B, Hq, T, D]
	kBH := toBH(k)     // [B, Hkv, T, D]
	vBH := toBH(v)     // [B, Hkv, T, D]

	headRatio := a.numHead / a.numKVHead
	outBH := tensor.New(b, a.numHead, t, a.headDim)
	d := a.headDim

	parallel.Default().For(0, b*a.numHead, func(i int) {
		bb := i / a.numHead
		hq := i % a.numHead
		kvh := hq / headRatio
		qh := qBH.Data[((bb*a.numHead+hq)*t)*d : ((bb*a.numHead+hq)*t+t)*d]
		var kFull, vFull []float32
		pos := t0
		if cache != nil {
			// Write k/v for this (batch, kv-head) into the cache and read the
			// full prefix as the available context.
			kNew := kBH.Data[((bb*a.numKVHead+kvh)*t)*d : ((bb*a.numKVHead+kvh)*t+t)*d]
			vNew := vBH.Data[((bb*a.numKVHead+kvh)*t)*d : ((bb*a.numKVHead+kvh)*t+t)*d]
			kFull, vFull = cache.writeKeyValue(layer, bb, kvh, kNew, vNew)
		} else {
			kFull = kBH.Data[((bb*a.numKVHead+kvh)*t)*d : ((bb*a.numKVHead+kvh)*t+t)*d]
			vFull = vBH.Data[((bb*a.numKVHead+kvh)*t)*d : ((bb*a.numKVHead+kvh)*t+t)*d]
		}
		tk := len(kFull) / d
		oh := attentionHead(qh, kFull, vFull, t, tk, d, pos, window[0])
		copy(outBH.Data[((bb*a.numHead+hq)*t)*d:], oh)
	})

	out := toBT(outBH).Reshape(b, t, a.numHead*a.headDim)
	return a.cproj.Forward(out)
}

// addGateTimesVE computes v + gate * ve, where gate (shape [B,T,Hkv]) scales
// each row of ve (shape [B,T,Hkv,D]) before adding to v.
func addGateTimesVE(v, gate, ve *tensor.Tensor) *tensor.Tensor {
	b, t, h, d := v.Shape[0], v.Shape[1], v.Shape[2], v.Shape[3]
	out := v.Clone()
	rows := b * t * h
	od, vd, gd := out.Data, ve.Data, gate.Data
	for r := 0; r < rows; r++ {
		g := gd[r]
		base := r * d
		for j := 0; j < d; j++ {
			od[base+j] += g * vd[base+j]
		}
	}
	return out
}
