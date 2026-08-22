package model

import (
	"math"

	"github.com/cookiengineer/gonano/tensor"
)

// rotaryBase is the RoPE frequency base used by nanochat.
const rotaryBase = 100000

// precomputeRotary returns cos and sin tables of shape [seqLen, headDim/2] for
// the given head dimension. Positions stride over half the head dimension, as
// in the standard RoPE formulation.
func precomputeRotary(seqLen, headDim int) (cos, sin *tensor.Tensor) {
	half := headDim / 2
	cos = tensor.New(seqLen, half)
	sin = tensor.New(seqLen, half)
	for t := 0; t < seqLen; t++ {
		for j := 0; j < half; j++ {
			// inv_freq[j] = base^(-2j/headDim), then freq = t * inv_freq.
			invFreq := math.Pow(rotaryBase, -2.0*float64(j)/float64(headDim))
			freq := float64(t) * invFreq
			cos.Set2(t, j, float32(math.Cos(freq)))
			sin.Set2(t, j, float32(math.Sin(freq)))
		}
	}
	return cos, sin
}

// ApplyRotary rotates the last dimension of x (shape [B,T,H,D]) using RoPE.
// cos/sin have shape [seqLen, D/2]; t0 offsets the position of the first
// token (nonzero during KV-cache decode). The result is a new tensor.
//
// For each token at position p = t0+t, the first and second halves of the last
// dimension are rotated pairwise: y1 = x1*cos + x2*sin, y2 = -x1*sin + x2*cos.
func ApplyRotary(x, cos, sin *tensor.Tensor, t0 int) *tensor.Tensor {
	b, t, h, d := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	half := d / 2
	out := tensor.New(b, t, h, d)
	xd, od := x.Data, out.Data
	cd, sd := cos.Data, sin.Data
	for bb := 0; bb < b; bb++ {
		for tt := 0; tt < t; tt++ {
			p := t0 + tt
			c := cd[p*half : (p+1)*half]
			s := sd[p*half : (p+1)*half]
			for hh := 0; hh < h; hh++ {
				base := ((bb*t+tt)*h + hh) * d
				for j := 0; j < half; j++ {
					x1 := xd[base+j]
					x2 := xd[base+half+j]
					od[base+j] = x1*c[j] + x2*s[j]
					od[base+half+j] = -x1*s[j] + x2*c[j]
				}
			}
		}
	}
	return out
}
