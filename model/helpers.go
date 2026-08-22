package model

import "github.com/cookiengineer/gonano/tensor"

// smearGateChannels is the number of leading embedding channels the smear
// gate reads (nanochat hardcodes 24).
const smearGateChannels = 24

// sliceChannels returns the first `channels` channels of x (shape [B,T,C]) for
// the given row range [rowStart, rowEnd), as a contiguous [B, rows, channels]
// tensor. It is used to feed the smear and value-embedding gates, which read
// only a small prefix of each token's embedding.
func sliceChannels(x *tensor.Tensor, rowStart, rowEnd, channels int) *tensor.Tensor {
	b, t, c := x.Shape[0], x.Shape[1], x.Shape[2]
	rows := rowEnd - rowStart
	out := tensor.New(b, rows, channels)
	xd, od := x.Data, out.Data
	for bb := 0; bb < b; bb++ {
		for r := rowStart; r < rowEnd; r++ {
			src := xd[(bb*t+r)*c : (bb*t+r)*c+channels]
			copy(od[(bb*rows+(r-rowStart))*channels:], src)
		}
	}
	return out
}

// smearAdd mixes the previous token's embedding into positions 1..T-1 in
// place: x[:, 1:, :] += gate[:, :, 0] * x[:, :-1, :], where gate has shape
// [B, T-1, 1] (one scalar per position).
func smearAdd(x, gate *tensor.Tensor) {
	b, t, c := x.Shape[0], x.Shape[1], x.Shape[2]
	xd, gd := x.Data, gate.Data
	for bb := 0; bb < b; bb++ {
		for r := 1; r < t; r++ {
			g := gd[bb*(t-1)+(r-1)]
			base := (bb*t + r) * c
			prev := (bb*t + (r - 1)) * c
			for j := 0; j < c; j++ {
				xd[base+j] += g * xd[prev+j]
			}
		}
	}
}

// hasVE reports whether a layer at layerIdx has a value embedding. Alternating
// layers have one, and the last layer always does.
func hasVE(layerIdx, numLayers int) bool {
	return layerIdx%2 == (numLayers-1)%2
}
