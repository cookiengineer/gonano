package model

import (
	"github.com/cookiengineer/gonano/tensor"
)

// trainCtx holds the activations saved by TrainForward for TrainBackward.
type trainCtx struct {
	idx *tensor.Int32s

	xEmbeddedNorm *tensor.Tensor // x after wte+norm, before smear
	xRawEmbedding *tensor.Tensor // x after wte, before norm
	smearSlice    *tensor.Tensor // [B,T-1,24] input to the smear gate
	smearSig      *tensor.Tensor // [B,T-1,1] sigmoid output (pre lambda)
	x0            *tensor.Tensor // x after smear

	xPrev    []*tensor.Tensor // x before combineResidual, per layer
	blockCtx []*blockCtx

	xBackout      *tensor.Tensor
	xFinalPreNorm *tensor.Tensor // input to the final norm
	xFinalNorm    *tensor.Tensor // input to lm_head
	logits        *tensor.Tensor // softcapped logits [B,T,vocab]
	valueEmbeds   map[int]*tensor.Tensor
}

// TrainForward runs the model forward, saving activations for backprop. It
// returns the softcapped logits and the training context.
func (m *Transformer) TrainForward(idx *tensor.Int32s) (*tensor.Tensor, *trainCtx) {
	b, t := idx.Shape[0], idx.Shape[1]
	ctx := &trainCtx{idx: idx, valueEmbeds: make(map[int]*tensor.Tensor)}

	x := m.wte.Forward(idx)
	ctx.xRawEmbedding = x
	x = normLastDim(x)
	ctx.xEmbeddedNorm = x

	smearSlice := sliceChannels(x, 1, t, smearGateChannels)
	ctx.smearSlice = smearSlice
	sig := tensor.Sigmoid(m.smearGate.Forward(smearSlice))
	ctx.smearSig = sig
	smearAdd(x, tensor.Scale(sig, m.smearLambda.Data[0]))

	x0 := x
	ctx.x0 = x0

	backoutLayer := m.Config.NumLayer / 2
	var xBackout *tensor.Tensor
	for i, block := range m.h {
		ctx.xPrev = append(ctx.xPrev, x)
		x = combineResidual(x, x0, m.residLambdas.Data[i], m.x0Lambdas.Data[i])
		var ve *tensor.Tensor
		if emb, ok := m.valueEmbeds[i]; ok {
			ve = emb.Forward(idx)
			ctx.valueEmbeds[i] = ve
		}
		var bctx *blockCtx
		x, bctx = block.forwardTrain(x, ve, m.cos, m.sin, 0, m.windowSizes[i])
		ctx.blockCtx = append(ctx.blockCtx, bctx)
		if i == backoutLayer {
			xBackout = x.Clone()
		}
	}
	ctx.xBackout = xBackout

	if xBackout != nil {
		x = tensor.AddScaled(x, xBackout, -m.backoutLambda.Data[0])
	}
	ctx.xFinalPreNorm = x
	x = normLastDim(x)
	ctx.xFinalNorm = x

	logits := m.lm.Forward(x)
	logits = trimVocab(logits, m.paddedVocab, m.Config.VocabSize)
	logits = tensor.Softcap(logits, softcap)
	ctx.logits = logits.Reshape(b, t, m.Config.VocabSize)
	return ctx.logits, ctx
}

// TrainBackward backpropagates gradLogits (the gradient of the loss with
// respect to the softcapped logits) through the model, accumulating gradients
// into every parameter's Grad buffer.
func (m *Transformer) TrainBackward(ctx *trainCtx, gradLogits *tensor.Tensor) {
	b, t := ctx.idx.Shape[0], ctx.idx.Shape[1]
	c := m.Config.EmbedDim

	gradSoft := tensor.SoftcapBackward(ctx.logits, gradLogits, softcap) // [B,T,vocab]
	gradPadded := tensor.New(b*t, m.paddedVocab)
	for i := 0; i < b*t; i++ {
		copy(gradPadded.Data[i*m.paddedVocab:], gradSoft.Data[i*m.Config.VocabSize:(i+1)*m.Config.VocabSize])
	}

	gradFinalNorm := m.lm.Backward(ctx.xFinalNorm, gradPadded).Reshape(b, t, c)
	gradX := normLastDimBackward(ctx.xFinalPreNorm, gradFinalNorm)

	var gradXBackout *tensor.Tensor
	if ctx.xBackout != nil {
		gradXBackout = tensor.Scale(gradX, -m.backoutLambda.Data[0])
		m.backoutLambda.EnsureGrad()
		var s float64
		for i := 0; i < gradX.Numel(); i++ {
			s += float64(gradX.Data[i]) * float64(-ctx.xBackout.Data[i])
		}
		m.backoutLambda.Grad[0] += float32(s)
	}

	n := m.Config.NumLayer
	backoutLayer := n / 2
	gradX0 := tensor.New(b, t, c)
	for i := n - 1; i >= 0; i-- {
		layerGrad := gradX
		if i == backoutLayer && gradXBackout != nil {
			layerGrad = tensor.Add(gradX, gradXBackout)
		}
		gradBlockIn := m.h[i].backwardTrain(layerGrad, ctx.blockCtx[i], m.cos, m.sin, 0)

		if gradVE := ctx.blockCtx[i].attnCtx.gradVE; gradVE != nil {
			kvDim := m.Config.NumKVHead * m.Config.HeadDim()
			m.valueEmbeds[i].Backward(ctx.idx, gradVE.Reshape(b, t, kvDim))
		}

		resid := m.residLambdas.Data[i]
		x0lamb := m.x0Lambdas.Data[i]
		xPrev := ctx.xPrev[i]
		gradX = tensor.Scale(gradBlockIn, resid)
		gradX0 = tensor.AddScaled(gradX0, gradBlockIn, x0lamb)

		m.residLambdas.EnsureGrad()
		m.x0Lambdas.EnsureGrad()
		var sr, sx float64
		for k := 0; k < gradBlockIn.Numel(); k++ {
			sr += float64(gradBlockIn.Data[k]) * float64(xPrev.Data[k])
			sx += float64(gradBlockIn.Data[k]) * float64(ctx.x0.Data[k])
		}
		m.residLambdas.Grad[i] += float32(sr)
		m.x0Lambdas.Grad[i] += float32(sx)
	}
	gradX = tensor.Add(gradX, gradX0)

	// Smear backward.
	gradEmbedded := smearBackward(m, ctx, gradX, b, t, c)

	// Norm between the embedding and the trunk.
	gradRaw := normLastDimBackward(ctx.xRawEmbedding, gradEmbedded)
	m.wte.Backward(ctx.idx, gradRaw)
}

// smearBackward backpropagates through the smear operation and the smear gate,
// returning the gradient with respect to the pre-smear embedding.
func smearBackward(m *Transformer, ctx *trainCtx, gradX *tensor.Tensor, b, t, c int) *tensor.Tensor {
	lambda := m.smearLambda.Data[0]
	gradEmbedded := tensor.New(b, t, c)
	gradZ := tensor.New(b, t-1, 1)
	var gradLambda float64
	xOld := ctx.xEmbeddedNorm

	for bb := 0; bb < b; bb++ {
		for r := 0; r < t; r++ {
			base := (bb*t + r) * c
			if r == 0 {
				for j := 0; j < c; j++ {
					gradEmbedded.Data[base+j] += gradX.Data[base+j]
				}
				continue
			}
			gIdx := bb*(t-1) + (r - 1)
			g := lambda * ctx.smearSig.Data[gIdx]
			prev := (bb*t + (r - 1)) * c
			var gradGate float64
			for j := 0; j < c; j++ {
				gradEmbedded.Data[base+j] += gradX.Data[base+j]
				gradEmbedded.Data[prev+j] += gradX.Data[base+j] * g
				gradGate += float64(gradX.Data[base+j]) * float64(xOld.Data[prev+j])
			}
			sig := ctx.smearSig.Data[gIdx]
			gradLambda += gradGate * float64(sig)
			gradZ.Data[gIdx] = float32(gradGate*float64(lambda)) * sig * (1 - sig)
		}
	}

	m.smearLambda.EnsureGrad()
	m.smearLambda.Grad[0] += float32(gradLambda)
	gradSlice := m.smearGate.Backward(ctx.smearSlice, gradZ)
	scatterAddChannels(gradEmbedded, 1, t, smearGateChannels, gradSlice)
	return gradEmbedded
}

// scatterAddChannels adds src into the first `channels` channels of dst for
// rows [rowStart, rowEnd), reversing sliceChannels.
func scatterAddChannels(dst *tensor.Tensor, rowStart, rowEnd, channels int, src *tensor.Tensor) {
	b, t, c := dst.Shape[0], dst.Shape[1], dst.Shape[2]
	for bb := 0; bb < b; bb++ {
		for r := rowStart; r < rowEnd; r++ {
			from := src.Data[(bb*(rowEnd-rowStart)+(r-rowStart))*channels:]
			to := dst.Data[(bb*t+r)*c : (bb*t+r)*c+channels]
			for j := 0; j < channels; j++ {
				to[j] += from[j]
			}
		}
	}
}
