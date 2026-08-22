package model

import (
	"github.com/cookiengineer/gonano/tensor"
)

// This file implements the training forward/backward passes. Unlike the
// inference Forward, the training forward saves intermediate activations so
// that TrainBackward can compute exact gradients via backpropagation. The
// analytic gradients are verified against finite differences in tests.

// softmaxBackward returns the gradient of a row-wise softmax given the
// probabilities probs and the gradient of the output: grad = probs * (g - <g·p>).
func softmaxBackward(probs, gradProbs *tensor.Tensor) *tensor.Tensor {
	rows, cols := probs.Shape[0], probs.Shape[1]
	out := tensor.New(rows, cols)
	for i := 0; i < rows; i++ {
		var dot float64
		for j := 0; j < cols; j++ {
			dot += float64(gradProbs.Data[i*cols+j]) * float64(probs.Data[i*cols+j])
		}
		for j := 0; j < cols; j++ {
			out.Data[i*cols+j] = probs.Data[i*cols+j] * (gradProbs.Data[i*cols+j] - float32(dot))
		}
	}
	return out
}

// scatterChannels is the reverse of sliceChannels: it places gradOut into the
// first `channels` channels of a zero tensor shaped like the original.
func scatterChannels(x *tensor.Tensor, rowStart, rowEnd, channels int, gradOut *tensor.Tensor) *tensor.Tensor {
	b, t, c := x.Shape[0], x.Shape[1], x.Shape[2]
	out := tensor.New(b, t, c)
	for bb := 0; bb < b; bb++ {
		for r := rowStart; r < rowEnd; r++ {
			src := gradOut.Data[(bb*(rowEnd-rowStart)+(r-rowStart))*channels : (bb*(rowEnd-rowStart)+(r-rowStart)+1)*channels]
			copy(out.Data[(bb*t+r)*c:(bb*t+r)*c+channels], src)
		}
	}
	return out
}

// negateSin returns a negated copy of the sin rotary table, used to transpose
// the rotary rotation during backpropagation.
func negateSin(sin *tensor.Tensor) *tensor.Tensor {
	out := tensor.New(sin.Shape...)
	for i, v := range sin.Data {
		out.Data[i] = -v
	}
	return out
}

// attnCtx holds the activations saved by the training attention forward.
type attnCtx struct {
	qRotary *tensor.Tensor // [B,T,Hq,D] post-rotary, pre-norm
	kRotary *tensor.Tensor // [B,T,Hkv,D] post-rotary, pre-norm
	qBH     *tensor.Tensor // [B,Hq,T,D] post-norm+scale
	kBH     *tensor.Tensor // [B,Hkv,T,D] post-norm+scale
	vBH     *tensor.Tensor // [B,Hkv,T,D] final v (post ve-gate)
	probs   *tensor.Tensor // [B,Hq,T,T]
	outBH   *tensor.Tensor // [B,Hq,T,D] attention output (pre cproj)

	veSig   *tensor.Tensor // [B,T,Hkv] sigmoid gate output (pre *3), nil if no VE
	ve      *tensor.Tensor // [B,T,Hkv,D] value embedding, nil if no VE
	gradVE  *tensor.Tensor // [B,T,Hkv,D] gradient wrt the value embedding (set by backward)
}

// forwardTrain runs the attention forward while saving activations.
func (a *CausalSelfAttention) forwardTrain(x, ve, cos, sin *tensor.Tensor, t0 int, window [2]int) (*tensor.Tensor, *attnCtx) {
	b, t := x.Shape[0], x.Shape[1]
	q := a.cq.Forward(x).Reshape(b, t, a.numHead, a.headDim)
	k := a.ck.Forward(x).Reshape(b, t, a.numKVHead, a.headDim)
	v := a.cv.Forward(x).Reshape(b, t, a.numKVHead, a.headDim)

	ctx := &attnCtx{}
	if ve != nil {
		ve4 := ve.Reshape(b, t, a.numKVHead, a.headDim)
		z := a.veGate.Forward(sliceChannels(x, 0, t, veGateChannels)) // [B,T,Hkv]
		sig := tensor.Sigmoid(z)
		ctx.veSig = sig
		gate := tensor.Scale(sig, veGateScale)
		v = addGateTimesVE(v, gate, ve4)
		ctx.ve = ve4
	}

	q = ApplyRotary(q, cos, sin, t0)
	k = ApplyRotary(k, cos, sin, t0)
	ctx.qRotary = q
	ctx.kRotary = k

	q = tensor.Scale(normLastDim(q), qkScale)
	k = tensor.Scale(normLastDim(k), qkScale)

	qBH := toBH(q)
	kBH := toBH(k)
	vBH := toBH(v)
	ctx.qBH = qBH
	ctx.kBH = kBH
	ctx.vBH = vBH

	headRatio := a.numHead / a.numKVHead
	d := a.headDim
	outBH := tensor.New(b, a.numHead, t, d)
	probs := tensor.New(b, a.numHead, t, t)

	for bb := 0; bb < b; bb++ {
		for hq := 0; hq < a.numHead; hq++ {
			kvh := hq / headRatio
			qh := qBH.Data[((bb*a.numHead+hq)*t)*d : ((bb*a.numHead+hq)*t+t)*d]
			kh := kBH.Data[((bb*a.numKVHead+kvh)*t)*d : ((bb*a.numKVHead+kvh)*t+t)*d]
			vh := vBH.Data[((bb*a.numKVHead+kvh)*t)*d : ((bb*a.numKVHead+kvh)*t+t)*d]
			oh, ph := attentionHeadWithProbs(qh, kh, vh, t, d, t0, window[0])
			copy(outBH.Data[((bb*a.numHead+hq)*t)*d:], oh)
			copy(probs.Data[((bb*a.numHead+hq)*t)*t:], ph)
		}
	}
	ctx.outBH = outBH
	ctx.probs = probs

	out := toBT(outBH).Reshape(b, t, a.numHead*a.headDim)
	return a.cproj.Forward(out), ctx
}

// attentionHeadWithProbs computes the attention output and returns the
// probability matrix for later backprop.
func attentionHeadWithProbs(q, k, v []float32, tq, d, t0, window int) ([]float32, []float32) {
	q2 := tensor.NewWithData([]int{tq, d}, q)
	k2 := tensor.NewWithData([]int{tq, d}, k)
	scores := tensor.MatMulTransB(q2, k2)
	sd := scores.Data
	for i := 0; i < tq; i++ {
		qpos := t0 + i
		for j := 0; j < tq; j++ {
			if j > qpos || (window >= 0 && qpos-j > window) {
				sd[i*tq+j] = maskedScore
			}
		}
	}
	probs := tensor.SoftmaxLastDim(scores)
	v2 := tensor.NewWithData([]int{tq, d}, v)
	return tensor.MatMul(probs, v2).Data, probs.Data
}

// backwardTrain runs the attention backward given the gradient of its output
// and the saved context. It returns the gradient wrt the attention input x.
func (a *CausalSelfAttention) backwardTrain(x *tensor.Tensor, gradOut *tensor.Tensor, ctx *attnCtx, cos, sin *tensor.Tensor, t0 int) *tensor.Tensor {
	b, t := x.Shape[0], x.Shape[1]
	d := a.headDim
	headRatio := a.numHead / a.numKVHead

	// cproj backward.
	outFlat := toBT(ctx.outBH).Reshape(b, t, a.numHead*a.headDim)
	gradOutBH := a.cproj.Backward(outFlat, gradOut).Reshape(b, t, a.numHead, a.headDim)
	gradOutBH = toBH(gradOutBH) // [B,Hq,T,D]

	gradQ := tensor.New(b, a.numHead, t, d)
	gradK := tensor.New(b, a.numKVHead, t, d)
	gradV := tensor.New(b, a.numKVHead, t, d)

	for bb := 0; bb < b; bb++ {
		for hq := 0; hq < a.numHead; hq++ {
			kvh := hq / headRatio
			oh := gradOutBH.Data[((bb*a.numHead+hq)*t)*d : ((bb*a.numHead+hq)*t+t)*d]
			vh := ctx.vBH.Data[((bb*a.numKVHead+kvh)*t)*d : ((bb*a.numKVHead+kvh)*t+t)*d]
			ph := ctx.probs.Data[((bb*a.numHead+hq)*t)*t : ((bb*a.numHead+hq)*t+t)*t]
			qh := ctx.qBH.Data[((bb*a.numHead+hq)*t)*d : ((bb*a.numHead+hq)*t+t)*d]
			kh := ctx.kBH.Data[((bb*a.numKVHead+kvh)*t)*d : ((bb*a.numKVHead+kvh)*t+t)*d]

			// grad_probs = grad_out @ v^T ; grad_v = probs^T @ grad_out.
			o2 := tensor.NewWithData([]int{t, d}, oh)
			v2 := tensor.NewWithData([]int{t, d}, vh)
			gradProbs := tensor.MatMulTransB(o2, v2)         // [T,T]
			gradVh := tensor.MatMul(tensor.Transpose(tensor.NewWithData([]int{t, t}, ph)), o2) // [T,D]

			gradScores := softmaxBackward(tensor.NewWithData([]int{t, t}, ph), gradProbs) // [T,T]
			q2 := tensor.NewWithData([]int{t, d}, qh)
			k2 := tensor.NewWithData([]int{t, d}, kh)
			gradQh := tensor.MatMul(gradScores, k2)             // [T,D]
			gradKh := tensor.MatMul(tensor.Transpose(gradScores), q2) // [T,D]

			copy(gradQ.Data[((bb*a.numHead+hq)*t)*d:], gradQh.Data)
			copy(gradK.Data[((bb*a.numKVHead+kvh)*t)*d:], gradKh.Data)
			copy(gradV.Data[((bb*a.numKVHead+kvh)*t)*d:], gradVh.Data)
		}
	}

	// Unscale QK (1.2) and back through QK norm.
	gradQ = tensor.Scale(gradQ, qkScale)
	gradK = tensor.Scale(gradK, qkScale)
	gradQBT := toBT(gradQ) // [B,T,Hq,D]
	gradKBT := toBT(gradK)

	// RMSNorm backward over the head dim.
	gradQPre := normLastDimBackward(ctx.qRotary, gradQBT)
	gradKPre := normLastDimBackward(ctx.kRotary, gradKBT)

	// Rotary backward: transpose the rotation (negate sin).
	nSin := negateSin(sin)
	gradQProj := ApplyRotary(gradQPre, cos, nSin, t0)
	gradKProj := ApplyRotary(gradKPre, cos, nSin, t0)

	// Accumulate into the input via the projection layers.
	gradX := a.cq.Backward(x, gradQProj.Reshape(b, t, a.numHead*a.headDim))
	gradX = tensor.Add(gradX, a.ck.Backward(x, gradKProj.Reshape(b, t, a.numKVHead*a.headDim)))

	// Value embedding gate backward.
	gradVBT := toBT(gradV) // [B,T,Hkv,D]
	if ctx.ve != nil {
		// grad_ve = grad_v * gate; grad_gate = sum_D(grad_v * ve).
		gate := tensor.Scale(ctx.veSig, veGateScale)
		ve4 := ctx.ve
		gd := gradVBT.Data
		gradVE := tensor.New(b, t, a.numKVHead, d)
		gradGate := tensor.New(b, t, a.numKVHead)
		for idx := 0; idx < b*t*a.numKVHead; idx++ {
			g := gate.Data[idx]
			base := idx * d
			var sum float64
			for j := 0; j < d; j++ {
				gradVE.Data[base+j] = gd[base+j] * g
				sum += float64(gd[base+j]) * float64(ve4.Data[base+j])
			}
			gradGate.Data[idx] = float32(sum)
		}
		ctx.gradVE = gradVE
		// sigmoid backward: gate = 3*sigmoid(z).
		gradSig := tensor.Scale(gradGate, veGateScale)
		gradZ := tensor.New(b, t, a.numKVHead)
		for i := range ctx.veSig.Data {
			s := ctx.veSig.Data[i]
			gradZ.Data[i] = gradSig.Data[i] * s * (1 - s)
		}
		gradGateSlice := a.veGate.Backward(sliceChannels(x, 0, t, veGateChannels), gradZ) // [B,T,12]
		scattered := tensor.New(b, t, a.embedDim)
		scatterAddChannels(scattered, 0, t, veGateChannels, gradGateSlice)
		gradX = tensor.Add(gradX, scattered)
	}

	// c_v projection gradient (gradV flows through cv).
	gradX = tensor.Add(gradX, a.cv.Backward(x, gradVBT.Reshape(b, t, a.numKVHead*a.headDim)))

	return gradX
}

// normLastDimBackward applies RMSNorm backward to an arbitrary-rank tensor.
func normLastDimBackward(x, gradOut *tensor.Tensor) *tensor.Tensor {
	last := x.Shape[len(x.Shape)-1]
	rows := x.Numel() / last
	return tensor.RMSNormBackward(x.Reshape(rows, last), gradOut.Reshape(rows, last), normEps).Reshape(x.Shape...)
}

// mlpCtx holds activations for the MLP backward.
type mlpCtx struct {
	fcIn  *tensor.Tensor // input to c_fc (== norm input)
	fcOut *tensor.Tensor // c_fc output (pre relu2)
	h     *tensor.Tensor // relu2 output (input to c_proj)
}

func (m *MLP) forwardTrain(x *tensor.Tensor) (*tensor.Tensor, *mlpCtx) {
	fc := m.CFc.Forward(x)
	h := tensor.Relu2(fc)
	out := m.CProj.Forward(h)
	return out, &mlpCtx{fcIn: x, fcOut: fc, h: h}
}

func (m *MLP) backwardTrain(gradOut *tensor.Tensor, ctx *mlpCtx) *tensor.Tensor {
	gradH := m.CProj.Backward(ctx.h, gradOut)
	gradFC := tensor.Relu2Backward(ctx.fcOut, gradH)
	return m.CFc.Backward(ctx.fcIn, gradFC)
}

// blockCtx holds activations for the block backward.
type blockCtx struct {
	xPre       *tensor.Tensor // block input (after combineResidual)
	attnNormIn *tensor.Tensor // norm(xPre), the attention's actual input
	attnOut    *tensor.Tensor // attention output (pre residual)
	attnCtx    *attnCtx
	mlpNormIn  *tensor.Tensor // x after attention residual (input to mlp norm)
	mlpOut     *tensor.Tensor
	mlpCtx     *mlpCtx
}

func (b *Block) forwardTrain(x, ve, cos, sin *tensor.Tensor, t0 int, window [2]int) (*tensor.Tensor, *blockCtx) {
	ctx := &blockCtx{xPre: x}
	attnIn := normLastDim(x)
	ctx.attnNormIn = attnIn
	attnOut, actx := b.Attn.forwardTrain(attnIn, ve, cos, sin, t0, window)
	ctx.attnOut = attnOut
	ctx.attnCtx = actx
	mid := tensor.Add(x, attnOut)
	ctx.mlpNormIn = mid
	mlpOut, mctx := b.MLP.forwardTrain(normLastDim(mid))
	ctx.mlpOut = mlpOut
	ctx.mlpCtx = mctx
	return tensor.Add(mid, mlpOut), ctx
}

func (b *Block) backwardTrain(gradOut *tensor.Tensor, ctx *blockCtx, cos, sin *tensor.Tensor, t0 int) *tensor.Tensor {
	// Forward: mid = x_pre + attn_out; out = mid + mlp_out. The residual
	// connections mean the gradient flows both through each sublayer and
	// directly (identity) across it.
	gradMid := gradOut
	gradMLPOut := gradOut

	gradMLPNormIn := b.MLP.backwardTrain(gradMLPOut, ctx.mlpCtx)
	gradMid = tensor.Add(gradMid, normLastDimBackward(ctx.mlpNormIn, gradMLPNormIn))

	gradAttnNormIn := b.Attn.backwardTrain(ctx.attnNormIn, gradMid, ctx.attnCtx, cos, sin, t0)
	gradXPre := normLastDimBackward(ctx.xPre, gradAttnNormIn)
	// Identity residual path.
	gradXPre = tensor.Add(gradXPre, gradMid)
	return gradXPre
}
