package optim

import (
	"math"

	"github.com/cookiengineer/gonano/tensor"
)

// polarExpressCoeffs are the quintic Polar Express coefficients (ns_steps=5),
// from the "Polar Express Sign Method" paper (arxiv 2505.16932), as used by
// nanochat's Muon.
var polarExpressCoeffs = [5][3]float64{
	{8.156554524902461, -22.48329292557795, 15.878769915207462},
	{4.042929935166739, -2.808917465908714, 0.5000178451051316},
	{3.8916678022926607, -2.772484153217685, 0.5060648178503393},
	{3.285753657755655, -2.3681294933425376, 0.46449024233003106},
	{2.3465413258596377, -1.7097828382687081, 0.42323551169305323},
}

// muonStep applies one Muon update to the 2D parameter p: Nesterov momentum,
// MuonEq row equilibration, Polar Express orthogonalization, Muon+
// renormalization, variance reduction, and cautious weight decay.
func (o *MuonAdamW) muonStep(g ParamGroup, p *tensor.Tensor) {
	st := o.muonStates[p]
	p.EnsureGrad()
	m, n := p.Shape[0], p.Shape[1]
	momentum := g.Momentum

	// Nesterov momentum:
	//   mom = momentum*mom + (1-momentum)*grad
	//   x   = (1-momentum)*grad + momentum*mom
	x := tensor.New(m, n)
	for i := range p.Data {
		st.momentum.Data[i] = momentum*st.momentum.Data[i] + (1-momentum)*p.Grad[i]
		x.Data[i] = (1-momentum)*p.Grad[i] + momentum*st.momentum.Data[i]
	}

	// MuonEq row equilibration: rescale each row to the mean row norm.
	target := frobeniusNorm(x) / float32(math.Sqrt(float64(m)))
	for i := 0; i < m; i++ {
		rn := rowL2Norm(x.Data[i*n : (i+1)*n])
		if rn < 1e-6 {
			rn = 1e-6
		}
		scale := target / rn
		for j := 0; j < n; j++ {
			x.Data[i*n+j] *= scale
		}
	}

	// Polar Express orthogonalization.
	norm := frobeniusNorm(x)
	scale := float32(1.0 / (float64(norm)*1.01 + 1e-6))
	for i := range x.Data {
		x.Data[i] *= scale
	}
	for step := 0; step < g.NSSteps; step++ {
		a := polarExpressCoeffs[step][0]
		b := polarExpressCoeffs[step][1]
		c := polarExpressCoeffs[step][2]
		var A *tensor.Tensor
		if m > n {
			A = tensor.MatMul(tensor.Transpose(x), x) // [n,n] tall
		} else {
			A = tensor.MatMul(x, tensor.Transpose(x)) // [m,m] wide
		}
		AA := tensor.MatMul(A, A)
		B := tensor.Add(tensor.Scale(A, float32(b)), tensor.Scale(AA, float32(c)))
		if m > n {
			x = tensor.Add(tensor.Scale(x, float32(a)), tensor.MatMul(x, B))
		} else {
			x = tensor.Add(tensor.Scale(x, float32(a)), tensor.MatMul(B, x))
		}
	}

	// Muon+ renormalization: snap the Frobenius norm to sqrt(min(m, n)).
	targetNorm := float32(math.Sqrt(float64(min(m, n))))
	cur := frobeniusNorm(x)
	if cur < 1e-6 {
		cur = 1e-6
	}
	rnScale := targetNorm / cur
	for i := range x.Data {
		x.Data[i] *= rnScale
	}

	// Variance reduction (NorMuon): per-row/column adaptive learning rate.
	var vMean []float32
	var redDimSize int
	if m >= n {
		// Reduce over columns (red_dim = -1): per-row means, shape [m].
		vMean = make([]float32, m)
		redDimSize = n
		for i := 0; i < m; i++ {
			var sum float64
			for j := 0; j < n; j++ {
				sum += float64(x.Data[i*n+j]) * float64(x.Data[i*n+j])
			}
			vMean[i] = float32(sum / float64(n))
		}
	} else {
		// Reduce over rows (red_dim = -2): per-column means, shape [n].
		vMean = make([]float32, n)
		redDimSize = m
		for j := 0; j < n; j++ {
			var sum float64
			for i := 0; i < m; i++ {
				sum += float64(x.Data[i*n+j]) * float64(x.Data[i*n+j])
			}
			vMean[j] = float32(sum / float64(m))
		}
	}

	var vNormSq float64
	for _, v := range vMean {
		vNormSq += float64(v)
	}
	vNormSq *= float64(redDimSize)
	vNorm := float32(math.Sqrt(vNormSq))

	beta2 := g.MuonBeta2
	stepSize := make([]float32, len(vMean))
	for i := range vMean {
		st.secondMom.Data[i] = beta2*st.secondMom.Data[i] + (1-beta2)*vMean[i]
		s := st.secondMom.Data[i]
		if s < 1e-10 {
			s = 1e-10
		}
		stepSize[i] = float32(1.0 / math.Sqrt(float64(s)))
	}

	var vNormNewSq float64
	for i := range vMean {
		term := float64(vMean[i]) * float64(redDimSize) * float64(stepSize[i]) * float64(stepSize[i])
		vNormNewSq += term
	}
	vNormNew := float32(math.Sqrt(vNormNewSq))
	if vNormNew < 1e-10 {
		vNormNew = 1e-10
	}
	ratio := vNorm / vNormNew
	finalScale := make([]float32, len(vMean))
	for i := range stepSize {
		finalScale[i] = stepSize[i] * ratio
	}

	if m >= n {
		// finalScale is per-row [m].
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				x.Data[i*n+j] *= finalScale[i]
			}
		}
	} else {
		// finalScale is per-column [n].
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				x.Data[i*n+j] *= finalScale[j]
			}
		}
	}

	// Cautious weight decay + parameter update.
	lr := g.LR
	wd := g.WeightDecay
	for i := range p.Data {
		var mask float32
		if x.Data[i]*p.Data[i] >= 0 {
			mask = 1
		}
		p.Data[i] -= lr*x.Data[i] + lr*wd*p.Data[i]*mask
	}
}

func frobeniusNorm(x *tensor.Tensor) float32 {
	var sum float64
	for _, v := range x.Data {
		sum += float64(v) * float64(v)
	}
	return float32(math.Sqrt(sum))
}

func rowL2Norm(s []float32) float32 {
	var sum float64
	for _, v := range s {
		sum += float64(v) * float64(v)
	}
	return float32(math.Sqrt(sum))
}
