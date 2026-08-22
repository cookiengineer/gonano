package model

import (
	"math"

	"github.com/cookiengineer/gonano/optim"
	"github.com/cookiengineer/gonano/tensor"
)

// SetupOptimizer builds the MuonAdamW parameter groups exactly as nanochat's
// setup_optimizer: embeddings, the unembedding layer, and scalars use AdamW;
// the 2D matrix parameters use Muon, grouped by shape. Learning rates for the
// AdamW groups are scaled by 1/sqrt(model_dim/768).
func (m *Transformer) SetupOptimizer(unembeddingLR, embeddingLR, matrixLR, weightDecay, scalarLR float32) []optim.ParamGroup {
	modelDim := m.Config.EmbedDim

	// Separate out the matrix parameters (transformer block Linear weights).
	var matrixParams []*tensor.Tensor
	for _, block := range m.h {
		matrixParams = append(matrixParams,
			block.Attn.cq.Weight, block.Attn.ck.Weight, block.Attn.cv.Weight, block.Attn.cproj.Weight,
			block.MLP.CFc.Weight, block.MLP.CProj.Weight,
		)
		if block.Attn.veGate != nil {
			matrixParams = append(matrixParams, block.Attn.veGate.Weight)
		}
	}

	var valueEmbedsParams []*tensor.Tensor
	for _, ve := range m.valueEmbeds {
		valueEmbedsParams = append(valueEmbedsParams, ve.Weight)
	}

	smearParams := []*tensor.Tensor{m.smearGate.Weight, m.smearLambda, m.backoutLambda}

	dmodelLRScale := float32(math.Pow(float64(modelDim)/768.0, -0.5))

	groups := []optim.ParamGroup{
		{Kind: optim.KindAdamW, Params: []*tensor.Tensor{m.lm.Weight}, LR: unembeddingLR * dmodelLRScale, Beta1: 0.8, Beta2: 0.96, Eps: 1e-10, WeightDecay: 0.01},
		{Kind: optim.KindAdamW, Params: []*tensor.Tensor{m.wte.Weight}, LR: embeddingLR * dmodelLRScale, Beta1: 0.8, Beta2: 0.995, Eps: 1e-10, WeightDecay: 0.001},
		{Kind: optim.KindAdamW, Params: valueEmbedsParams, LR: embeddingLR * dmodelLRScale * 0.5, Beta1: 0.8, Beta2: 0.995, Eps: 1e-10, WeightDecay: 0.01},
		{Kind: optim.KindAdamW, Params: []*tensor.Tensor{m.residLambdas}, LR: scalarLR * 0.01, Beta1: 0.8, Beta2: 0.95, Eps: 1e-10, WeightDecay: 0.05},
		{Kind: optim.KindAdamW, Params: []*tensor.Tensor{m.x0Lambdas}, LR: scalarLR, Beta1: 0.96, Beta2: 0.95, Eps: 1e-10, WeightDecay: 0.0},
		{Kind: optim.KindAdamW, Params: smearParams, LR: 0.2, Beta1: 0.8, Beta2: 0.95, Eps: 1e-10, WeightDecay: 0.0},
	}

	// Muon groups, binned by shape.
	byShape := map[[2]int][]*tensor.Tensor{}
	for _, p := range matrixParams {
		key := [2]int{p.Shape[0], p.Shape[1]}
		byShape[key] = append(byShape[key], p)
	}
	shapes := make([][2]int, 0, len(byShape))
	for shape := range byShape {
		shapes = append(shapes, shape)
	}
	sortShapes(shapes)
	for _, shape := range shapes {
		groups = append(groups, optim.ParamGroup{
			Kind: optim.KindMuon, Params: byShape[shape], LR: matrixLR,
			Momentum: 0.95, NSSteps: 5, MuonBeta2: 0.9, WeightDecay: weightDecay,
		})
	}
	return groups
}

func sortShapes(shapes [][2]int) {
	for i := 1; i < len(shapes); i++ {
		for j := i; j > 0; j-- {
			a, b := shapes[j-1], shapes[j]
			if a[0] > b[0] || (a[0] == b[0] && a[1] > b[1]) {
				shapes[j-1], shapes[j] = b, a
			} else {
				break
			}
		}
	}
}
