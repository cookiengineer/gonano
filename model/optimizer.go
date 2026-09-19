package model

import (
	"math"

	"github.com/cookiengineer/gonano/optimizer"
	"github.com/cookiengineer/gonano/tensors"
)

// SetupOptimizer builds the MuonAdamW parameter groups exactly as nanochat's
// setup_optimizer: embeddings, the unembedding layer, and scalars use AdamW;
// the 2D matrix parameters use Muon, grouped by shape. Learning rates for the
// AdamW groups are scaled by 1/sqrt(model_dim/768).
func (model *Transformer) SetupOptimizer(unembeddingLR, embeddingLR, matrixLR, weightDecay, scalarLR float32) []optimizer.ParamGroup {
	modelDimension := model.Config.EmbedDim

	// Separate out the matrix parameters (transformer block Linear weights).
	var matrixParameters []*tensors.Tensor
	for _, block := range model.blocks {
		matrixParameters = append(matrixParameters,
			block.attention.queryProjection.Weight, block.attention.keyProjection.Weight, block.attention.valueProjection.Weight, block.attention.outputProjection.Weight,
			block.mlp.inputProjection.Weight, block.mlp.outputProjection.Weight,
		)
		if block.attention.valueEmbeddingGate != nil {
			matrixParameters = append(matrixParameters, block.attention.valueEmbeddingGate.Weight)
		}
	}

	var valueEmbeddingParameters []*tensors.Tensor
	for _, valueEmbedding := range model.valueEmbeds {
		valueEmbeddingParameters = append(valueEmbeddingParameters, valueEmbedding.Weight)
	}

	smearParameters := []*tensors.Tensor{model.smearGate.Weight, model.smearLambda, model.backoutLambda}

	modelDimensionLRScale := float32(math.Pow(float64(modelDimension)/768.0, -0.5))

	groups := []optimizer.ParamGroup{
		{Kind: optimizer.KindAdamW, Params: []*tensors.Tensor{model.lmHead.Weight}, LR: unembeddingLR * modelDimensionLRScale, Beta1: 0.8, Beta2: 0.96, Eps: 1e-10, WeightDecay: 0.01},
		{Kind: optimizer.KindAdamW, Params: []*tensors.Tensor{model.tokenEmbedding.Weight}, LR: embeddingLR * modelDimensionLRScale, Beta1: 0.8, Beta2: 0.995, Eps: 1e-10, WeightDecay: 0.001},
		{Kind: optimizer.KindAdamW, Params: valueEmbeddingParameters, LR: embeddingLR * modelDimensionLRScale * 0.5, Beta1: 0.8, Beta2: 0.995, Eps: 1e-10, WeightDecay: 0.01},
		{Kind: optimizer.KindAdamW, Params: []*tensors.Tensor{model.residLambdas}, LR: scalarLR * 0.01, Beta1: 0.8, Beta2: 0.95, Eps: 1e-10, WeightDecay: 0.05},
		{Kind: optimizer.KindAdamW, Params: []*tensors.Tensor{model.x0Lambdas}, LR: scalarLR, Beta1: 0.96, Beta2: 0.95, Eps: 1e-10, WeightDecay: 0.0},
		{Kind: optimizer.KindAdamW, Params: smearParameters, LR: 0.2, Beta1: 0.8, Beta2: 0.95, Eps: 1e-10, WeightDecay: 0.0},
	}

	// Muon groups, binned by shape.
	parametersByShape := map[[2]int][]*tensors.Tensor{}
	for _, parameter := range matrixParameters {
		shapeKey := [2]int{parameter.Shape[0], parameter.Shape[1]}
		parametersByShape[shapeKey] = append(parametersByShape[shapeKey], parameter)
	}
	shapes := make([][2]int, 0, len(parametersByShape))
	for shape := range parametersByShape {
		shapes = append(shapes, shape)
	}
	sortShapes(shapes)
	for _, shape := range shapes {
		groups = append(groups, optimizer.ParamGroup{
			Kind: optimizer.KindMuon, Params: parametersByShape[shape], LR: matrixLR,
			Momentum: 0.95, NSSteps: 5, MuonBeta2: 0.9, WeightDecay: weightDecay,
		})
	}
	return groups
}

func sortShapes(shapes [][2]int) {
	for index := 1; index < len(shapes); index++ {
		for scanIndex := index; scanIndex > 0; scanIndex-- {
			previousShape, currentShape := shapes[scanIndex-1], shapes[scanIndex]
			if previousShape[0] > currentShape[0] || (previousShape[0] == currentShape[0] && previousShape[1] > currentShape[1]) {
				shapes[scanIndex-1], shapes[scanIndex] = currentShape, previousShape
			} else {
				break
			}
		}
	}
}
