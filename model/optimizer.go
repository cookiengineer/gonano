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
//
// When sinkhorn is true the embedding table, the unembedding layer, and the
// value embeddings are routed to the Sinkhorn-balanced momentum update
// (DeepSeek-V4.1 §2.5, Algorithm 1) instead of AdamW, which halves their
// optimizer-state memory.
func (model *Transformer) SetupOptimizer(unembeddingLR, embeddingLR, matrixLR, weightDecay, scalarLR float32, sinkhorn bool) []optimizer.ParamGroup {
	modelDimension := model.Config.EmbedDim

	// Separate out the matrix parameters (transformer block Linear weights).
	var matrixParameters []*tensors.Tensor
	headWise := model.Config.HeadWiseMuon
	for _, block := range model.blocks {
		if headWise {
			// Split Q and K by attention head so each head is orthogonalized
			// independently (DeepSeek-V4.1 §2.5). The views share storage with
			// the parent weights and their gradients.
			matrixParameters = append(matrixParameters,
				headWiseViews(block.attention.queryProjection.Weight, model.Config.NumHead, model.Config.HeadDim())...)
			matrixParameters = append(matrixParameters,
				headWiseViews(block.attention.keyProjection.Weight, model.Config.NumKVHead, model.Config.HeadDim())...)
		} else {
			matrixParameters = append(matrixParameters,
				block.attention.queryProjection.Weight, block.attention.keyProjection.Weight)
		}
		matrixParameters = append(matrixParameters,
			block.attention.valueProjection.Weight, block.attention.outputProjection.Weight,
			block.mlp.inputProjection.Weight, block.mlp.outputProjection.Weight,
		)
		if block.attention.queryDown != nil {
			matrixParameters = append(matrixParameters, block.attention.queryDown.Weight)
		}
		if block.attention.kvDown != nil {
			matrixParameters = append(matrixParameters, block.attention.kvDown.Weight)
		}
		if block.attention.valueEmbeddingGate != nil {
			matrixParameters = append(matrixParameters, block.attention.valueEmbeddingGate.Weight)
		}
		if block.attention.compressor != nil {
			matrixParameters = append(matrixParameters, block.attention.compressor.logitWeight.Weight)
		}
		if block.attention.indexer != nil {
			matrixParameters = append(matrixParameters,
				block.attention.indexer.query.Weight, block.attention.indexer.key.Weight)
		}
	}

	var valueEmbeddingParameters []*tensors.Tensor
	for _, valueEmbedding := range model.valueEmbeds {
		valueEmbeddingParameters = append(valueEmbeddingParameters, valueEmbedding.Weight)
	}

	smearParameters := []*tensors.Tensor{model.smearGate.Weight, model.smearLambda, model.backoutLambda}
	for _, block := range model.blocks {
		if block.attention.compressor != nil {
			smearParameters = append(smearParameters, block.attention.compressor.bias)
		}
		if block.attention.indexer != nil {
			smearParameters = append(smearParameters, block.attention.indexer.headWeights)
		}
	}

	modelDimensionLRScale := float32(math.Pow(float64(modelDimension)/768.0, -0.5))

	groups := make([]optimizer.ParamGroup, 0, 8)
	if sinkhorn {
		groups = append(groups,
			sinkhornGroup([]*tensors.Tensor{model.lmHead.Weight}, unembeddingLR*modelDimensionLRScale),
			sinkhornGroup([]*tensors.Tensor{model.tokenEmbedding.Weight}, embeddingLR*modelDimensionLRScale),
			sinkhornGroup(valueEmbeddingParameters, embeddingLR*modelDimensionLRScale*0.5),
		)
	} else {
		groups = append(groups,
			optimizer.ParamGroup{Kind: optimizer.KindAdamW, Params: []*tensors.Tensor{model.lmHead.Weight}, LR: unembeddingLR * modelDimensionLRScale, Beta1: 0.8, Beta2: 0.96, Eps: 1e-10, WeightDecay: 0.01},
			optimizer.ParamGroup{Kind: optimizer.KindAdamW, Params: []*tensors.Tensor{model.tokenEmbedding.Weight}, LR: embeddingLR * modelDimensionLRScale, Beta1: 0.8, Beta2: 0.995, Eps: 1e-10, WeightDecay: 0.001},
			optimizer.ParamGroup{Kind: optimizer.KindAdamW, Params: valueEmbeddingParameters, LR: embeddingLR * modelDimensionLRScale * 0.5, Beta1: 0.8, Beta2: 0.995, Eps: 1e-10, WeightDecay: 0.01},
		)
	}
	groups = append(groups,
		optimizer.ParamGroup{Kind: optimizer.KindAdamW, Params: []*tensors.Tensor{model.residLambdas}, LR: scalarLR * 0.01, Beta1: 0.8, Beta2: 0.95, Eps: 1e-10, WeightDecay: 0.05},
		optimizer.ParamGroup{Kind: optimizer.KindAdamW, Params: []*tensors.Tensor{model.x0Lambdas}, LR: scalarLR, Beta1: 0.96, Beta2: 0.95, Eps: 1e-10, WeightDecay: 0.0},
		optimizer.ParamGroup{Kind: optimizer.KindAdamW, Params: smearParameters, LR: 0.2, Beta1: 0.8, Beta2: 0.95, Eps: 1e-10, WeightDecay: 0.0},
	)

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

// sinkhornGroup builds a Sinkhorn-balanced parameter group with the
// DeepSeek-V4.1 §2.5 configuration: K=11 odd alternating normalizations,
// tau=1e-3, epsilon=1e-20, gamma=0.18, and the same momentum as Muon.
func sinkhornGroup(params []*tensors.Tensor, learningRate float32) optimizer.ParamGroup {
	return optimizer.ParamGroup{
		Kind:          optimizer.KindSinkhorn,
		Params:        params,
		LR:            learningRate,
		Momentum:      0.95,
		Gamma:         0.18,
		SinkhornSteps: 11,
		SinkhornTau:   1e-3,
		SinkhornEps:   1e-20,
	}
}

// headWiseViews splits a [heads*headDim, columns] weight matrix into `heads`
// row-block views of shape [headDim, columns]. Each view shares the parent's
// Data and Grad backing arrays, so the Muon update and the backpropagated
// gradient both operate in place on the same storage.
func headWiseViews(weight *tensors.Tensor, heads, headDim int) []*tensors.Tensor {
	if heads < 1 {
		return nil
	}
	columns := weight.Shape[1]
	if weight.Shape[0] != heads*headDim {
		panic("model: head-wise Muon weight shape does not match heads*headDim")
	}
	weight.EnsureGrad()
	views := make([]*tensors.Tensor, heads)
	for head := 0; head < heads; head++ {
		start := head * headDim * columns
		end := start + headDim*columns
		views[head] = &tensors.Tensor{
			Shape: []int{headDim, columns},
			Data:  weight.Data[start:end],
			Grad:  weight.Grad[start:end],
		}
	}
	return views
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
