package model

import (
	"fmt"
	"sort"

	"github.com/cookiengineer/gonano/tensors"
)

// NamedParameters returns every trainable parameter of the model keyed by a
// state-dict-style name matching nanochat's checkpoint keys. The order is
// deterministic (sorted by name). This is the single source of truth used by
// weight initialization verification, checkpointing, and optimizer setup.
func (model *Transformer) NamedParameters() map[string]*tensors.Tensor {
	parameters := make(map[string]*tensors.Tensor)
	parameters["transformer.wte.weight"] = model.tokenEmbedding.Weight
	parameters["lm_head.weight"] = model.lmHead.Weight
	for layerIndex, block := range model.blocks {
		parameters[fmt.Sprintf("transformer.h.%d.attn.c_q.weight", layerIndex)] = block.attention.queryProjection.Weight
		parameters[fmt.Sprintf("transformer.h.%d.attn.c_k.weight", layerIndex)] = block.attention.keyProjection.Weight
		parameters[fmt.Sprintf("transformer.h.%d.attn.c_v.weight", layerIndex)] = block.attention.valueProjection.Weight
		parameters[fmt.Sprintf("transformer.h.%d.attn.c_proj.weight", layerIndex)] = block.attention.outputProjection.Weight
		if block.attention.valueEmbeddingGate != nil {
			parameters[fmt.Sprintf("transformer.h.%d.attn.ve_gate.weight", layerIndex)] = block.attention.valueEmbeddingGate.Weight
		}
		if block.attention.compressor != nil {
			parameters[fmt.Sprintf("transformer.h.%d.attn.compress_logits.weight", layerIndex)] = block.attention.compressor.logitWeight.Weight
			parameters[fmt.Sprintf("transformer.h.%d.attn.compress_bias", layerIndex)] = block.attention.compressor.bias
		}
		if block.attention.indexer != nil {
			parameters[fmt.Sprintf("transformer.h.%d.attn.indexer_query.weight", layerIndex)] = block.attention.indexer.query.Weight
			parameters[fmt.Sprintf("transformer.h.%d.attn.indexer_key.weight", layerIndex)] = block.attention.indexer.key.Weight
			parameters[fmt.Sprintf("transformer.h.%d.attn.indexer_heads", layerIndex)] = block.attention.indexer.headWeights
		}
		parameters[fmt.Sprintf("transformer.h.%d.mlp.c_fc.weight", layerIndex)] = block.mlp.inputProjection.Weight
		parameters[fmt.Sprintf("transformer.h.%d.mlp.c_proj.weight", layerIndex)] = block.mlp.outputProjection.Weight
	}
	for layerIndex, valueEmbedding := range model.valueEmbeds {
		parameters[fmt.Sprintf("value_embeds.%d.weight", layerIndex)] = valueEmbedding.Weight
	}
	parameters["resid_lambdas"] = model.residLambdas
	parameters["x0_lambdas"] = model.x0Lambdas
	parameters["smear_gate.weight"] = model.smearGate.Weight
	parameters["smear_lambda"] = model.smearLambda
	parameters["backout_lambda"] = model.backoutLambda
	return parameters
}

// ParameterNames returns the sorted parameter names.
func (model *Transformer) ParameterNames() []string {
	parameters := model.NamedParameters()
	names := make([]string, 0, len(parameters))
	for name := range parameters {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// TotalParams returns the total number of parameters.
func (model *Transformer) TotalParams() int {
	total := 0
	for _, parameter := range model.NamedParameters() {
		total += parameter.Numel()
	}
	return total
}

// ZeroGrad zeroes the gradients of all parameters.
func (model *Transformer) ZeroGrad() {
	for _, parameter := range model.NamedParameters() {
		parameter.ZeroGrad()
	}
}

// Parameters returns every trainable parameter as a flat slice (deterministic
// order by name).
func (model *Transformer) Parameters() []*tensors.Tensor {
	names := model.ParameterNames()
	output := make([]*tensors.Tensor, 0, len(names))
	parameters := model.NamedParameters()
	for _, name := range names {
		output = append(output, parameters[name])
	}
	return output
}
