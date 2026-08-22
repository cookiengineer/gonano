package model

import (
	"fmt"
	"sort"

	"github.com/cookiengineer/gonano/tensor"
)

// NamedParameters returns every trainable parameter of the model keyed by a
// state-dict-style name matching nanochat's checkpoint keys. The order is
// deterministic (sorted by name). This is the single source of truth used by
// weight initialization verification, checkpointing, and optimizer setup.
func (m *Transformer) NamedParameters() map[string]*tensor.Tensor {
	params := make(map[string]*tensor.Tensor)
	params["transformer.wte.weight"] = m.wte.Weight
	params["lm_head.weight"] = m.lm.Weight
	for i, block := range m.h {
		params[fmt.Sprintf("transformer.h.%d.attn.c_q.weight", i)] = block.Attn.cq.Weight
		params[fmt.Sprintf("transformer.h.%d.attn.c_k.weight", i)] = block.Attn.ck.Weight
		params[fmt.Sprintf("transformer.h.%d.attn.c_v.weight", i)] = block.Attn.cv.Weight
		params[fmt.Sprintf("transformer.h.%d.attn.c_proj.weight", i)] = block.Attn.cproj.Weight
		if block.Attn.veGate != nil {
			params[fmt.Sprintf("transformer.h.%d.attn.ve_gate.weight", i)] = block.Attn.veGate.Weight
		}
		params[fmt.Sprintf("transformer.h.%d.mlp.c_fc.weight", i)] = block.MLP.CFc.Weight
		params[fmt.Sprintf("transformer.h.%d.mlp.c_proj.weight", i)] = block.MLP.CProj.Weight
	}
	for i, ve := range m.valueEmbeds {
		params[fmt.Sprintf("value_embeds.%d.weight", i)] = ve.Weight
	}
	params["resid_lambdas"] = m.residLambdas
	params["x0_lambdas"] = m.x0Lambdas
	params["smear_gate.weight"] = m.smearGate.Weight
	params["smear_lambda"] = m.smearLambda
	params["backout_lambda"] = m.backoutLambda
	return params
}

// ParameterNames returns the sorted parameter names.
func (m *Transformer) ParameterNames() []string {
	params := m.NamedParameters()
	names := make([]string, 0, len(params))
	for name := range params {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// TotalParams returns the total number of parameters.
func (m *Transformer) TotalParams() int {
	total := 0
	for _, p := range m.NamedParameters() {
		total += p.Numel()
	}
	return total
}

// ZeroGrad zeroes the gradients of all parameters.
func (m *Transformer) ZeroGrad() {
	for _, p := range m.NamedParameters() {
		p.ZeroGrad()
	}
}

// Parameters returns every trainable parameter as a flat slice (deterministic
// order by name).
func (m *Transformer) Parameters() []*tensor.Tensor {
	names := m.ParameterNames()
	out := make([]*tensor.Tensor, 0, len(names))
	params := m.NamedParameters()
	for _, name := range names {
		out = append(out, params[name])
	}
	return out
}
