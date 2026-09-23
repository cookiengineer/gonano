package model

import (
	"github.com/cookiengineer/gonano/model/layers"
	"github.com/cookiengineer/gonano/tensors"
)

// Router is the per-token expert gate of a DeepSeekMoE feed-forward. Its
// weight maps the hidden state to one affinity score per routed expert, and a
// non-trainable correction bias implements auxiliary-loss-free load balancing
// (DeepSeek-V4.1 §2.1.1): the bias participates in expert selection but not in
// the output weighting.
type Router struct {
	weight *layers.Linear  // [NumExperts, Dim]
	bias   *tensors.Tensor // [NumExperts], updated by load, never by gradient
}

// MoE is the routed mixture-of-experts feed-forward. Each token selects
// TopK routed experts whose SwiGLU outputs are combined by the router
// affinities and scaled by Scale; the always-active shared expert lives in the
// block's MLP. Expert weights are packed contiguously ([E, Hidden, Dim] for the
// gate/up projections and [E, Dim, Hidden] for the down projection) so the
// grouped GEMM kernel can walk them without pointer chasing, while the
// optimizer consumes per-expert rank-2 views.
type MoE struct {
	NumExperts int
	TopK       int
	Scale      float32
	Dim        int
	Hidden     int
	Clamp      float32

	router     *Router
	gateWeight *tensors.Tensor // [E, Hidden, Dim]
	upWeight   *tensors.Tensor // [E, Hidden, Dim]
	downWeight *tensors.Tensor // [E, Dim, Hidden]

	// loadCounts and loadTotal accumulate routed-token counts across the
	// micro-batches of one optimizer step for the bias update.
	loadCounts []float64
	loadTotal  float64
}

// NewMoE builds a routed MoE feed-forward for a config. The shared expert is
// owned by the block's MLP and is not part of MoE.
func NewMoE(configuration Config) *MoE {
	experts := configuration.NumExperts
	hidden := configuration.MoEExpertHidden()
	return &MoE{
		NumExperts: experts,
		TopK:       configuration.MoEActiveExperts(),
		Scale:      configuration.MoEEffectiveScale(),
		Dim:        configuration.EmbedDim,
		Hidden:     hidden,
		Clamp:      configuration.MoEEffectiveClamp(),
		router: &Router{
			weight: layers.NewLinear(configuration.EmbedDim, experts),
			bias:   tensors.New(experts),
		},
		gateWeight: tensors.New(experts, hidden, configuration.EmbedDim),
		upWeight:   tensors.New(experts, hidden, configuration.EmbedDim),
		downWeight: tensors.New(experts, configuration.EmbedDim, hidden),
		loadCounts: make([]float64, experts),
	}
}

// expertForward holds one expert's forward state for backprop. The tensors are
// views into the grouped buffers produced by forwardTraining.
type expertForward struct {
	id      int
	rows    []int
	weights []float32
	input   *tensors.Tensor // [n, Dim]
	gate    *tensors.Tensor // [n, Hidden]
	up      *tensors.Tensor // [n, Hidden]
	hidden  *tensors.Tensor // [n, Hidden]
	output  *tensors.Tensor // [n, Dim]
}

// moeContext holds the routing and expert state saved by forwardTraining.
type moeContext struct {
	shape   []int
	rows    int
	input   *tensors.Tensor // [rows, Dim]
	probs   *tensors.Tensor // [rows, E]
	experts []expertForward
}

// Forward computes the routed expert contribution for input of shape
// [..., Dim] and returns the same shape. The shared expert is added by the
// caller.
func (moe *MoE) Forward(input *tensors.Tensor) *tensors.Tensor {
	output, _ := moe.forward(input, false)
	return output
}

// forwardTraining computes the routed contribution while saving the routing
// and expert activations for backprop.
func (moe *MoE) forwardTraining(input *tensors.Tensor) (*tensors.Tensor, *moeContext) {
	return moe.forward(input, true)
}

func (moe *MoE) forward(input *tensors.Tensor, training bool) (*tensors.Tensor, *moeContext) {
	shape := append([]int(nil), input.Shape...)
	rows := input.Numel() / moe.Dim
	flat := input.Reshape(rows, moe.Dim)

	logits := moe.router.weight.Forward(flat) // [rows, E]
	probs, selected, selectedWeights := tensors.MoEGateTopK(logits, moe.router.bias, rows, moe.NumExperts, moe.TopK)

	// Count and offset the selected experts, then gather the selected rows into
	// expert-major order (one row per (token, selected expert) pair).
	total := rows * moe.TopK
	tokenCounts := make([]int32, moe.NumExperts)
	for index := 0; index < total; index++ {
		tokenCounts[selected.Data[index]]++
	}
	offsets := make([]int, moe.NumExperts)
	cursor := 0
	for expert := 0; expert < moe.NumExperts; expert++ {
		offsets[expert] = cursor
		cursor += int(tokenCounts[expert])
	}
	gathered := tensors.New(total, moe.Dim)
	gatheredRows := make([]int, total)
	gatheredWeights := make([]float32, total)
	fill := make([]int, moe.NumExperts)
	for row := 0; row < rows; row++ {
		for slot := 0; slot < moe.TopK; slot++ {
			index := row*moe.TopK + slot
			expert := int(selected.Data[index])
			position := offsets[expert] + fill[expert]
			fill[expert]++
			copy(gathered.Data[position*moe.Dim:(position+1)*moe.Dim], flat.Data[row*moe.Dim:(row+1)*moe.Dim])
			gatheredRows[position] = row
			gatheredWeights[position] = selectedWeights.Data[index]
		}
	}

	// One grouped GEMM per projection over all experts.
	groupedGate := tensors.GroupedMatMulTransposed(gathered, moe.gateWeight, tokenCounts, moe.NumExperts, moe.Hidden, moe.Dim)
	groupedUp := tensors.GroupedMatMulTransposed(gathered, moe.upWeight, tokenCounts, moe.NumExperts, moe.Hidden, moe.Dim)
	groupedHidden := tensors.SwiGLU(groupedGate, groupedUp, moe.Clamp)
	groupedOutput := tensors.GroupedMatMulTransposed(groupedHidden, moe.downWeight, tokenCounts, moe.NumExperts, moe.Dim, moe.Hidden)

	output := tensors.New(rows, moe.Dim)
	for index := 0; index < total; index++ {
		weight := gatheredWeights[index] * moe.Scale
		rowBase := gatheredRows[index] * moe.Dim
		expertBase := index * moe.Dim
		for column := 0; column < moe.Dim; column++ {
			output.Data[rowBase+column] += weight * groupedOutput.Data[expertBase+column]
		}
	}

	if !training {
		return output.Reshape(shape...), nil
	}

	experts := make([]expertForward, 0, moe.NumExperts)
	for expert := 0; expert < moe.NumExperts; expert++ {
		count := int(tokenCounts[expert])
		if count == 0 {
			continue
		}
		offset := offsets[expert]
		experts = append(experts, expertForward{
			id:      expert,
			rows:    gatheredRows[offset : offset+count],
			weights: gatheredWeights[offset : offset+count],
			input:   sliceRows(gathered, offset, count, moe.Dim),
			gate:    sliceRows(groupedGate, offset, count, moe.Hidden),
			up:      sliceRows(groupedUp, offset, count, moe.Hidden),
			hidden:  sliceRows(groupedHidden, offset, count, moe.Hidden),
			output:  sliceRows(groupedOutput, offset, count, moe.Dim),
		})
		moe.loadCounts[expert] += float64(count)
		moe.loadTotal += float64(count)
	}
	context := &moeContext{shape: shape, rows: rows, input: flat, probs: probs, experts: experts}
	return output.Reshape(shape...), context
}

// backwardTraining backpropagates the routed contribution. It accumulates
// expert and router weight gradients and returns the gradient with respect to
// the MoE input.
func (moe *MoE) backwardTraining(outputGradient *tensors.Tensor, context *moeContext) *tensors.Tensor {
	rows := context.rows
	gradOutput := outputGradient.Reshape(rows, moe.Dim)
	gradInput := tensors.New(rows, moe.Dim)

	// dLoss/dProbs[row, expert] for the softmax backward below.
	gradProbs := make([]float64, rows*moe.NumExperts)

	for expertIndex := range context.experts {
		expert := &context.experts[expertIndex]
		expertID := expert.id
		count := len(expert.rows)
		gradExpertOutput := tensors.New(count, moe.Dim)
		for index, row := range expert.rows {
			weight := expert.weights[index] * moe.Scale
			base := row * moe.Dim
			expertBase := index * moe.Dim
			var dot float64
			for column := 0; column < moe.Dim; column++ {
				gradient := gradOutput.Data[base+column]
				gradExpertOutput.Data[expertBase+column] = gradient * weight
				dot += float64(gradient) * float64(expert.output.Data[expertBase+column])
			}
			gradProbs[row*moe.NumExperts+expertID] += dot * float64(moe.Scale)
		}

		gradHidden := moe.downLinear(expertID).Backward(expert.hidden, gradExpertOutput)
		gradGate, gradUp := tensors.SwiGLUBackward(expert.gate, expert.up, gradHidden, moe.Clamp)
		gradFromGate := moe.gateLinear(expertID).Backward(expert.input, gradGate)
		gradFromUp := moe.upLinear(expertID).Backward(expert.input, gradUp)
		for index, row := range expert.rows {
			base := row * moe.Dim
			expertBase := index * moe.Dim
			for column := 0; column < moe.Dim; column++ {
				gradInput.Data[base+column] += gradFromGate.Data[expertBase+column] + gradFromUp.Data[expertBase+column]
			}
		}
	}

	// Router softmax backward: dLogits = probs * (dProbs - <dProbs, probs>).
	gradLogits := tensors.New(rows, moe.NumExperts)
	for row := 0; row < rows; row++ {
		base := row * moe.NumExperts
		var dot float64
		for expert := 0; expert < moe.NumExperts; expert++ {
			dot += float64(context.probs.Data[base+expert]) * gradProbs[base+expert]
		}
		for expert := 0; expert < moe.NumExperts; expert++ {
			probability := float64(context.probs.Data[base+expert])
			gradLogits.Data[base+expert] = float32(probability * (gradProbs[base+expert] - dot))
		}
	}
	gradRouterInput := moe.router.weight.Backward(context.input, gradLogits)
	gradInput = tensors.Add(gradInput, gradRouterInput)
	return gradInput.Reshape(context.shape...)
}

// updateRouterBias applies the auxiliary-loss-free load-balancing update and
// resets the accumulated load counts.
func (moe *MoE) updateRouterBias(speed float32) {
	if moe.loadTotal <= 0 {
		return
	}
	mean := moe.loadTotal / float64(moe.NumExperts)
	for expert := 0; expert < moe.NumExperts; expert++ {
		difference := mean - moe.loadCounts[expert]
		moe.router.bias.Data[expert] += speed * float32(signFloat(difference))
		moe.loadCounts[expert] = 0
	}
	moe.loadTotal = 0
}

// sliceRows returns a [count, width] view of source rows [offset, offset+count)
// sharing source's Data. It is used to expose each expert's slice of the
// grouped forward buffers for backprop.
func sliceRows(source *tensors.Tensor, offset, count, width int) *tensors.Tensor {
	return &tensors.Tensor{
		Shape: []int{count, width},
		Data:  source.Data[offset*width : (offset+count)*width],
	}
}

// packedView returns a rank-2 view of one expert's slice of a packed
// [experts, rows, columns] tensor. The view shares the parent's Data and Grad
// backing arrays so linear-layer backward accumulates into the parent.
func packedView(packed *tensors.Tensor, index, rows, columns int) *tensors.Tensor {
	packed.EnsureGrad()
	start := index * rows * columns
	end := start + rows*columns
	return &tensors.Tensor{
		Shape: []int{rows, columns},
		Data:  packed.Data[start:end],
		Grad:  packed.Grad[start:end],
	}
}

func (moe *MoE) gateLinear(expert int) *layers.Linear {
	return &layers.Linear{InFeatures: moe.Dim, OutFeatures: moe.Hidden, Weight: packedView(moe.gateWeight, expert, moe.Hidden, moe.Dim)}
}

func (moe *MoE) upLinear(expert int) *layers.Linear {
	return &layers.Linear{InFeatures: moe.Dim, OutFeatures: moe.Hidden, Weight: packedView(moe.upWeight, expert, moe.Hidden, moe.Dim)}
}

func (moe *MoE) downLinear(expert int) *layers.Linear {
	return &layers.Linear{InFeatures: moe.Hidden, OutFeatures: moe.Dim, Weight: packedView(moe.downWeight, expert, moe.Dim, moe.Hidden)}
}

// expertViews returns per-expert rank-2 views of a packed expert tensor for the
// optimizer. The views share the parent's Data and Grad backing arrays, so a
// Muon update and the backpropagated gradient act on the same storage.
func expertViews(packed *tensors.Tensor, experts, rows, columns int) []*tensors.Tensor {
	views := make([]*tensors.Tensor, experts)
	for expert := 0; expert < experts; expert++ {
		views[expert] = packedView(packed, expert, rows, columns)
	}
	return views
}

// signFloat returns -1, 0, or 1 for the sign of value.
func signFloat(value float64) float64 {
	switch {
	case value > 0:
		return 1
	case value < 0:
		return -1
	default:
		return 0
	}
}
