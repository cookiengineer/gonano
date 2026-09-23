package optimizer

import (
	"math"

	"github.com/cookiengineer/gonano/tensors"
)

// sinkhornState is the per-parameter optimizer state for the Sinkhorn-balanced
// momentum update (DeepSeek-V4.1 §2.5, Algorithm 1). Unlike AdamW and Muon it
// keeps only a momentum buffer, which is the memory saving the paper relies on
// for the large embedding and prediction-head matrices.
type sinkhornState struct {
	momentum *tensors.Tensor
}

// sinkhornStep applies one Sinkhorn-balanced momentum update to the 2D
// parameter param: Nesterov momentum, masking of near-zero rows, K alternating
// row/column L2 normalizations, a unit-RMS rescale, and the gamma-corrected
// learning-rate update. It applies no weight decay.
//
// The rows correspond to token (or n-gram) identities and the columns to hidden
// features, so balancing the update along both axes equalizes the per-token and
// per-feature step sizes.
func (optimizer *MuonAdamW) sinkhornStep(group ParamGroup, param *tensors.Tensor) {
	state := optimizer.sinkhornStates[param]
	param.EnsureGrad()
	rows, columns := param.Shape[0], param.Shape[1]
	beta := float64(group.Momentum)

	// Nesterov momentum:
	//   M_t     = beta*M_{t-1} + (1-beta)*G_t
	//   G_hat_t = beta*M_t + (1-beta)*G_t
	update := tensors.New(rows, columns)
	for index := range param.Data {
		momentumValue := beta*float64(state.momentum.Data[index]) + (1-beta)*float64(param.Grad[index])
		state.momentum.Data[index] = float32(momentumValue)
		update.Data[index] = float32(beta*momentumValue + (1-beta)*float64(param.Grad[index]))
	}

	// Row norms and the mean row norm. Rows whose norm is a small fraction tau
	// of the mean carry no useful update and are masked to zero.
	rowNorms := make([]float64, rows)
	var normSum float64
	for row := 0; row < rows; row++ {
		norm := rowL2Norm(update.Data[row*columns : (row+1)*columns])
		rowNorms[row] = norm
		normSum += norm
	}
	threshold := float64(group.SinkhornTau) * normSum / float64(rows)
	for row := 0; row < rows; row++ {
		if rowNorms[row] <= threshold {
			row := update.Data[row*columns : (row+1)*columns]
			for index := range row {
				row[index] = 0
			}
		}
	}

	// Sinkhorn balancing: alternate row normalization on odd steps and column
	// normalization on even steps, which approximately equalizes the row-wise
	// and column-wise RMS of the update matrix.
	epsilon := float64(group.SinkhornEps)
	for step := 1; step <= group.SinkhornSteps; step++ {
		if step%2 == 1 {
			for row := 0; row < rows; row++ {
				values := update.Data[row*columns : (row+1)*columns]
				scale := 1.0 / (rowL2Norm(values) + epsilon)
				for index := range values {
					values[index] = float32(float64(values[index]) * scale)
				}
			}
		} else {
			for column := 0; column < columns; column++ {
				var sum float64
				for row := 0; row < rows; row++ {
					value := float64(update.Data[row*columns+column])
					sum += value * value
				}
				scale := 1.0 / (math.Sqrt(sum) + epsilon)
				for row := 0; row < rows; row++ {
					update.Data[row*columns+column] = float32(float64(update.Data[row*columns+column]) * scale)
				}
			}
		}
	}

	// Convert the unit row L2 norm into a unit row RMS (factor sqrt(columns))
	// and match the Adam update magnitude with the gamma correction factor:
	//   W_{t+1} = W_t - gamma*eta_t * sqrt(n) * U^(K)
	learningRate := float64(group.Gamma) * float64(group.LR) * math.Sqrt(float64(columns))
	for index := range param.Data {
		param.Data[index] -= float32(learningRate * float64(update.Data[index]))
	}
}

// rowL2Norm returns the L2 norm of a row slice in float64 for numerical
// stability.
func rowL2Norm(values []float32) float64 {
	var sum float64
	for _, value := range values {
		sum += float64(value) * float64(value)
	}
	return math.Sqrt(sum)
}
