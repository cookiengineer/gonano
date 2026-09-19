package optimizer

import (
	"math"

	"github.com/cookiengineer/gonano/tensors"
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

// muonStep applies one Muon update to the 2D parameter param: Nesterov
// momentum, MuonEq row equilibration, Polar Express orthogonalization, Muon+
// renormalization, variance reduction, and cautious weight decay.
func (optimizer *MuonAdamW) muonStep(group ParamGroup, param *tensors.Tensor) {
	state := optimizer.muonStates[param]
	param.EnsureGrad()
	rows, columns := param.Shape[0], param.Shape[1]
	momentum := group.Momentum

	// Nesterov momentum:
	//   mom = momentum*mom + (1-momentum)*grad
	//   update = (1-momentum)*grad + momentum*mom
	update := tensors.New(rows, columns)
	for index := range param.Data {
		state.momentum.Data[index] = momentum*state.momentum.Data[index] + (1-momentum)*param.Grad[index]
		update.Data[index] = (1-momentum)*param.Grad[index] + momentum*state.momentum.Data[index]
	}

	// MuonEq row equilibration: rescale each row to the mean row norm.
	target := computeFrobeniusNorm(update) / float32(math.Sqrt(float64(rows)))
	for row := 0; row < rows; row++ {
		rowNorm := computeRowL2Norm(update.Data[row*columns : (row+1)*columns])
		if rowNorm < 1e-6 {
			rowNorm = 1e-6
		}
		scale := target / rowNorm
		for column := 0; column < columns; column++ {
			update.Data[row*columns+column] *= scale
		}
	}

	// Polar Express orthogonalization.
	norm := computeFrobeniusNorm(update)
	scale := float32(1.0 / (float64(norm)*1.01 + 1e-6))
	for index := range update.Data {
		update.Data[index] *= scale
	}
	for step := 0; step < group.NSSteps; step++ {
		coeffA := polarExpressCoeffs[step][0]
		coeffB := polarExpressCoeffs[step][1]
		coeffC := polarExpressCoeffs[step][2]
		var gramMatrix *tensors.Tensor
		if rows > columns {
			gramMatrix = tensors.MatMul(tensors.Transpose(update), update) // [columns,columns] tall
		} else {
			gramMatrix = tensors.MatMul(update, tensors.Transpose(update)) // [rows,rows] wide
		}
		gramSquared := tensors.MatMul(gramMatrix, gramMatrix)
		polyTerm := tensors.Add(tensors.Scale(gramMatrix, float32(coeffB)), tensors.Scale(gramSquared, float32(coeffC)))
		if rows > columns {
			update = tensors.Add(tensors.Scale(update, float32(coeffA)), tensors.MatMul(update, polyTerm))
		} else {
			update = tensors.Add(tensors.Scale(update, float32(coeffA)), tensors.MatMul(polyTerm, update))
		}
	}

	// Muon+ renormalization: snap the Frobenius norm to sqrt(min(rows, columns)).
	targetNorm := float32(math.Sqrt(float64(min(rows, columns))))
	currentNorm := computeFrobeniusNorm(update)
	if currentNorm < 1e-6 {
		currentNorm = 1e-6
	}
	renormScale := targetNorm / currentNorm
	for index := range update.Data {
		update.Data[index] *= renormScale
	}

	// Variance reduction (NorMuon): per-row/column adaptive learning rate.
	var varianceMean []float32
	var reduceSize int
	if rows >= columns {
		// Reduce over columns (red_dim = -1): per-row means, shape [rows].
		varianceMean = make([]float32, rows)
		reduceSize = columns
		for row := 0; row < rows; row++ {
			var sum float64
			for column := 0; column < columns; column++ {
				sum += float64(update.Data[row*columns+column]) * float64(update.Data[row*columns+column])
			}
			varianceMean[row] = float32(sum / float64(columns))
		}
	} else {
		// Reduce over rows (red_dim = -2): per-column means, shape [columns].
		varianceMean = make([]float32, columns)
		reduceSize = rows
		for column := 0; column < columns; column++ {
			var sum float64
			for row := 0; row < rows; row++ {
				sum += float64(update.Data[row*columns+column]) * float64(update.Data[row*columns+column])
			}
			varianceMean[column] = float32(sum / float64(rows))
		}
	}

	var varianceNormSquared float64
	for _, mean := range varianceMean {
		varianceNormSquared += float64(mean)
	}
	varianceNormSquared *= float64(reduceSize)
	varianceNorm := float32(math.Sqrt(varianceNormSquared))

	beta2 := group.MuonBeta2
	stepSize := make([]float32, len(varianceMean))
	for index := range varianceMean {
		state.secondMom.Data[index] = beta2*state.secondMom.Data[index] + (1-beta2)*varianceMean[index]
		secondMoment := state.secondMom.Data[index]
		if secondMoment < 1e-10 {
			secondMoment = 1e-10
		}
		stepSize[index] = float32(1.0 / math.Sqrt(float64(secondMoment)))
	}

	var newVarianceNormSquared float64
	for index := range varianceMean {
		term := float64(varianceMean[index]) * float64(reduceSize) * float64(stepSize[index]) * float64(stepSize[index])
		newVarianceNormSquared += term
	}
	newVarianceNorm := float32(math.Sqrt(newVarianceNormSquared))
	if newVarianceNorm < 1e-10 {
		newVarianceNorm = 1e-10
	}
	ratio := varianceNorm / newVarianceNorm
	finalScale := make([]float32, len(varianceMean))
	for index := range stepSize {
		finalScale[index] = stepSize[index] * ratio
	}

	if rows >= columns {
		// finalScale is per-row [rows].
		for row := 0; row < rows; row++ {
			for column := 0; column < columns; column++ {
				update.Data[row*columns+column] *= finalScale[row]
			}
		}
	} else {
		// finalScale is per-column [columns].
		for row := 0; row < rows; row++ {
			for column := 0; column < columns; column++ {
				update.Data[row*columns+column] *= finalScale[column]
			}
		}
	}

	// Cautious weight decay + parameter update.
	learningRate := group.LR
	weightDecay := group.WeightDecay
	for index := range param.Data {
		var mask float32
		if update.Data[index]*param.Data[index] >= 0 {
			mask = 1
		}
		param.Data[index] -= learningRate*update.Data[index] + learningRate*weightDecay*param.Data[index]*mask
	}
}

func computeFrobeniusNorm(matrix *tensors.Tensor) float32 {
	var sum float64
	for _, value := range matrix.Data {
		sum += float64(value) * float64(value)
	}
	return float32(math.Sqrt(sum))
}

func computeRowL2Norm(values []float32) float32 {
	var sum float64
	for _, value := range values {
		sum += float64(value) * float64(value)
	}
	return float32(math.Sqrt(sum))
}
