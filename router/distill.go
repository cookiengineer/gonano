package router

// DistillNgram trains the n-gram fast path from a transformer teacher: every
// example is relabeled with the teacher's argmax prediction and the n-gram
// classifier is fit on those labels. This is the knowledge-distillation step
// that lets the cheap classifier approximate the meta model.
func DistillNgram(teacher *TransformerClassifier, examples []Example, configuration NgramConfig) *NgramClassifier {
	classes := teacher.NumClasses()
	labeled := make([]Example, len(examples))
	for index, example := range examples {
		probabilities := teacher.Predict(example.Tokens)
		labeled[index] = Example{Tokens: example.Tokens, Label: argMax(probabilities)}
	}
	return TrainNgram(labeled, classes, configuration)
}

// DistillAccuracy returns the fraction of examples on which the student's
// argmax matches the teacher's argmax — the distillation fidelity.
func DistillAccuracy(teacher *TransformerClassifier, student *NgramClassifier, examples []Example) float32 {
	if len(examples) == 0 {
		return 0
	}
	matches := 0
	for _, example := range examples {
		teacherLabel := argMax(teacher.Predict(example.Tokens))
		studentLabel := argMax(student.Predict(example.Tokens))
		if teacherLabel == studentLabel {
			matches++
		}
	}
	return float32(matches) / float32(len(examples))
}
