package evaluator

import (
	"math"
	"math/rand"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
)

// TaskMeta describes a CORE benchmark task.
type TaskMeta struct {
	TaskType              string // "multiple_choice", "schema", "language_modeling"
	DatasetURI            string
	NumFewshot            int
	ContinuationDelimiter string
}

// CoreExample is a single CORE benchmark example. It is a JSON object with the
// fields used by the task type.
type CoreExample map[string]any

// renderMCPrompts renders the multiple-choice prompts for one example, one per
// choice.
func renderMCPrompts(item CoreExample, delimiter string, fewshot []CoreExample) []string {
	var prefix string
	for _, example := range fewshot {
		choices := example["choices"].([]string)
		gold := example["gold"].(int)
		prefix += example["query"].(string) + delimiter + choices[gold] + "\n\n"
	}
	choices := item["choices"].([]string)
	prompts := make([]string, len(choices))
	for choiceIndex, choice := range choices {
		prompts[choiceIndex] = prefix + item["query"].(string) + delimiter + choice
	}
	return prompts
}

// renderSchemaPrompts renders the schema prompts (context varies, continuation
// fixed).
func renderSchemaPrompts(item CoreExample, delimiter string, fewshot []CoreExample) []string {
	var prefix string
	for _, example := range fewshot {
		options := example["context_options"].([]string)
		gold := example["gold"].(int)
		prefix += options[gold] + delimiter + example["continuation"].(string) + "\n\n"
	}
	options := item["context_options"].([]string)
	prompts := make([]string, len(options))
	for optionIndex, option := range options {
		prompts[optionIndex] = prefix + option + delimiter + item["continuation"].(string)
	}
	return prompts
}

// renderLMPrompts renders the two language-modeling prompts (without and with
// the continuation).
func renderLMPrompts(item CoreExample, delimiter string, fewshot []CoreExample) []string {
	var prefix string
	for _, example := range fewshot {
		prefix += trimRight(example["context"].(string)) + delimiter + example["continuation"].(string) + "\n\n"
	}
	context := trimRight(item["context"].(string))
	withoutContinuation := prefix + context + delimiter
	withContinuation := prefix + context + delimiter + item["continuation"].(string)
	return []string{withoutContinuation, withContinuation}
}

func trimRight(text string) string {
	for len(text) > 0 && (text[len(text)-1] == ' ' || text[len(text)-1] == '\n' || text[len(text)-1] == '\t') {
		text = text[:len(text)-1]
	}
	return text
}

// findCommonPrefixLength returns the length of the common prefix of token
// sequences.
func findCommonPrefixLength(sequences [][]int) int {
	minLength := len(sequences[0])
	for _, sequence := range sequences {
		if len(sequence) < minLength {
			minLength = len(sequence)
		}
	}
	for position := 0; position < minLength; position++ {
		token := sequences[0][position]
		for _, sequence := range sequences {
			if sequence[position] != token {
				return position
			}
		}
	}
	return minLength
}

// findCommonSuffixLength returns the length of the common suffix.
func findCommonSuffixLength(sequences [][]int) int {
	minLength := len(sequences[0])
	for _, sequence := range sequences {
		if len(sequence) < minLength {
			minLength = len(sequence)
		}
	}
	for offset := 1; offset <= minLength; offset++ {
		token := sequences[0][len(sequences[0])-offset]
		for _, sequence := range sequences {
			if sequence[len(sequence)-offset] != token {
				return offset - 1
			}
		}
	}
	return minLength
}

// stackSequences pads token sequences to a common length.
func stackSequences(sequences [][]int, pad int) *tensors.Int32s {
	maxLength := 0
	for _, sequence := range sequences {
		if len(sequence) > maxLength {
			maxLength = len(sequence)
		}
	}
	stacked := tensors.NewInt32s(len(sequences), maxLength)
	for sequenceIndex, sequence := range sequences {
		for position := 0; position < maxLength; position++ {
			if position < len(sequence) {
				stacked.Set2(sequenceIndex, position, int32(sequence[position]))
			} else {
				stacked.Set2(sequenceIndex, position, int32(pad))
			}
		}
	}
	return stacked
}

// forwardLosses returns per-position cross-entropy losses and argmax
// predictions for a batch of token ids.
func forwardLosses(transformer *model.Transformer, input *tensors.Int32s) (*tensors.Tensor, *tensors.Int32s) {
	logits := transformer.Forward(input, nil) // [B,T,vocab]
	batchSize, sequenceLen, vocabSize := logits.Shape[0], logits.Shape[1], logits.Shape[2]
	flat := logits.Reshape(batchSize*sequenceLen, vocabSize)
	// targets are the next token (roll left).
	targets := tensors.NewInt32s(batchSize * sequenceLen)
	for position := 0; position < batchSize*sequenceLen; position++ {
		if (position+1)%sequenceLen == 0 {
			targets.Data[position] = -1
		} else {
			targets.Data[position] = input.Data[position+1]
		}
	}
	losses, _ := tensors.CrossEntropyPerPosition(flat, targets, -1)
	lossesTensor := tensors.NewWithData([]int{batchSize, sequenceLen}, losses)
	argmax := tensors.ArgMaxLastDim(flat).Reshape(batchSize, sequenceLen)
	return lossesTensor, argmax
}

// EvaluateExample scores a single CORE example, returning true if correct.
func EvaluateExample(transformer *model.Transformer, tokenizer *tokenizer.Tokenizer, data []CoreExample, exampleIndex int, taskMeta TaskMeta) bool {
	item := data[exampleIndex]

	var fewshot []CoreExample
	if taskMeta.NumFewshot > 0 {
		randomGenerator := rand.New(rand.NewSource(1234 + int64(exampleIndex)))
		permutation := randomGenerator.Perm(len(data))
		for _, candidateIndex := range permutation {
			if candidateIndex == exampleIndex {
				continue
			}
			fewshot = append(fewshot, data[candidateIndex])
			if len(fewshot) == taskMeta.NumFewshot {
				break
			}
		}
	}

	bos := tokenizer.BOSTokenID()
	var prompts []string
	var startIndices, endIndices []int
	var sequences [][]int

	switch taskMeta.TaskType {
	case "multiple_choice":
		prompts = renderMCPrompts(item, taskMeta.ContinuationDelimiter, fewshot)
		for _, prompt := range prompts {
			sequences = append(sequences, append([]int{bos}, tokenizer.Encode(prompt)...))
		}
		prefix := findCommonPrefixLength(sequences)
		for _, sequence := range sequences {
			startIndices = append(startIndices, prefix)
			endIndices = append(endIndices, len(sequence))
		}
	case "schema":
		prompts = renderSchemaPrompts(item, taskMeta.ContinuationDelimiter, fewshot)
		for _, prompt := range prompts {
			sequences = append(sequences, append([]int{bos}, tokenizer.Encode(prompt)...))
		}
		suffix := findCommonSuffixLength(sequences)
		for _, sequence := range sequences {
			startIndices = append(startIndices, len(sequence)-suffix)
			endIndices = append(endIndices, len(sequence))
		}
	case "language_modeling":
		prompts = renderLMPrompts(item, taskMeta.ContinuationDelimiter, fewshot)
		withoutContinuation := append([]int{bos}, tokenizer.Encode(prompts[0])...)
		withContinuation := append([]int{bos}, tokenizer.Encode(prompts[1])...)
		sequences = [][]int{withContinuation}
		startIndices = []int{len(withoutContinuation)}
		endIndices = []int{len(withContinuation)}
	default:
		return false
	}

	input := stackSequences(sequences, bos)
	losses, predictions := forwardLosses(transformer, input)

	if taskMeta.TaskType == "language_modeling" {
		startIndex, endIndex := startIndices[0], endIndices[0]
		ok := true
		for position := startIndex - 1; position < endIndex-1; position++ {
			if predictions.Data[position] != input.Data[position+1] {
				ok = false
				break
			}
		}
		return ok
	}

	// Multiple choice / schema: pick the option with the lowest mean loss.
	bestIndex := -1
	bestLoss := float32(math.Inf(1))
	for optionIndex := range startIndices {
		startIndex, endIndex := startIndices[optionIndex], endIndices[optionIndex]
		var sum float32
		count := 0
		for position := startIndex - 1; position < endIndex-1; position++ {
			if position >= 0 && position < losses.Shape[1] {
				sum += losses.Data[optionIndex*losses.Shape[1]+position]
				count++
			}
		}
		if count > 0 {
			mean := sum / float32(count)
			if mean < bestLoss {
				bestLoss = mean
				bestIndex = optionIndex
			}
		}
	}
	gold := item["gold"].(int)
	return bestIndex == gold
}

// EvaluateTask evaluates a task across examples and returns the mean accuracy.
func EvaluateTask(transformer *model.Transformer, tokenizer *tokenizer.Tokenizer, data []CoreExample, taskMeta TaskMeta) float32 {
	if len(data) == 0 {
		return 0
	}
	correct := 0
	for exampleIndex := range data {
		if EvaluateExample(transformer, tokenizer, data, exampleIndex, taskMeta) {
			correct++
		}
	}
	return float32(correct) / float32(len(data))
}
