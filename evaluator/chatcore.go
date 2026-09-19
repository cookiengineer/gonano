package evaluator

import (
	"github.com/cookiengineer/gonano/evaluator/tasks"
	"github.com/cookiengineer/gonano/inference"
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
)

// CategoricalAccuracy evaluates a categorical (multiple-choice) task by
// checking, for each example, which answer letter has the highest logit at the
// answer position.
func CategoricalAccuracy(task tasks.Task, transformer *model.Transformer, tokenizer *tokenizer.Tokenizer) float32 {
	total := task.NumExamples()
	if total == 0 {
		return 0
	}
	correct := 0
	for exampleIndex := 0; exampleIndex < total; exampleIndex++ {
		conversation := task.GetExample(exampleIndex)
		ids := tokenizer.RenderForCompletion(conversation)
		letters, _ := conversation.Extra["letters"].([]string)

		inputIDs := tensors.NewInt32sWithData([]int{1, len(ids)}, convertToInt32s(ids))
		logits := transformer.Forward(inputIDs, nil) // [1, T, vocab]
		vocab := transformer.Config.VocabSize
		lastLogits := logits.Reshape(len(ids), vocab).Data[(len(ids)-1)*vocab:]

		// Encode each letter and find the argmax among them.
		bestLetter, bestLogit := -1, float32(-1e30)
		for letterIndex, letter := range letters {
			letterIDs := tokenizer.Encode(letter)
			if len(letterIDs) != 1 {
				continue
			}
			if lastLogits[letterIDs[0]] > bestLogit {
				bestLogit = lastLogits[letterIDs[0]]
				bestLetter = letterIndex
			}
		}
		if bestLetter >= 0 && task.Evaluate(conversation, letters[bestLetter]) {
			correct++
		}
	}
	return float32(correct) / float32(total)
}

// GenerativeAccuracy evaluates a generative task by sampling a completion for
// each example and checking it against the task's evaluation criterion.
func GenerativeAccuracy(task tasks.Task, transformer *model.Transformer, tokenizer *tokenizer.Tokenizer, engine *inference.Engine, numSamples, maxTokens int, temperature float32, topK int) float32 {
	total := task.NumExamples()
	if total == 0 {
		return 0
	}
	passed := 0
	for exampleIndex := 0; exampleIndex < total; exampleIndex++ {
		conversation := task.GetExample(exampleIndex)
		prompt := tokenizer.RenderForCompletion(conversation)
		results, _ := engine.GenerateBatch(prompt, numSamples, maxTokens, temperature, topK, uint64(42+exampleIndex))
		prefix := len(prompt)
		ok := false
		for _, result := range results {
			completion := tokenizer.Decode(result[prefix:])
			if task.Evaluate(conversation, completion) {
				ok = true
				break
			}
		}
		if ok {
			passed++
		}
	}
	return float32(passed) / float32(total)
}

func convertToInt32s(ids []int) []int32 {
	converted := make([]int32, len(ids))
	for index, value := range ids {
		converted[index] = int32(value)
	}
	return converted
}

// ChatCORE computes the mean centered accuracy over the given tasks: each
// task's accuracy is normalized against its random baseline so the metric
// ranges from 0 (random) to 1 (perfect).
func ChatCORE(accuracies map[string]float32, baselines map[string]float32) float32 {
	var sum float32
	count := 0
	for name, accuracy := range accuracies {
		baseline := baselines[name]
		sum += (accuracy - baseline) / (1 - baseline)
		count++
	}
	if count == 0 {
		return 0
	}
	return sum / float32(count)
}
