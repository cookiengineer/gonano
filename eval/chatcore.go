package eval

import (
	"github.com/cookiengineer/gonano/eval/tasks"
	"github.com/cookiengineer/gonano/infer"
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensor"
	"github.com/cookiengineer/gonano/tokenizer"
)

// CategoricalAccuracy evaluates a categorical (multiple-choice) task by
// checking, for each example, which answer letter has the highest logit at the
// answer position.
func CategoricalAccuracy(task tasks.Task, m *model.Transformer, tok *tokenizer.Tokenizer) float32 {
	total := task.NumExamples()
	if total == 0 {
		return 0
	}
	correct := 0
	for i := 0; i < total; i++ {
		conv := task.GetExample(i)
		ids := tok.RenderForCompletion(conv)
		letters, _ := conv.Extra["letters"].([]string)

		idx := tensor.NewInt32sWithData([]int{1, len(ids)}, toI32(ids))
		logits := m.Forward(idx, nil) // [1, T, vocab]
		vocab := m.Config.VocabSize
		last := logits.Reshape(len(ids), vocab).Data[(len(ids)-1)*vocab:]

		// Encode each letter and find the argmax among them.
		best, bestLogit := -1, float32(-1e30)
		for li, letter := range letters {
			letterIDs := tok.Encode(letter)
			if len(letterIDs) != 1 {
				continue
			}
			if last[letterIDs[0]] > bestLogit {
				bestLogit = last[letterIDs[0]]
				best = li
			}
		}
		if best >= 0 && task.Evaluate(conv, letters[best]) {
			correct++
		}
	}
	return float32(correct) / float32(total)
}

// GenerativeAccuracy evaluates a generative task by sampling a completion for
// each example and checking it against the task's evaluation criterion.
func GenerativeAccuracy(task tasks.Task, m *model.Transformer, tok *tokenizer.Tokenizer, engine *infer.Engine, numSamples, maxTokens int, temperature float32, topK int) float32 {
	total := task.NumExamples()
	if total == 0 {
		return 0
	}
	passed := 0
	for i := 0; i < total; i++ {
		conv := task.GetExample(i)
		prompt := tok.RenderForCompletion(conv)
		results, _ := engine.GenerateBatch(prompt, numSamples, maxTokens, temperature, topK, uint64(42+i))
		prefix := len(prompt)
		ok := false
		for _, r := range results {
			completion := tok.Decode(r[prefix:])
			if task.Evaluate(conv, completion) {
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

func toI32(ids []int) []int32 {
	out := make([]int32, len(ids))
	for i, v := range ids {
		out[i] = int32(v)
	}
	return out
}

// ChatCORE computes the mean centered accuracy over the given tasks: each
// task's accuracy is normalized against its random baseline so the metric
// ranges from 0 (random) to 1 (perfect).
func ChatCORE(accuracies map[string]float32, baselines map[string]float32) float32 {
	var sum float32
	n := 0
	for name, acc := range accuracies {
		base := baselines[name]
		sum += (acc - base) / (1 - base)
		n++
	}
	if n == 0 {
		return 0
	}
	return sum / float32(n)
}
