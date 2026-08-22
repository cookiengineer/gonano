package eval

import (
	"math"
	"math/rand"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensor"
	"github.com/cookiengineer/gonano/tokenizer"
)

// TaskMeta describes a CORE benchmark task.
type TaskMeta struct {
	TaskType             string // "multiple_choice", "schema", "language_modeling"
	DatasetURI           string
	NumFewshot           int
	ContinuationDelimiter string
}

// CoreExample is a single CORE benchmark example. It is a JSON object with the
// fields used by the task type.
type CoreExample map[string]any

// renderMCPrompts renders the multiple-choice prompts for one example, one per
// choice.
func renderMCPrompts(item CoreExample, delimiter string, fewshot []CoreExample) []string {
	var sb string
	for _, ex := range fewshot {
		choices := ex["choices"].([]string)
		gold := ex["gold"].(int)
		sb += ex["query"].(string) + delimiter + choices[gold] + "\n\n"
	}
	choices := item["choices"].([]string)
	prompts := make([]string, len(choices))
	for i, choice := range choices {
		prompts[i] = sb + item["query"].(string) + delimiter + choice
	}
	return prompts
}

// renderSchemaPrompts renders the schema prompts (context varies, continuation
// fixed).
func renderSchemaPrompts(item CoreExample, delimiter string, fewshot []CoreExample) []string {
	var sb string
	for _, ex := range fewshot {
		opts := ex["context_options"].([]string)
		gold := ex["gold"].(int)
		sb += opts[gold] + delimiter + ex["continuation"].(string) + "\n\n"
	}
	opts := item["context_options"].([]string)
	prompts := make([]string, len(opts))
	for i, opt := range opts {
		prompts[i] = sb + opt + delimiter + item["continuation"].(string)
	}
	return prompts
}

// renderLMPrompts renders the two language-modeling prompts (without and with
// the continuation).
func renderLMPrompts(item CoreExample, delimiter string, fewshot []CoreExample) []string {
	var sb string
	for _, ex := range fewshot {
		sb += trimRight(ex["context"].(string)) + delimiter + ex["continuation"].(string) + "\n\n"
	}
	context := trimRight(item["context"].(string))
	without := sb + context + delimiter
	with := sb + context + delimiter + item["continuation"].(string)
	return []string{without, with}
}

func trimRight(s string) string {
	for len(s) > 0 && (s[len(s)-1] == ' ' || s[len(s)-1] == '\n' || s[len(s)-1] == '\t') {
		s = s[:len(s)-1]
	}
	return s
}

// commonPrefixLength returns the length of the common prefix of token
// sequences.
func commonPrefixLength(seqs [][]int) int {
	minLen := len(seqs[0])
	for _, s := range seqs {
		if len(s) < minLen {
			minLen = len(s)
		}
	}
	for i := 0; i < minLen; i++ {
		token := seqs[0][i]
		for _, s := range seqs {
			if s[i] != token {
				return i
			}
		}
	}
	return minLen
}

// commonSuffixLength returns the length of the common suffix.
func commonSuffixLength(seqs [][]int) int {
	minLen := len(seqs[0])
	for _, s := range seqs {
		if len(s) < minLen {
			minLen = len(s)
		}
	}
	for i := 1; i <= minLen; i++ {
		token := seqs[0][len(seqs[0])-i]
		for _, s := range seqs {
			if s[len(s)-i] != token {
				return i - 1
			}
		}
	}
	return minLen
}

// stackSequences pads token sequences to a common length.
func stackSequences(seqs [][]int, pad int) *tensor.Int32s {
	maxLen := 0
	for _, s := range seqs {
		if len(s) > maxLen {
			maxLen = len(s)
		}
	}
	out := tensor.NewInt32s(len(seqs), maxLen)
	for i, s := range seqs {
		for j := 0; j < maxLen; j++ {
			if j < len(s) {
				out.Set2(i, j, int32(s[j]))
			} else {
				out.Set2(i, j, int32(pad))
			}
		}
	}
	return out
}

// forwardLosses returns per-position cross-entropy losses and argmax
// predictions for a batch of token ids.
func forwardLosses(m *model.Transformer, input *tensor.Int32s) (*tensor.Tensor, *tensor.Int32s) {
	logits := m.Forward(input, nil) // [B,T,vocab]
	b, t, vocab := logits.Shape[0], logits.Shape[1], logits.Shape[2]
	flat := logits.Reshape(b*t, vocab)
	// targets are the next token (roll left).
	targets := tensor.NewInt32s(b * t)
	for i := 0; i < b*t; i++ {
		if (i+1)%t == 0 {
			targets.Data[i] = -1
		} else {
			targets.Data[i] = input.Data[i+1]
		}
	}
	losses, _ := tensor.CrossEntropyPerPosition(flat, targets, -1)
	lossesT := tensor.NewWithData([]int{b, t}, losses)
	argmax := tensor.ArgMaxLastDim(flat).Reshape(b, t)
	return lossesT, argmax
}

// EvaluateExample scores a single CORE example, returning true if correct.
func EvaluateExample(m *model.Transformer, tok *tokenizer.Tokenizer, data []CoreExample, idx int, meta TaskMeta) bool {
	item := data[idx]

	var fewshot []CoreExample
	if meta.NumFewshot > 0 {
		rng := rand.New(rand.NewSource(1234 + int64(idx)))
		perm := rng.Perm(len(data))
		for _, i := range perm {
			if i == idx {
				continue
			}
			fewshot = append(fewshot, data[i])
			if len(fewshot) == meta.NumFewshot {
				break
			}
		}
	}

	bos := tok.BOSTokenID()
	var prompts []string
	var startIdxs, endIdxs []int
	var seqs [][]int

	switch meta.TaskType {
	case "multiple_choice":
		prompts = renderMCPrompts(item, meta.ContinuationDelimiter, fewshot)
		for _, p := range prompts {
			seqs = append(seqs, append([]int{bos}, tok.Encode(p)...))
		}
		prefix := commonPrefixLength(seqs)
		for _, s := range seqs {
			startIdxs = append(startIdxs, prefix)
			endIdxs = append(endIdxs, len(s))
		}
	case "schema":
		prompts = renderSchemaPrompts(item, meta.ContinuationDelimiter, fewshot)
		for _, p := range prompts {
			seqs = append(seqs, append([]int{bos}, tok.Encode(p)...))
		}
		suffix := commonSuffixLength(seqs)
		for _, s := range seqs {
			startIdxs = append(startIdxs, len(s)-suffix)
			endIdxs = append(endIdxs, len(s))
		}
	case "language_modeling":
		prompts = renderLMPrompts(item, meta.ContinuationDelimiter, fewshot)
		without := append([]int{bos}, tok.Encode(prompts[0])...)
		with := append([]int{bos}, tok.Encode(prompts[1])...)
		seqs = [][]int{with}
		startIdxs = []int{len(without)}
		endIdxs = []int{len(with)}
	default:
		return false
	}

	input := stackSequences(seqs, bos)
	losses, predictions := forwardLosses(m, input)

	if meta.TaskType == "language_modeling" {
		si, ei := startIdxs[0], endIdxs[0]
		ok := true
		for k := si - 1; k < ei-1; k++ {
			if predictions.Data[k] != input.Data[k+1] {
				ok = false
				break
			}
		}
		return ok
	}

	// Multiple choice / schema: pick the option with the lowest mean loss.
	bestIdx := -1
	bestLoss := float32(math.Inf(1))
	for i := range startIdxs {
		si, ei := startIdxs[i], endIdxs[i]
		var sum float32
		count := 0
		for k := si - 1; k < ei-1; k++ {
			if k >= 0 && k < losses.Shape[1] {
				sum += losses.Data[i*losses.Shape[1]+k]
				count++
			}
		}
		if count > 0 {
			mean := sum / float32(count)
			if mean < bestLoss {
				bestLoss = mean
				bestIdx = i
			}
		}
	}
	gold := item["gold"].(int)
	return bestIdx == gold
}

// EvaluateTask evaluates a task across examples and returns the mean accuracy.
func EvaluateTask(m *model.Transformer, tok *tokenizer.Tokenizer, data []CoreExample, meta TaskMeta) float32 {
	if len(data) == 0 {
		return 0
	}
	correct := 0
	for i := range data {
		if EvaluateExample(m, tok, data, i, meta) {
			correct++
		}
	}
	return float32(correct) / float32(len(data))
}
