// Package tasks implements the evaluation datasets (MMLU, GSM8K, ARC,
// HumanEval, SmolTalk) and the Task abstraction used for SFT training and
// ChatCORE evaluation.
package tasks

import (
	"math/rand"

	"github.com/cookiengineer/gonano/tokenizer"
)

// EvalType selects how a task is evaluated.
type EvalType int

const (
	// Generative tasks are evaluated by sampling completions.
	Generative EvalType = iota
	// Categorical tasks are evaluated by checking the argmax logit over
	// answer letters.
	Categorical
)

// Task is a dataset of conversations with an evaluation criterion.
type Task interface {
	EvalType() EvalType
	NumExamples() int
	GetExample(index int) *tokenizer.Conversation
	// Evaluate reports whether the assistant response is correct.
	Evaluate(conv *tokenizer.Conversation, response string) bool
	// Reward returns a scalar reward for RL (defaults to Evaluate as 0/1).
	Reward(conv *tokenizer.Conversation, response string) float32
}

// base provides the lightweight slicing of a Task (start/stop/step).
type base struct {
	start, stop, step int
}

func (b *base) len(total int) int {
	stop := b.stop
	if stop == 0 {
		stop = total
	}
	span := stop - b.start
	if span <= 0 {
		return 0
	}
	return (span + b.step - 1) / b.step
}

func (b *base) index(i int) int {
	return b.start + i*b.step
}

// reward0 implements the default Reward for tasks that reuse Evaluate.
func reward0(t Task, conv *tokenizer.Conversation, response string) float32 {
	if t.Evaluate(conv, response) {
		return 1
	}
	return 0
}

// TaskMixture combines several tasks with a deterministic shuffle, so tasks
// are interleaved throughout training.
type TaskMixture struct {
	tasks    []Task
	indexMap [][2]int // (taskIdx, localIdx)
}

// NewTaskMixture builds a mixture of tasks. Passing a task multiple times
// oversamples it.
func NewTaskMixture(tasks []Task) *TaskMixture {
	var lengths []int
	total := 0
	for _, t := range tasks {
		n := t.NumExamples()
		lengths = append(lengths, n)
		total += n
	}
	indexMap := make([][2]int, total)
	i := 0
	for ti := range tasks {
		for li := 0; li < lengths[ti]; li++ {
			indexMap[i] = [2]int{ti, li}
			i++
		}
	}
	rng := rand.New(rand.NewSource(42))
	rng.Shuffle(len(indexMap), func(i, j int) { indexMap[i], indexMap[j] = indexMap[j], indexMap[i] })
	return &TaskMixture{tasks: tasks, indexMap: indexMap}
}

func (m *TaskMixture) EvalType() EvalType { return Generative }

func (m *TaskMixture) NumExamples() int { return len(m.indexMap) }

func (m *TaskMixture) GetExample(index int) *tokenizer.Conversation {
	ti, li := m.indexMap[index][0], m.indexMap[index][1]
	return m.tasks[ti].GetExample(li)
}

func (m *TaskMixture) Evaluate(conv *tokenizer.Conversation, response string) bool { return false }
func (m *TaskMixture) Reward(conv *tokenizer.Conversation, response string) float32 { return 0 }

// RenderMC renders a multiple-choice question in nanochat's format: the letter
// is placed after the choice with no whitespace so the assistant's answer
// letter is the exact token that appears in the prompt.
func RenderMC(question string, letters, choices []string) string {
	query := "Multiple Choice question: " + question + "\n"
	for i := range letters {
		query += "- " + choices[i] + "=" + letters[i] + "\n"
	}
	query += "\nRespond only with the letter of the correct answer."
	return query
}
