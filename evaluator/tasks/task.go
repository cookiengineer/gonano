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
	Evaluate(conversation *tokenizer.Conversation, response string) bool
	// Reward returns a scalar reward for RL (defaults to Evaluate as 0/1).
	Reward(conversation *tokenizer.Conversation, response string) float32
}

// base provides the lightweight slicing of a Task (start/stop/step).
type base struct {
	start, stop, step int
}

func (baseTask *base) slicedCount(total int) int {
	stop := baseTask.stop
	if stop == 0 {
		stop = total
	}
	span := stop - baseTask.start
	if span <= 0 {
		return 0
	}
	return (span + baseTask.step - 1) / baseTask.step
}

func (baseTask *base) resolveIndex(localIndex int) int {
	return baseTask.start + localIndex*baseTask.step
}

// evaluateReward implements the default Reward for tasks that reuse Evaluate.
func evaluateReward(task Task, conversation *tokenizer.Conversation, response string) float32 {
	if task.Evaluate(conversation, response) {
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
	for _, task := range tasks {
		count := task.NumExamples()
		lengths = append(lengths, count)
		total += count
	}
	indexMap := make([][2]int, total)
	index := 0
	for taskIndex := range tasks {
		for localIndex := 0; localIndex < lengths[taskIndex]; localIndex++ {
			indexMap[index] = [2]int{taskIndex, localIndex}
			index++
		}
	}
	randomGenerator := rand.New(rand.NewSource(42))
	randomGenerator.Shuffle(len(indexMap), func(first, second int) {
		indexMap[first], indexMap[second] = indexMap[second], indexMap[first]
	})
	return &TaskMixture{tasks: tasks, indexMap: indexMap}
}

func (mixture *TaskMixture) EvalType() EvalType { return Generative }

func (mixture *TaskMixture) NumExamples() int { return len(mixture.indexMap) }

func (mixture *TaskMixture) GetExample(index int) *tokenizer.Conversation {
	taskIndex, localIndex := mixture.indexMap[index][0], mixture.indexMap[index][1]
	return mixture.tasks[taskIndex].GetExample(localIndex)
}

func (mixture *TaskMixture) Evaluate(conversation *tokenizer.Conversation, response string) bool {
	return false
}

func (mixture *TaskMixture) Reward(conversation *tokenizer.Conversation, response string) float32 {
	return 0
}

// RenderMC renders a multiple-choice question in nanochat's format: the letter
// is placed after the choice with no whitespace so the assistant's answer
// letter is the exact token that appears in the prompt.
func RenderMC(question string, letters, choices []string) string {
	query := "Multiple Choice question: " + question + "\n"
	for index := range letters {
		query += "- " + choices[index] + "=" + letters[index] + "\n"
	}
	query += "\nRespond only with the letter of the correct answer."
	return query
}
