package tasks

import (
	"regexp"
	"strings"

	"github.com/cookiengineer/gonano/tokenizer"
)

// codeBlockRE matches a fenced Python code block.
var codeBlockRE = regexp.MustCompile("(?s)```(?:python)?\\s*\\n(.*?)\\n```")

// ExtractProgram extracts the first Python code block from a completion, or
// the whole completion if no fenced block is present.
func ExtractProgram(completion string) string {
	if matches := codeBlockRE.FindStringSubmatch(completion); matches != nil {
		return strings.TrimSpace(matches[1])
	}
	return strings.TrimSpace(completion)
}

// ExtractImports extracts leading import statements from a program.
func ExtractImports(prompt string) string {
	var imports []string
	for _, line := range strings.Split(prompt, "\n") {
		stripped := strings.TrimSpace(line)
		if strings.HasPrefix(stripped, "import ") || strings.HasPrefix(stripped, "from ") {
			imports = append(imports, stripped)
		} else if stripped != "" && !strings.HasPrefix(stripped, "#") {
			break
		}
	}
	return strings.Join(imports, "\n")
}

// HumanEval is a coding benchmark: the assistant completes a Python function
// that is then checked against unit tests in a sandbox.
type HumanEval struct {
	examples []*tokenizer.Conversation
	// Execute runs a program and reports success (injectable for tests).
	Execute func(program string) bool
}

// HumanEvalRow is a single raw HumanEval example.
type HumanEvalRow struct {
	Prompt     string
	Solution   string
	EntryPoint string
	Test       string
}

// NewHumanEvalFromRows builds a HumanEval task from raw rows.
func NewHumanEvalFromRows(rows []HumanEvalRow) *HumanEval {
	task := &HumanEval{}
	for _, row := range rows {
		complete := row.Prompt + "\n" + row.Solution
		task.examples = append(task.examples, &tokenizer.Conversation{
			Messages: []tokenizer.Message{
				{Role: "user", Content: row.Prompt},
				{Role: "assistant", Content: complete},
			},
			Extra: map[string]any{"entry_point": row.EntryPoint, "test": row.Test},
		})
	}
	return task
}

func (humaneval *HumanEval) EvalType() EvalType { return Generative }

func (humaneval *HumanEval) NumExamples() int { return len(humaneval.examples) }

func (humaneval *HumanEval) GetExample(index int) *tokenizer.Conversation {
	return humaneval.examples[index]
}

func (humaneval *HumanEval) Evaluate(conversation *tokenizer.Conversation, completion string) bool {
	imports := ExtractImports(conversation.Messages[0].Content)
	code := ExtractProgram(completion)
	entryPoint, _ := conversation.Extra["entry_point"].(string)
	test, _ := conversation.Extra["test"].(string)
	program := imports + "\n\n" + code + "\n\n" + test + "\n" + "check(" + entryPoint + ")"
	if humaneval.Execute != nil {
		return humaneval.Execute(program)
	}
	return false
}

func (humaneval *HumanEval) Reward(conversation *tokenizer.Conversation, response string) float32 {
	return evaluateReward(humaneval, conversation, response)
}
