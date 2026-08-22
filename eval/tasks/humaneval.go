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
	if m := codeBlockRE.FindStringSubmatch(completion); m != nil {
		return strings.TrimSpace(m[1])
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
	Prompt    string
	Solution  string
	EntryPoint string
	Test      string
}

// NewHumanEvalFromRows builds a HumanEval task from raw rows.
func NewHumanEvalFromRows(rows []HumanEvalRow) *HumanEval {
	t := &HumanEval{}
	for _, row := range rows {
		complete := row.Prompt + "\n" + row.Solution
		t.examples = append(t.examples, &tokenizer.Conversation{
			Messages: []tokenizer.Message{
				{Role: "user", Content: row.Prompt},
				{Role: "assistant", Content: complete},
			},
			Extra: map[string]any{"entry_point": row.EntryPoint, "test": row.Test},
		})
	}
	return t
}

func (h *HumanEval) EvalType() EvalType { return Generative }

func (h *HumanEval) NumExamples() int { return len(h.examples) }

func (h *HumanEval) GetExample(index int) *tokenizer.Conversation { return h.examples[index] }

func (h *HumanEval) Evaluate(conv *tokenizer.Conversation, completion string) bool {
	imports := ExtractImports(conv.Messages[0].Content)
	code := ExtractProgram(completion)
	entryPoint, _ := conv.Extra["entry_point"].(string)
	test, _ := conv.Extra["test"].(string)
	program := imports + "\n\n" + code + "\n\n" + test + "\n" + "check(" + entryPoint + ")"
	if h.Execute != nil {
		return h.Execute(program)
	}
	return false
}

func (h *HumanEval) Reward(conv *tokenizer.Conversation, response string) float32 {
	return reward0(h, conv, response)
}
