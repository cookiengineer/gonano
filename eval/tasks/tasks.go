package tasks

import (
	"regexp"

	"github.com/cookiengineer/gonano/tokenizer"
)

// MMLU is the massive multitask language understanding dataset (4-way multiple
// choice). It is a categorical task.
type MMLU struct {
	examples []*tokenizer.Conversation
}

var mmluLetters = []string{"A", "B", "C", "D"}

// NewMMLUFromRows builds an MMLU task from raw dataset rows. Each row has
// "question" (string), "choices" ([]string), and "answer" (int index).
func NewMMLUFromRows(rows []MMLURow) *MMLU {
	t := &MMLU{}
	for _, row := range rows {
		if len(row.Choices) != 4 {
			continue
		}
		user := RenderMC(row.Question, mmluLetters, row.Choices)
		t.examples = append(t.examples, &tokenizer.Conversation{
			Messages: []tokenizer.Message{
				{Role: "user", Content: user},
				{Role: "assistant", Content: mmluLetters[row.Answer]},
			},
			Extra: map[string]any{"letters": mmluLetters},
		})
	}
	return t
}

// MMLURow is a single raw MMLU example.
type MMLURow struct {
	Question string
	Choices  []string
	Answer   int
}

func (m *MMLU) EvalType() EvalType { return Categorical }

func (m *MMLU) NumExamples() int { return len(m.examples) }

func (m *MMLU) GetExample(index int) *tokenizer.Conversation { return m.examples[index] }

func (m *MMLU) Evaluate(conv *tokenizer.Conversation, response string) bool {
	return response == conv.Messages[len(conv.Messages)-1].Content
}

func (m *MMLU) Reward(conv *tokenizer.Conversation, response string) float32 {
	return reward0(m, conv, response)
}

// ARC is the AI2 Reasoning Challenge dataset (multiple choice).
type ARC struct {
	examples []*tokenizer.Conversation
}

// ARCrow is a single raw ARC example.
type ARCRow struct {
	Question string
	Choices  []string
	Letters  []string
	Answer   string // one of the letters
}

// NewARCFromRows builds an ARC task from raw rows.
func NewARCFromRows(rows []ARCRow) *ARC {
	t := &ARC{}
	for _, row := range rows {
		user := RenderMC(row.Question, row.Letters, row.Choices)
		t.examples = append(t.examples, &tokenizer.Conversation{
			Messages: []tokenizer.Message{
				{Role: "user", Content: user},
				{Role: "assistant", Content: row.Answer},
			},
			Extra: map[string]any{"letters": row.Letters},
		})
	}
	return t
}

func (a *ARC) EvalType() EvalType { return Categorical }

func (a *ARC) NumExamples() int { return len(a.examples) }

func (a *ARC) GetExample(index int) *tokenizer.Conversation { return a.examples[index] }

func (a *ARC) Evaluate(conv *tokenizer.Conversation, response string) bool {
	return response == conv.Messages[len(conv.Messages)-1].Content
}

func (a *ARC) Reward(conv *tokenizer.Conversation, response string) float32 {
	return reward0(a, conv, response)
}

// gsmRE extracts the numerical answer after the "####" marker.
var gsmRE = regexp.MustCompile(`####\s*([0-9.,-]+)`)

// ExtractAnswer returns the answer after "####" in a GSM8K completion, or an
// empty string if none is found.
func ExtractAnswer(completion string) string {
	m := gsmRE.FindStringSubmatch(completion)
	if m == nil {
		return ""
	}
	return m[1]
}

// GSM8K is the grade-school math dataset. Assistant messages contain tool
// calls (the <<expr=result>> calculator syntax).
type GSM8K struct {
	examples []*tokenizer.Conversation
}

// NewGSM8KFromRows builds a GSM8K task from raw rows (question, answer).
func NewGSM8KFromRows(rows []GSM8KRow) *GSM8K {
	t := &GSM8K{}
	for _, row := range rows {
		parts := splitGSM8KAnswer(row.Answer)
		t.examples = append(t.examples, &tokenizer.Conversation{
			Messages: []tokenizer.Message{
				{Role: "user", Content: row.Question},
				{Role: "assistant", Parts: parts},
			},
		})
	}
	return t
}

// GSM8KRow is a single raw GSM8K example.
type GSM8KRow struct {
	Question string
	Answer   string
}

// splitGSM8KAnswer parses the <<expr=result>> tool calls in a GSM8K answer.
func splitGSM8KAnswer(answer string) []tokenizer.MessagePart {
	var parts []tokenizer.MessagePart
	re := regexp.MustCompile(`<<([^>]+)>>`)
	last := 0
	for _, loc := range re.FindAllStringIndex(answer, -1) {
		if loc[0] > last {
			parts = append(parts, tokenizer.MessagePart{Type: "text", Text: answer[last:loc[0]]})
		}
		inner := answer[loc[0]+2 : loc[1]-2]
		expr, result := inner, ""
		if idx := lastIndexByte(inner, '='); idx >= 0 {
			expr, result = inner[:idx], inner[idx+1:]
		}
		parts = append(parts, tokenizer.MessagePart{Type: "tool_call", Text: expr})
		parts = append(parts, tokenizer.MessagePart{Type: "tool_output", Text: result})
		last = loc[1]
	}
	if last < len(answer) {
		parts = append(parts, tokenizer.MessagePart{Type: "text", Text: answer[last:]})
	}
	return parts
}

func lastIndexByte(s string, b byte) int {
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] == b {
			return i
		}
	}
	return -1
}

func (g *GSM8K) EvalType() EvalType { return Generative }

func (g *GSM8K) NumExamples() int { return len(g.examples) }

func (g *GSM8K) GetExample(index int) *tokenizer.Conversation { return g.examples[index] }

func (g *GSM8K) Evaluate(conv *tokenizer.Conversation, response string) bool {
	// The ground truth answer is the last text part of the assistant message.
	assistant := conv.Messages[len(conv.Messages)-1]
	ref := assistant.Parts[len(assistant.Parts)-1].Text
	return ExtractAnswer(response) == ExtractAnswer(ref)
}

func (g *GSM8K) Reward(conv *tokenizer.Conversation, response string) float32 {
	return reward0(g, conv, response)
}

// SmolTalk is a general conversational dataset (SFT training data).
type SmolTalk struct {
	examples []*tokenizer.Conversation
}

// NewSmolTalkFromRows builds a SmolTalk task from raw rows (each has
// "messages", a list of {role, content}).
func NewSmolTalkFromRows(rows []SmolTalkRow) *SmolTalk {
	t := &SmolTalk{}
	for _, row := range rows {
		conv := &tokenizer.Conversation{}
		for _, m := range row.Messages {
			conv.Messages = append(conv.Messages, tokenizer.Message{Role: m.Role, Content: m.Content})
		}
		t.examples = append(t.examples, conv)
	}
	return t
}

// SmolTalkRow is a single SmolTalk example.
type SmolTalkRow struct {
	Messages []SmolTalkMessage
}

// SmolTalkMessage is a single message in a SmolTalk conversation.
type SmolTalkMessage struct {
	Role    string
	Content string
}

func (s *SmolTalk) EvalType() EvalType { return Generative }

func (s *SmolTalk) NumExamples() int { return len(s.examples) }

func (s *SmolTalk) GetExample(index int) *tokenizer.Conversation { return s.examples[index] }

func (s *SmolTalk) Evaluate(conv *tokenizer.Conversation, response string) bool { return true }

func (s *SmolTalk) Reward(conv *tokenizer.Conversation, response string) float32 { return 0 }
