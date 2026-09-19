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
	task := &MMLU{}
	for _, row := range rows {
		if len(row.Choices) != 4 {
			continue
		}
		user := RenderMC(row.Question, mmluLetters, row.Choices)
		task.examples = append(task.examples, &tokenizer.Conversation{
			Messages: []tokenizer.Message{
				{Role: "user", Content: user},
				{Role: "assistant", Content: mmluLetters[row.Answer]},
			},
			Extra: map[string]any{"letters": mmluLetters},
		})
	}
	return task
}

// MMLURow is a single raw MMLU example.
type MMLURow struct {
	Question string
	Choices  []string
	Answer   int
}

func (mmlu *MMLU) EvalType() EvalType { return Categorical }

func (mmlu *MMLU) NumExamples() int { return len(mmlu.examples) }

func (mmlu *MMLU) GetExample(index int) *tokenizer.Conversation { return mmlu.examples[index] }

func (mmlu *MMLU) Evaluate(conversation *tokenizer.Conversation, response string) bool {
	return response == conversation.Messages[len(conversation.Messages)-1].Content
}

func (mmlu *MMLU) Reward(conversation *tokenizer.Conversation, response string) float32 {
	return evaluateReward(mmlu, conversation, response)
}

// ARC is the AI2 Reasoning Challenge dataset (multiple choice).
type ARC struct {
	examples []*tokenizer.Conversation
}

// ARCRow is a single raw ARC example.
type ARCRow struct {
	Question string
	Choices  []string
	Letters  []string
	Answer   string // one of the letters
}

// NewARCFromRows builds an ARC task from raw rows.
func NewARCFromRows(rows []ARCRow) *ARC {
	task := &ARC{}
	for _, row := range rows {
		user := RenderMC(row.Question, row.Letters, row.Choices)
		task.examples = append(task.examples, &tokenizer.Conversation{
			Messages: []tokenizer.Message{
				{Role: "user", Content: user},
				{Role: "assistant", Content: row.Answer},
			},
			Extra: map[string]any{"letters": row.Letters},
		})
	}
	return task
}

func (arc *ARC) EvalType() EvalType { return Categorical }

func (arc *ARC) NumExamples() int { return len(arc.examples) }

func (arc *ARC) GetExample(index int) *tokenizer.Conversation { return arc.examples[index] }

func (arc *ARC) Evaluate(conversation *tokenizer.Conversation, response string) bool {
	return response == conversation.Messages[len(conversation.Messages)-1].Content
}

func (arc *ARC) Reward(conversation *tokenizer.Conversation, response string) float32 {
	return evaluateReward(arc, conversation, response)
}

// gsmRE extracts the numerical answer after the "####" marker.
var gsmRE = regexp.MustCompile(`####\s*([0-9.,-]+)`)

// ExtractAnswer returns the answer after "####" in a GSM8K completion, or an
// empty string if none is found.
func ExtractAnswer(completion string) string {
	matches := gsmRE.FindStringSubmatch(completion)
	if matches == nil {
		return ""
	}
	return matches[1]
}

// GSM8K is the grade-school math dataset. Assistant messages contain tool
// calls (the <<expr=result>> calculator syntax).
type GSM8K struct {
	examples []*tokenizer.Conversation
}

// NewGSM8KFromRows builds a GSM8K task from raw rows (question, answer).
func NewGSM8KFromRows(rows []GSM8KRow) *GSM8K {
	task := &GSM8K{}
	for _, row := range rows {
		parts := splitGSM8KAnswer(row.Answer)
		task.examples = append(task.examples, &tokenizer.Conversation{
			Messages: []tokenizer.Message{
				{Role: "user", Content: row.Question},
				{Role: "assistant", Parts: parts},
			},
		})
	}
	return task
}

// GSM8KRow is a single raw GSM8K example.
type GSM8KRow struct {
	Question string
	Answer   string
}

// splitGSM8KAnswer parses the <<expr=result>> tool calls in a GSM8K answer.
func splitGSM8KAnswer(answer string) []tokenizer.MessagePart {
	var parts []tokenizer.MessagePart
	pattern := regexp.MustCompile(`<<([^>]+)>>`)
	lastIndex := 0
	for _, location := range pattern.FindAllStringIndex(answer, -1) {
		if location[0] > lastIndex {
			parts = append(parts, tokenizer.MessagePart{Type: "text", Text: answer[lastIndex:location[0]]})
		}
		inner := answer[location[0]+2 : location[1]-2]
		expression, result := inner, ""
		if separatorIndex := findLastIndexByte(inner, '='); separatorIndex >= 0 {
			expression, result = inner[:separatorIndex], inner[separatorIndex+1:]
		}
		parts = append(parts, tokenizer.MessagePart{Type: "tool_call", Text: expression})
		parts = append(parts, tokenizer.MessagePart{Type: "tool_output", Text: result})
		lastIndex = location[1]
	}
	if lastIndex < len(answer) {
		parts = append(parts, tokenizer.MessagePart{Type: "text", Text: answer[lastIndex:]})
	}
	return parts
}

func findLastIndexByte(text string, target byte) int {
	for index := len(text) - 1; index >= 0; index-- {
		if text[index] == target {
			return index
		}
	}
	return -1
}

func (gsm8k *GSM8K) EvalType() EvalType { return Generative }

func (gsm8k *GSM8K) NumExamples() int { return len(gsm8k.examples) }

func (gsm8k *GSM8K) GetExample(index int) *tokenizer.Conversation { return gsm8k.examples[index] }

func (gsm8k *GSM8K) Evaluate(conversation *tokenizer.Conversation, response string) bool {
	// The ground truth answer is the last text part of the assistant message.
	assistant := conversation.Messages[len(conversation.Messages)-1]
	reference := assistant.Parts[len(assistant.Parts)-1].Text
	return ExtractAnswer(response) == ExtractAnswer(reference)
}

func (gsm8k *GSM8K) Reward(conversation *tokenizer.Conversation, response string) float32 {
	return evaluateReward(gsm8k, conversation, response)
}

// SmolTalk is a general conversational dataset (SFT training data).
type SmolTalk struct {
	examples []*tokenizer.Conversation
}

// NewSmolTalkFromRows builds a SmolTalk task from raw rows (each has
// "messages", a list of {role, content}).
func NewSmolTalkFromRows(rows []SmolTalkRow) *SmolTalk {
	task := &SmolTalk{}
	for _, row := range rows {
		conversation := &tokenizer.Conversation{}
		for _, message := range row.Messages {
			conversation.Messages = append(conversation.Messages, tokenizer.Message{Role: message.Role, Content: message.Content})
		}
		task.examples = append(task.examples, conversation)
	}
	return task
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

func (smoltalk *SmolTalk) EvalType() EvalType { return Generative }

func (smoltalk *SmolTalk) NumExamples() int { return len(smoltalk.examples) }

func (smoltalk *SmolTalk) GetExample(index int) *tokenizer.Conversation {
	return smoltalk.examples[index]
}

func (smoltalk *SmolTalk) Evaluate(conversation *tokenizer.Conversation, response string) bool {
	return true
}

func (smoltalk *SmolTalk) Reward(conversation *tokenizer.Conversation, response string) float32 {
	return 0
}
