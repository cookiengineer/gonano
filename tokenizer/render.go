package tokenizer

import (
	"fmt"
	"strings"
)

// ReasoningEffortInstruction returns the system-prompt instruction that selects
// a reasoning-effort level b in [1,100] (DeepSeek-V4.1 §5.1.4). Higher values
// request more thorough reasoning.
func ReasoningEffortInstruction(effort int) string {
	return fmt.Sprintf("Reasoning Effort: %d (range 1--100; higher values request more thorough reasoning)", effort)
}

// ThinkingInstruction returns the system-prompt instruction that enables or
// disables an explicit reasoning trace delimited by <|think_start|> and
// <|think_end|>. It conditions the model the same way the reasoning-effort
// instruction does, so a single checkpoint can serve both modes.
func ThinkingInstruction(enabled bool) string {
	if enabled {
		return "Thinking mode: enabled (reason inside <|think_start|>...<|think_end|> before the final answer)"
	}
	return "Thinking mode: disabled (answer directly, without a reasoning trace)"
}

// ThinkingInstruction is the method form for callers that only hold a Tokenizer.
func (tokenizer *Tokenizer) ThinkingInstruction(enabled bool) string {
	return ThinkingInstruction(enabled)
}

// thinkingEnabled reads the optional "thinking" entry from a conversation's
// Extra metadata. The second result reports whether the entry was present.
func thinkingEnabled(conv *Conversation) (bool, bool) {
	if conv == nil || conv.Extra == nil {
		return false, false
	}
	switch value := conv.Extra["thinking"].(type) {
	case bool:
		return value, true
	case int:
		return value != 0, true
	case int32:
		return value != 0, true
	case int64:
		return value != 0, true
	case float64:
		return value != 0, true
	case float32:
		return value != 0, true
	default:
		return false, false
	}
}

// ReasoningTokenCount returns the number of tokens inside the first
// <|think_start|>...<|think_end|> span of ids (zero when there is none). This
// is the ℓ quantity of the reasoning-length penalty (DeepSeek-V4.1 §5.1.4).
func (tokenizer *Tokenizer) ReasoningTokenCount(ids []int) int {
	start := tokenizer.EncodeSpecial("<|think_start|>")
	end := tokenizer.EncodeSpecial("<|think_end|>")
	count := 0
	inThinking := false
	for _, id := range ids {
		switch {
		case id == start:
			inThinking = true
		case id == end:
			inThinking = false
		case inThinking:
			count++
		}
	}
	return count
}

// ReasoningEffortInstruction is the method form of ReasoningEffortInstruction
// for callers that only hold a Tokenizer value.
func (tokenizer *Tokenizer) ReasoningEffortInstruction(effort int) string {
	return ReasoningEffortInstruction(effort)
}

// reasoningEffort reads the optional "effort" entry from a conversation's Extra
// metadata.
func reasoningEffort(conv *Conversation) (int, bool) {
	if conv == nil || conv.Extra == nil {
		return 0, false
	}
	switch value := conv.Extra["effort"].(type) {
	case int:
		return value, true
	case int32:
		return int(value), true
	case int64:
		return int(value), true
	case float64:
		return int(value), true
	case float32:
		return int(value), true
	default:
		return 0, false
	}
}

// MessagePart is one part of an assistant message: plain text, a tool call, or
// a tool output.
type MessagePart struct {
	Type string // "text", "tool_call", "tool_output"
	Text string
}

// Message is a single turn in a conversation.
type Message struct {
	Role     string        // "system", "user", or "assistant"
	Content  string        // simple string content
	Thinking string        // optional assistant reasoning trace (rendered as a think block)
	Parts    []MessagePart // assistant parts with tool calls (takes precedence over Content)
}

// Conversation is a chat document: a list of messages, plus optional metadata
// used by evaluation tasks (letters for multiple-choice, entry_point/test for
// HumanEval, subject, ...).
type Conversation struct {
	Messages []Message
	Extra    map[string]any
}

// RenderConversation tokenizes a conversation and returns the token ids and a
// loss mask of the same length. mask is 1 for assistant completion tokens (the
// tokens the model is trained to produce) and 0 for user prompts, BOS, special
// tokens, and tool outputs.
func (tokenizer *Tokenizer) RenderConversation(conv *Conversation, maxTokens int) ([]int, []int) {
	ids, mask := make([]int, 0), make([]int, 0)
	add := func(tokenIDs []int, maskVal int) {
		ids = append(ids, tokenIDs...)
		for index := 0; index < len(tokenIDs); index++ {
			mask = append(mask, maskVal)
		}
	}

	messages := append([]Message(nil), conv.Messages...)
	var instructions []string
	if effort, ok := reasoningEffort(conv); ok {
		instructions = append(instructions, ReasoningEffortInstruction(effort))
	}
	if enabled, ok := thinkingEnabled(conv); ok {
		instructions = append(instructions, ThinkingInstruction(enabled))
	}
	if len(instructions) > 0 {
		instruction := strings.Join(instructions, "\n")
		if len(messages) > 0 && messages[0].Role == "system" {
			messages[0] = Message{Role: "system", Content: instruction + "\n\n" + messages[0].Content}
		} else {
			messages = append([]Message{{Role: "system", Content: instruction}}, messages...)
		}
	}
	if len(messages) == 0 {
		return ids, mask
	}
	if len(messages) >= 2 && messages[0].Role == "system" {
		merged := messages[1]
		merged.Content = messages[0].Content + "\n\n" + messages[1].Content
		messages = append([]Message{merged}, messages[2:]...)
	}

	bos := tokenizer.BOSTokenID()
	userStart, userEnd := tokenizer.EncodeSpecial("<|user_start|>"), tokenizer.EncodeSpecial("<|user_end|>")
	assistantStart, assistantEnd := tokenizer.EncodeSpecial("<|assistant_start|>"), tokenizer.EncodeSpecial("<|assistant_end|>")
	toolStart, toolEnd := tokenizer.EncodeSpecial("<|tool_start|>"), tokenizer.EncodeSpecial("<|tool_end|>")
	toolOutputStart, toolOutputEnd := tokenizer.EncodeSpecial("<|tool_output_start|>"), tokenizer.EncodeSpecial("<|tool_output_end|>")
	thinkStart, thinkEnd := tokenizer.EncodeSpecial("<|think_start|>"), tokenizer.EncodeSpecial("<|think_end|>")
	addThinking := func(text string) {
		add([]int{thinkStart}, 1)
		if text != "" {
			add(tokenizer.Encode(text), 1)
		}
		add([]int{thinkEnd}, 1)
	}

	add([]int{bos}, 0)
	for messageIndex, message := range messages {
		if message.Role == "user" {
			add([]int{userStart}, 0)
			add(tokenizer.Encode(message.Content), 0)
			add([]int{userEnd}, 0)
			continue
		}
		// assistant
		add([]int{assistantStart}, 0)
		if len(message.Parts) == 0 {
			if message.Thinking != "" {
				addThinking(message.Thinking)
			}
			add(tokenizer.Encode(message.Content), 1)
		} else {
			for _, part := range message.Parts {
				switch part.Type {
				case "thinking":
					addThinking(part.Text)
				case "text":
					add(tokenizer.Encode(part.Text), 1)
				case "tool_call":
					add([]int{toolStart}, 1)
					add(tokenizer.Encode(part.Text), 1)
					add([]int{toolEnd}, 1)
				case "tool_output":
					add([]int{toolOutputStart}, 0)
					add(tokenizer.Encode(part.Text), 0)
					add([]int{toolOutputEnd}, 0)
				}
			}
		}
		add([]int{assistantEnd}, 1)
		_ = messageIndex
	}

	if len(ids) > maxTokens {
		ids = ids[:maxTokens]
		mask = mask[:maxTokens]
	}
	return ids, mask
}

// RenderForCompletion tokenizes a conversation primed for the assistant to
// complete it: the last assistant message is removed and the assistant-start
// token is appended. When the conversation requests thinking, the think-start
// token is primed too, so the model begins its reasoning trace. Used during RL
// rollout generation.
func (tokenizer *Tokenizer) RenderForCompletion(conv *Conversation) []int {
	messages := append([]Message(nil), conv.Messages...)
	messages = messages[:len(messages)-1]
	ids, _ := tokenizer.RenderConversation(&Conversation{Messages: messages, Extra: conv.Extra}, 1<<30)
	ids = append(ids, tokenizer.EncodeSpecial("<|assistant_start|>"))
	if enabled, ok := thinkingEnabled(conv); ok && enabled {
		ids = append(ids, tokenizer.EncodeSpecial("<|think_start|>"))
	}
	return ids
}
