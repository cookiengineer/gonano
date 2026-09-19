// Package tokenizer implements a byte-level BPE tokenizer in the style of
// GPT-4: train with our own BPE trainer, infer with a tiktoken-compatible
// split pattern. It also renders chat conversations into token id sequences.
package tokenizer

// SpecialTokens are the control tokens used to delimit documents, render
// conversations, and mark tool calls. They are assigned ids in this order,
// appended after the mergeable vocabulary.
//
// The tool-call vocabulary is language-agnostic: the assistant emits
// <|tool_start|> … <|tool_end|> to invoke a tool, and the runtime replies with
// <|tool_output_start|> … <|tool_output_end|>. The tool itself is executed by
// Go code (see package inference), not by any particular scripting language.
var SpecialTokens = []string{
	"<|bos|>",
	"<|user_start|>",
	"<|user_end|>",
	"<|assistant_start|>",
	"<|assistant_end|>",
	"<|tool_start|>",
	"<|tool_end|>",
	"<|tool_output_start|>",
	"<|tool_output_end|>",
}

// splitPattern is the GPT-4/tiktoken split pattern, described in prose because
// it is implemented by a hand-written scanner in splitter.go (Go's regexp/RE2
// rejects the possessive quantifiers and lookahead used in the original):
//
//	'(?i:[sdmt]|ll|ve|re)
//	|[^\r\n\p{L}\p{N}]?+\p{L}+
//	|\p{N}{1,2}
//	| ?[^\s\p{L}\p{N}]++[\r\n]*
//	|\s*[\r\n]
//	|\s+(?!\S)
//	|\s+
//
// The pattern deviates from GPT-4 by using \p{N}{1,2} instead of {1,3}, so
// numbers do not consume too many tokens at smaller vocab sizes.
const splitPatternDescription = "GPT-4 split pattern with \\p{N}{1,2}"
