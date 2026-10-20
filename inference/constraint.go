package inference

import (
	"fmt"
	"strings"

	"github.com/cookiengineer/gonano/tokenizer"
)

// Constraint is a byte-level incremental grammar used to mask token sampling.
// Allowed must not mutate the state; Advance commits a token that Allowed
// accepted.
type Constraint interface {
	Clone() Constraint
	Allowed(tokenBytes []byte) bool
	Advance(tokenBytes []byte)
	Finished() bool
}

type jsonFrame struct {
	object bool
}

type jsonState uint8

const (
	jsonValue jsonState = iota
	jsonObjectKeyOrEnd
	jsonObjectKey
	jsonColon
	jsonAfterObjectValue
	jsonArrayValueOrEnd
	jsonAfterArrayValue
	jsonString
	jsonStringEscape
	jsonStringUnicode
	jsonNumber
	jsonLiteral
	jsonDone
)

const (
	jsonNumberStart uint8 = iota
	jsonNumberMinus
	jsonNumberZero
	jsonNumberInteger
	jsonNumberFractionStart
	jsonNumberFraction
	jsonNumberExponentStart
	jsonNumberExponentSign
	jsonNumberExponent
)

// JSONConstraint accepts any complete JSON value. It is intentionally
// independent of the model tokenizer: the sampler converts each token id to
// its raw bytes and feeds them through the state machine.
type JSONConstraint struct {
	state         jsonState
	stack         []jsonFrame
	stringKey     bool
	numberStage   uint8
	literal       string
	literalIndex  int
	unicodeDigits int
}

// NewJSONConstraint returns a constraint for one complete JSON value.
func NewJSONConstraint() *JSONConstraint {
	return &JSONConstraint{state: jsonValue}
}

// Clone returns an independent copy of the constraint state.
func (constraint *JSONConstraint) Clone() Constraint {
	clone := *constraint
	clone.stack = append([]jsonFrame(nil), constraint.stack...)
	return &clone
}

// StateKey identifies the incremental parser state for mask caching.
func (constraint *JSONConstraint) StateKey() string {
	builder := strings.Builder{}
	fmt.Fprintf(&builder, "json:%d:%d:%t:%d:%s:%d:%d:", constraint.state, len(constraint.stack), constraint.stringKey, constraint.numberStage, constraint.literal, constraint.literalIndex, constraint.unicodeDigits)
	for _, frame := range constraint.stack {
		if frame.object {
			builder.WriteByte('o')
		} else {
			builder.WriteByte('a')
		}
	}
	return builder.String()
}

// Finished reports whether a complete top-level JSON value has been produced.
func (constraint *JSONConstraint) Finished() bool {
	if constraint.state == jsonDone {
		return true
	}
	if constraint.state == jsonNumber && len(constraint.stack) == 0 {
		switch constraint.numberStage {
		case jsonNumberZero, jsonNumberInteger, jsonNumberFraction, jsonNumberExponent:
			return true
		}
	}
	return false
}

// Allowed reports whether appending tokenBytes keeps the JSON prefix valid.
func (constraint *JSONConstraint) Allowed(tokenBytes []byte) bool {
	clone := constraint.Clone().(*JSONConstraint)
	return clone.advanceBytes(tokenBytes)
}

// Advance commits tokenBytes.
func (constraint *JSONConstraint) Advance(tokenBytes []byte) { constraint.advanceBytes(tokenBytes) }

func (constraint *JSONConstraint) advanceBytes(bytes []byte) bool {
	for _, value := range bytes {
		if !constraint.advanceByte(value) {
			return false
		}
	}
	return true
}

func isJSONWhitespace(value byte) bool {
	return value == ' ' || value == '\t' || value == '\n' || value == '\r'
}

func isJSONDigit(value byte) bool { return value >= '0' && value <= '9' }

func (constraint *JSONConstraint) finishValue() {
	if len(constraint.stack) == 0 {
		constraint.state = jsonDone
		return
	}
	top := constraint.stack[len(constraint.stack)-1]
	if top.object {
		constraint.state = jsonAfterObjectValue
	} else {
		constraint.state = jsonAfterArrayValue
	}
}

func (constraint *JSONConstraint) advanceByte(value byte) bool {
	for {
		switch constraint.state {
		case jsonValue:
			if isJSONWhitespace(value) {
				return true
			}
			switch value {
			case '{':
				constraint.stack = append(constraint.stack, jsonFrame{object: true})
				constraint.state = jsonObjectKeyOrEnd
			case '[':
				constraint.stack = append(constraint.stack, jsonFrame{object: false})
				constraint.state = jsonArrayValueOrEnd
			case '"':
				constraint.stringKey = false
				constraint.state = jsonString
			case 't':
				constraint.literal = "true"
				constraint.literalIndex = 1
				constraint.state = jsonLiteral
			case 'f':
				constraint.literal = "false"
				constraint.literalIndex = 1
				constraint.state = jsonLiteral
			case 'n':
				constraint.literal = "null"
				constraint.literalIndex = 1
				constraint.state = jsonLiteral
			case '-':
				constraint.numberStage = jsonNumberMinus
				constraint.state = jsonNumber
			case '0':
				constraint.numberStage = jsonNumberZero
				constraint.state = jsonNumber
			case '1', '2', '3', '4', '5', '6', '7', '8', '9':
				constraint.numberStage = jsonNumberInteger
				constraint.state = jsonNumber
			default:
				return false
			}
			return true
		case jsonObjectKeyOrEnd:
			if isJSONWhitespace(value) {
				return true
			}
			if value == '}' {
				constraint.stack = constraint.stack[:len(constraint.stack)-1]
				constraint.finishValue()
				return true
			}
			if value == '"' {
				constraint.stringKey = true
				constraint.state = jsonString
				return true
			}
			return false
		case jsonObjectKey:
			if isJSONWhitespace(value) {
				return true
			}
			if value == '"' {
				constraint.stringKey = true
				constraint.state = jsonString
				return true
			}
			return false
		case jsonColon:
			if isJSONWhitespace(value) {
				return true
			}
			if value == ':' {
				constraint.state = jsonValue
				return true
			}
			return false
		case jsonAfterObjectValue:
			if isJSONWhitespace(value) {
				return true
			}
			if value == ',' {
				constraint.state = jsonObjectKey
				return true
			}
			if value == '}' {
				constraint.stack = constraint.stack[:len(constraint.stack)-1]
				constraint.finishValue()
				return true
			}
			return false
		case jsonArrayValueOrEnd:
			if isJSONWhitespace(value) {
				return true
			}
			if value == ']' {
				constraint.stack = constraint.stack[:len(constraint.stack)-1]
				constraint.finishValue()
				return true
			}
			constraint.state = jsonValue
			continue
		case jsonAfterArrayValue:
			if isJSONWhitespace(value) {
				return true
			}
			if value == ',' {
				constraint.state = jsonValue
				return true
			}
			if value == ']' {
				constraint.stack = constraint.stack[:len(constraint.stack)-1]
				constraint.finishValue()
				return true
			}
			return false
		case jsonString:
			if value == '"' {
				if constraint.stringKey {
					constraint.state = jsonColon
				} else {
					constraint.finishValue()
				}
				return true
			}
			if value == '\\' {
				constraint.state = jsonStringEscape
				return true
			}
			if value < 0x20 {
				return false
			}
			return true
		case jsonStringEscape:
			switch value {
			case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
				constraint.state = jsonString
				return true
			case 'u':
				constraint.unicodeDigits = 0
				constraint.state = jsonStringUnicode
				return true
			default:
				return false
			}
		case jsonStringUnicode:
			if !((value >= '0' && value <= '9') || (value >= 'a' && value <= 'f') || (value >= 'A' && value <= 'F')) {
				return false
			}
			constraint.unicodeDigits++
			if constraint.unicodeDigits == 4 {
				constraint.state = jsonString
			}
			return true
		case jsonNumber:
			switch constraint.numberStage {
			case jsonNumberMinus:
				if value == '0' {
					constraint.numberStage = jsonNumberZero
					return true
				}
				if value >= '1' && value <= '9' {
					constraint.numberStage = jsonNumberInteger
					return true
				}
				return false
			case jsonNumberZero:
				if value == '.' {
					constraint.numberStage = jsonNumberFractionStart
					return true
				}
				if value == 'e' || value == 'E' {
					constraint.numberStage = jsonNumberExponentStart
					return true
				}
				constraint.finishValue()
				continue
			case jsonNumberInteger:
				if isJSONDigit(value) {
					return true
				}
				if value == '.' {
					constraint.numberStage = jsonNumberFractionStart
					return true
				}
				if value == 'e' || value == 'E' {
					constraint.numberStage = jsonNumberExponentStart
					return true
				}
				constraint.finishValue()
				continue
			case jsonNumberFractionStart:
				if !isJSONDigit(value) {
					return false
				}
				constraint.numberStage = jsonNumberFraction
				return true
			case jsonNumberFraction:
				if isJSONDigit(value) {
					return true
				}
				if value == 'e' || value == 'E' {
					constraint.numberStage = jsonNumberExponentStart
					return true
				}
				constraint.finishValue()
				continue
			case jsonNumberExponentStart:
				if value == '+' || value == '-' {
					constraint.numberStage = jsonNumberExponentSign
					return true
				}
				if !isJSONDigit(value) {
					return false
				}
				constraint.numberStage = jsonNumberExponent
				return true
			case jsonNumberExponentSign:
				if !isJSONDigit(value) {
					return false
				}
				constraint.numberStage = jsonNumberExponent
				return true
			case jsonNumberExponent:
				if isJSONDigit(value) {
					return true
				}
				constraint.finishValue()
				continue
			}
			return false
		case jsonLiteral:
			if constraint.literalIndex >= len(constraint.literal) {
				constraint.finishValue()
				continue
			}
			if value != constraint.literal[constraint.literalIndex] {
				return false
			}
			constraint.literalIndex++
			if constraint.literalIndex == len(constraint.literal) {
				constraint.finishValue()
			}
			return true
		case jsonDone:
			if isJSONWhitespace(value) {
				return true
			}
			return false
		}
		return false
	}
}

// ConstraintMask builds the allowed-token mask for the current constraint
// state. Special stop tokens are allowed only after the grammar is complete.
func ConstraintMask(tokenizerImpl *tokenizer.Tokenizer, constraint Constraint, vocab int, stopTokens []int) []bool {
	mask := make([]bool, vocab)
	if constraint == nil {
		for index := range mask {
			mask[index] = true
		}
		return mask
	}
	finished := constraint.Finished()
	stop := make(map[int]bool, len(stopTokens))
	for _, id := range stopTokens {
		stop[id] = true
	}
	for id := 0; id < vocab; id++ {
		if id >= tokenizerImpl.VocabSize() {
			continue
		}
		if tokenizerImpl.IsSpecial(id) {
			if finished && stop[id] {
				mask[id] = true
			}
			continue
		}
		mask[id] = constraint.Allowed(tokenizerImpl.DecodeSingleTokenBytes(id))
	}
	return mask
}
