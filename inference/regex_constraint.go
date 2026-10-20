package inference

import (
	"fmt"
	"regexp/syntax"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

// RegexConstraint constrains generation to a complete regex match. It uses the
// standard-library regexp syntax compiler as an incremental NFA, so Allowed is
// cheap and side-effect free. Empty-width assertions (^, $, \b) are treated as
// transparent because generation is a single contiguous stream.
type RegexConstraint struct {
	program *syntax.Prog
	states  map[int]bool
	pending []byte
}

// NewRegexConstraint compiles pattern with Perl syntax.
func NewRegexConstraint(pattern string) (*RegexConstraint, error) {
	expression, err := syntax.Parse(pattern, syntax.Perl)
	if err != nil {
		return nil, err
	}
	program, err := syntax.Compile(expression.Simplify())
	if err != nil {
		return nil, err
	}
	constraint := &RegexConstraint{program: program, states: make(map[int]bool)}
	constraint.addClosure(constraint.states, program.Start)
	return constraint, nil
}

// Clone returns an independent copy of the constraint state.
func (constraint *RegexConstraint) Clone() Constraint {
	clone := &RegexConstraint{
		program: constraint.program,
		states:  make(map[int]bool, len(constraint.states)),
		pending: append([]byte(nil), constraint.pending...),
	}
	for state := range constraint.states {
		clone.states[state] = true
	}
	return clone
}

// StateKey identifies the NFA state set and pending bytes for mask caching.
func (constraint *RegexConstraint) StateKey() string {
	states := make([]int, 0, len(constraint.states))
	for state := range constraint.states {
		states = append(states, state)
	}
	sort.Ints(states)
	builder := strings.Builder{}
	fmt.Fprintf(&builder, "regex:%p:", constraint.program)
	for _, state := range states {
		builder.WriteString(strconv.Itoa(state))
		builder.WriteByte(',')
	}
	builder.WriteByte(':')
	for _, value := range constraint.pending {
		builder.WriteString(strconv.Itoa(int(value)))
		builder.WriteByte(',')
	}
	return builder.String()
}

// Finished reports whether the input is a complete match.
func (constraint *RegexConstraint) Finished() bool {
	if len(constraint.pending) != 0 {
		return false
	}
	for state := range constraint.states {
		if constraint.program.Inst[state].Op == syntax.InstMatch {
			return true
		}
	}
	return false
}

// Allowed reports whether appending tokenBytes keeps a full match reachable.
func (constraint *RegexConstraint) Allowed(tokenBytes []byte) bool {
	clone := constraint.Clone().(*RegexConstraint)
	return clone.advanceBytes(tokenBytes)
}

// Advance commits tokenBytes.
func (constraint *RegexConstraint) Advance(tokenBytes []byte) { constraint.advanceBytes(tokenBytes) }

func (constraint *RegexConstraint) advanceBytes(tokenBytes []byte) bool {
	data := append(append([]byte(nil), constraint.pending...), tokenBytes...)
	constraint.pending = nil
	for len(data) > 0 {
		if !utf8.FullRune(data) {
			// A valid multi-byte sequence may be completed by a later token.
			if !validRunePrefix(data) {
				return false
			}
			constraint.pending = append([]byte(nil), data...)
			return true
		}
		value, size := utf8.DecodeRune(data)
		if value == utf8.RuneError && size == 1 {
			return false
		}
		if !constraint.step(value) {
			return false
		}
		data = data[size:]
	}
	return true
}

func (constraint *RegexConstraint) step(value rune) bool {
	next := make(map[int]bool)
	for state := range constraint.states {
		instruction := &constraint.program.Inst[state]
		switch instruction.Op {
		case syntax.InstRune, syntax.InstRune1, syntax.InstRuneAny, syntax.InstRuneAnyNotNL:
			if instruction.MatchRune(value) {
				constraint.addClosure(next, int(instruction.Out))
			}
		}
	}
	constraint.states = next
	return len(next) > 0
}

func (constraint *RegexConstraint) addClosure(states map[int]bool, state int) {
	if state < 0 || state >= len(constraint.program.Inst) || states[state] {
		return
	}
	states[state] = true
	instruction := &constraint.program.Inst[state]
	switch instruction.Op {
	case syntax.InstAlt, syntax.InstAltMatch:
		constraint.addClosure(states, int(instruction.Out))
		constraint.addClosure(states, int(instruction.Arg))
	case syntax.InstCapture, syntax.InstEmptyWidth, syntax.InstNop:
		constraint.addClosure(states, int(instruction.Out))
	}
}

func validRunePrefix(data []byte) bool {
	if len(data) == 0 {
		return true
	}
	if !utf8.RuneStart(data[0]) {
		return false
	}
	expected := 0
	switch {
	case data[0]&0xE0 == 0xC0:
		expected = 2
	case data[0]&0xF0 == 0xE0:
		expected = 3
	case data[0]&0xF8 == 0xF0:
		expected = 4
	default:
		return false
	}
	if len(data) >= expected {
		return false
	}
	for _, value := range data[1:] {
		if value&0xC0 != 0x80 {
			return false
		}
	}
	return true
}
