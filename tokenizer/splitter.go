package tokenizer

import "unicode"

// SplitPieces splits text into GPT-4-style pieces by matching the split
// pattern (see special.go). It scans Unicode code points left to right and,
// at each position, applies the ordered alternatives of the pattern. This is a
// faithful hand-written implementation of the regex, since Go's RE2 engine
// rejects the possessive quantifiers and lookahead of the original.
func SplitPieces(text string) []string {
	runes := []rune(text)
	n := len(runes)
	pieces := make([]string, 0, n/4+1)
	i := 0
	for i < n {
		if p, next := matchContraction(runes, i); p >= 0 {
			pieces = append(pieces, string(runes[i:next]))
			i = next
			continue
		}
		if next := matchLetter(runes, i); next > i {
			pieces = append(pieces, string(runes[i:next]))
			i = next
			continue
		}
		if next := matchNumber(runes, i); next > i {
			pieces = append(pieces, string(runes[i:next]))
			i = next
			continue
		}
		if next := matchPunctuation(runes, i); next > i {
			pieces = append(pieces, string(runes[i:next]))
			i = next
			continue
		}
		if next := matchNewline(runes, i); next > i {
			pieces = append(pieces, string(runes[i:next]))
			i = next
			continue
		}
		if next := matchWhitespace(runes, i); next > i {
			pieces = append(pieces, string(runes[i:next]))
			i = next
			continue
		}
		// Unreachable in practice, but never loop forever.
		pieces = append(pieces, string(runes[i]))
		i++
	}
	return pieces
}

// isNewline reports whether r is \r or \n (the chars explicitly excluded from
// the letter-alternative's optional prefix).
func isNewline(r rune) bool { return r == '\r' || r == '\n' }

// matchContraction matches ' followed by s/d/m/t/ll/ve/re (case-insensitive).
// Returns the end index, or -1 if no match.
func matchContraction(runes []rune, i int) (start, end int) {
	if runes[i] != '\'' {
		return -1, -1
	}
	rest := runes[i+1:]
	lower := func(r rune) rune { return unicode.ToLower(r) }
	// Two-letter contractions first.
	if len(rest) >= 2 {
		pair := string([]rune{lower(rest[0]), lower(rest[1])})
		switch pair {
		case "ll", "ve", "re":
			return i, i + 3
		}
	}
	if len(rest) >= 1 {
		switch lower(rest[0]) {
		case 's', 'd', 'm', 't':
			return i, i + 2
		}
	}
	return -1, -1
}

// matchLetter matches [^\r\n\p{L}\p{N}]?+ \p{L}+.
func matchLetter(runes []rune, i int) int {
	j := i
	if !isNewline(runes[j]) && !unicode.IsLetter(runes[j]) && !unicode.IsNumber(runes[j]) {
		j++
		if j >= len(runes) || !unicode.IsLetter(runes[j]) {
			return i
		}
	}
	if j >= len(runes) || !unicode.IsLetter(runes[j]) {
		return i
	}
	for j < len(runes) && unicode.IsLetter(runes[j]) {
		j++
	}
	return j
}

// matchNumber matches \p{N}{1,2}.
func matchNumber(runes []rune, i int) int {
	if !unicode.IsNumber(runes[i]) {
		return i
	}
	j := i + 1
	if j < len(runes) && unicode.IsNumber(runes[j]) {
		j++
	}
	return j
}

// matchPunctuation matches " ?" [^\s\p{L}\p{N}]++ [\r\n]*.
func matchPunctuation(runes []rune, i int) int {
	j := i
	if runes[j] == ' ' {
		j++
	}
	// One or more punctuation/symbol chars (not whitespace, letter, number).
	start := j
	for j < len(runes) && !unicode.IsSpace(runes[j]) && !unicode.IsLetter(runes[j]) && !unicode.IsNumber(runes[j]) {
		j++
	}
	if j == start {
		return i
	}
	for j < len(runes) && isNewline(runes[j]) {
		j++
	}
	return j
}

// matchNewline matches \s* [\r\n]: a maximal whitespace run ending in a
// newline. The regex backtracks \s* to leave the last \r or \n for [\r\n], so
// a whitespace run with a newline consumes everything up to and including its
// last newline.
func matchNewline(runes []rune, i int) int {
	j := i
	lastNewline := -1
	for j < len(runes) && unicode.IsSpace(runes[j]) {
		if isNewline(runes[j]) {
			lastNewline = j
		}
		j++
	}
	if lastNewline >= 0 {
		return lastNewline + 1
	}
	return i
}

// matchWhitespace matches \s+ (and \s+(?!\S)).
func matchWhitespace(runes []rune, i int) int {
	if !unicode.IsSpace(runes[i]) {
		return i
	}
	j := i
	for j < len(runes) && unicode.IsSpace(runes[j]) {
		j++
	}
	return j
}
