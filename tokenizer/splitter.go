package tokenizer

import "unicode"

// SplitPieces splits text into GPT-4-style pieces by matching the split
// pattern (see special.go). It scans Unicode code points left to right and,
// at each position, applies the ordered alternatives of the pattern. This is a
// faithful hand-written implementation of the regex, since Go's RE2 engine
// rejects the possessive quantifiers and lookahead of the original.
func SplitPieces(text string) []string {
	runes := []rune(text)
	runeCount := len(runes)
	pieces := make([]string, 0, runeCount/4+1)
	index := 0
	for index < runeCount {
		if start, end := matchContraction(runes, index); start >= 0 {
			pieces = append(pieces, string(runes[index:end]))
			index = end
			continue
		}
		if end := matchLetter(runes, index); end > index {
			pieces = append(pieces, string(runes[index:end]))
			index = end
			continue
		}
		if end := matchNumber(runes, index); end > index {
			pieces = append(pieces, string(runes[index:end]))
			index = end
			continue
		}
		if end := matchPunctuation(runes, index); end > index {
			pieces = append(pieces, string(runes[index:end]))
			index = end
			continue
		}
		if end := matchNewline(runes, index); end > index {
			pieces = append(pieces, string(runes[index:end]))
			index = end
			continue
		}
		if end := matchWhitespace(runes, index); end > index {
			pieces = append(pieces, string(runes[index:end]))
			index = end
			continue
		}
		// Unreachable in practice, but never loop forever.
		pieces = append(pieces, string(runes[index]))
		index++
	}
	return pieces
}

// isNewline reports whether character is \r or \n (the chars explicitly
// excluded from the letter-alternative's optional prefix).
func isNewline(character rune) bool { return character == '\r' || character == '\n' }

// matchContraction matches ' followed by s/d/m/t/ll/ve/re (case-insensitive).
// Returns the end index, or -1 if no match.
func matchContraction(runes []rune, index int) (start, end int) {
	if runes[index] != '\'' {
		return -1, -1
	}
	rest := runes[index+1:]
	lower := func(character rune) rune { return unicode.ToLower(character) }
	// Two-letter contractions first.
	if len(rest) >= 2 {
		twoLetter := string([]rune{lower(rest[0]), lower(rest[1])})
		switch twoLetter {
		case "ll", "ve", "re":
			return index, index + 3
		}
	}
	if len(rest) >= 1 {
		switch lower(rest[0]) {
		case 's', 'd', 'm', 't':
			return index, index + 2
		}
	}
	return -1, -1
}

// matchLetter matches [^\r\n\p{L}\p{N}]?+ \p{L}+.
func matchLetter(runes []rune, index int) int {
	cursor := index
	if !isNewline(runes[cursor]) && !unicode.IsLetter(runes[cursor]) && !unicode.IsNumber(runes[cursor]) {
		cursor++
		if cursor >= len(runes) || !unicode.IsLetter(runes[cursor]) {
			return index
		}
	}
	if cursor >= len(runes) || !unicode.IsLetter(runes[cursor]) {
		return index
	}
	for cursor < len(runes) && unicode.IsLetter(runes[cursor]) {
		cursor++
	}
	return cursor
}

// matchNumber matches \p{N}{1,2}.
func matchNumber(runes []rune, index int) int {
	if !unicode.IsNumber(runes[index]) {
		return index
	}
	cursor := index + 1
	if cursor < len(runes) && unicode.IsNumber(runes[cursor]) {
		cursor++
	}
	return cursor
}

// matchPunctuation matches " ?" [^\s\p{L}\p{N}]++ [\r\n]*.
func matchPunctuation(runes []rune, index int) int {
	cursor := index
	if runes[cursor] == ' ' {
		cursor++
	}
	// One or more punctuation/symbol chars (not whitespace, letter, number).
	start := cursor
	for cursor < len(runes) && !unicode.IsSpace(runes[cursor]) && !unicode.IsLetter(runes[cursor]) && !unicode.IsNumber(runes[cursor]) {
		cursor++
	}
	if cursor == start {
		return index
	}
	for cursor < len(runes) && isNewline(runes[cursor]) {
		cursor++
	}
	return cursor
}

// matchNewline matches \s* [\r\n]: a maximal whitespace run ending in a
// newline. The regex backtracks \s* to leave the last \r or \n for [\r\n], so
// a whitespace run with a newline consumes everything up to and including its
// last newline.
func matchNewline(runes []rune, index int) int {
	cursor := index
	lastNewline := -1
	for cursor < len(runes) && unicode.IsSpace(runes[cursor]) {
		if isNewline(runes[cursor]) {
			lastNewline = cursor
		}
		cursor++
	}
	if lastNewline >= 0 {
		return lastNewline + 1
	}
	return index
}

// matchWhitespace matches \s+ (and \s+(?!\S)).
func matchWhitespace(runes []rune, index int) int {
	if !unicode.IsSpace(runes[index]) {
		return index
	}
	cursor := index
	for cursor < len(runes) && unicode.IsSpace(runes[cursor]) {
		cursor++
	}
	return cursor
}
