package inference

import (
	"fmt"
	"regexp"
	"strings"
	"unicode/utf8"
)

// EBNFConstraint compiles a non-recursive EBNF grammar to a byte-level regex
// constraint. Supported syntax: quoted terminals, identifiers, sequences,
// alternation (|), grouping, optional [ ... ], repetition { ... }, postfix
// ?, * and +, comma-separated sequences, and single-rune ranges a..z.
// Recursive rules are rejected because the constrained decoder uses a finite
// automaton.
type EBNFConstraint struct {
	*RegexConstraint
}

// NewEBNFConstraint parses grammar and builds a full-match constraint. The
// start rule is named "start" when present, otherwise the first rule.
func NewEBNFConstraint(grammar string) (*EBNFConstraint, error) {
	parser := &ebnfParser{lexer: &ebnfLexer{input: []rune(grammar)}}
	rules, first, err := parser.parseGrammar()
	if err != nil {
		return nil, err
	}
	start := "start"
	if _, ok := rules[start]; !ok {
		start = first
	}
	if start == "" {
		return nil, fmt.Errorf("ebnf: no rules")
	}
	pattern, err := expandEBNFRule(rules, start, map[string]bool{})
	if err != nil {
		return nil, err
	}
	regex, err := NewRegexConstraint(pattern)
	if err != nil {
		return nil, err
	}
	return &EBNFConstraint{RegexConstraint: regex}, nil
}

func expandEBNFRule(rules map[string]*ebnfNode, name string, visiting map[string]bool) (string, error) {
	node, ok := rules[name]
	if !ok {
		return "", fmt.Errorf("ebnf: undefined rule %q", name)
	}
	if visiting[name] {
		return "", fmt.Errorf("ebnf: recursive rule %q is not supported", name)
	}
	visiting[name] = true
	pattern, err := node.regex(func(reference string) (string, error) {
		return expandEBNFRule(rules, reference, visiting)
	})
	delete(visiting, name)
	return pattern, err
}

type ebnfKind uint8

const (
	ebnfLiteral ebnfKind = iota
	ebnfReference
	ebnfSequence
	ebnfAlternation
	ebnfOptional
	ebnfStar
	ebnfPlus
	ebnfRange
)

type ebnfNode struct {
	kind     ebnfKind
	text     string
	children []*ebnfNode
	low      rune
	high     rune
}

func (node *ebnfNode) regex(expand func(string) (string, error)) (string, error) {
	switch node.kind {
	case ebnfLiteral:
		return regexp.QuoteMeta(node.text), nil
	case ebnfReference:
		return expand(node.text)
	case ebnfRange:
		return fmt.Sprintf("[\\x{%x}-\\x{%x}]", node.low, node.high), nil
	case ebnfSequence:
		var builder strings.Builder
		for _, child := range node.children {
			pattern, err := child.regex(expand)
			if err != nil {
				return "", err
			}
			builder.WriteString(pattern)
		}
		return builder.String(), nil
	case ebnfAlternation:
		parts := make([]string, 0, len(node.children))
		for _, child := range node.children {
			pattern, err := child.regex(expand)
			if err != nil {
				return "", err
			}
			parts = append(parts, pattern)
		}
		return "(" + strings.Join(parts, "|") + ")", nil
	case ebnfOptional, ebnfStar, ebnfPlus:
		pattern, err := node.children[0].regex(expand)
		if err != nil {
			return "", err
		}
		switch node.kind {
		case ebnfOptional:
			return "(" + pattern + ")?", nil
		case ebnfStar:
			return "(" + pattern + ")*", nil
		default:
			return "(" + pattern + ")+", nil
		}
	}
	return "", fmt.Errorf("ebnf: unsupported node")
}

type ebnfTokenKind uint8

const (
	ebnfEOF ebnfTokenKind = iota
	ebnfIdent
	ebnfString
	ebnfAssign
	ebnfSemicolon
	ebnfPipe
	ebnfLParen
	ebnfRParen
	ebnfLBracket
	ebnfRBracket
	ebnfLBrace
	ebnfRBrace
	ebnfComma
	ebnfRangeToken
	ebnfOptionalToken
	ebnfStarToken
	ebnfPlusToken
)

type ebnfToken struct {
	kind ebnfTokenKind
	text string
}

type ebnfLexer struct {
	input []rune
	pos   int
}

func (lexer *ebnfLexer) next() (ebnfToken, error) {
	for lexer.pos < len(lexer.input) {
		value := lexer.input[lexer.pos]
		if value == ' ' || value == '\t' || value == '\n' || value == '\r' {
			lexer.pos++
			continue
		}
		if value == '(' && lexer.pos+1 < len(lexer.input) && lexer.input[lexer.pos+1] == '*' {
			lexer.pos += 2
			for lexer.pos+1 < len(lexer.input) && !(lexer.input[lexer.pos] == '*' && lexer.input[lexer.pos+1] == ')') {
				lexer.pos++
			}
			if lexer.pos+1 >= len(lexer.input) {
				return ebnfToken{}, fmt.Errorf("ebnf: unterminated comment")
			}
			lexer.pos += 2
			continue
		}
		break
	}
	if lexer.pos >= len(lexer.input) {
		return ebnfToken{kind: ebnfEOF}, nil
	}
	value := lexer.input[lexer.pos]
	if value == '"' || value == '\'' {
		quote := value
		lexer.pos++
		var builder strings.Builder
		for lexer.pos < len(lexer.input) {
			current := lexer.input[lexer.pos]
			lexer.pos++
			if current == quote {
				return ebnfToken{kind: ebnfString, text: builder.String()}, nil
			}
			if current != '\\' {
				builder.WriteRune(current)
				continue
			}
			if lexer.pos >= len(lexer.input) {
				return ebnfToken{}, fmt.Errorf("ebnf: trailing escape")
			}
			escaped := lexer.input[lexer.pos]
			lexer.pos++
			switch escaped {
			case 'n':
				builder.WriteRune('\n')
			case 'r':
				builder.WriteRune('\r')
			case 't':
				builder.WriteRune('\t')
			case '\\', '"', '\'':
				builder.WriteRune(escaped)
			case 'u':
				if lexer.pos+4 > len(lexer.input) {
					return ebnfToken{}, fmt.Errorf("ebnf: short unicode escape")
				}
				var codepoint rune
				for index := 0; index < 4; index++ {
					digit := lexer.input[lexer.pos+index]
					codepoint <<= 4
					switch {
					case digit >= '0' && digit <= '9':
						codepoint += digit - '0'
					case digit >= 'a' && digit <= 'f':
						codepoint += digit - 'a' + 10
					case digit >= 'A' && digit <= 'F':
						codepoint += digit - 'A' + 10
					default:
						return ebnfToken{}, fmt.Errorf("ebnf: invalid unicode escape")
					}
				}
				lexer.pos += 4
				builder.WriteRune(codepoint)
			default:
				builder.WriteRune(escaped)
			}
		}
		return ebnfToken{}, fmt.Errorf("ebnf: unterminated string")
	}
	if value == ':' && lexer.pos+2 < len(lexer.input) && lexer.input[lexer.pos+1] == ':' && lexer.input[lexer.pos+2] == '=' {
		lexer.pos += 3
		return ebnfToken{kind: ebnfAssign}, nil
	}
	single := map[rune]ebnfTokenKind{
		'=': ebnfAssign, ';': ebnfSemicolon, '|': ebnfPipe, '(': ebnfLParen, ')': ebnfRParen,
		'[': ebnfLBracket, ']': ebnfRBracket, '{': ebnfLBrace, '}': ebnfRBrace, ',': ebnfComma,
		'?': ebnfOptionalToken, '*': ebnfStarToken, '+': ebnfPlusToken,
	}
	if kind, ok := single[value]; ok {
		lexer.pos++
		return ebnfToken{kind: kind}, nil
	}
	if value == '.' && lexer.pos+1 < len(lexer.input) && lexer.input[lexer.pos+1] == '.' {
		lexer.pos += 2
		return ebnfToken{kind: ebnfRangeToken}, nil
	}
	if isEBNFIdentStart(value) {
		start := lexer.pos
		for lexer.pos < len(lexer.input) && isEBNFIdentPart(lexer.input[lexer.pos]) {
			lexer.pos++
		}
		return ebnfToken{kind: ebnfIdent, text: string(lexer.input[start:lexer.pos])}, nil
	}
	return ebnfToken{}, fmt.Errorf("ebnf: unexpected character %q", value)
}

func isEBNFIdentStart(value rune) bool {
	return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') || value == '_'
}

func isEBNFIdentPart(value rune) bool {
	return isEBNFIdentStart(value) || (value >= '0' && value <= '9') || value == '-'
}

type ebnfParser struct {
	lexer   *ebnfLexer
	current ebnfToken
	has     bool
}

func (parser *ebnfParser) peek() (ebnfToken, error) {
	if parser.has {
		return parser.current, nil
	}
	token, err := parser.lexer.next()
	if err != nil {
		return ebnfToken{}, err
	}
	parser.current = token
	parser.has = true
	return token, nil
}

func (parser *ebnfParser) take() (ebnfToken, error) {
	token, err := parser.peek()
	if err != nil {
		return ebnfToken{}, err
	}
	parser.has = false
	return token, nil
}

func (parser *ebnfParser) parseGrammar() (map[string]*ebnfNode, string, error) {
	rules := make(map[string]*ebnfNode)
	first := ""
	for {
		token, err := parser.peek()
		if err != nil {
			return nil, "", err
		}
		if token.kind == ebnfEOF {
			break
		}
		if token.kind != ebnfIdent {
			return nil, "", fmt.Errorf("ebnf: expected rule name")
		}
		name := token.text
		if _, exists := rules[name]; exists {
			return nil, "", fmt.Errorf("ebnf: duplicate rule %q", name)
		}
		parser.has = false
		assign, err := parser.take()
		if err != nil {
			return nil, "", err
		}
		if assign.kind != ebnfAssign {
			return nil, "", fmt.Errorf("ebnf: expected = after %q", name)
		}
		node, err := parser.parseExpression()
		if err != nil {
			return nil, "", err
		}
		rules[name] = node
		if first == "" {
			first = name
		}
		end, err := parser.peek()
		if err != nil {
			return nil, "", err
		}
		if end.kind == ebnfSemicolon {
			parser.has = false
		}
	}
	return rules, first, nil
}

func (parser *ebnfParser) parseExpression() (*ebnfNode, error) {
	node, err := parser.parseSequence()
	if err != nil {
		return nil, err
	}
	alternatives := []*ebnfNode{node}
	for {
		token, err := parser.peek()
		if err != nil {
			return nil, err
		}
		if token.kind != ebnfPipe {
			break
		}
		parser.has = false
		next, err := parser.parseSequence()
		if err != nil {
			return nil, err
		}
		alternatives = append(alternatives, next)
	}
	if len(alternatives) == 1 {
		return alternatives[0], nil
	}
	return &ebnfNode{kind: ebnfAlternation, children: alternatives}, nil
}

func (parser *ebnfParser) parseSequence() (*ebnfNode, error) {
	var children []*ebnfNode
	for {
		token, err := parser.peek()
		if err != nil {
			return nil, err
		}
		if token.kind == ebnfComma {
			parser.has = false
			continue
		}
		if !ebnfStartsTerm(token.kind) {
			break
		}
		term, err := parser.parseTerm()
		if err != nil {
			return nil, err
		}
		children = append(children, term)
	}
	if len(children) == 0 {
		return &ebnfNode{kind: ebnfLiteral, text: ""}, nil
	}
	if len(children) == 1 {
		return children[0], nil
	}
	return &ebnfNode{kind: ebnfSequence, children: children}, nil
}

func ebnfStartsTerm(kind ebnfTokenKind) bool {
	switch kind {
	case ebnfIdent, ebnfString, ebnfLParen, ebnfLBracket, ebnfLBrace:
		return true
	}
	return false
}

func (parser *ebnfParser) parseTerm() (*ebnfNode, error) {
	node, err := parser.parsePrimary()
	if err != nil {
		return nil, err
	}
	for {
		token, err := parser.peek()
		if err != nil {
			return nil, err
		}
		switch token.kind {
		case ebnfOptionalToken:
			node = &ebnfNode{kind: ebnfOptional, children: []*ebnfNode{node}}
		case ebnfStarToken:
			node = &ebnfNode{kind: ebnfStar, children: []*ebnfNode{node}}
		case ebnfPlusToken:
			node = &ebnfNode{kind: ebnfPlus, children: []*ebnfNode{node}}
		default:
			return node, nil
		}
		parser.has = false
	}
}

func (parser *ebnfParser) parsePrimary() (*ebnfNode, error) {
	token, err := parser.take()
	if err != nil {
		return nil, err
	}
	switch token.kind {
	case ebnfIdent:
		return &ebnfNode{kind: ebnfReference, text: token.text}, nil
	case ebnfString:
		next, err := parser.peek()
		if err != nil {
			return nil, err
		}
		if next.kind == ebnfRangeToken {
			parser.has = false
			end, err := parser.take()
			if err != nil {
				return nil, err
			}
			if end.kind != ebnfString {
				return nil, fmt.Errorf("ebnf: range end must be a string")
			}
			low, lowSize := utf8.DecodeRuneInString(token.text)
			high, highSize := utf8.DecodeRuneInString(end.text)
			if lowSize != len(token.text) || highSize != len(end.text) || low > high {
				return nil, fmt.Errorf("ebnf: invalid character range %q..%q", token.text, end.text)
			}
			return &ebnfNode{kind: ebnfRange, low: low, high: high}, nil
		}
		return &ebnfNode{kind: ebnfLiteral, text: token.text}, nil
	case ebnfLParen:
		node, err := parser.parseExpression()
		if err != nil {
			return nil, err
		}
		closeToken, err := parser.take()
		if err != nil || closeToken.kind != ebnfRParen {
			return nil, fmt.Errorf("ebnf: expected )")
		}
		return node, nil
	case ebnfLBracket:
		node, err := parser.parseExpression()
		if err != nil {
			return nil, err
		}
		closeToken, err := parser.take()
		if err != nil || closeToken.kind != ebnfRBracket {
			return nil, fmt.Errorf("ebnf: expected ]")
		}
		return &ebnfNode{kind: ebnfOptional, children: []*ebnfNode{node}}, nil
	case ebnfLBrace:
		node, err := parser.parseExpression()
		if err != nil {
			return nil, err
		}
		closeToken, err := parser.take()
		if err != nil || closeToken.kind != ebnfRBrace {
			return nil, fmt.Errorf("ebnf: expected }")
		}
		return &ebnfNode{kind: ebnfStar, children: []*ebnfNode{node}}, nil
	}
	return nil, fmt.Errorf("ebnf: unexpected token")
}
