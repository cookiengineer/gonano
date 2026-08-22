package infer

import (
	"strconv"
	"strings"
)

// UseCalculator evaluates a safe expression emitted by the model: either a
// pure arithmetic expression or a string .count() operation. It returns the
// result as a string, or nil if the expression is unsupported or invalid.
//
// This is a dependency-free Go reimplementation of nanochat's use_calculator:
// there is no Python subprocess and no arbitrary code execution.
func UseCalculator(expr string) *string {
	expr = strings.ReplaceAll(expr, ",", "")

	if isArithmetic(expr) {
		if strings.Contains(expr, "**") {
			return nil
		}
		if v, ok := evalArithmetic(expr); ok {
			s := formatNumber(v)
			return &s
		}
		return nil
	}

	if isCountExpression(expr) {
		if n, ok := evalCount(expr); ok {
			s := formatNumber(float64(n))
			return &s
		}
		return nil
	}
	return nil
}

func isArithmetic(expr string) bool {
	for _, r := range expr {
		switch {
		case r >= '0' && r <= '9', r == '*', r == '+', r == '-', r == '/', r == '.', r == '(', r == ')', r == ' ':
		default:
			return false
		}
	}
	return len(expr) > 0
}

func isCountExpression(expr string) bool {
	// Only allow letters, digits, quotes, dot, parens, underscore, space.
	for _, r := range expr {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '\'', r == '"', r == '.', r == '(', r == ')', r == '_', r == ' ':
		default:
			return false
		}
	}
	lower := strings.ToLower(expr)
	for _, bad := range []string{"__", "import", "exec", "eval", "compile", "open", "file", "input", "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr", "hasattr"} {
		if strings.Contains(lower, bad) {
			return false
		}
	}
	return strings.Contains(expr, ".count(")
}

// evalCount evaluates "literal".count("sub") and returns the occurrence count.
func evalCount(expr string) (int, bool) {
	dot := strings.Index(expr, ".count(")
	if dot < 0 {
		return 0, false
	}
	literal := strings.TrimSpace(expr[:dot])
	arg := strings.TrimSpace(expr[dot+len(".count("):])
	if !strings.HasSuffix(arg, ")") {
		return 0, false
	}
	arg = strings.TrimSpace(arg[:len(arg)-1])

	lit, ok := unquote(literal)
	if !ok {
		return 0, false
	}
	sub, ok := unquote(arg)
	if !ok {
		return 0, false
	}
	return strings.Count(lit, sub), true
}

func unquote(s string) (string, bool) {
	if len(s) < 2 {
		return "", false
	}
	if (s[0] == '"' && s[len(s)-1] == '"') || (s[0] == '\'' && s[len(s)-1] == '\'') {
		return s[1 : len(s)-1], true
	}
	return "", false
}

// arithmetic parser: a small recursive-descent evaluator.
type arithParser struct {
	s   string
	pos int
}

func (p *arithParser) parse() (float64, bool) {
	v, ok := p.expr()
	if !ok {
		return 0, false
	}
	p.skipSpace()
	if p.pos != len(p.s) {
		return 0, false
	}
	return v, true
}

func (p *arithParser) expr() (float64, bool) {
	v, ok := p.term()
	if !ok {
		return 0, false
	}
	for {
		p.skipSpace()
		if p.pos >= len(p.s) {
			return v, true
		}
		op := p.s[p.pos]
		if op != '+' && op != '-' {
			return v, true
		}
		p.pos++
		rhs, ok := p.term()
		if !ok {
			return 0, false
		}
		if op == '+' {
			v += rhs
		} else {
			v -= rhs
		}
	}
}

func (p *arithParser) term() (float64, bool) {
	v, ok := p.factor()
	if !ok {
		return 0, false
	}
	for {
		p.skipSpace()
		if p.pos >= len(p.s) {
			return v, true
		}
		op := p.s[p.pos]
		if op != '*' && op != '/' {
			return v, true
		}
		p.pos++
		rhs, ok := p.factor()
		if !ok {
			return 0, false
		}
		if op == '*' {
			v *= rhs
		} else {
			if rhs == 0 {
				return 0, false
			}
			v /= rhs
		}
	}
}

func (p *arithParser) factor() (float64, bool) {
	p.skipSpace()
	if p.pos >= len(p.s) {
		return 0, false
	}
	if p.s[p.pos] == '-' {
		p.pos++
		v, ok := p.factor()
		return -v, ok
	}
	if p.s[p.pos] == '(' {
		p.pos++
		v, ok := p.expr()
		if !ok {
			return 0, false
		}
		p.skipSpace()
		if p.pos >= len(p.s) || p.s[p.pos] != ')' {
			return 0, false
		}
		p.pos++
		return v, true
	}
	return p.number()
}

func (p *arithParser) number() (float64, bool) {
	start := p.pos
	seenDigit := false
	for p.pos < len(p.s) && p.s[p.pos] >= '0' && p.s[p.pos] <= '9' {
		p.pos++
		seenDigit = true
	}
	if p.pos < len(p.s) && p.s[p.pos] == '.' {
		p.pos++
		for p.pos < len(p.s) && p.s[p.pos] >= '0' && p.s[p.pos] <= '9' {
			p.pos++
			seenDigit = true
		}
	}
	if !seenDigit {
		return 0, false
	}
	var v float64
	intPart := true
	frac := 0.1
	for _, c := range p.s[start:p.pos] {
		if c == '.' {
			intPart = false
			frac = 0.1
			continue
		}
		d := float64(c - '0')
		if intPart {
			v = v*10 + d
		} else {
			v += d * frac
			frac *= 0.1
		}
	}
	return v, true
}

func (p *arithParser) skipSpace() {
	for p.pos < len(p.s) && p.s[p.pos] == ' ' {
		p.pos++
	}
}

func evalArithmetic(expr string) (float64, bool) {
	p := &arithParser{s: expr}
	return p.parse()
}

func formatNumber(v float64) string {
	if v == float64(int64(v)) {
		return strconv.FormatInt(int64(v), 10)
	}
	return strconv.FormatFloat(v, 'f', -1, 64)
}
