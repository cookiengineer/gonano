package infer

import (
	"strconv"
	"strings"
)

// Tool is a callable function the model can invoke through the tool-call
// protocol: the assistant emits <|tool_start|> … <|tool_end|>, and the runtime
// dispatches the enclosed text to a Tool and feeds its result back as
// <|tool_output_start|> … <|tool_output_end|>.
type Tool interface {
	// Name returns the tool's identifier (used for logging/registration).
	Name() string
	// Call evaluates expr and reports whether it understood it. When ok is
	// false the expression is passed on to the next tool.
	Call(expr string) (result string, ok bool)
}

// Calculator is a safe arithmetic and string-counting tool. It evaluates pure
// arithmetic expressions (no **, no function calls) and "literal".count("sub")
// expressions. There is no arbitrary code execution: everything runs in Go.
type Calculator struct{}

func (Calculator) Name() string { return "calculator" }

func (Calculator) Call(expr string) (string, bool) {
	expr = strings.ReplaceAll(expr, ",", "")

	if isArithmetic(expr) {
		if strings.Contains(expr, "**") {
			return "", false
		}
		if v, ok := evalArithmetic(expr); ok {
			return formatNumber(v), true
		}
		return "", false
	}

	if isCountExpression(expr) {
		if n, ok := evalCount(expr); ok {
			return formatNumber(float64(n)), true
		}
		return "", false
	}
	return "", false
}

// Registry dispatches an expression to the first registered tool that
// understands it. Callers can register additional Go-implemented tools
// (e.g. a unit converter or a weather lookup) to extend the model's abilities.
type Registry struct {
	tools []Tool
}

// Register adds a tool. Tools are consulted in registration order.
func (r *Registry) Register(t Tool) { r.tools = append(r.tools, t) }

// Execute runs the first tool that understands expr.
func (r *Registry) Execute(expr string) (string, bool) {
	_, result, ok := r.ExecuteTool(expr)
	return result, ok
}

// ExecuteTool runs the first tool that understands expr and also returns the
// tool's name (so callers can report which tool handled a call).
func (r *Registry) ExecuteTool(expr string) (name, result string, ok bool) {
	for _, t := range r.tools {
		if res, ok := t.Call(expr); ok {
			return t.Name(), res, true
		}
	}
	return "", "", false
}

// NewCalculator returns a registry containing the built-in calculator tool.
func NewCalculator() *Registry {
	r := &Registry{}
	r.Register(Calculator{})
	return r
}

// UseCalculator evaluates expr with the built-in calculator tool. It returns
// nil if the expression is unsupported or invalid.
func UseCalculator(expr string) *string {
	result, ok := (Calculator{}).Call(expr)
	if !ok {
		return nil
	}
	return &result
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
