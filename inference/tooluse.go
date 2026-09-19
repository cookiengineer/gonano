package inference

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
		if value, ok := evalArithmetic(expr); ok {
			return formatNumber(value), true
		}
		return "", false
	}

	if isCountExpression(expr) {
		if count, ok := evalCount(expr); ok {
			return formatNumber(float64(count)), true
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
func (registry *Registry) Register(tool Tool) { registry.tools = append(registry.tools, tool) }

// Execute runs the first tool that understands expr.
func (registry *Registry) Execute(expr string) (string, bool) {
	_, result, ok := registry.ExecuteTool(expr)
	return result, ok
}

// ExecuteTool runs the first tool that understands expr and also returns the
// tool's name (so callers can report which tool handled a call).
func (registry *Registry) ExecuteTool(expr string) (name, result string, ok bool) {
	for _, tool := range registry.tools {
		if callResult, ok := tool.Call(expr); ok {
			return tool.Name(), callResult, true
		}
	}
	return "", "", false
}

// NewCalculator returns a registry containing the built-in calculator tool.
func NewCalculator() *Registry {
	registry := &Registry{}
	registry.Register(Calculator{})
	return registry
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
	for _, char := range expr {
		switch {
		case char >= '0' && char <= '9', char == '*', char == '+', char == '-', char == '/', char == '.', char == '(', char == ')', char == ' ':
		default:
			return false
		}
	}
	return len(expr) > 0
}

func isCountExpression(expr string) bool {
	// Only allow letters, digits, quotes, dot, parens, underscore, space.
	for _, char := range expr {
		switch {
		case char >= 'a' && char <= 'z', char >= 'A' && char <= 'Z', char >= '0' && char <= '9', char == '\'', char == '"', char == '.', char == '(', char == ')', char == '_', char == ' ':
		default:
			return false
		}
	}
	lower := strings.ToLower(expr)
	for _, forbidden := range []string{"__", "import", "exec", "eval", "compile", "open", "file", "input", "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr", "hasattr"} {
		if strings.Contains(lower, forbidden) {
			return false
		}
	}
	return strings.Contains(expr, ".count(")
}

// evalCount evaluates "literal".count("sub") and returns the occurrence count.
func evalCount(expr string) (int, bool) {
	dotIndex := strings.Index(expr, ".count(")
	if dotIndex < 0 {
		return 0, false
	}
	literal := strings.TrimSpace(expr[:dotIndex])
	argument := strings.TrimSpace(expr[dotIndex+len(".count("):])
	if !strings.HasSuffix(argument, ")") {
		return 0, false
	}
	argument = strings.TrimSpace(argument[:len(argument)-1])

	literalText, ok := unquote(literal)
	if !ok {
		return 0, false
	}
	substring, ok := unquote(argument)
	if !ok {
		return 0, false
	}
	return strings.Count(literalText, substring), true
}

func unquote(text string) (string, bool) {
	if len(text) < 2 {
		return "", false
	}
	if (text[0] == '"' && text[len(text)-1] == '"') || (text[0] == '\'' && text[len(text)-1] == '\'') {
		return text[1 : len(text)-1], true
	}
	return "", false
}

// arithmetic parser: a small recursive-descent evaluator.
type arithParser struct {
	source   string
	position int
}

func (parser *arithParser) parse() (float64, bool) {
	value, ok := parser.expr()
	if !ok {
		return 0, false
	}
	parser.skipSpace()
	if parser.position != len(parser.source) {
		return 0, false
	}
	return value, true
}

func (parser *arithParser) expr() (float64, bool) {
	value, ok := parser.term()
	if !ok {
		return 0, false
	}
	for {
		parser.skipSpace()
		if parser.position >= len(parser.source) {
			return value, true
		}
		operator := parser.source[parser.position]
		if operator != '+' && operator != '-' {
			return value, true
		}
		parser.position++
		rightHandSide, ok := parser.term()
		if !ok {
			return 0, false
		}
		if operator == '+' {
			value += rightHandSide
		} else {
			value -= rightHandSide
		}
	}
}

func (parser *arithParser) term() (float64, bool) {
	value, ok := parser.factor()
	if !ok {
		return 0, false
	}
	for {
		parser.skipSpace()
		if parser.position >= len(parser.source) {
			return value, true
		}
		operator := parser.source[parser.position]
		if operator != '*' && operator != '/' {
			return value, true
		}
		parser.position++
		rightHandSide, ok := parser.factor()
		if !ok {
			return 0, false
		}
		if operator == '*' {
			value *= rightHandSide
		} else {
			if rightHandSide == 0 {
				return 0, false
			}
			value /= rightHandSide
		}
	}
}

func (parser *arithParser) factor() (float64, bool) {
	parser.skipSpace()
	if parser.position >= len(parser.source) {
		return 0, false
	}
	if parser.source[parser.position] == '-' {
		parser.position++
		value, ok := parser.factor()
		return -value, ok
	}
	if parser.source[parser.position] == '(' {
		parser.position++
		value, ok := parser.expr()
		if !ok {
			return 0, false
		}
		parser.skipSpace()
		if parser.position >= len(parser.source) || parser.source[parser.position] != ')' {
			return 0, false
		}
		parser.position++
		return value, true
	}
	return parser.number()
}

func (parser *arithParser) number() (float64, bool) {
	startIndex := parser.position
	seenDigit := false
	for parser.position < len(parser.source) && parser.source[parser.position] >= '0' && parser.source[parser.position] <= '9' {
		parser.position++
		seenDigit = true
	}
	if parser.position < len(parser.source) && parser.source[parser.position] == '.' {
		parser.position++
		for parser.position < len(parser.source) && parser.source[parser.position] >= '0' && parser.source[parser.position] <= '9' {
			parser.position++
			seenDigit = true
		}
	}
	if !seenDigit {
		return 0, false
	}
	var value float64
	integerPart := true
	fraction := 0.1
	for _, char := range parser.source[startIndex:parser.position] {
		if char == '.' {
			integerPart = false
			fraction = 0.1
			continue
		}
		digit := float64(char - '0')
		if integerPart {
			value = value*10 + digit
		} else {
			value += digit * fraction
			fraction *= 0.1
		}
	}
	return value, true
}

func (parser *arithParser) skipSpace() {
	for parser.position < len(parser.source) && parser.source[parser.position] == ' ' {
		parser.position++
	}
}

func evalArithmetic(expr string) (float64, bool) {
	parser := &arithParser{source: expr}
	return parser.parse()
}

func formatNumber(value float64) string {
	if value == float64(int64(value)) {
		return strconv.FormatInt(int64(value), 10)
	}
	return strconv.FormatFloat(value, 'f', -1, 64)
}
