package evaluator

import (
	"fmt"
	"strconv"
	"strings"
)

// A minimal YAML-subset parser sufficient for the CORE benchmark's core.yaml.
// It supports nested maps, lists of maps, and scalar values (strings, numbers,
// booleans, null), with indentation-based nesting.

type yamlLine struct {
	indent  int
	content string
}

// ParseYAML parses a YAML document into nested Go values.
func ParseYAML(data string) (map[string]any, error) {
	var lines []yamlLine
	for _, rawLine := range strings.Split(data, "\n") {
		trimmed := strings.TrimRight(rawLine, " \t\r")
		if strings.TrimSpace(trimmed) == "" || strings.HasPrefix(strings.TrimSpace(trimmed), "#") {
			continue
		}
		indent := len(trimmed) - len(strings.TrimLeft(trimmed, " "))
		lines = append(lines, yamlLine{indent: indent, content: strings.TrimSpace(trimmed)})
	}
	if len(lines) == 0 {
		return map[string]any{}, nil
	}
	position := 0
	value, err := parseYAMLValue(lines, &position, lines[0].indent)
	if err != nil {
		return nil, err
	}
	result, ok := value.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("yaml: top-level value is not a mapping")
	}
	return result, nil
}

func parseYAMLValue(lines []yamlLine, position *int, indent int) (any, error) {
	if *position >= len(lines) {
		return nil, nil
	}
	line := lines[*position]
	if line.indent < indent {
		return nil, nil
	}
	if line.indent > indent {
		return nil, fmt.Errorf("yaml: unexpected indent at %q", line.content)
	}

	if strings.HasPrefix(line.content, "- ") || line.content == "-" {
		return parseYAMLList(lines, position, indent)
	}
	return parseYAMLMap(lines, position, indent)
}

func parseYAMLMap(lines []yamlLine, position *int, indent int) (map[string]any, error) {
	result := map[string]any{}
	for *position < len(lines) {
		line := lines[*position]
		if line.indent < indent {
			break
		}
		if line.indent > indent {
			return nil, fmt.Errorf("yaml: unexpected indent at %q", line.content)
		}
		key, rest, ok := splitKeyValue(line.content)
		if !ok {
			return nil, fmt.Errorf("yaml: expected key: value, got %q", line.content)
		}
		*position++
		rest = strings.TrimSpace(rest)
		if rest == "" || rest == "|" || rest == ">" {
			// Nested block or empty.
			if *position < len(lines) && lines[*position].indent > indent {
				value, err := parseYAMLValue(lines, position, lines[*position].indent)
				if err != nil {
					return nil, err
				}
				result[key] = value
			} else {
				result[key] = nil
			}
		} else {
			result[key] = parseScalar(rest)
		}
	}
	return result, nil
}

func parseYAMLList(lines []yamlLine, position *int, indent int) ([]any, error) {
	var result []any
	for *position < len(lines) {
		line := lines[*position]
		if line.indent < indent {
			break
		}
		if line.indent > indent {
			return nil, fmt.Errorf("yaml: unexpected list indent at %q", line.content)
		}
		content := line.content
		if strings.HasPrefix(content, "- ") {
			content = strings.TrimSpace(content[2:])
		} else if content == "-" {
			content = ""
		} else {
			break
		}
		*position++
		if content == "" {
			// Nested block under the list item.
			if *position < len(lines) && lines[*position].indent > indent {
				value, err := parseYAMLValue(lines, position, lines[*position].indent)
				if err != nil {
					return nil, err
				}
				result = append(result, value)
			} else {
				result = append(result, nil)
			}
		} else if key, rest, ok := splitKeyValue(content); ok {
			// List item is a map: start with this key, then continue the map.
			mapping := map[string]any{}
			rest = strings.TrimSpace(rest)
			if rest == "" {
				if *position < len(lines) && lines[*position].indent > indent {
					value, err := parseYAMLValue(lines, position, lines[*position].indent)
					if err != nil {
						return nil, err
					}
					mapping[key] = value
				} else {
					mapping[key] = nil
				}
			} else {
				mapping[key] = parseScalar(rest)
			}
			// Continue parsing keys of this map (indented deeper than the dash).
			itemIndent := line.indent + 2
			remaining, err := parseYAMLMap(lines, position, itemIndent)
			if err != nil {
				return nil, err
			}
			for key, value := range remaining {
				mapping[key] = value
			}
			result = append(result, mapping)
		} else {
			result = append(result, parseScalar(content))
		}
	}
	return result, nil
}

func splitKeyValue(text string) (string, string, bool) {
	colonIndex := strings.Index(text, ":")
	if colonIndex < 0 {
		return "", "", false
	}
	return strings.TrimSpace(text[:colonIndex]), text[colonIndex+1:], true
}

func parseScalar(text string) any {
	text = strings.TrimSpace(text)
	// Strip a trailing comment.
	if commentIndex := strings.Index(text, " #"); commentIndex >= 0 {
		text = strings.TrimSpace(text[:commentIndex])
	}
	if len(text) >= 2 && ((text[0] == '"' && text[len(text)-1] == '"') || (text[0] == '\'' && text[len(text)-1] == '\'')) {
		return text[1 : len(text)-1]
	}
	switch text {
	case "true", "True", "TRUE":
		return true
	case "false", "False", "FALSE":
		return false
	case "null", "Null", "NULL", "~":
		return nil
	}
	if integer, err := strconv.ParseInt(text, 10, 64); err == nil {
		return integer
	}
	if floatValue, err := strconv.ParseFloat(text, 64); err == nil {
		return floatValue
	}
	return text
}
