package eval

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
	for _, raw := range strings.Split(data, "\n") {
		trimmed := strings.TrimRight(raw, " \t\r")
		if strings.TrimSpace(trimmed) == "" || strings.HasPrefix(strings.TrimSpace(trimmed), "#") {
			continue
		}
		indent := len(trimmed) - len(strings.TrimLeft(trimmed, " "))
		lines = append(lines, yamlLine{indent: indent, content: strings.TrimSpace(trimmed)})
	}
	if len(lines) == 0 {
		return map[string]any{}, nil
	}
	idx := 0
	v, err := parseYAMLValue(lines, &idx, lines[0].indent)
	if err != nil {
		return nil, err
	}
	m, ok := v.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("yaml: top-level value is not a mapping")
	}
	return m, nil
}

func parseYAMLValue(lines []yamlLine, idx *int, indent int) (any, error) {
	if *idx >= len(lines) {
		return nil, nil
	}
	line := lines[*idx]
	if line.indent < indent {
		return nil, nil
	}
	if line.indent > indent {
		return nil, fmt.Errorf("yaml: unexpected indent at %q", line.content)
	}

	if strings.HasPrefix(line.content, "- ") || line.content == "-" {
		return parseYAMLList(lines, idx, indent)
	}
	return parseYAMLMap(lines, idx, indent)
}

func parseYAMLMap(lines []yamlLine, idx *int, indent int) (map[string]any, error) {
	out := map[string]any{}
	for *idx < len(lines) {
		line := lines[*idx]
		if line.indent < indent {
			break
		}
		if line.indent > indent {
			return nil, fmt.Errorf("yaml: unexpected indent at %q", line.content)
		}
		key, rest, ok := splitKV(line.content)
		if !ok {
			return nil, fmt.Errorf("yaml: expected key: value, got %q", line.content)
		}
		*idx++
		rest = strings.TrimSpace(rest)
		if rest == "" || rest == "|" || rest == ">" {
			// Nested block or empty.
			if *idx < len(lines) && lines[*idx].indent > indent {
				v, err := parseYAMLValue(lines, idx, lines[*idx].indent)
				if err != nil {
					return nil, err
				}
				out[key] = v
			} else {
				out[key] = nil
			}
		} else {
			out[key] = parseScalar(rest)
		}
	}
	return out, nil
}

func parseYAMLList(lines []yamlLine, idx *int, indent int) ([]any, error) {
	var out []any
	for *idx < len(lines) {
		line := lines[*idx]
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
		*idx++
		if content == "" {
			// Nested block under the list item.
			if *idx < len(lines) && lines[*idx].indent > indent {
				v, err := parseYAMLValue(lines, idx, lines[*idx].indent)
				if err != nil {
					return nil, err
				}
				out = append(out, v)
			} else {
				out = append(out, nil)
			}
		} else if key, rest, ok := splitKV(content); ok {
			// List item is a map: start with this key, then continue the map.
			m := map[string]any{}
			rest = strings.TrimSpace(rest)
			if rest == "" {
				if *idx < len(lines) && lines[*idx].indent > indent {
					v, err := parseYAMLValue(lines, idx, lines[*idx].indent)
					if err != nil {
						return nil, err
					}
					m[key] = v
				} else {
					m[key] = nil
				}
			} else {
				m[key] = parseScalar(rest)
			}
			// Continue parsing keys of this map (indented deeper than the dash).
			itemIndent := line.indent + 2
			more, err := parseYAMLMap(lines, idx, itemIndent)
			if err != nil {
				return nil, err
			}
			for k, v := range more {
				m[k] = v
			}
			out = append(out, m)
		} else {
			out = append(out, parseScalar(content))
		}
	}
	return out, nil
}

func splitKV(s string) (string, string, bool) {
	i := strings.Index(s, ":")
	if i < 0 {
		return "", "", false
	}
	return strings.TrimSpace(s[:i]), s[i+1:], true
}

func parseScalar(s string) any {
	s = strings.TrimSpace(s)
	// Strip a trailing comment.
	if idx := strings.Index(s, " #"); idx >= 0 {
		s = strings.TrimSpace(s[:idx])
	}
	if len(s) >= 2 && ((s[0] == '"' && s[len(s)-1] == '"') || (s[0] == '\'' && s[len(s)-1] == '\'')) {
		return s[1 : len(s)-1]
	}
	switch s {
	case "true", "True", "TRUE":
		return true
	case "false", "False", "FALSE":
		return false
	case "null", "Null", "NULL", "~":
		return nil
	}
	if i, err := strconv.ParseInt(s, 10, 64); err == nil {
		return i
	}
	if f, err := strconv.ParseFloat(s, 64); err == nil {
		return f
	}
	return s
}
