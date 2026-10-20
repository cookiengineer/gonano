package inference

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"
)

// JSONSchema is the subset of JSON Schema used by constrained decoding.
type JSONSchema struct {
	Type                string
	Properties          map[string]*JSONSchema
	Required            []string
	Items               *JSONSchema
	Enum                []any
	AdditionalAllowed   bool
	Additional          *JSONSchema
	InferredObject      bool
	InferredArray       bool
	enumStringStarts    map[byte]bool
	enumNumberStart     bool
	enumLiteralStart    map[byte]bool
	enumStructuralStart map[byte]bool
}

// JSONSchemaConstraint incrementally validates a JSON value against a schema.
// Supported keywords: type, properties, required, items, enum,
// additionalProperties (bool or schema). Unsupported keywords are ignored.
type JSONSchemaConstraint struct {
	root  *JSONSchema
	state jsonState
	stack []schemaFrame

	valueSchema   *JSONSchema
	raw           []byte
	decoded       []byte
	stringKey     bool
	numberStage   uint8
	literal       string
	literalIndex  int
	unicodeValue  rune
	unicodeDigits int
}

type schemaFrame struct {
	schema *JSONSchema
	object bool
	key    string
	seen   map[string]bool
}

// NewJSONSchemaConstraint compiles a schema value (map, RawMessage, []byte or
// JSON string) into a constraint.
func NewJSONSchemaConstraint(value any) (*JSONSchemaConstraint, error) {
	root, err := parseJSONSchema(value)
	if err != nil {
		return nil, err
	}
	return &JSONSchemaConstraint{root: root, state: jsonValue}, nil
}

func parseJSONSchema(value any) (*JSONSchema, error) {
	switch typed := value.(type) {
	case nil:
		return &JSONSchema{AdditionalAllowed: true}, nil
	case json.RawMessage:
		var decoded any
		if err := json.Unmarshal(typed, &decoded); err != nil {
			return nil, err
		}
		return parseJSONSchema(decoded)
	case []byte:
		var decoded any
		if err := json.Unmarshal(typed, &decoded); err != nil {
			return nil, err
		}
		return parseJSONSchema(decoded)
	case string:
		var decoded any
		if err := json.Unmarshal([]byte(typed), &decoded); err != nil {
			return nil, err
		}
		return parseJSONSchema(decoded)
	case map[string]any:
		schema := &JSONSchema{AdditionalAllowed: true, enumStringStarts: map[byte]bool{}, enumLiteralStart: map[byte]bool{}, enumStructuralStart: map[byte]bool{}}
		if typeName, ok := typed["type"].(string); ok {
			schema.Type = typeName
		}
		if properties, ok := typed["properties"].(map[string]any); ok {
			schema.Properties = make(map[string]*JSONSchema, len(properties))
			for name, child := range properties {
				parsed, err := parseJSONSchema(child)
				if err != nil {
					return nil, fmt.Errorf("schema property %q: %w", name, err)
				}
				schema.Properties[name] = parsed
			}
		}
		if required, ok := typed["required"].([]any); ok {
			for _, item := range required {
				if name, ok := item.(string); ok {
					schema.Required = append(schema.Required, name)
				}
			}
		}
		if items, exists := typed["items"]; exists {
			parsed, err := parseJSONSchema(items)
			if err != nil {
				return nil, fmt.Errorf("schema items: %w", err)
			}
			schema.Items = parsed
		}
		if enum, ok := typed["enum"].([]any); ok {
			schema.Enum = append([]any(nil), enum...)
		}
		if additional, exists := typed["additionalProperties"]; exists {
			switch value := additional.(type) {
			case bool:
				schema.AdditionalAllowed = value
			case map[string]any:
				parsed, err := parseJSONSchema(value)
				if err != nil {
					return nil, fmt.Errorf("schema additionalProperties: %w", err)
				}
				schema.Additional = parsed
				schema.AdditionalAllowed = true
			default:
				schema.AdditionalAllowed = true
			}
		}
		schema.InferredObject = schema.Type == "object" || (schema.Type == "" && (schema.Properties != nil || schema.Required != nil))
		schema.InferredArray = schema.Type == "array" || (schema.Type == "" && schema.Items != nil)
		schema.indexEnumStarts()
		return schema, nil
	default:
		return nil, fmt.Errorf("schema: unsupported value %T", value)
	}
}

func (schema *JSONSchema) indexEnumStarts() {
	if len(schema.Enum) == 0 {
		return
	}
	for _, value := range schema.Enum {
		switch typed := value.(type) {
		case string:
			schema.enumStringStarts['"'] = true
		case float64, int, int64, json.Number:
			schema.enumNumberStart = true
		case bool:
			if typed {
				schema.enumLiteralStart['t'] = true
			} else {
				schema.enumLiteralStart['f'] = true
			}
		case nil:
			schema.enumLiteralStart['n'] = true
		case map[string]any:
			schema.enumStructuralStart['{'] = true
		case []any:
			schema.enumStructuralStart['['] = true
		}
	}
}

// Clone returns an independent copy of the constraint.
func (constraint *JSONSchemaConstraint) Clone() Constraint {
	clone := &JSONSchemaConstraint{
		root:          constraint.root,
		state:         constraint.state,
		valueSchema:   constraint.valueSchema,
		raw:           append([]byte(nil), constraint.raw...),
		decoded:       append([]byte(nil), constraint.decoded...),
		stringKey:     constraint.stringKey,
		numberStage:   constraint.numberStage,
		literal:       constraint.literal,
		literalIndex:  constraint.literalIndex,
		unicodeValue:  constraint.unicodeValue,
		unicodeDigits: constraint.unicodeDigits,
	}
	clone.stack = make([]schemaFrame, len(constraint.stack))
	for index, frame := range constraint.stack {
		clone.stack[index] = frame
		if frame.seen != nil {
			clone.stack[index].seen = make(map[string]bool, len(frame.seen))
			for key, value := range frame.seen {
				clone.stack[index].seen[key] = value
			}
		}
	}
	return clone
}

// StateKey identifies the schema parser state for mask caching.
func (constraint *JSONSchemaConstraint) StateKey() string {
	builder := strings.Builder{}
	fmt.Fprintf(&builder, "schema:%d:%t:%d:%s:%d:%d:%d:%p:%s:%s:", constraint.state, constraint.stringKey, constraint.numberStage, constraint.literal, constraint.literalIndex, constraint.unicodeDigits, constraint.unicodeValue, constraint.valueSchema, constraint.raw, constraint.decoded)
	for index, frame := range constraint.stack {
		keys := make([]string, 0, len(frame.seen))
		for key := range frame.seen {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		fmt.Fprintf(&builder, "%d:%p:%t:%s:[%s]:", index, frame.schema, frame.object, frame.key, strings.Join(keys, ","))
	}
	return builder.String()
}

// Finished reports whether a complete schema-valid value has been produced.
func (constraint *JSONSchemaConstraint) Finished() bool {
	if constraint.state == jsonDone {
		return true
	}
	if constraint.state == jsonNumber && len(constraint.stack) == 0 {
		switch constraint.numberStage {
		case jsonNumberZero, jsonNumberInteger, jsonNumberFraction, jsonNumberExponent:
			return constraint.scalarValid()
		}
	}
	return false
}

// Allowed reports whether appending tokenBytes keeps the value schema-valid.
func (constraint *JSONSchemaConstraint) Allowed(tokenBytes []byte) bool {
	clone := constraint.Clone().(*JSONSchemaConstraint)
	return clone.advanceBytes(tokenBytes)
}

// Advance commits tokenBytes.
func (constraint *JSONSchemaConstraint) Advance(tokenBytes []byte) {
	constraint.advanceBytes(tokenBytes)
}

func (constraint *JSONSchemaConstraint) advanceBytes(bytes []byte) bool {
	for _, value := range bytes {
		if !constraint.advanceByte(value) {
			return false
		}
	}
	return true
}

func (constraint *JSONSchemaConstraint) currentValueSchema() *JSONSchema {
	if len(constraint.stack) == 0 {
		return constraint.root
	}
	top := constraint.stack[len(constraint.stack)-1]
	if top.object {
		if schema, ok := top.schema.Properties[top.key]; ok {
			return schema
		}
		if top.schema.Additional != nil {
			return top.schema.Additional
		}
		return &JSONSchema{AdditionalAllowed: true}
	}
	if top.schema.Items != nil {
		return top.schema.Items
	}
	return &JSONSchema{AdditionalAllowed: true}
}

func (constraint *JSONSchemaConstraint) keyAllowed(frame *schemaFrame, key string) bool {
	if _, ok := frame.schema.Properties[key]; ok {
		return true
	}
	for _, required := range frame.schema.Required {
		if required == key {
			return true
		}
	}
	if frame.schema.Additional != nil {
		return true
	}
	return frame.schema.AdditionalAllowed
}

func (constraint *JSONSchemaConstraint) appendStringByte(value byte) bool {
	candidate := append(constraint.decoded, value)
	if !constraint.stringPrefixAllowed(candidate) {
		return false
	}
	constraint.decoded = candidate
	return true
}

func (constraint *JSONSchemaConstraint) stringPrefixAllowed(candidate []byte) bool {
	if constraint.stringKey {
		return constraint.keyPrefixAllowed(candidate)
	}
	if constraint.valueSchema == nil || len(constraint.valueSchema.Enum) == 0 {
		return true
	}
	hasString := false
	for _, value := range constraint.valueSchema.Enum {
		text, ok := value.(string)
		if !ok {
			continue
		}
		hasString = true
		if strings.HasPrefix(text, string(candidate)) {
			return true
		}
	}
	return !hasString
}

func (constraint *JSONSchemaConstraint) keyPrefixAllowed(candidate []byte) bool {
	if len(constraint.stack) == 0 {
		return true
	}
	frame := &constraint.stack[len(constraint.stack)-1]
	if !frame.object {
		return true
	}
	if frame.schema.Additional != nil || frame.schema.AdditionalAllowed {
		return true
	}
	text := string(candidate)
	for key := range frame.schema.Properties {
		if strings.HasPrefix(key, text) {
			return true
		}
	}
	for _, key := range frame.schema.Required {
		if strings.HasPrefix(key, text) {
			return true
		}
	}
	return false
}

func (constraint *JSONSchemaConstraint) finishValue() bool {
	if len(constraint.stack) == 0 {
		constraint.state = jsonDone
		return true
	}
	top := &constraint.stack[len(constraint.stack)-1]
	if top.object {
		constraint.state = jsonAfterObjectValue
	} else {
		constraint.state = jsonAfterArrayValue
	}
	return true
}

func (constraint *JSONSchemaConstraint) closeObject() bool {
	if len(constraint.stack) == 0 {
		return false
	}
	frame := &constraint.stack[len(constraint.stack)-1]
	for _, required := range frame.schema.Required {
		if !frame.seen[required] {
			return false
		}
	}
	constraint.stack = constraint.stack[:len(constraint.stack)-1]
	return constraint.finishValue()
}

func (constraint *JSONSchemaConstraint) closeArray() bool {
	if len(constraint.stack) == 0 {
		return false
	}
	constraint.stack = constraint.stack[:len(constraint.stack)-1]
	return constraint.finishValue()
}

func (constraint *JSONSchemaConstraint) startValue(value byte) bool {
	schema := constraint.currentValueSchema()
	constraint.valueSchema = schema
	constraint.raw = constraint.raw[:0]
	constraint.decoded = constraint.decoded[:0]
	switch value {
	case '{':
		if !constraint.valueTypeAllows(value) {
			return false
		}
		constraint.stack = append(constraint.stack, schemaFrame{schema: schema, object: true, seen: map[string]bool{}})
		constraint.state = jsonObjectKeyOrEnd
		return true
	case '[':
		if !constraint.valueTypeAllows(value) {
			return false
		}
		constraint.stack = append(constraint.stack, schemaFrame{schema: schema})
		constraint.state = jsonArrayValueOrEnd
		return true
	case '"':
		if !constraint.valueTypeAllows('"') {
			return false
		}
		constraint.state = jsonString
		return true
	case 't', 'f', 'n':
		if !constraint.valueTypeAllows(value) {
			return false
		}
		switch value {
		case 't':
			constraint.literal = "true"
		case 'f':
			constraint.literal = "false"
		case 'n':
			constraint.literal = "null"
		}
		constraint.literalIndex = 1
		constraint.raw = append(constraint.raw, value)
		constraint.state = jsonLiteral
		return true
	case '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		if !constraint.valueTypeAllows(value) {
			return false
		}
		constraint.raw = append(constraint.raw, value)
		if value == '-' {
			constraint.numberStage = jsonNumberMinus
		} else if value == '0' {
			constraint.numberStage = jsonNumberZero
		} else {
			constraint.numberStage = jsonNumberInteger
		}
		constraint.state = jsonNumber
		return true
	}
	return false
}

func (constraint *JSONSchemaConstraint) valueTypeAllows(value byte) bool {
	schema := constraint.valueSchema
	if schema == nil {
		return true
	}
	if len(schema.Enum) > 0 {
		switch value {
		case '"':
			return schema.enumStringStarts['"']
		case '{', '[':
			return schema.enumStructuralStart[value]
		case 't', 'f', 'n':
			return schema.enumLiteralStart[value]
		default:
			return schema.enumNumberStart && (value == '-' || isJSONDigit(value))
		}
	}
	if schema.InferredObject {
		return value == '{'
	}
	if schema.InferredArray {
		return value == '['
	}
	switch schema.Type {
	case "object":
		return value == '{'
	case "array":
		return value == '['
	case "string":
		return value == '"'
	case "number", "integer":
		return value == '-' || isJSONDigit(value)
	case "boolean":
		return value == 't' || value == 'f'
	case "null":
		return value == 'n'
	}
	return true
}

func (constraint *JSONSchemaConstraint) finishScalar() bool {
	if !constraint.scalarValid() {
		return false
	}
	return constraint.finishValue()
}

func (constraint *JSONSchemaConstraint) scalarValid() bool {
	schema := constraint.valueSchema
	if schema == nil {
		return true
	}
	raw := string(constraint.raw)
	if constraint.state == jsonString || (constraint.state != jsonNumber && constraint.state != jsonLiteral && len(constraint.decoded) > 0) {
		// String values are validated at the closing quote.
	}
	if len(schema.Enum) > 0 {
		value, err := constraint.scalarValue()
		if err != nil {
			return false
		}
		found := false
		for _, candidate := range schema.Enum {
			if reflect.DeepEqual(value, candidate) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	switch schema.Type {
	case "string":
		return true
	case "number":
		return true
	case "integer":
		for _, value := range raw {
			if value == '.' || value == 'e' || value == 'E' {
				return false
			}
		}
		return true
	case "boolean":
		return raw == "true" || raw == "false"
	case "null":
		return raw == "null"
	}
	return true
}

func (constraint *JSONSchemaConstraint) scalarValue() (any, error) {
	switch constraint.state {
	case jsonString:
		return string(constraint.decoded), nil
	case jsonLiteral, jsonNumber:
		var value any
		if err := json.Unmarshal([]byte(constraint.raw), &value); err != nil {
			return nil, err
		}
		return value, nil
	}
	if len(constraint.decoded) > 0 {
		return string(constraint.decoded), nil
	}
	if len(constraint.raw) > 0 {
		var value any
		if err := json.Unmarshal(constraint.raw, &value); err != nil {
			return nil, err
		}
		return value, nil
	}
	return nil, nil
}

func (constraint *JSONSchemaConstraint) finishStringValue() bool {
	if !constraint.scalarValid() {
		return false
	}
	return constraint.finishValue()
}

func (constraint *JSONSchemaConstraint) finishLiteral() bool {
	if !constraint.scalarValid() {
		return false
	}
	return constraint.finishValue()
}

func (constraint *JSONSchemaConstraint) advanceByte(value byte) bool {
	for {
		switch constraint.state {
		case jsonValue:
			if isJSONWhitespace(value) {
				return true
			}
			return constraint.startValue(value)
		case jsonObjectKeyOrEnd:
			if isJSONWhitespace(value) {
				return true
			}
			if value == '}' {
				return constraint.closeObject()
			}
			if value == '"' {
				constraint.stringKey = true
				constraint.decoded = constraint.decoded[:0]
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
				constraint.decoded = constraint.decoded[:0]
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
				return constraint.closeObject()
			}
			return false
		case jsonArrayValueOrEnd:
			if isJSONWhitespace(value) {
				return true
			}
			if value == ']' {
				return constraint.closeArray()
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
				return constraint.closeArray()
			}
			return false
		case jsonString:
			if value == '"' {
				if constraint.stringKey {
					frame := &constraint.stack[len(constraint.stack)-1]
					key := string(constraint.decoded)
					if !constraint.keyAllowed(frame, key) {
						return false
					}
					frame.key = key
					frame.seen[key] = true
					constraint.stringKey = false
					constraint.state = jsonColon
					return true
				}
				return constraint.finishStringValue()
			}
			if value == '\\' {
				constraint.state = jsonStringEscape
				return true
			}
			if value < 0x20 {
				return false
			}
			return constraint.appendStringByte(value)
		case jsonStringEscape:
			switch value {
			case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
				decoded := value
				if value == 'b' {
					decoded = '\b'
				} else if value == 'f' {
					decoded = '\f'
				} else if value == 'n' {
					decoded = '\n'
				} else if value == 'r' {
					decoded = '\r'
				} else if value == 't' {
					decoded = '\t'
				}
				if !constraint.appendStringByte(decoded) {
					return false
				}
				constraint.state = jsonString
				return true
			case 'u':
				constraint.unicodeDigits = 0
				constraint.unicodeValue = 0
				constraint.state = jsonStringUnicode
				return true
			default:
				return false
			}
		case jsonStringUnicode:
			digit := hexValue(value)
			if digit < 0 {
				return false
			}
			constraint.unicodeValue = constraint.unicodeValue*16 + rune(digit)
			constraint.unicodeDigits++
			if constraint.unicodeDigits == 4 {
				for _, encoded := range []byte(string(constraint.unicodeValue)) {
					if !constraint.appendStringByte(encoded) {
						return false
					}
				}
				constraint.state = jsonString
			}
			return true
		case jsonNumber:
			if isNumberContinuation(value, &constraint.numberStage) {
				constraint.raw = append(constraint.raw, value)
				return true
			}
			if !constraint.finishScalar() {
				return false
			}
			continue
		case jsonLiteral:
			if constraint.literalIndex >= len(constraint.literal) {
				if !constraint.finishLiteral() {
					return false
				}
				continue
			}
			if value != constraint.literal[constraint.literalIndex] {
				return false
			}
			constraint.raw = append(constraint.raw, value)
			constraint.literalIndex++
			if constraint.literalIndex == len(constraint.literal) {
				if !constraint.finishLiteral() {
					return false
				}
			}
			return true
		case jsonDone:
			return isJSONWhitespace(value)
		}
		return false
	}
}

func hexValue(value byte) int {
	switch {
	case value >= '0' && value <= '9':
		return int(value - '0')
	case value >= 'a' && value <= 'f':
		return int(value-'a') + 10
	case value >= 'A' && value <= 'F':
		return int(value-'A') + 10
	}
	return -1
}

// isNumberContinuation applies JSON number grammar. It returns true when the
// byte belongs to the current number; false means the caller should finish the
// number and reprocess the byte.
func isNumberContinuation(value byte, stage *uint8) bool {
	switch *stage {
	case jsonNumberMinus:
		if value == '0' {
			*stage = jsonNumberZero
			return true
		}
		if value >= '1' && value <= '9' {
			*stage = jsonNumberInteger
			return true
		}
		return false
	case jsonNumberZero:
		if value == '.' {
			*stage = jsonNumberFractionStart
			return true
		}
		if value == 'e' || value == 'E' {
			*stage = jsonNumberExponentStart
			return true
		}
		return false
	case jsonNumberInteger:
		if isJSONDigit(value) {
			return true
		}
		if value == '.' {
			*stage = jsonNumberFractionStart
			return true
		}
		if value == 'e' || value == 'E' {
			*stage = jsonNumberExponentStart
			return true
		}
		return false
	case jsonNumberFractionStart:
		if isJSONDigit(value) {
			*stage = jsonNumberFraction
			return true
		}
		return false
	case jsonNumberFraction:
		if isJSONDigit(value) {
			return true
		}
		if value == 'e' || value == 'E' {
			*stage = jsonNumberExponentStart
			return true
		}
		return false
	case jsonNumberExponentStart:
		if value == '+' || value == '-' {
			*stage = jsonNumberExponentSign
			return true
		}
		if isJSONDigit(value) {
			*stage = jsonNumberExponent
			return true
		}
		return false
	case jsonNumberExponentSign:
		if isJSONDigit(value) {
			*stage = jsonNumberExponent
			return true
		}
		return false
	case jsonNumberExponent:
		return isJSONDigit(value)
	}
	return false
}
