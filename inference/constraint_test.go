package inference

import (
	"encoding/json"
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
)

func TestJSONConstraintValidAndInvalid(t *testing.T) {
	valid := []string{
		`null`, `true`, `false`, `0`, `-12.5e+3`,
		`{}`, `[]`, `{"a":1}`, `[1,2,{"b":"x"}]`,
		` { "a" : [ true, null, -0.5 ] } `,
	}
	for _, input := range valid {
		constraint := NewJSONConstraint()
		for index := 0; index < len(input); index++ {
			if !constraint.Allowed([]byte{input[index]}) {
				t.Fatalf("valid JSON %q rejected at byte %d (%q)", input, index, input[index])
			}
			constraint.Advance([]byte{input[index]})
		}
		if !constraint.Finished() {
			t.Fatalf("valid JSON %q did not finish", input)
		}
	}
	invalid := []string{`{`, `{"a"}`, `{"a":}`, `[1,]`, `tru`, `01`, `1.`, `{"a":1} x`}
	for _, input := range invalid {
		constraint := NewJSONConstraint()
		ok := true
		for index := 0; index < len(input); index++ {
			if !constraint.Allowed([]byte{input[index]}) {
				ok = false
				break
			}
			constraint.Advance([]byte{input[index]})
		}
		if ok && constraint.Finished() {
			t.Fatalf("invalid JSON %q accepted", input)
		}
	}
}

func jsonTestTokenizer() *tokenizer.Tokenizer {
	ranks := map[string]int{}
	next := 0
	for _, token := range []string{"{", "}", "[", "]", `"`, ":", ",", "a", "b", "1", "2", " ", "true", "false", "null"} {
		ranks[token] = next
		next++
	}
	return tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
}

func TestConstraintMaskAllowsOnlyGrammarTokens(t *testing.T) {
	tokenizerImpl := jsonTestTokenizer()
	constraint := NewJSONConstraint()
	stop := []int{tokenizerImpl.EncodeSpecial("<|assistant_end|>")}
	mask := ConstraintMask(tokenizerImpl, constraint, tokenizerImpl.VocabSize(), stop)
	for id, allowed := range mask {
		if tokenizerImpl.IsSpecial(id) {
			if allowed {
				t.Fatalf("special token %d allowed before JSON completion", id)
			}
			continue
		}
		text := string(tokenizerImpl.DecodeSingleTokenBytes(id))
		if text == "{" || text == "[" || text == `"` || text == "true" || text == "false" || text == "null" || text == "1" {
			if !allowed {
				t.Fatalf("token %q should be allowed at JSON start", text)
			}
		}
	}
}

func TestEngineJSONConstraintProducesValidJSON(t *testing.T) {
	tokenizerImpl := jsonTestTokenizer()
	config := model.Config{
		SequenceLen: 32, VocabSize: tokenizerImpl.VocabSize(), NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	engine := NewEngine(transformer, tokenizerImpl)
	prompt := []int{tokenizerImpl.EncodeSpecial("<|bos|>")}
	got, _ := engine.GenerateBatchWith(prompt, 1, 16, 0, 0, 1, GenerateOptions{Constraint: NewJSONConstraint()})
	output := got[0][len(prompt):]
	text := tokenizerImpl.Decode(output)
	if text != "" && !json.Valid([]byte(text)) {
		// The generation may stop on max tokens before completing a value; in
		// that case verify that every prefix was at least grammar-valid.
		constraint := NewJSONConstraint()
		for _, id := range output {
			bytes := tokenizerImpl.DecodeSingleTokenBytes(id)
			if tokenizerImpl.IsSpecial(id) {
				break
			}
			if !constraint.Allowed(bytes) {
				t.Fatalf("engine emitted disallowed token %q", string(bytes))
			}
			constraint.Advance(bytes)
		}
	}
	if text != "" && json.Valid([]byte(text)) {
		return
	}
	if text == "" {
		t.Fatal("engine produced no constrained output")
	}
}

func TestRegexConstraintIncrementalMatch(t *testing.T) {
	constraint, err := NewRegexConstraint(`a(b|c)*d`)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	for _, prefix := range []string{"a", "ab", "abc", "ad", "abcbcd"} {
		probe := constraint.Clone()
		for _, value := range []byte(prefix) {
			if !probe.Allowed([]byte{value}) {
				t.Fatalf("prefix %q rejected at %q", prefix, value)
			}
			probe.Advance([]byte{value})
		}
		if got := probe.Finished(); got != (prefix[len(prefix)-1] == 'd') {
			t.Fatalf("prefix %q finished = %v", prefix, got)
		}
	}
	for _, input := range []string{"x", "ac e", "ade"} {
		probe := constraint.Clone()
		accepted := true
		for _, value := range []byte(input) {
			if !probe.Allowed([]byte{value}) {
				accepted = false
				break
			}
			probe.Advance([]byte{value})
		}
		if accepted && probe.Finished() {
			t.Fatalf("invalid input %q accepted", input)
		}
	}
}

func TestRegexConstraintUTF8(t *testing.T) {
	constraint, err := NewRegexConstraint(`é+`)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	if !constraint.Allowed([]byte("é")) {
		t.Fatal("valid UTF-8 token rejected")
	}
	constraint.Advance([]byte("é"))
	if !constraint.Finished() {
		t.Fatal("single é should finish")
	}
}

func TestEBNFConstraintGrammar(t *testing.T) {
	constraint, err := NewEBNFConstraint(`start = "a" { "b" } [ "x" ] "c" ;`)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	valid := []string{"ac", "abc", "abbc", "axc", "abbxc"}
	for _, input := range valid {
		probe := constraint.Clone()
		for _, value := range []byte(input) {
			if !probe.Allowed([]byte{value}) {
				t.Fatalf("valid %q rejected at %q", input, value)
			}
			probe.Advance([]byte{value})
		}
		if !probe.Finished() {
			t.Fatalf("valid %q did not finish", input)
		}
	}
	invalid := []string{"a", "ab", "bbc", "acx", "axbc"}
	for _, input := range invalid {
		probe := constraint.Clone()
		accepted := true
		for _, value := range []byte(input) {
			if !probe.Allowed([]byte{value}) {
				accepted = false
				break
			}
			probe.Advance([]byte{value})
		}
		if accepted && probe.Finished() {
			t.Fatalf("invalid %q accepted", input)
		}
	}
}

func TestEBNFConstraintRangesAndRecursion(t *testing.T) {
	constraint, err := NewEBNFConstraint(`start = digit { digit } ; digit = "0".."9" ;`)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	probe := constraint.Clone()
	for _, value := range []byte("12345") {
		if !probe.Allowed([]byte{value}) {
			t.Fatalf("digit %q rejected", value)
		}
		probe.Advance([]byte{value})
	}
	if !probe.Finished() {
		t.Fatal("digit sequence did not finish")
	}
	if _, err := NewEBNFConstraint(`start = start ;`); err == nil {
		t.Fatal("recursive grammar accepted")
	}
}

func schemaTestConstraint(t *testing.T, schema any) *JSONSchemaConstraint {
	t.Helper()
	constraint, err := NewJSONSchemaConstraint(schema)
	if err != nil {
		t.Fatalf("schema: %v", err)
	}
	return constraint
}

func acceptsWhole(constraint Constraint, input string) bool {
	probe := constraint.Clone()
	for _, value := range []byte(input) {
		if !probe.Allowed([]byte{value}) {
			return false
		}
		probe.Advance([]byte{value})
	}
	return probe.Finished()
}

func TestJSONSchemaConstraintObjectAndTypes(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
			"age":  map[string]any{"type": "integer"},
		},
		"required":             []any{"name"},
		"additionalProperties": false,
	}
	constraint := schemaTestConstraint(t, schema)
	if !acceptsWhole(constraint, `{"name":"a","age":1}`) {
		t.Fatal("valid object rejected")
	}
	if acceptsWhole(constraint, `{"age":1}`) {
		t.Fatal("missing required property accepted")
	}
	if acceptsWhole(constraint, `{"name":1}`) {
		t.Fatal("wrong property type accepted")
	}
	if acceptsWhole(constraint, `{"other":1}`) {
		t.Fatal("unknown property accepted")
	}
	if acceptsWhole(constraint, `{"name":"a","age":1.5}`) {
		t.Fatal("fractional integer accepted")
	}
}

func TestJSONSchemaConstraintEnumAndArray(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"color": map[string]any{"type": "string", "enum": []any{"red", "blue"}},
			"tags":  map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
		},
		"required":             []any{"color"},
		"additionalProperties": false,
	}
	constraint := schemaTestConstraint(t, schema)
	if !acceptsWhole(constraint, `{"color":"red","tags":["a","b"]}`) {
		t.Fatal("valid enum/array rejected")
	}
	if acceptsWhole(constraint, `{"color":"green"}`) {
		t.Fatal("invalid enum accepted")
	}
	if acceptsWhole(constraint, `{"color":"red","tags":[1]}`) {
		t.Fatal("wrong array item type accepted")
	}
}

func TestJSONSchemaConstraintAdditionalSchema(t *testing.T) {
	schema := map[string]any{
		"type":                 "object",
		"properties":           map[string]any{},
		"additionalProperties": map[string]any{"type": "integer"},
	}
	constraint := schemaTestConstraint(t, schema)
	if !acceptsWhole(constraint, `{"a":1,"b":2}`) {
		t.Fatal("additionalProperties schema rejected valid object")
	}
	if acceptsWhole(constraint, `{"a":"x"}`) {
		t.Fatal("additionalProperties schema accepted wrong type")
	}
}

func schemaTestTokenizer() *tokenizer.Tokenizer {
	ranks := map[string]int{}
	next := 0
	for _, token := range []string{"{", "}", "[", "]", `"`, ":", ",", "a", "1", "2"} {
		ranks[token] = next
		next++
	}
	return tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
}

func TestEngineJSONSchemaConstraint(t *testing.T) {
	tokenizerImpl := schemaTestTokenizer()
	config := model.Config{
		SequenceLen: 128, VocabSize: tokenizerImpl.VocabSize(), NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	engine := NewEngine(transformer, tokenizerImpl)
	schema := map[string]any{
		"type":                 "object",
		"properties":           map[string]any{"a": map[string]any{"type": "string", "enum": []any{"a"}}},
		"required":             []any{"a"},
		"additionalProperties": false,
	}
	constraint, err := NewJSONSchemaConstraint(schema)
	if err != nil {
		t.Fatalf("schema: %v", err)
	}
	prompt := []int{tokenizerImpl.EncodeSpecial("<|bos|>")}
	got, _ := engine.GenerateBatchWith(prompt, 1, 32, 0, 0, 1, GenerateOptions{Constraint: constraint})
	output := got[0][len(prompt):]
	text := tokenizerImpl.Decode(output)
	if text != "" {
		var decoded map[string]any
		if err := json.Unmarshal([]byte(text), &decoded); err != nil {
			t.Fatalf("generated invalid JSON %q: %v", text, err)
		}
		if _, ok := decoded["a"].(string); !ok {
			t.Fatalf("generated value does not satisfy schema: %q", text)
		}
	}
}

func TestConstraintMaskCacheReusesState(t *testing.T) {
	tokenizerImpl := jsonTestTokenizer()
	constraint := NewJSONConstraint()
	stop := []int{tokenizerImpl.EncodeSpecial("<|assistant_end|>")}
	beforeHits, _, _ := ConstraintMaskCacheStats()
	first := ConstraintMaskCached(tokenizerImpl, constraint, tokenizerImpl.VocabSize(), stop)
	second := ConstraintMaskCached(tokenizerImpl, constraint, tokenizerImpl.VocabSize(), stop)
	afterHits, _, _ := ConstraintMaskCacheStats()
	if afterHits <= beforeHits {
		t.Fatal("mask cache did not record a hit")
	}
	if len(first) != len(second) {
		t.Fatal("cached mask length mismatch")
	}
	for index := range first {
		if first[index] != second[index] {
			t.Fatalf("cached mask mismatch at %d", index)
		}
	}
	constraint.Advance([]byte("{"))
	third := ConstraintMaskCached(tokenizerImpl, constraint, tokenizerImpl.VocabSize(), stop)
	if len(third) != len(first) {
		t.Fatal("advanced mask length mismatch")
	}
	different := false
	for index := range first {
		if first[index] != third[index] {
			different = true
			break
		}
	}
	if !different {
		t.Fatal("mask cache did not distinguish constraint states")
	}
}

func TestConstraintMaskCacheMatchesScanForAllGrammars(t *testing.T) {
	tokenizerImpl := jsonTestTokenizer()
	stop := []int{tokenizerImpl.EncodeSpecial("<|assistant_end|>")}
	schema, err := NewJSONSchemaConstraint(map[string]any{
		"type":                 "object",
		"properties":           map[string]any{"a": map[string]any{"type": "string"}},
		"required":             []any{"a"},
		"additionalProperties": false,
	})
	if err != nil {
		t.Fatalf("schema: %v", err)
	}
	regex, err := NewRegexConstraint(`a(b|c)*d`)
	if err != nil {
		t.Fatalf("regex: %v", err)
	}
	ebnf, err := NewEBNFConstraint(`start = "a" { "b" } "c" ;`)
	if err != nil {
		t.Fatalf("ebnf: %v", err)
	}
	cases := []struct {
		name       string
		constraint Constraint
		advance    string
	}{
		{"json", NewJSONConstraint(), `{"a":`},
		{"schema", schema, `{"a":"x`},
		{"regex", regex, "ab"},
		{"ebnf", ebnf, "abb"},
	}
	for _, testCase := range cases {
		scanned := ConstraintMask(tokenizerImpl, testCase.constraint, tokenizerImpl.VocabSize(), stop)
		cached := ConstraintMaskCached(tokenizerImpl, testCase.constraint, tokenizerImpl.VocabSize(), stop)
		for index := range scanned {
			if scanned[index] != cached[index] {
				t.Fatalf("%s: cached mask mismatch at %d", testCase.name, index)
			}
		}
		testCase.constraint.Advance([]byte(testCase.advance))
		scanned = ConstraintMask(tokenizerImpl, testCase.constraint, tokenizerImpl.VocabSize(), stop)
		cached = ConstraintMaskCached(tokenizerImpl, testCase.constraint, tokenizerImpl.VocabSize(), stop)
		for index := range scanned {
			if scanned[index] != cached[index] {
				t.Fatalf("%s: advanced cached mask mismatch at %d", testCase.name, index)
			}
		}
	}
}

func TestEngineRegexAndEBNFConstraints(t *testing.T) {
	tokenizerImpl := jsonTestTokenizer()
	config := model.Config{
		SequenceLen: 16, VocabSize: tokenizerImpl.VocabSize(), NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	engine := NewEngine(transformer, tokenizerImpl)
	prompt := []int{tokenizerImpl.EncodeSpecial("<|bos|>")}

	regex, err := NewRegexConstraint(`ab`)
	if err != nil {
		t.Fatalf("regex: %v", err)
	}
	got, _ := engine.GenerateBatchWith(prompt, 1, 8, 0, 0, 1, GenerateOptions{Constraint: regex})
	if text := tokenizerImpl.Decode(got[0][len(prompt):]); text != "ab" {
		t.Fatalf("regex output = %q, want ab", text)
	}

	ebnf, err := NewEBNFConstraint(`start = "a" "b" ;`)
	if err != nil {
		t.Fatalf("ebnf: %v", err)
	}
	got, _ = engine.GenerateBatchWith(prompt, 1, 8, 0, 0, 1, GenerateOptions{Constraint: ebnf})
	if text := tokenizerImpl.Decode(got[0][len(prompt):]); text != "ab" {
		t.Fatalf("ebnf output = %q, want ab", text)
	}
}
