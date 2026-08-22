package infer

import (
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensor"
	"github.com/cookiengineer/gonano/tokenizer"
)

func TestSampleNextTokenArgmax(t *testing.T) {
	logits := tensor.NewWithData([]int{2, 4}, []float32{
		1, 9, 3, 5,
		7, 2, 8, 1,
	})
	got := SampleNextToken(logits, tensor.NewRNG(1), 0, 0)
	if got.Data[0] != 1 || got.Data[1] != 2 {
		t.Fatalf("argmax = %v, want [1 2]", got.Data)
	}
}

func TestSampleNextTokenTopK(t *testing.T) {
	logits := tensor.NewWithData([]int{1, 8}, []float32{1, 2, 3, 4, 5, 6, 7, 8})
	rng := tensor.NewRNG(42)
	got := SampleNextToken(logits, rng, 1.0, 3)
	// Sampled token must be within the top 3 (indices 5, 6, 7).
	if int(got.Data[0]) < 5 {
		t.Fatalf("top-k sample = %d, want in [5,7]", got.Data[0])
	}
}

func TestUseCalculatorArithmetic(t *testing.T) {
	cases := map[string]string{
		"1+2*3":    "7",
		"(1+2)*3":  "9",
		"10/4":     "2.5",
		"1,000+2":  "1002",
		"7-3":      "4",
	}
	for expr, want := range cases {
		got := UseCalculator(expr)
		if got == nil || *got != want {
			t.Errorf("UseCalculator(%q) = %v, want %q", expr, got, want)
		}
	}
}

func TestUseCalculatorCount(t *testing.T) {
	got := UseCalculator(`"hello world".count("l")`)
	if got == nil || *got != "3" {
		t.Fatalf("count = %v, want 3", got)
	}
}

func TestUseCalculatorRejects(t *testing.T) {
	for _, expr := range []string{
		"2**3",
		"__import__('os')",
		"open('file')",
		"1+abc",
	} {
		if got := UseCalculator(expr); got != nil {
			t.Errorf("UseCalculator(%q) = %v, want nil", expr, got)
		}
	}
}

// echoTool is a tiny custom tool used to verify the registry is extensible
// beyond the built-in calculator.
type echoTool struct{}

func (echoTool) Name() string                     { return "echo" }
func (echoTool) Call(expr string) (string, bool)  { return expr, true }

func TestToolRegistry(t *testing.T) {
	reg := NewCalculator()
	if got, ok := reg.Execute("1+2*3"); !ok || got != "7" {
		t.Fatalf("calculator Execute = %q, %v; want 7, true", got, ok)
	}
	if got, ok := reg.Execute(`"hello world".count("l")`); !ok || got != "3" {
		t.Fatalf("count Execute = %q, %v; want 3, true", got, ok)
	}
	// Unhandled expression: only the calculator is registered.
	if _, ok := reg.Execute("hello"); ok {
		t.Fatal("calculator should not handle plain text")
	}

	// Register a custom tool; it is consulted after the calculator.
	reg.Register(echoTool{})
	if got, ok := reg.Execute("hello"); !ok || got != "hello" {
		t.Fatalf("custom tool = %q, %v; want hello, true", got, ok)
	}
}

func testEngineModel() (*model.Transformer, *tokenizer.Tokenizer) {
	cfg := model.Config{
		SequenceLen: 16, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	m := model.NewTransformer(cfg)
	m.InitWeights(tensor.NewRNG(42))
	ranks := make(map[string]int, 256)
	for i := 0; i < 256; i++ {
		ranks[string([]byte{byte(i)})] = i
	}
	tok := tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
	return m, tok
}

func naiveGenerate(m *model.Transformer, tokens []int, maxTokens int, temperature float32, topK int, seed uint64) []int {
	rng := tensor.NewRNG(seed)
	ids := append([]int(nil), tokens...)
	vocab := m.Config.VocabSize
	for i := 0; i < maxTokens; i++ {
		idx := tensor.NewInt32sWithData([]int{1, len(ids)}, toI32(ids))
		logits := m.Forward(idx, nil)
		flat := logits.Reshape(len(ids), vocab)
		row := flat.Data[(len(ids)-1)*vocab : len(ids)*vocab]
		next := sampleRow(row, rng, temperature, topK)
		ids = append(ids, next)
	}
	return ids
}

func TestEngineMatchesNaiveGenerate(t *testing.T) {
	m, tok := testEngineModel()
	engine := NewEngine(m, tok)
	prompt := []int{1, 5, 2, 8}

	want := naiveGenerate(m, prompt, 6, 0, 0, 7)
	got, _ := engine.GenerateBatch(prompt, 1, 6, 0, 0, 7)

	for i := range want {
		if got[0][i] != want[i] {
			t.Fatalf("token %d: engine=%d naive=%d", i, got[0][i], want[i])
		}
	}
}

func TestEngineGenerateBatchShapes(t *testing.T) {
	m, tok := testEngineModel()
	engine := NewEngine(m, tok)
	prompt := []int{1, 2, 3}
	results, masks := engine.GenerateBatch(prompt, 3, 4, 0.0, 0, 42)
	if len(results) != 3 || len(masks) != 3 {
		t.Fatalf("expected 3 rows, got %d", len(results))
	}
	for i := 0; i < 3; i++ {
		if len(results[i]) != len(masks[i]) {
			t.Fatalf("row %d: results/masks length mismatch", i)
		}
		// Results start with the prompt.
		for j := 0; j < len(prompt); j++ {
			if results[i][j] != prompt[j] {
				t.Fatalf("row %d does not start with prompt", i)
			}
		}
	}
}

func TestMeasure(t *testing.T) {
	m, tok := testEngineModel()
	engine := NewEngine(m, tok)
	meas := Measure(engine, []int{1, 2, 3}, 2, 5, 0.0, 0, 42)
	if meas.NumTokens != 5 {
		t.Fatalf("NumTokens = %d, want 5", meas.NumTokens)
	}
	if meas.TTFT < 0 {
		t.Fatalf("TTFT = %v, want >= 0", meas.TTFT)
	}
	if len(meas.StepTimes) != 4 {
		t.Fatalf("StepTimes = %d, want 4", len(meas.StepTimes))
	}
}
