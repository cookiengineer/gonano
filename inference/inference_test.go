package inference

import (
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
)

func TestSampleNextTokenArgmax(test *testing.T) {
	logits := tensors.NewWithData([]int{2, 4}, []float32{
		1, 9, 3, 5,
		7, 2, 8, 1,
	})
	got := SampleNextToken(logits, tensors.NewRNG(1), 0, 0)
	if got.Data[0] != 1 || got.Data[1] != 2 {
		test.Fatalf("argmax = %v, want [1 2]", got.Data)
	}
}

func TestSampleNextTokenTopK(test *testing.T) {
	logits := tensors.NewWithData([]int{1, 8}, []float32{1, 2, 3, 4, 5, 6, 7, 8})
	randomGenerator := tensors.NewRNG(42)
	got := SampleNextToken(logits, randomGenerator, 1.0, 3)
	// Sampled token must be within the top 3 (indices 5, 6, 7).
	if int(got.Data[0]) < 5 {
		test.Fatalf("top-k sample = %d, want in [5,7]", got.Data[0])
	}
}

func TestUseCalculatorArithmetic(test *testing.T) {
	cases := map[string]string{
		"1+2*3":   "7",
		"(1+2)*3": "9",
		"10/4":    "2.5",
		"1,000+2": "1002",
		"7-3":     "4",
	}
	for expr, want := range cases {
		got := UseCalculator(expr)
		if got == nil || *got != want {
			test.Errorf("UseCalculator(%q) = %v, want %q", expr, got, want)
		}
	}
}

func TestUseCalculatorCount(test *testing.T) {
	got := UseCalculator(`"hello world".count("l")`)
	if got == nil || *got != "3" {
		test.Fatalf("count = %v, want 3", got)
	}
}

func TestUseCalculatorRejects(test *testing.T) {
	for _, expr := range []string{
		"2**3",
		"__import__('os')",
		"open('file')",
		"1+abc",
	} {
		if got := UseCalculator(expr); got != nil {
			test.Errorf("UseCalculator(%q) = %v, want nil", expr, got)
		}
	}
}

// echoTool is a tiny custom tool used to verify the registry is extensible
// beyond the built-in calculator.
type echoTool struct{}

func (echoTool) Name() string                    { return "echo" }
func (echoTool) Call(expr string) (string, bool) { return expr, true }

func TestToolRegistry(test *testing.T) {
	registry := NewCalculator()
	if got, ok := registry.Execute("1+2*3"); !ok || got != "7" {
		test.Fatalf("calculator Execute = %q, %v; want 7, true", got, ok)
	}
	if got, ok := registry.Execute(`"hello world".count("l")`); !ok || got != "3" {
		test.Fatalf("count Execute = %q, %v; want 3, true", got, ok)
	}
	// Unhandled expression: only the calculator is registered.
	if _, ok := registry.Execute("hello"); ok {
		test.Fatal("calculator should not handle plain text")
	}

	// Register a custom tool; it is consulted after the calculator.
	registry.Register(echoTool{})
	if got, ok := registry.Execute("hello"); !ok || got != "hello" {
		test.Fatalf("custom tool = %q, %v; want hello, true", got, ok)
	}
}

func testEngineModel() (*model.Transformer, *tokenizer.Tokenizer) {
	config := model.Config{
		SequenceLen: 16, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	ranks := make(map[string]int, 256)
	for index := 0; index < 256; index++ {
		ranks[string([]byte{byte(index)})] = index
	}
	tokenizerImpl := tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
	return transformer, tokenizerImpl
}

func naiveGenerate(transformer *model.Transformer, tokens []int, maxTokens int, temperature float32, topK int, seed uint64) []int {
	randomGenerator := tensors.NewRNG(seed)
	ids := append([]int(nil), tokens...)
	vocab := transformer.Config.VocabSize
	for index := 0; index < maxTokens; index++ {
		inputIDs := tensors.NewInt32sWithData([]int{1, len(ids)}, toI32(ids))
		logits := transformer.Forward(inputIDs, nil)
		flat := logits.Reshape(len(ids), vocab)
		row := flat.Data[(len(ids)-1)*vocab : len(ids)*vocab]
		nextToken := sampleRow(row, randomGenerator, temperature, topK)
		ids = append(ids, nextToken)
	}
	return ids
}

func testGQAEngineModel() (*model.Transformer, *tokenizer.Tokenizer) {
	config := model.Config{
		SequenceLen: 16, VocabSize: 32, NumLayer: 2, NumHead: 4, NumKVHead: 1,
		EmbedDim: 32, WindowPattern: "L",
	}
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	ranks := make(map[string]int, 256)
	for index := 0; index < 256; index++ {
		ranks[string([]byte{byte(index)})] = index
	}
	tokenizerImpl := tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
	return transformer, tokenizerImpl
}

// TestEngineMatchesNaiveGenerateGQA exercises the grouped-query KV-cache path,
// where several query heads share one KV head (and therefore one cache slot).
func testCompressedEngineModel() (*model.Transformer, *tokenizer.Tokenizer) {
	config := model.Config{
		SequenceLen: 16, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L", CompressionRatio: 2,
	}
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	ranks := make(map[string]int, 256)
	for index := 0; index < 256; index++ {
		ranks[string([]byte{byte(index)})] = index
	}
	tokenizerImpl := tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
	return transformer, tokenizerImpl
}

// compressedReference greedily generates by recomputing the full compressed
// forward from scratch at every step, providing an independent reference for
// the incremental compressed KV cache.
func compressedReference(transformer *model.Transformer, tokens []int, maxTokens int) []int {
	config := transformer.Config
	headDim := config.HeadDim()
	ids := append([]int(nil), tokens...)
	for step := 0; step < maxTokens; step++ {
		cache := model.NewKVBuffer(1, len(ids), config.NumLayer, config.NumKVHead, headDim)
		if ratio := config.Compression(); ratio > 1 {
			cache.EnableCompression(ratio, config.EmbedDim, config.NumKVHead*headDim, len(ids)/ratio+1)
		}
		inputIDs := tensors.NewInt32sWithData([]int{1, len(ids)}, toI32(ids))
		logits := transformer.Forward(inputIDs, cache)
		flat := logits.Reshape(len(ids), config.VocabSize)
		row := flat.Data[(len(ids)-1)*config.VocabSize:]
		best := 0
		for index := 1; index < config.VocabSize; index++ {
			if row[index] > row[best] {
				best = index
			}
		}
		ids = append(ids, best)
	}
	return ids
}

func testSparseCompressedEngineModel() (*model.Transformer, *tokenizer.Tokenizer) {
	config := model.Config{
		SequenceLen: 16, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L", CompressionRatio: 2, SparseTopK: 2, IndexerDim: 4,
	}
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	ranks := make(map[string]int, 256)
	for index := 0; index < 256; index++ {
		ranks[string([]byte{byte(index)})] = index
	}
	tokenizerImpl := tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
	return transformer, tokenizerImpl
}

// testReuseEngineModel exercises cross-layer compressed reuse during
// autoregressive inference: layer 0 is full, layer 1 reindexes, and layers 2-3
// reuse the shared compressed KV and selection.
func testReuseEngineModel() (*model.Transformer, *tokenizer.Tokenizer) {
	config := model.Config{
		SequenceLen: 16, VocabSize: 32, NumLayer: 4, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L", CompressionRatio: 2, SparseTopK: 2,
		IndexerDim: 4, ReusePattern: "FRUU",
	}
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	ranks := make(map[string]int, 256)
	for index := 0; index < 256; index++ {
		ranks[string([]byte{byte(index)})] = index
	}
	tokenizerImpl := tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
	return transformer, tokenizerImpl
}

// testDenseReuseEngineModel exercises dense (non-sparse) cross-layer reuse,
// covering the dense compressed branch of the reuse inference path.
func testDenseReuseEngineModel() (*model.Transformer, *tokenizer.Tokenizer) {
	config := model.Config{
		SequenceLen: 16, VocabSize: 32, NumLayer: 4, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L", CompressionRatio: 2, ReusePattern: "FRUU",
	}
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	ranks := make(map[string]int, 256)
	for index := 0; index < 256; index++ {
		ranks[string([]byte{byte(index)})] = index
	}
	tokenizerImpl := tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
	return transformer, tokenizerImpl
}

func TestEngineMatchesRecomputeReuseDense(test *testing.T) {
	transformer, tokenizerImpl := testDenseReuseEngineModel()
	engine := NewEngine(transformer, tokenizerImpl)
	prompt := []int{1, 5, 2, 8, 3, 7, 4, 6}

	want := compressedReference(transformer, prompt, 6)
	got, _ := engine.GenerateBatch(prompt, 1, 6, 0, 0, 7)

	limit := len(want)
	if len(got[0]) < limit {
		limit = len(got[0])
	}
	for index := 0; index < limit; index++ {
		if got[0][index] != want[index] {
			test.Fatalf("token %d: engine=%d recompute=%d", index, got[0][index], want[index])
		}
	}
}

func TestEngineMatchesRecomputeReuse(test *testing.T) {
	transformer, tokenizerImpl := testReuseEngineModel()
	engine := NewEngine(transformer, tokenizerImpl)
	prompt := []int{1, 5, 2, 8, 3, 7, 4, 6}

	want := compressedReference(transformer, prompt, 6)
	got, _ := engine.GenerateBatch(prompt, 1, 6, 0, 0, 7)

	limit := len(want)
	if len(got[0]) < limit {
		limit = len(got[0])
	}
	for index := 0; index < limit; index++ {
		if got[0][index] != want[index] {
			test.Fatalf("token %d: engine=%d recompute=%d", index, got[0][index], want[index])
		}
	}
}

func TestEngineMatchesRecomputeSparseCompressed(test *testing.T) {
	transformer, tokenizerImpl := testSparseCompressedEngineModel()
	engine := NewEngine(transformer, tokenizerImpl)
	prompt := []int{1, 5, 2, 8, 3, 7, 4, 6}

	want := compressedReference(transformer, prompt, 6)
	got, _ := engine.GenerateBatch(prompt, 1, 6, 0, 0, 7)

	limit := len(want)
	if len(got[0]) < limit {
		limit = len(got[0])
	}
	for index := 0; index < limit; index++ {
		if got[0][index] != want[index] {
			test.Fatalf("token %d: engine=%d recompute=%d", index, got[0][index], want[index])
		}
	}
}

func TestEngineMatchesRecomputeHierarchicalSparse(test *testing.T) {
	config := model.Config{
		SequenceLen: 16, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L", CompressionRatio: 2, SparseTopK: 2,
		IndexerDim: 4, IndexerPool: 2, IndexerCandidates: 1000,
	}
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	ranks := make(map[string]int, 256)
	for index := 0; index < 256; index++ {
		ranks[string([]byte{byte(index)})] = index
	}
	tokenizerImpl := tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
	engine := NewEngine(transformer, tokenizerImpl)
	prompt := []int{1, 5, 2, 8, 3, 7, 4, 6}

	want := compressedReference(transformer, prompt, 6)
	got, _ := engine.GenerateBatch(prompt, 1, 6, 0, 0, 7)

	limit := len(want)
	if len(got[0]) < limit {
		limit = len(got[0])
	}
	for index := 0; index < limit; index++ {
		if got[0][index] != want[index] {
			test.Fatalf("token %d: engine=%d recompute=%d", index, got[0][index], want[index])
		}
	}
}

func TestEngineMatchesRecomputeCompressed(test *testing.T) {
	transformer, tokenizerImpl := testCompressedEngineModel()
	engine := NewEngine(transformer, tokenizerImpl)
	prompt := []int{1, 5, 2, 8}

	want := compressedReference(transformer, prompt, 6)
	got, _ := engine.GenerateBatch(prompt, 1, 6, 0, 0, 7)

	limit := len(want)
	if len(got[0]) < limit {
		limit = len(got[0])
	}
	for index := 0; index < limit; index++ {
		if got[0][index] != want[index] {
			test.Fatalf("token %d: engine=%d recompute=%d", index, got[0][index], want[index])
		}
	}
}

func TestEngineMatchesNaiveGenerateGQA(test *testing.T) {
	transformer, tokenizerImpl := testGQAEngineModel()
	engine := NewEngine(transformer, tokenizerImpl)
	prompt := []int{1, 5, 2, 8}

	want := naiveGenerate(transformer, prompt, 6, 0, 0, 7)
	got, _ := engine.GenerateBatch(prompt, 1, 6, 0, 0, 7)

	for index := range want {
		if got[0][index] != want[index] {
			test.Fatalf("token %d: engine=%d naive=%d", index, got[0][index], want[index])
		}
	}
}

func TestEngineMatchesNaiveGenerate(test *testing.T) {
	transformer, tokenizerImpl := testEngineModel()
	engine := NewEngine(transformer, tokenizerImpl)
	prompt := []int{1, 5, 2, 8}

	want := naiveGenerate(transformer, prompt, 6, 0, 0, 7)
	got, _ := engine.GenerateBatch(prompt, 1, 6, 0, 0, 7)

	for index := range want {
		if got[0][index] != want[index] {
			test.Fatalf("token %d: engine=%d naive=%d", index, got[0][index], want[index])
		}
	}
}

func TestEngineGenerateBatchShapes(test *testing.T) {
	transformer, tokenizerImpl := testEngineModel()
	engine := NewEngine(transformer, tokenizerImpl)
	prompt := []int{1, 2, 3}
	results, masks := engine.GenerateBatch(prompt, 3, 4, 0.0, 0, 42)
	if len(results) != 3 || len(masks) != 3 {
		test.Fatalf("expected 3 rows, got %d", len(results))
	}
	for index := 0; index < 3; index++ {
		if len(results[index]) != len(masks[index]) {
			test.Fatalf("row %d: results/masks length mismatch", index)
		}
		// Results start with the prompt.
		for promptIndex := 0; promptIndex < len(prompt); promptIndex++ {
			if results[index][promptIndex] != prompt[promptIndex] {
				test.Fatalf("row %d does not start with prompt", index)
			}
		}
	}
}

func TestMeasure(test *testing.T) {
	transformer, tokenizerImpl := testEngineModel()
	engine := NewEngine(transformer, tokenizerImpl)
	measurement := Measure(engine, []int{1, 2, 3}, 2, 5, 0.0, 0, 42)
	if measurement.NumTokens != 5 {
		test.Fatalf("NumTokens = %d, want 5", measurement.NumTokens)
	}
	if measurement.TTFT < 0 {
		test.Fatalf("TTFT = %v, want >= 0", measurement.TTFT)
	}
	if len(measurement.StepTimes) != 4 {
		test.Fatalf("StepTimes = %d, want 4", len(measurement.StepTimes))
	}
}

func TestMeasureTrafficAccounting(test *testing.T) {
	transformer, tokenizerImpl := testEngineModel()
	engine := NewEngine(transformer, tokenizerImpl)
	prompt := []int{1, 2, 3}
	numSamples := 2
	measurement := Measure(engine, prompt, numSamples, 5, 0.0, 0, 42)

	if measurement.DecodeSteps != len(measurement.StepTimes) {
		test.Fatalf("DecodeSteps = %d, want %d", measurement.DecodeSteps, len(measurement.StepTimes))
	}
	wantWeight := int64(measurement.DecodeSteps) * int64(transformer.WeightReadBytes())
	if measurement.WeightBytes != wantWeight {
		test.Fatalf("WeightBytes = %d, want %d", measurement.WeightBytes, wantWeight)
	}
	var wantKV int64
	for step := 0; step < measurement.DecodeSteps; step++ {
		wantKV += int64(numSamples) * int64(transformer.KVReadBytes(len(prompt)+1+step))
	}
	if measurement.KVBytes != wantKV {
		test.Fatalf("KVBytes = %d, want %d", measurement.KVBytes, wantKV)
	}
	if measurement.KVBytes <= 0 || measurement.WeightBytes <= 0 {
		test.Fatalf("traffic accounting must be positive")
	}
}
