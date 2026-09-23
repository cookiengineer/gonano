package gonano_test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cookiengineer/gonano/bank"
	"github.com/cookiengineer/gonano/data"
	"github.com/cookiengineer/gonano/inference"
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/model/checkpoint"
	"github.com/cookiengineer/gonano/router"
	"github.com/cookiengineer/gonano/server"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
	"github.com/cookiengineer/gonano/trainer"
)

// e2eDomains are the dataset directories and their router label order.
var e2eDomains = []string{"physics", "math", "cooking", "trivia"}

// e2eTokenizer loads the bundled Markdown BPE tokenizer. Subword tokens make
// the n-gram router meaningful; a byte-level tokenizer degenerates to
// character n-grams and conflates topics that share common words.
func e2eTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	tok, err := tokenizer.LoadTokenizer(filepath.Join("tokenizer", "defaults", "markdown.json"))
	if err != nil {
		t.Fatalf("load bundled tokenizer: %v", err)
	}
	return tok
}

func e2eToI32(ids []int) []int32 {
	converted := make([]int32, len(ids))
	for index, value := range ids {
		converted[index] = int32(value)
	}
	return converted
}

// e2eTrainModel trains a tiny LM checkpoint on one markdown corpus.
func e2eTrainModel(t *testing.T, tok *tokenizer.Tokenizer, dir, path string, steps int, seed uint64) {
	t.Helper()
	configuration := model.Config{
		SequenceLen: 128, VocabSize: tok.VocabSize(), NumLayer: 1, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	transformer := model.NewTransformer(configuration)
	transformer.InitWeights(tensors.NewRNG(seed))
	groups := transformer.SetupOptimizer(0.01, 0.1, 0.02, 0.28, 0.5, false)
	training := trainer.NewTrainer(transformer, groups, 1)
	source := data.NewMarkdownSource(dir, 16)
	loader := data.NewPretrainLoader(tok, 1, configuration.SequenceLen, func() ([]string, data.State) { return source.Next() }, 100)
	for step := 0; step < steps; step++ {
		inputs, targets, _ := loader.Next()
		training.TrainStep(inputs, targets)
		training.StepOptimizer(step, steps)
	}
	if err := checkpoint.Save(path, checkpoint.Meta{Step: steps, ModelConfig: configuration}, transformer.NamedParameters()); err != nil {
		t.Fatalf("save domain model %s: %v", path, err)
	}
}

// e2eCollectExamples samples labeled token sequences from a markdown corpus.
func e2eCollectExamples(tok *tokenizer.Tokenizer, dir string, label, limit, maxLen int) []router.Example {
	source := data.NewMarkdownSource(dir, 16)
	examples := make([]router.Example, 0, limit)
	guard := 0
	for len(examples) < limit && guard < limit*8 {
		guard++
		documents, _ := source.Next()
		if len(documents) == 0 {
			break
		}
		for _, document := range documents {
			tokens := tok.Encode(document)
			if len(tokens) == 0 {
				continue
			}
			if len(tokens) > maxLen {
				tokens = tokens[:maxLen]
			}
			examples = append(examples, router.Example{Tokens: e2eToI32(tokens), Label: label})
			if len(examples) >= limit {
				break
			}
		}
	}
	return examples
}

// e2eCombinedProvider cycles documents from every domain directory.
func e2eCombinedProvider(dirs []string) data.DocProvider {
	sources := make([]func() ([]string, data.State), len(dirs))
	for index, dir := range dirs {
		source := data.NewMarkdownSource(dir, 16)
		sources[index] = source.Next
	}
	cursor := 0
	return func() ([]string, data.State) {
		for tries := 0; tries < len(sources); tries++ {
			documents, state := sources[cursor%len(sources)]()
			cursor++
			if len(documents) > 0 {
				return documents, state
			}
		}
		return nil, data.State{}
	}
}

// e2eTestBank trains the domain models, encoder, and router from the datasets/
// corpus and returns them for the end-to-end assertions.
func e2eTestBank(t *testing.T) (tok *tokenizer.Tokenizer, work string, domains []string, checkpointPaths map[string]string, trained *router.Router, routerPath string) {
	t.Helper()
	tok = e2eTokenizer(t)
	work = t.TempDir()
	domains = e2eDomains
	checkpointPaths = make(map[string]string, len(domains))
	examples := make([]router.Example, 0, len(domains)*200)

	dirs := make([]string, len(domains))
	for index, domain := range domains {
		dir := filepath.Join("datasets", domain)
		dirs[index] = dir
		path := filepath.Join(work, domain+".gn")
		e2eTrainModel(t, tok, dir, path, 40, uint64(index+1))
		checkpointPaths[domain] = path
		examples = append(examples, e2eCollectExamples(tok, dir, index, 200, 256)...)
	}

	// A small encoder LM trained on all domains; its hidden states feed the
	// router's linear probe.
	encoderConfig := model.ConfigForDepth(2, tok.VocabSize(), 64, 64, 256, "SSSL")
	encoder := model.NewTransformer(encoderConfig)
	encoder.InitWeights(tensors.NewRNG(1))
	groups := encoder.SetupOptimizer(0.01, 0.1, 0.02, 0.28, 0.5, false)
	training := trainer.NewTrainer(encoder, groups, 1)
	loader := data.NewPretrainLoader(tok, 1, encoderConfig.SequenceLen, e2eCombinedProvider(dirs), 100)
	for step := 0; step < 200; step++ {
		inputs, targets, _ := loader.Next()
		training.TrainStep(inputs, targets)
		training.StepOptimizer(step, 200)
	}
	encoderPath := filepath.Join(work, "encoder.gn")
	if err := checkpoint.Save(encoderPath, checkpoint.Meta{Step: 200, ModelConfig: encoderConfig}, encoder.NamedParameters()); err != nil {
		t.Fatalf("save encoder: %v", err)
	}

	teacher := router.NewTransformerClassifier(encoder, len(domains), 256)
	teacher.Train(examples, router.HeadConfig{Epochs: 120, LearningRate: 0.05, BatchSize: 16, Seed: 3})

	// The per-domain directories provide ground-truth labels, so the fast path
	// is trained supervised. Distillation is verified separately because it is
	// the fallback when only the teacher's labels are available.
	ngram := router.TrainNgram(examples, len(domains), router.NgramConfig{Epochs: 40, Buckets: 1 << 18, Seed: 7})
	distilled := router.DistillNgram(teacher, examples, router.NgramConfig{Epochs: 40, Buckets: 1 << 18, Seed: 7})
	if fidelity := router.DistillAccuracy(teacher, distilled, examples); fidelity < 0.8 {
		t.Fatalf("distillation fidelity %.3f < 0.8", fidelity)
	}

	trained = &router.Router{Domains: domains, Ngram: ngram, Transformer: teacher, EncoderPath: encoderPath}
	routerPath = filepath.Join(work, "router.gn")
	if err := trained.Save(routerPath); err != nil {
		t.Fatalf("save router: %v", err)
	}
	return tok, work, domains, checkpointPaths, trained, routerPath
}

// TestDomainBankEndToEnd trains each domain model from datasets/, trains the
// meta-router, reloads the bundle, routes held-out prompts, builds the bank,
// blends the selected domains, and exercises the server in bank mode.
func TestDomainBankEndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping domain bank end-to-end test in short mode")
	}
	tok, work, domains, checkpointPaths, _, routerPath := e2eTestBank(t)

	loaded, err := router.LoadRouterAuto(routerPath)
	if err != nil {
		t.Fatalf("LoadRouterAuto: %v", err)
	}
	if len(loaded.Domains) != len(domains) {
		t.Fatalf("domains = %v, want %v", loaded.Domains, domains)
	}

	direct := []struct{ text, want string }{
		{"newton laws describe force and momentum", "physics"},
		{"entropy and heat flow in an isolated system", "physics"},
		{"solve a system of linear equations", "math"},
		{"the derivative and integral of a function", "math"},
		{"knead the dough and bake the bread", "cooking"},
		{"simmer onions and season the soup", "cooking"},
		{"a group of flamingos is called a flamboyance", "trivia"},
		{"vanilla comes from the orchid family", "trivia"},
	}
	for _, probe := range direct {
		scored := loaded.Route(e2eToI32(tok.Encode(probe.text)))
		if len(scored) == 0 || scored[0].Domain != probe.want {
			t.Fatalf("route(%q) = %v, want top-1 %q", probe.text, scored, probe.want)
		}
	}

	overlaps := []struct {
		text    string
		allowed []string
	}{
		{"the rate of change of energy in a field", []string{"physics", "math"}},
		{"vectors describe velocity and force in space", []string{"math", "physics"}},
		{"heat and temperature while roasting food", []string{"physics", "cooking"}},
	}
	for _, probe := range overlaps {
		scored := loaded.Route(e2eToI32(tok.Encode(probe.text)))
		if len(scored) == 0 {
			t.Fatalf("route(%q) returned no scores", probe.text)
		}
		if !containsString(probe.allowed, scored[0].Domain) {
			t.Fatalf("overlap route(%q) = %q, want one of %v", probe.text, scored[0].Domain, probe.allowed)
		}
	}

	manifestDomains := make([]bank.Domain, len(domains))
	for index, domain := range domains {
		manifestDomains[index] = bank.Domain{ID: domain, Checkpoint: checkpointPaths[domain]}
	}
	manifest := &bank.Manifest{Version: 1, Router: routerPath, Domains: manifestDomains}
	manifestPath := filepath.Join(work, "bank.json")
	if err := manifest.Save(manifestPath); err != nil {
		t.Fatalf("save manifest: %v", err)
	}
	reloaded, err := bank.LoadManifest(manifestPath)
	if err != nil {
		t.Fatalf("LoadManifest: %v", err)
	}

	domainBank := bank.NewBank(reloaded, tok, bank.BankOptions{MaxModels: 2})

	// Route a physics prompt, acquire the selected domains, and blend.
	scored := loaded.TopDomains(e2eToI32(tok.Encode("momentum and collisions of particles")), 2, 0)
	if len(scored) == 0 {
		t.Fatal("router returned no domains")
	}
	ids := make([]string, len(scored))
	weights := make([]inference.ModelWeight, len(scored))
	for index, score := range scored {
		ids[index] = score.Domain
	}
	models, loadedIDs, err := domainBank.AcquireMany(ids)
	if err != nil {
		t.Fatalf("AcquireMany: %v", err)
	}
	for index := range models {
		weights[index] = inference.ModelWeight{Model: models[index], Weight: scored[index].Value, Domain: loadedIDs[index]}
	}
	ensemble := inference.NewEnsemble(weights, tok)
	results, _ := ensemble.GenerateBatch([]int{tok.BOSTokenID(), 10, 20}, 1, 4, 0, 0, 42)
	if len(results) != 1 || len(results[0]) < 3 {
		t.Fatalf("unexpected blended generation %v", results)
	}

	// HTTP smoke test in bank mode.
	firstModel, err := domainBank.Acquire(domains[0])
	if err != nil {
		t.Fatalf("acquire first domain: %v", err)
	}
	httpServer := server.NewServer(firstModel, tok, nil, "gonano")
	httpServer.Bank = domainBank
	httpServer.Router = loaded
	httpServer.MaxDomains = 2
	httpServer.MinDomainScore = 0
	httpServer.RouteScope = "last-turn"

	post := func(payload map[string]any) *httptest.ResponseRecorder {
		body, _ := json.Marshal(payload)
		request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
		recorder := httptest.NewRecorder()
		httpServer.Handler().ServeHTTP(recorder, request)
		if recorder.Code != http.StatusOK {
			t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
		}
		return recorder
	}

	// Single-turn physics request reports the routed domain.
	physics := post(map[string]any{
		"model":      "gonano",
		"max_tokens": 4,
		"messages":   []map[string]string{{"role": "user", "content": "why do waves carry energy"}},
	})
	if header := physics.Header().Get("X-Gonano-Domains"); !strings.Contains(header, "physics") {
		t.Fatalf("X-Gonano-Domains = %q, want it to contain physics", header)
	}
	var response struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(physics.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v (body %s)", err, physics.Body.String())
	}
	if len(response.Choices) == 0 {
		t.Fatalf("no choices in response: %s", physics.Body.String())
	}

	// Per-turn re-routing: a cooking final turn must override the physics
	// context of the earlier turns.
	multiTurn := post(map[string]any{
		"model":      "gonano",
		"max_tokens": 2,
		"messages": []map[string]string{
			{"role": "user", "content": "newton laws describe force and momentum"},
			{"role": "assistant", "content": "momentum is conserved"},
			{"role": "user", "content": "knead the dough and bake the bread"},
		},
	})
	if header := multiTurn.Header().Get("X-Gonano-Domains"); !strings.Contains(header, "cooking") {
		t.Fatalf("per-turn routing header = %q, want cooking", header)
	}

	// A requested model that names a domain pins routing.
	pinned := post(map[string]any{
		"model":      "math",
		"max_tokens": 2,
		"messages":   []map[string]string{{"role": "user", "content": "knead the dough and bake the bread"}},
	})
	if header := pinned.Header().Get("X-Gonano-Domains"); header != "math" {
		t.Fatalf("pinned routing header = %q, want math", header)
	}
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
