package bank

import (
	"path/filepath"
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/model/checkpoint"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
)

func byteTokenizer() *tokenizer.Tokenizer {
	ranks := make(map[string]int, 256)
	for index := 0; index < 256; index++ {
		ranks[string([]byte{byte(index)})] = index
	}
	return tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
}

func writeModelCheckpoint(t *testing.T, path string, vocab int) {
	t.Helper()
	configuration := model.Config{
		SequenceLen: 8, VocabSize: vocab, NumLayer: 1, NumHead: 2, NumKVHead: 2,
		EmbedDim: 16, WindowPattern: "L",
	}
	transformer := model.NewTransformer(configuration)
	transformer.InitWeights(tensors.NewRNG(1))
	if err := checkpoint.Save(path, checkpoint.Meta{Step: 0, ModelConfig: configuration}, transformer.NamedParameters()); err != nil {
		t.Fatalf("save checkpoint: %v", err)
	}
}

func TestManifestRoundTrip(t *testing.T) {
	manifest := &Manifest{
		Version:   1,
		Tokenizer: "/tmp/tokenizer.json",
		Domains: []Domain{
			{ID: "physics", Name: "Physics", Checkpoint: "/tmp/physics.gn", DataDirs: []string{"/data/physics"}},
			{ID: "math", Checkpoint: "/tmp/math.gn"},
		},
	}
	path := filepath.Join(t.TempDir(), "bank.json")
	if err := manifest.Save(path); err != nil {
		t.Fatalf("save: %v", err)
	}
	loaded, err := LoadManifest(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(loaded.Domains) != 2 || loaded.Domains[0].ID != "physics" || loaded.Tokenizer != manifest.Tokenizer {
		t.Fatalf("manifest not restored: %+v", loaded)
	}
	if _, ok := loaded.DomainByID("math"); !ok {
		t.Fatal("DomainByID(math) failed")
	}
}

func TestManifestRejectsDuplicateDomain(t *testing.T) {
	manifest := &Manifest{Domains: []Domain{{ID: "a"}, {ID: "a"}}}
	path := filepath.Join(t.TempDir(), "bank.json")
	if err := manifest.Save(path); err != nil {
		t.Fatalf("save: %v", err)
	}
	if _, err := LoadManifest(path); err == nil {
		t.Fatal("expected duplicate-domain error")
	}
}

func TestBankAcquireAndBytes(t *testing.T) {
	tok := byteTokenizer()
	modelPath := filepath.Join(t.TempDir(), "a.gn")
	writeModelCheckpoint(t, modelPath, tok.VocabSize())

	manifest := &Manifest{Domains: []Domain{{ID: "a", Checkpoint: modelPath}}}
	bank := NewBank(manifest, tok, BankOptions{})
	transformer, err := bank.Acquire("a")
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	if transformer == nil || bank.Bytes() <= 0 {
		t.Fatalf("expected a resident model with bytes, got %v bytes=%d", transformer, bank.Bytes())
	}
	if resident := bank.Resident(); len(resident) != 1 || resident[0] != "a" {
		t.Fatalf("resident = %v, want [a]", resident)
	}
}

func TestBankRejectsVocabMismatch(t *testing.T) {
	tok := byteTokenizer()
	modelPath := filepath.Join(t.TempDir(), "mismatch.gn")
	writeModelCheckpoint(t, modelPath, 256) // tokenizer vocab is 256 + specials

	manifest := &Manifest{Domains: []Domain{{ID: "bad", Checkpoint: modelPath}}}
	bank := NewBank(manifest, tok, BankOptions{})
	if _, err := bank.Acquire("bad"); err == nil {
		t.Fatal("expected vocab mismatch error")
	}
}

func TestBankEvictionByModelCount(t *testing.T) {
	tok := byteTokenizer()
	firstPath := filepath.Join(t.TempDir(), "a.gn")
	secondPath := filepath.Join(t.TempDir(), "b.gn")
	writeModelCheckpoint(t, firstPath, tok.VocabSize())
	writeModelCheckpoint(t, secondPath, tok.VocabSize())

	manifest := &Manifest{Domains: []Domain{
		{ID: "a", Checkpoint: firstPath},
		{ID: "b", Checkpoint: secondPath},
	}}
	bank := NewBank(manifest, tok, BankOptions{MaxModels: 1})
	if _, err := bank.Acquire("a"); err != nil {
		t.Fatalf("acquire a: %v", err)
	}
	if _, err := bank.Acquire("b"); err != nil {
		t.Fatalf("acquire b: %v", err)
	}
	if resident := bank.Resident(); len(resident) != 1 || resident[0] != "b" {
		t.Fatalf("resident = %v, want [b]", resident)
	}
}
