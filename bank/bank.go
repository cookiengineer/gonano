// Package bank manages a bank of independently trained domain models for the
// domain meta-router (architecture A). A manifest lists one checkpoint per
// semantic domain; the bank loads models on demand and evicts the least
// recently used ones to stay within a byte budget.
//
// All models in a bank must share one tokenizer and VocabSize, otherwise their
// logits could not be blended. The bank enforces this at load time.
package bank

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"sync"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/model/checkpoint"
	"github.com/cookiengineer/gonano/tokenizer"
)

// Domain describes one bank domain.
type Domain struct {
	// ID is the stable domain identifier used for routing and CLI flags.
	ID string `json:"id"`
	// Name is a human-readable label.
	Name string `json:"name,omitempty"`
	// Checkpoint is the path to the domain model (.gn or .gguf).
	Checkpoint string `json:"checkpoint"`
	// DataDirs lists the training/evaluation text sources for the domain. The
	// router trainer reads labeled examples from these.
	DataDirs []string `json:"data_dirs,omitempty"`
	// Description is free-form metadata.
	Description string `json:"description,omitempty"`
}

// Manifest is the bank descriptor, stored as JSON.
type Manifest struct {
	Version   int      `json:"version"`
	Tokenizer string   `json:"tokenizer"`
	Router    string   `json:"router,omitempty"`
	Domains   []Domain `json:"domains"`
}

// LoadManifest reads a bank manifest.
func LoadManifest(path string) (*Manifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var manifest Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return nil, fmt.Errorf("bank: invalid manifest %s: %w", path, err)
	}
	if len(manifest.Domains) == 0 {
		return nil, fmt.Errorf("bank: manifest %s has no domains", path)
	}
	seen := make(map[string]bool, len(manifest.Domains))
	for _, domain := range manifest.Domains {
		if domain.ID == "" {
			return nil, fmt.Errorf("bank: manifest %s has a domain without an id", path)
		}
		if seen[domain.ID] {
			return nil, fmt.Errorf("bank: manifest %s has duplicate domain %q", path, domain.ID)
		}
		seen[domain.ID] = true
	}
	return &manifest, nil
}

// Save writes the manifest as indented JSON.
func (manifest *Manifest) Save(path string) error {
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// DomainByID returns the domain with the given id.
func (manifest *Manifest) DomainByID(id string) (Domain, bool) {
	for _, domain := range manifest.Domains {
		if domain.ID == id {
			return domain, true
		}
	}
	return Domain{}, false
}

// IDs returns the domain ids in manifest order.
func (manifest *Manifest) IDs() []string {
	ids := make([]string, len(manifest.Domains))
	for index, domain := range manifest.Domains {
		ids[index] = domain.ID
	}
	return ids
}

// BankOptions bounds resident memory.
type BankOptions struct {
	// MaxBytes is the resident byte budget for loaded models. Zero is
	// unbounded.
	MaxBytes int64
	// MaxModels bounds the number of simultaneously resident models. Zero is
	// unbounded.
	MaxModels int
}

// entry is one resident (or loadable) domain model.
type entry struct {
	domain Domain
	model  *model.Transformer
	bytes  int64
	used   uint64
}

// Bank loads and caches domain models within a byte/model budget.
//
// The bank is safe for concurrent use. Acquire returns a pointer the caller may
// keep using even if the bank later evicts the entry (Go's GC keeps it alive);
// eviction only shrinks the bank's own retained set.
type Bank struct {
	mutex     sync.Mutex
	manifest  *Manifest
	tokenizer *tokenizer.Tokenizer
	vocabSize int
	options   BankOptions
	entries   map[string]*entry
	order     []string
	bytes     int64
	clock     uint64
}

// NewBank builds a bank over a manifest and the shared tokenizer.
func NewBank(manifest *Manifest, tokenizerImpl *tokenizer.Tokenizer, options BankOptions) *Bank {
	return &Bank{
		manifest:  manifest,
		tokenizer: tokenizerImpl,
		vocabSize: tokenizerImpl.VocabSize(),
		options:   options,
		entries:   make(map[string]*entry),
	}
}

// Manifest returns the bank's manifest.
func (bank *Bank) Manifest() *Manifest { return bank.manifest }

// Tokenizer returns the shared tokenizer.
func (bank *Bank) Tokenizer() *tokenizer.Tokenizer { return bank.tokenizer }

// Acquire returns the model for a domain, loading it if needed. It validates
// that the model's vocabulary matches the bank's tokenizer.
func (bank *Bank) Acquire(id string) (*model.Transformer, error) {
	bank.mutex.Lock()
	defer bank.mutex.Unlock()

	bank.clock++
	if existing, ok := bank.entries[id]; ok {
		existing.used = bank.clock
		bank.touchLocked(id)
		return existing.model, nil
	}

	domain, ok := bank.manifest.DomainByID(id)
	if !ok {
		return nil, fmt.Errorf("bank: unknown domain %q", id)
	}
	meta, parameters, err := checkpoint.LoadAny(domain.Checkpoint)
	if err != nil {
		return nil, fmt.Errorf("bank: load domain %q: %w", id, err)
	}
	if meta.ModelConfig.VocabSize != bank.vocabSize {
		return nil, fmt.Errorf("bank: domain %q vocab %d does not match tokenizer vocab %d",
			id, meta.ModelConfig.VocabSize, bank.vocabSize)
	}
	transformer := checkpoint.LoadModel(meta, parameters)
	resident := &entry{
		domain: domain,
		model:  transformer,
		bytes:  int64(transformer.TotalParams()) * 4,
		used:   bank.clock,
	}
	bank.entries[id] = resident
	bank.order = append(bank.order, id)
	bank.bytes += resident.bytes
	bank.evictLocked(id)
	return resident.model, nil
}

// AcquireMany loads several domains, returning the models and the ids actually
// acquired, in the requested order. It is the entry point used by blended
// generation.
func (bank *Bank) AcquireMany(ids []string) ([]*model.Transformer, []string, error) {
	models := make([]*model.Transformer, 0, len(ids))
	loadedIDs := make([]string, 0, len(ids))
	for _, id := range ids {
		transformer, err := bank.Acquire(id)
		if err != nil {
			return models, loadedIDs, err
		}
		models = append(models, transformer)
		loadedIDs = append(loadedIDs, id)
	}
	return models, loadedIDs, nil
}

// Resident returns the ids currently held by the bank, most recently used
// first.
func (bank *Bank) Resident() []string {
	bank.mutex.Lock()
	defer bank.mutex.Unlock()
	resident := append([]string(nil), bank.order...)
	for left, right := 0, len(resident)-1; left < right; left, right = left+1, right-1 {
		resident[left], resident[right] = resident[right], resident[left]
	}
	return resident
}

// Bytes returns the approximate resident byte count.
func (bank *Bank) Bytes() int64 {
	bank.mutex.Lock()
	defer bank.mutex.Unlock()
	return bank.bytes
}

// touchLocked moves an id to the most-recently-used end of the order.
func (bank *Bank) touchLocked(id string) {
	for index, candidate := range bank.order {
		if candidate == id {
			bank.order = append(bank.order[:index], bank.order[index+1:]...)
			break
		}
	}
	bank.order = append(bank.order, id)
}

// evictLocked removes least-recently-used entries until the budget is met,
// never evicting the just-acquired keep id.
func (bank *Bank) evictLocked(keep string) {
	for {
		overCount := bank.options.MaxModels > 0 && len(bank.entries) > bank.options.MaxModels
		overBytes := bank.options.MaxBytes > 0 && bank.bytes > bank.options.MaxBytes
		if !overCount && !overBytes {
			return
		}
		victim := ""
		for _, id := range bank.order {
			if id != keep {
				victim = id
				break
			}
		}
		if victim == "" {
			return
		}
		bank.removeLocked(victim)
	}
}

// removeLocked drops an entry and updates accounting.
func (bank *Bank) removeLocked(id string) {
	entry, ok := bank.entries[id]
	if !ok {
		return
	}
	delete(bank.entries, id)
	bank.bytes -= entry.bytes
	for index, candidate := range bank.order {
		if candidate == id {
			bank.order = append(bank.order[:index], bank.order[index+1:]...)
			break
		}
	}
}

// SortedDomains returns the manifest domains sorted by id.
func (bank *Bank) SortedDomains() []Domain {
	domains := append([]Domain(nil), bank.manifest.Domains...)
	sort.SliceStable(domains, func(left, right int) bool { return domains[left].ID < domains[right].ID })
	return domains
}
