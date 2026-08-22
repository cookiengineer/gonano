// Package data provides the datasets and dataloaders for gonano: reading
// Parquet text shards, downloading them from the HuggingFace Hub, and packing
// documents into BOS-aligned training batches.
package data

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cookiengineer/gonano/data/parquet"
)

// State records the current position in a dataset for checkpoint/resume.
type State struct {
	PQIndex int // parquet file index
	RGIndex int // row group index within the file
	Epoch   int // number of times the dataset has been cycled (1-based)
}

// ParquetSource yields document batches from a set of Parquet files, cycling
// infinitely across epochs. Each file must contain a flat "text" BYTE_ARRAY
// column. It is safe for use from a single goroutine.
type ParquetSource struct {
	paths     []string
	batchSize int

	pqIdx int
	epoch int

	reader *parquet.Reader
	curRG  int

	pending   []string
	lastState State
}

// NewParquetSource returns a source over the given parquet file paths.
func NewParquetSource(paths []string, batchSize int) *ParquetSource {
	if batchSize <= 0 {
		batchSize = 128
	}
	return &ParquetSource{paths: paths, batchSize: batchSize, epoch: 1}
}

// Next returns the next batch of up to batchSize documents. It cycles forever,
// wrapping to the start of the dataset (and incrementing the epoch) at the end.
func (s *ParquetSource) Next() ([]string, State) {
	for {
		if len(s.pending) > 0 {
			n := min(s.batchSize, len(s.pending))
			batch := s.pending[:n]
			s.pending = s.pending[n:]
			return batch, s.lastState
		}
		if s.reader == nil {
			if s.pqIdx >= len(s.paths) {
				s.pqIdx = 0
				s.epoch++
			}
			r, err := parquet.Open(s.paths[s.pqIdx])
			if err != nil {
				panic(fmt.Sprintf("data: open parquet %s: %v", s.paths[s.pqIdx], err))
			}
			s.reader = r
			s.curRG = 0
		}
		if s.curRG >= s.reader.NumRowGroups() {
			s.reader.Close()
			s.reader = nil
			s.pqIdx++
			continue
		}
		docs, err := s.reader.ReadColumnStrings(s.curRG, "text")
		if err != nil {
			panic(fmt.Sprintf("data: read %s row group %d: %v", s.paths[s.pqIdx], s.curRG, err))
		}
		s.lastState = State{PQIndex: s.pqIdx, RGIndex: s.curRG, Epoch: s.epoch}
		s.pending = docs
		s.curRG++
	}
}

// ListParquetFiles returns the sorted full paths of all .parquet files in dir,
// ignoring temporary (.tmp) files.
func ListParquetFiles(dir string) []string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var paths []string
	for _, e := range entries {
		name := e.Name()
		if strings.HasSuffix(name, ".parquet") && !strings.HasSuffix(name, ".tmp") {
			paths = append(paths, filepath.Join(dir, name))
		}
	}
	sort.Strings(paths)
	return paths
}
