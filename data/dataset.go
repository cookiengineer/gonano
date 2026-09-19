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
func (source *ParquetSource) Next() ([]string, State) {
	for {
		if len(source.pending) > 0 {
			count := min(source.batchSize, len(source.pending))
			batch := source.pending[:count]
			source.pending = source.pending[count:]
			return batch, source.lastState
		}
		if source.reader == nil {
			if source.pqIdx >= len(source.paths) {
				source.pqIdx = 0
				source.epoch++
			}
			openedReader, err := parquet.Open(source.paths[source.pqIdx])
			if err != nil {
				panic(fmt.Sprintf("data: open parquet %s: %v", source.paths[source.pqIdx], err))
			}
			source.reader = openedReader
			source.curRG = 0
		}
		if source.curRG >= source.reader.NumRowGroups() {
			source.reader.Close()
			source.reader = nil
			source.pqIdx++
			continue
		}
		docs, err := source.reader.ReadColumnStrings(source.curRG, "text")
		if err != nil {
			panic(fmt.Sprintf("data: read %s row group %d: %v", source.paths[source.pqIdx], source.curRG, err))
		}
		source.lastState = State{PQIndex: source.pqIdx, RGIndex: source.curRG, Epoch: source.epoch}
		source.pending = docs
		source.curRG++
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
	for _, entry := range entries {
		name := entry.Name()
		if strings.HasSuffix(name, ".parquet") && !strings.HasSuffix(name, ".tmp") {
			paths = append(paths, filepath.Join(dir, name))
		}
	}
	sort.Strings(paths)
	return paths
}
