package data

import (
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// MarkdownSource yields documents from Markdown (.md) files in a directory
// tree, one document per file. Files are read in sorted order and cycled
// forever across epochs. This is the entry point for "training on webdata that
// was encoded into Markdown".
type MarkdownSource struct {
	paths     []string
	batchSize int

	idx   int
	epoch int

	pending   []string
	lastState State
}

// NewMarkdownSource walks dir for .md files and returns a source over them.
func NewMarkdownSource(dir string, batchSize int) *MarkdownSource {
	if batchSize <= 0 {
		batchSize = 128
	}
	var paths []string
	_ = filepath.WalkDir(dir, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if !entry.IsDir() && strings.HasSuffix(strings.ToLower(entry.Name()), ".md") {
			paths = append(paths, path)
		}
		return nil
	})
	sort.Strings(paths)
	return &MarkdownSource{paths: paths, batchSize: batchSize, epoch: 1}
}

// NumFiles returns the number of Markdown files found.
func (source *MarkdownSource) NumFiles() int { return len(source.paths) }

// Next returns the next batch of up to batchSize documents.
func (source *MarkdownSource) Next() ([]string, State) {
	for {
		if len(source.pending) > 0 {
			count := min(source.batchSize, len(source.pending))
			batch := source.pending[:count]
			source.pending = source.pending[count:]
			return batch, source.lastState
		}
		if len(source.paths) == 0 {
			// No data: yield an empty batch (the caller decides how to handle it).
			return nil, source.lastState
		}
		if source.idx >= len(source.paths) {
			source.idx = 0
			source.epoch++
		}
		doc, err := os.ReadFile(source.paths[source.idx])
		source.lastState = State{PQIndex: source.idx, RGIndex: 0, Epoch: source.epoch}
		source.idx++
		if err != nil {
			continue
		}
		text := strings.TrimPrefix(string(doc), "\ufeff") // strip a UTF-8 BOM
		source.pending = []string{text}
	}
}
