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
	_ = filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if !d.IsDir() && strings.HasSuffix(strings.ToLower(d.Name()), ".md") {
			paths = append(paths, path)
		}
		return nil
	})
	sort.Strings(paths)
	return &MarkdownSource{paths: paths, batchSize: batchSize, epoch: 1}
}

// NumFiles returns the number of Markdown files found.
func (s *MarkdownSource) NumFiles() int { return len(s.paths) }

// Next returns the next batch of up to batchSize documents.
func (s *MarkdownSource) Next() ([]string, State) {
	for {
		if len(s.pending) > 0 {
			n := min(s.batchSize, len(s.pending))
			batch := s.pending[:n]
			s.pending = s.pending[n:]
			return batch, s.lastState
		}
		if len(s.paths) == 0 {
			// No data: yield an empty batch (the caller decides how to handle it).
			return nil, s.lastState
		}
		if s.idx >= len(s.paths) {
			s.idx = 0
			s.epoch++
		}
		doc, err := os.ReadFile(s.paths[s.idx])
		s.lastState = State{PQIndex: s.idx, RGIndex: 0, Epoch: s.epoch}
		s.idx++
		if err != nil {
			continue
		}
		text := strings.TrimPrefix(string(doc), "\ufeff") // strip a UTF-8 BOM
		s.pending = []string{text}
	}
}
