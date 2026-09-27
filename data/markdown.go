package data

import (
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
// Unlike filepath.WalkDir, it follows symlinked directories (and files), so a
// corpus can be assembled from links into shared extraction trees. Visited
// real directories are tracked to make symlink cycles terminate.
func NewMarkdownSource(dir string, batchSize int) *MarkdownSource {
	if batchSize <= 0 {
		batchSize = 128
	}
	paths := collectMarkdownFiles(dir)
	sort.Strings(paths)
	return &MarkdownSource{paths: paths, batchSize: batchSize, epoch: 1}
}

// collectMarkdownFiles recursively gathers .md file paths below root, following
// symbolic links in both directions. A directory reached through more than one
// link is traversed only once; a symlink cycle therefore terminates.
func collectMarkdownFiles(root string) []string {
	var paths []string
	visited := make(map[string]bool)

	var walk func(dir string)
	walk = func(dir string) {
		if real, err := filepath.EvalSymlinks(dir); err == nil {
			if visited[real] {
				return
			}
			visited[real] = true
		}
		entries, err := os.ReadDir(dir)
		if err != nil {
			return
		}
		for _, entry := range entries {
			path := filepath.Join(dir, entry.Name())
			info, err := os.Stat(path) // follows symlinks
			if err != nil {
				continue
			}
			if info.IsDir() {
				walk(path)
				continue
			}
			if strings.HasSuffix(strings.ToLower(entry.Name()), ".md") {
				paths = append(paths, path)
			}
		}
	}

	walk(root)
	return paths
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
