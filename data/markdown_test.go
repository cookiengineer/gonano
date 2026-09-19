package data

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestMarkdownSource(tests *testing.T) {
	dir := tests.TempDir()
	files := map[string]string{
		"a.md":  "# First\nThis is the first document.",
		"b.md":  "# Second\nThis is the second document.",
		"c.txt": "ignored",
	}
	for name, content := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			tests.Fatal(err)
		}
	}

	src := NewMarkdownSource(dir, 2)
	if src.NumFiles() != 2 {
		tests.Fatalf("NumFiles = %d, want 2", src.NumFiles())
	}

	// First epoch: two documents (sorted order a.md then b.md).
	batch1, _ := src.Next()
	batch2, _ := src.Next()
	got := append(append([]string{}, batch1...), batch2...)
	want := []string{"# First\nThis is the first document.", "# Second\nThis is the second document."}
	if !reflect.DeepEqual(got, want) {
		tests.Fatalf("got %q, want %q", got, want)
	}

	// Wraps: epoch increments.
	_, state := src.Next()
	if state.Epoch != 2 {
		tests.Fatalf("epoch = %d, want 2", state.Epoch)
	}
}

func TestMarkdownSourceStripsBOM(tests *testing.T) {
	dir := tests.TempDir()
	os.WriteFile(filepath.Join(dir, "x.md"), []byte("\ufeff# Title\nbody"), 0o644)
	src := NewMarkdownSource(dir, 1)
	batch, _ := src.Next()
	if len(batch) != 1 || len(batch[0]) == 0 || batch[0][0] == 0xEF {
		tests.Fatalf("BOM not stripped: %q", batch)
	}
}

func TestMarkdownSourceEmptyDir(tests *testing.T) {
	src := NewMarkdownSource(tests.TempDir(), 1)
	batch, _ := src.Next()
	if batch != nil {
		tests.Fatalf("expected nil batch for empty dir, got %v", batch)
	}
}
