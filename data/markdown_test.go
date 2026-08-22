package data

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestMarkdownSource(t *testing.T) {
	dir := t.TempDir()
	files := map[string]string{
		"a.md": "# First\nThis is the first document.",
		"b.md": "# Second\nThis is the second document.",
		"c.txt": "ignored",
	}
	for name, content := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	src := NewMarkdownSource(dir, 2)
	if src.NumFiles() != 2 {
		t.Fatalf("NumFiles = %d, want 2", src.NumFiles())
	}

	// First epoch: two documents (sorted order a.md then b.md).
	batch1, _ := src.Next()
	batch2, _ := src.Next()
	got := append(append([]string{}, batch1...), batch2...)
	want := []string{"# First\nThis is the first document.", "# Second\nThis is the second document."}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q, want %q", got, want)
	}

	// Wraps: epoch increments.
	_, state := src.Next()
	if state.Epoch != 2 {
		t.Fatalf("epoch = %d, want 2", state.Epoch)
	}
}

func TestMarkdownSourceStripsBOM(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "x.md"), []byte("\ufeff# Title\nbody"), 0o644)
	src := NewMarkdownSource(dir, 1)
	batch, _ := src.Next()
	if len(batch) != 1 || len(batch[0]) == 0 || batch[0][0] == 0xEF {
		t.Fatalf("BOM not stripped: %q", batch)
	}
}

func TestMarkdownSourceEmptyDir(t *testing.T) {
	src := NewMarkdownSource(t.TempDir(), 1)
	batch, _ := src.Next()
	if batch != nil {
		t.Fatalf("expected nil batch for empty dir, got %v", batch)
	}
}
