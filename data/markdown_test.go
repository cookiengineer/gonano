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

func TestMarkdownSourceFollowsSymlinkedDirectories(tests *testing.T) {
	dir := tests.TempDir()
	real := filepath.Join(dir, "real")
	if err := os.MkdirAll(real, 0o755); err != nil {
		tests.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(real, "doc.md"), []byte("linked body"), 0o644); err != nil {
		tests.Fatal(err)
	}
	link := filepath.Join(dir, "link")
	if err := os.Symlink(real, link); err != nil {
		tests.Skipf("symlinks unsupported: %v", err)
	}
	// Create a symlink cycle: link/self -> link -> real.
	if err := os.Symlink(link, filepath.Join(link, "self")); err != nil {
		tests.Fatal(err)
	}

	src := NewMarkdownSource(link, 1)
	if src.NumFiles() != 1 {
		tests.Fatalf("NumFiles = %d, want 1 (symlinked dir + cycle)", src.NumFiles())
	}
	batch, _ := src.Next()
	if len(batch) != 1 || batch[0] != "linked body" {
		tests.Fatalf("got %q, want linked body", batch)
	}
}

func TestMarkdownSourceFollowsSymlinkedFiles(tests *testing.T) {
	dir := tests.TempDir()
	target := filepath.Join(dir, "target.md")
	if err := os.WriteFile(target, []byte("target body"), 0o644); err != nil {
		tests.Fatal(err)
	}
	alias := filepath.Join(dir, "alias.md")
	if err := os.Symlink(target, alias); err != nil {
		tests.Skipf("symlinks unsupported: %v", err)
	}

	src := NewMarkdownSource(dir, 10)
	if src.NumFiles() != 2 {
		tests.Fatalf("NumFiles = %d, want 2 (target + alias)", src.NumFiles())
	}
}

func TestMarkdownSourceSkipsBrokenLinks(tests *testing.T) {
	dir := tests.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "ok.md"), []byte("ok"), 0o644); err != nil {
		tests.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(dir, "missing"), filepath.Join(dir, "broken.md")); err != nil {
		tests.Skipf("symlinks unsupported: %v", err)
	}
	src := NewMarkdownSource(dir, 10)
	if src.NumFiles() != 1 {
		tests.Fatalf("NumFiles = %d, want 1 (broken link skipped)", src.NumFiles())
	}
}
