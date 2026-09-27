package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNewDocProviderMarkdown(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "page.md"), []byte("# hello world"), 0o644); err != nil {
		t.Fatalf("write markdown: %v", err)
	}
	provider, err := newDocProvider(dir, "markdown")
	if err != nil {
		t.Fatalf("newDocProvider: %v", err)
	}
	docs, _ := provider()
	if len(docs) == 0 {
		t.Fatal("provider returned no documents")
	}
}

func TestNewDocProviderEmptyMarkdown(t *testing.T) {
	if _, err := newDocProvider(t.TempDir(), "markdown"); err == nil {
		t.Fatal("expected error for a directory with no .md files")
	}
}

func TestNewDocProviderUnknownFormat(t *testing.T) {
	if _, err := newDocProvider(t.TempDir(), "csv"); err == nil {
		t.Fatal("expected error for an unknown format")
	}
}
