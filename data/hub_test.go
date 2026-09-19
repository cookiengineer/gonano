package data

import (
	"strings"
	"testing"
)

func TestBaseDirHonorsEnvironment(t *testing.T) {
	directory := t.TempDir()
	t.Setenv("GONANO_BASE_DIR", directory)
	if got := BaseDir(); got != directory {
		t.Fatalf("BaseDir = %q, want %q", got, directory)
	}
}

func TestBaseDirDefault(t *testing.T) {
	t.Setenv("GONANO_BASE_DIR", "")
	if got := BaseDir(); !strings.HasSuffix(got, ".cache/gonano") {
		t.Fatalf("BaseDir = %q, want a path ending in .cache/gonano", got)
	}
}
