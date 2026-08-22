package logging

import (
	"bytes"
	"strings"
	"testing"
)

func TestPrintBanner(t *testing.T) {
	var b bytes.Buffer
	PrintBanner(&b)
	if !strings.Contains(b.String(), "████") {
		t.Fatalf("banner missing block art: %q", b.String()[:40])
	}
}

func TestNopMetrics(t *testing.T) {
	m := NopMetrics{}
	m.Log(map[string]any{"x": 1})
	if err := m.Close(); err != nil {
		t.Fatalf("NopMetrics.Close returned error: %v", err)
	}
}

func TestJSONLMetrics(t *testing.T) {
	var b bytes.Buffer
	m := NewJSONLMetrics(&b)
	m.Log(map[string]any{"loss": 1.5, "step": 3})
	m.Log(map[string]any{"loss": 1.4, "step": 4})
	if err := m.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(b.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected 2 lines, got %d: %q", len(lines), b.String())
	}
	for _, line := range lines {
		if !strings.HasPrefix(line, "{") || !strings.HasSuffix(line, "}") {
			t.Fatalf("line not a JSON object: %q", line)
		}
	}
}

func TestJSONLMetricsNilWriter(t *testing.T) {
	m := &JSONLMetrics{}
	m.Log(map[string]any{"x": 1}) // must not panic
}
