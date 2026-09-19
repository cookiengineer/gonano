package logging

import (
	"bytes"
	"strings"
	"testing"
)

func TestPrintBanner(test *testing.T) {
	var buffer bytes.Buffer
	PrintBanner(&buffer)
	if !strings.Contains(buffer.String(), "████") {
		test.Fatalf("banner missing block art: %q", buffer.String()[:40])
	}
}

func TestNopMetrics(test *testing.T) {
	metrics := NopMetrics{}
	metrics.Log(map[string]any{"x": 1})
	if err := metrics.Close(); err != nil {
		test.Fatalf("NopMetrics.Close returned error: %v", err)
	}
}

func TestJSONLMetrics(test *testing.T) {
	var buffer bytes.Buffer
	metrics := NewJSONLMetrics(&buffer)
	metrics.Log(map[string]any{"loss": 1.5, "step": 3})
	metrics.Log(map[string]any{"loss": 1.4, "step": 4})
	if err := metrics.Close(); err != nil {
		test.Fatalf("Close: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(buffer.String()), "\n")
	if len(lines) != 2 {
		test.Fatalf("expected 2 lines, got %d: %q", len(lines), buffer.String())
	}
	for _, line := range lines {
		if !strings.HasPrefix(line, "{") || !strings.HasSuffix(line, "}") {
			test.Fatalf("line not a JSON object: %q", line)
		}
	}
}

func TestJSONLMetricsNilWriter(test *testing.T) {
	metrics := &JSONLMetrics{}
	metrics.Log(map[string]any{"x": 1}) // must not panic
}
