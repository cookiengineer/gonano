// Package logging provides shared logging, banner printing, and a metrics
// sink for the gonano framework.
//
// It wraps the standard library log/slog so that commands and library callers
// get consistent, leveled output without any third-party dependencies.
package logging

import (
	"encoding/json"
	"io"
	"log/slog"
	"os"
	"sync"
)

// Default returns a slog.Logger writing text to stderr at the given level.
// It is the default used by the gonano command-line tools.
func Default(level slog.Level) *slog.Logger {
	h := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level})
	return slog.New(h)
}

// banner is the ASCII banner printed by the command-line tools.
const banner = `
                                                       █████                █████
                                                      ░░███                ░░███
    ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
   ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
    ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
    ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
    ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
   ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
`

// PrintBanner writes the gonano banner to w.
func PrintBanner(w io.Writer) {
	_, _ = io.WriteString(w, banner+"\n")
}

// Metrics is a minimal sink for scalar metrics emitted during training and
// evaluation. The reference implementation logs to wandb; here we provide a
// local, dependency-free equivalent. The nil value is a no-op sink.
type Metrics interface {
	// Log records a set of named scalar values for a single event.
	Log(values map[string]any)
	// Close flushes and releases any resources held by the sink.
	Close() error
}

// NopMetrics is a Metrics sink that discards everything.
type NopMetrics struct{}

func (NopMetrics) Log(map[string]any) {}
func (NopMetrics) Close() error       { return nil }

// JSONLMetrics writes one JSON object per line to w. It is safe for
// concurrent use and is the default sink for offline runs.
type JSONLMetrics struct {
	mu sync.Mutex
	w  io.Writer
}

// NewJSONLMetrics returns a Metrics sink writing newline-delimited JSON to w.
func NewJSONLMetrics(w io.Writer) *JSONLMetrics {
	return &JSONLMetrics{w: w}
}

// Log appends a single JSON line for values to the underlying writer.
func (m *JSONLMetrics) Log(values map[string]any) {
	if m == nil || m.w == nil {
		return
	}
	b, err := json.Marshal(values)
	if err != nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	_, _ = m.w.Write(append(b, '\n'))
}

// Close flushes the underlying writer if it implements interface{ Flush() error }.
func (m *JSONLMetrics) Close() error {
	if m == nil {
		return nil
	}
	if f, ok := m.w.(interface{ Flush() error }); ok {
		return f.Flush()
	}
	return nil
}
