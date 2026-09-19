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
	handler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level})
	return slog.New(handler)
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

// PrintBanner writes the gonano banner to writer.
func PrintBanner(writer io.Writer) {
	_, _ = io.WriteString(writer, banner+"\n")
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

// JSONLMetrics writes one JSON object per line to writer. It is safe for
// concurrent use and is the default sink for offline runs.
type JSONLMetrics struct {
	mutex  sync.Mutex
	writer io.Writer
}

// NewJSONLMetrics returns a Metrics sink writing newline-delimited JSON to writer.
func NewJSONLMetrics(writer io.Writer) *JSONLMetrics {
	return &JSONLMetrics{writer: writer}
}

// Log appends a single JSON line for values to the underlying writer.
func (metrics *JSONLMetrics) Log(values map[string]any) {
	if metrics == nil || metrics.writer == nil {
		return
	}
	encoded, err := json.Marshal(values)
	if err != nil {
		return
	}
	metrics.mutex.Lock()
	defer metrics.mutex.Unlock()
	_, _ = metrics.writer.Write(append(encoded, '\n'))
}

// Close flushes the underlying writer if it implements interface{ Flush() error }.
func (metrics *JSONLMetrics) Close() error {
	if metrics == nil {
		return nil
	}
	if flusher, ok := metrics.writer.(interface{ Flush() error }); ok {
		return flusher.Flush()
	}
	return nil
}
