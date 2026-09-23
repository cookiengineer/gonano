//go:build linux

package device

import "testing"

func TestAvailableMemoryBytes(t *testing.T) {
	available, ok := AvailableMemoryBytes()
	if !ok {
		t.Skip("/proc/meminfo unavailable; memory preflight is skipped here")
	}
	if available == 0 {
		t.Fatal("AvailableMemoryBytes returned zero with ok=true")
	}
}
