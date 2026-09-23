//go:build linux

package device

import (
	"os"
	"strconv"
	"strings"
)

// AvailableMemoryBytes returns the memory the kernel estimates is available to
// new allocations without swapping, read from MemAvailable in /proc/meminfo.
// The second result is false when it cannot be determined.
func AvailableMemoryBytes() (uint64, bool) {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, false
	}
	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasPrefix(line, "MemAvailable:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return 0, false
		}
		kilobytes, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			return 0, false
		}
		return kilobytes * 1024, true
	}
	return 0, false
}
