//go:build !linux

package device

// AvailableMemoryBytes returns (0, false) on platforms without a portable
// memory-availability query, so callers skip the training-memory preflight.
func AvailableMemoryBytes() (uint64, bool) { return 0, false }
