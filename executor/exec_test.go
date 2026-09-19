package executor

import (
	"strings"
	"testing"
)

func TestFormatInt(test *testing.T) {
	if got := formatInt(0); got != "None" {
		test.Fatalf("formatInt(0) = %q, want None", got)
	}
	if got := formatInt(1024); got != "1024" {
		test.Fatalf("formatInt(1024) = %q", got)
	}
}

func TestQuote(test *testing.T) {
	got := quote("a'b\nc")
	want := "'a\\'b\\nc'"
	if got != want {
		test.Fatalf("quote = %q, want %q", got, want)
	}
}

func TestGuardTemplate(test *testing.T) {
	// The guard must format the memory limit and be valid enough to contain
	// the key restrictions.
	if !strings.Contains(guard, "setrlimit") {
		test.Fatal("guard missing rlimit")
	}
	if !strings.Contains(guard, "subprocess.Popen = None") {
		test.Fatal("guard missing subprocess restriction")
	}
}

func TestExecuteCodeHello(test *testing.T) {
	if !Available() {
		test.Skip("python3 not available")
	}
	result := ExecuteCode("print('hello')", 5e9, 256*1024*1024)
	if !result.Success {
		test.Fatalf("expected success, got %+v", result)
	}
	if strings.TrimSpace(result.Stdout) != "hello" {
		test.Fatalf("stdout = %q", result.Stdout)
	}
}
