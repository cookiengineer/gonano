package exec

import (
	"strings"
	"testing"
)

func TestFormatInt(t *testing.T) {
	if got := formatInt(0); got != "None" {
		t.Fatalf("formatInt(0) = %q, want None", got)
	}
	if got := formatInt(1024); got != "1024" {
		t.Fatalf("formatInt(1024) = %q", got)
	}
}

func TestQuote(t *testing.T) {
	got := quote("a'b\nc")
	want := "'a\\'b\\nc'"
	if got != want {
		t.Fatalf("quote = %q, want %q", got, want)
	}
}

func TestGuardTemplate(t *testing.T) {
	// The guard must format the memory limit and be valid enough to contain
	// the key restrictions.
	if !strings.Contains(guard, "setrlimit") {
		t.Fatal("guard missing rlimit")
	}
	if !strings.Contains(guard, "subprocess.Popen = None") {
		t.Fatal("guard missing subprocess restriction")
	}
}

func TestExecuteCodeHello(t *testing.T) {
	if !Available() {
		t.Skip("python3 not available")
	}
	res := ExecuteCode("print('hello')", 5e9, 256*1024*1024)
	if !res.Success {
		t.Fatalf("expected success, got %+v", res)
	}
	if strings.TrimSpace(res.Stdout) != "hello" {
		t.Fatalf("stdout = %q", res.Stdout)
	}
}
