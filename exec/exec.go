// Package exec runs untrusted Python code produced by an LLM in a sandboxed
// subprocess, mirroring nanochat's execution.py. It is used by the HumanEval
// evaluation. This is not a true security sandbox: it protects against
// accidental destructive behavior (resource limits, disabled destructive
// calls) but not against adversarial code.
package exec

import (
	"context"
	"os/exec"
	"strings"
	"time"
)

// Result is the outcome of a sandboxed execution.
type Result struct {
	Success        bool
	Stdout         string
	Stderr         string
	Error          string
	Timeout        bool
	MemoryExceeded bool
}

// guard is prepended to the untrusted code. It applies memory limits and
// disables destructive standard-library functions.
const guard = `
import faulthandler, builtins, os, shutil, subprocess, sys
maximum_memory_bytes = {maximum_memory_bytes}
if maximum_memory_bytes is not None and sys.platform != "darwin":
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
    resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
    resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))
faulthandler.disable()
builtins.exit = None
builtins.quit = None
builtins.help = None
os.environ["OMP_NUM_THREADS"] = "1"
for name in ("kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir",
             "setuid", "fork", "forkpty", "killpg", "rename", "renames", "truncate",
             "replace", "unlink", "fchmod", "fchown", "chmod", "chown", "chroot",
             "lchflags", "lchmod", "lchown", "getcwd", "chdir"):
    setattr(os, name, None)
for name in ("rmtree", "move", "chown"):
    setattr(shutil, name, None)
subprocess.Popen = None
for name in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
    sys.modules[name] = None
`

// ExecuteCode runs code in a fresh Python subprocess with resource limits and a
// scrubbed environment. It returns the outcome.
func ExecuteCode(code string, timeout time.Duration, maxMemoryBytes int64) Result {
	guard := strings.ReplaceAll(guard, "{maximum_memory_bytes}", formatInt(maxMemoryBytes))
	program := guard + "\nexec(compile(" + quote(code) + ", '<llm>', 'exec'), {'__name__': '__main__'})\n"

	ctx := context.Background()
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	cmd := exec.CommandContext(ctx, "python3", "-c", program)
	cmd.Env = []string{"PATH=/usr/bin:/bin"}
	cmd.Stdin = nil

	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	if ctx.Err() == context.DeadlineExceeded {
		return Result{Success: false, Timeout: true, Error: "Execution timed out (process killed)"}
	}
	success := err == nil
	se := stderr.String()
	errorMsg := ""
	if !success {
		lines := strings.Split(strings.TrimSpace(se), "\n")
		if len(lines) > 0 {
			errorMsg = lines[len(lines)-1]
		}
	}
	return Result{
		Success:        success,
		Stdout:         stdout.String(),
		Stderr:         se,
		Error:          errorMsg,
		MemoryExceeded: strings.Contains(se, "MemoryError"),
	}
}

func formatInt(v int64) string {
	if v <= 0 {
		return "None"
	}
	return itoa(v)
}

func itoa(v int64) string {
	if v == 0 {
		return "0"
	}
	neg := v < 0
	if neg {
		v = -v
	}
	var buf [20]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

// quote produces a repr()-style Python string literal for code.
func quote(s string) string {
	var b strings.Builder
	b.WriteByte('\'')
	for _, r := range s {
		switch r {
		case '\\':
			b.WriteString("\\\\")
		case '\'':
			b.WriteString("\\'")
		case '\n':
			b.WriteString("\\n")
		case '\r':
			b.WriteString("\\r")
		case '\t':
			b.WriteString("\\t")
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte('\'')
	return b.String()
}

// Available reports whether a Python interpreter is available on this host.
func Available() bool {
	_, err := exec.LookPath("python3")
	return err == nil
}
