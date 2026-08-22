package tasks

import (
	"testing"
)

func TestRenderMC(t *testing.T) {
	q := RenderMC("What is 2+2?", []string{"A", "B", "C", "D"}, []string{"3", "4", "5", "6"})
	want := "Multiple Choice question: What is 2+2?\n- 3=A\n- 4=B\n- 5=C\n- 6=D\n\nRespond only with the letter of the correct answer."
	if q != want {
		t.Fatalf("RenderMC = %q, want %q", q, want)
	}
}

func TestTaskMixture(t *testing.T) {
	m1 := NewMMLUFromRows([]MMLURow{
		{Question: "q1", Choices: []string{"a", "b", "c", "d"}, Answer: 0},
		{Question: "q2", Choices: []string{"a", "b", "c", "d"}, Answer: 1},
	})
	g := NewGSM8KFromRows([]GSM8KRow{
		{Question: "q3", Answer: "#### 42"},
	})
	mix := NewTaskMixture([]Task{m1, g})
	if mix.NumExamples() != 3 {
		t.Fatalf("NumExamples = %d, want 3", mix.NumExamples())
	}
	// Every example is accessible and has the right shape.
	seen := map[int]bool{}
	for i := 0; i < 3; i++ {
		conv := mix.GetExample(i)
		if conv == nil || len(conv.Messages) != 2 {
			t.Fatalf("example %d malformed", i)
		}
		seen[i] = true
	}
	if len(seen) != 3 {
		t.Fatal("did not see all examples")
	}
}

func TestMMLUEvaluate(t *testing.T) {
	m := NewMMLUFromRows([]MMLURow{
		{Question: "q1", Choices: []string{"a", "b", "c", "d"}, Answer: 1},
	})
	conv := m.GetExample(0)
	if !m.Evaluate(conv, "B") {
		t.Fatal("expected B to be correct")
	}
	if m.Evaluate(conv, "A") {
		t.Fatal("expected A to be wrong")
	}
	if m.EvalType() != Categorical {
		t.Fatal("MMLU should be categorical")
	}
}

func TestGSM8KEvaluateAndExtract(t *testing.T) {
	if ExtractAnswer("The answer is #### 42") != "42" {
		t.Fatal("ExtractAnswer failed")
	}
	if ExtractAnswer("no answer") != "" {
		t.Fatal("ExtractAnswer should be empty")
	}
	g := NewGSM8KFromRows([]GSM8KRow{
		{Question: "q", Answer: "Work: 12/60 = <<12/60=0.2>>0.2. #### 10"},
	})
	conv := g.GetExample(0)
	// The answer contains tool-call parts and a final text part.
	parts := conv.Messages[1].Parts
	if len(parts) == 0 {
		t.Fatal("expected tool parts")
	}
	if !g.Evaluate(conv, "some work #### 10") {
		t.Fatal("expected correct answer")
	}
	if g.Evaluate(conv, "#### 11") {
		t.Fatal("expected wrong answer")
	}
}

func TestExtractProgram(t *testing.T) {
	got := ExtractProgram("here is code:\n```python\nprint(1)\n```\nmore")
	if got != "print(1)" {
		t.Fatalf("ExtractProgram = %q", got)
	}
	plain := ExtractProgram("print(2)")
	if plain != "print(2)" {
		t.Fatalf("ExtractProgram plain = %q", plain)
	}
}

func TestExtractImports(t *testing.T) {
	got := ExtractImports("import os\nfrom math import sqrt\n\ndef f():\n    pass")
	if got != "import os\nfrom math import sqrt" {
		t.Fatalf("ExtractImports = %q", got)
	}
}

func TestHumanEvalEvaluate(t *testing.T) {
	h := NewHumanEvalFromRows([]HumanEvalRow{
		{Prompt: "def double(x):\n", Solution: "    return x * 2", EntryPoint: "double", Test: "def check(f):\n    assert f(2) == 4"},
	})
	conv := h.GetExample(0)
	// Inject a fake executor.
	h.Execute = func(program string) bool {
		return len(program) > 0 && contains(program, "check(double)")
	}
	if !h.Evaluate(conv, "```python\n    return x * 2\n```") {
		t.Fatal("expected correct completion")
	}
	h.Execute = func(program string) bool { return false }
	if h.Evaluate(conv, "    return x * 3") {
		t.Fatal("expected wrong completion")
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

func TestSmolTalkFromRows(t *testing.T) {
	s := NewSmolTalkFromRows([]SmolTalkRow{
		{Messages: []SmolTalkMessage{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "hello"}}},
	})
	if s.NumExamples() != 1 {
		t.Fatalf("NumExamples = %d, want 1", s.NumExamples())
	}
	conv := s.GetExample(0)
	if conv.Messages[0].Role != "user" {
		t.Fatal("first message should be user")
	}
}
