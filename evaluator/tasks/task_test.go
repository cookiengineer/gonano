package tasks

import (
	"testing"
)

func TestRenderMC(tester *testing.T) {
	question := RenderMC("What is 2+2?", []string{"A", "B", "C", "D"}, []string{"3", "4", "5", "6"})
	want := "Multiple Choice question: What is 2+2?\n- 3=A\n- 4=B\n- 5=C\n- 6=D\n\nRespond only with the letter of the correct answer."
	if question != want {
		tester.Fatalf("RenderMC = %q, want %q", question, want)
	}
}

func TestTaskMixture(tester *testing.T) {
	mmluTask := NewMMLUFromRows([]MMLURow{
		{Question: "q1", Choices: []string{"a", "b", "c", "d"}, Answer: 0},
		{Question: "q2", Choices: []string{"a", "b", "c", "d"}, Answer: 1},
	})
	gsm8kTask := NewGSM8KFromRows([]GSM8KRow{
		{Question: "q3", Answer: "#### 42"},
	})
	mix := NewTaskMixture([]Task{mmluTask, gsm8kTask})
	if mix.NumExamples() != 3 {
		tester.Fatalf("NumExamples = %d, want 3", mix.NumExamples())
	}
	// Every example is accessible and has the right shape.
	seen := map[int]bool{}
	for index := 0; index < 3; index++ {
		conversation := mix.GetExample(index)
		if conversation == nil || len(conversation.Messages) != 2 {
			tester.Fatalf("example %d malformed", index)
		}
		seen[index] = true
	}
	if len(seen) != 3 {
		tester.Fatal("did not see all examples")
	}
}

func TestMMLUEvaluate(tester *testing.T) {
	mmlu := NewMMLUFromRows([]MMLURow{
		{Question: "q1", Choices: []string{"a", "b", "c", "d"}, Answer: 1},
	})
	conversation := mmlu.GetExample(0)
	if !mmlu.Evaluate(conversation, "B") {
		tester.Fatal("expected B to be correct")
	}
	if mmlu.Evaluate(conversation, "A") {
		tester.Fatal("expected A to be wrong")
	}
	if mmlu.EvalType() != Categorical {
		tester.Fatal("MMLU should be categorical")
	}
}

func TestGSM8KEvaluateAndExtract(tester *testing.T) {
	if ExtractAnswer("The answer is #### 42") != "42" {
		tester.Fatal("ExtractAnswer failed")
	}
	if ExtractAnswer("no answer") != "" {
		tester.Fatal("ExtractAnswer should be empty")
	}
	gsm8k := NewGSM8KFromRows([]GSM8KRow{
		{Question: "q", Answer: "Work: 12/60 = <<12/60=0.2>>0.2. #### 10"},
	})
	conversation := gsm8k.GetExample(0)
	// The answer contains tool-call parts and a final text part.
	parts := conversation.Messages[1].Parts
	if len(parts) == 0 {
		tester.Fatal("expected tool parts")
	}
	if !gsm8k.Evaluate(conversation, "some work #### 10") {
		tester.Fatal("expected correct answer")
	}
	if gsm8k.Evaluate(conversation, "#### 11") {
		tester.Fatal("expected wrong answer")
	}
}

func TestExtractProgram(tester *testing.T) {
	got := ExtractProgram("here is code:\n```python\nprint(1)\n```\nmore")
	if got != "print(1)" {
		tester.Fatalf("ExtractProgram = %q", got)
	}
	plain := ExtractProgram("print(2)")
	if plain != "print(2)" {
		tester.Fatalf("ExtractProgram plain = %q", plain)
	}
}

func TestExtractImports(tester *testing.T) {
	got := ExtractImports("import os\nfrom math import sqrt\n\ndef f():\n    pass")
	if got != "import os\nfrom math import sqrt" {
		tester.Fatalf("ExtractImports = %q", got)
	}
}

func TestHumanEvalEvaluate(tester *testing.T) {
	humaneval := NewHumanEvalFromRows([]HumanEvalRow{
		{Prompt: "def double(x):\n", Solution: "    return x * 2", EntryPoint: "double", Test: "def check(f):\n    assert f(2) == 4"},
	})
	conversation := humaneval.GetExample(0)
	// Inject a fake executor.
	humaneval.Execute = func(program string) bool {
		return len(program) > 0 && contains(program, "check(double)")
	}
	if !humaneval.Evaluate(conversation, "```python\n    return x * 2\n```") {
		tester.Fatal("expected correct completion")
	}
	humaneval.Execute = func(program string) bool { return false }
	if humaneval.Evaluate(conversation, "    return x * 3") {
		tester.Fatal("expected wrong completion")
	}
}

func contains(text, substring string) bool {
	for index := 0; index+len(substring) <= len(text); index++ {
		if text[index:index+len(substring)] == substring {
			return true
		}
	}
	return false
}

func TestSmolTalkFromRows(tester *testing.T) {
	smoltalk := NewSmolTalkFromRows([]SmolTalkRow{
		{Messages: []SmolTalkMessage{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "hello"}}},
	})
	if smoltalk.NumExamples() != 1 {
		tester.Fatalf("NumExamples = %d, want 1", smoltalk.NumExamples())
	}
	conversation := smoltalk.GetExample(0)
	if conversation.Messages[0].Role != "user" {
		tester.Fatal("first message should be user")
	}
}
