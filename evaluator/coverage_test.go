package evaluator

import (
	"strings"
	"testing"

	"github.com/cookiengineer/gonano/evaluator/tasks"
	"github.com/cookiengineer/gonano/inference"
)

func TestRenderSchemaPrompts(tester *testing.T) {
	item := CoreExample{"context_options": []string{"alpha", "beta"}, "continuation": "done"}
	prompts := renderSchemaPrompts(item, " ", nil)
	if len(prompts) != 2 {
		tester.Fatalf("prompt count = %d, want 2", len(prompts))
	}
	if !strings.Contains(prompts[0], "alpha") || !strings.Contains(prompts[1], "beta") {
		tester.Fatalf("prompts did not carry the context options: %v", prompts)
	}
}

func TestRenderLMPrompts(tester *testing.T) {
	item := CoreExample{"context": "the cat sat", "continuation": "on the mat"}
	prompts := renderLMPrompts(item, " ", nil)
	if len(prompts) != 2 {
		tester.Fatalf("prompt count = %d, want 2", len(prompts))
	}
	if !strings.Contains(prompts[0], "the cat sat") || strings.Contains(prompts[0], "on the mat") {
		tester.Fatalf("without-continuation prompt = %q", prompts[0])
	}
	if !strings.Contains(prompts[1], "on the mat") {
		tester.Fatalf("with-continuation prompt = %q", prompts[1])
	}
}

func TestTrimRight(tester *testing.T) {
	if trimmed := trimRight("hello  \n\t"); trimmed != "hello" {
		tester.Fatalf("trimRight = %q, want %q", trimmed, "hello")
	}
}

func TestCommonSuffixLength(tester *testing.T) {
	if suffix := findCommonSuffixLength([][]int{{1, 2, 3, 4}, {9, 2, 3, 4}}); suffix != 3 {
		tester.Fatalf("commonSuffixLength = %d, want 3", suffix)
	}
	if suffix := findCommonSuffixLength([][]int{{1, 2}, {3, 4}}); suffix != 0 {
		tester.Fatalf("commonSuffixLength = %d, want 0", suffix)
	}
}

func TestCategoricalAccuracy(tester *testing.T) {
	transformer := evalModel()
	tokenizer := evalTokenizer()
	task := tasks.NewMMLUFromRows([]tasks.MMLURow{
		{Question: "What is 2+2?", Choices: []string{"3", "4", "5", "6"}, Answer: 1},
	})
	accuracy := CategoricalAccuracy(task, transformer, tokenizer)
	if accuracy < 0 || accuracy > 1 {
		tester.Fatalf("accuracy = %v, want in [0,1]", accuracy)
	}
}

func TestGenerativeAccuracy(tester *testing.T) {
	transformer := evalModel()
	tokenizer := evalTokenizer()
	engine := inference.NewEngine(transformer, tokenizer)
	task := tasks.NewGSM8KFromRows([]tasks.GSM8KRow{
		{Question: "1+1?", Answer: "1+1=2\n#### 2"},
	})
	accuracy := GenerativeAccuracy(task, transformer, tokenizer, engine, 1, 2, 0.0, 0)
	if accuracy < 0 || accuracy > 1 {
		tester.Fatalf("accuracy = %v, want in [0,1]", accuracy)
	}
}
