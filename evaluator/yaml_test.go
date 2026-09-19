package evaluator

import (
	"testing"
)

func TestParseYAML(tester *testing.T) {
	doc := `
icl_tasks:
  - label: arc_easy
    icl_task_type: multiple_choice
    dataset_uri: arc_easy.jsonl
    num_fewshot:
      - 0
    continuation_delimiter: ' '
  - label: squad
    icl_task_type: language_modeling
    dataset_uri: squad.jsonl
    num_fewshot:
      - 1
    continuation_delimiter: ' '
`
	root, err := ParseYAML(doc)
	if err != nil {
		tester.Fatalf("ParseYAML: %v", err)
	}
	tasks, ok := root["icl_tasks"].([]any)
	if !ok {
		tester.Fatalf("icl_tasks = %#v, want list", root["icl_tasks"])
	}
	if len(tasks) != 2 {
		tester.Fatalf("len = %d, want 2", len(tasks))
	}
	first := tasks[0].(map[string]any)
	if first["label"] != "arc_easy" {
		tester.Fatalf("label = %v, want arc_easy", first["label"])
	}
	if first["icl_task_type"] != "multiple_choice" {
		tester.Fatalf("type = %v", first["icl_task_type"])
	}
	numFewshot := first["num_fewshot"].([]any)
	if numFewshot[0] != int64(0) {
		tester.Fatalf("num_fewshot = %v", numFewshot)
	}
	if first["continuation_delimiter"] != " " {
		tester.Fatalf("delimiter = %q", first["continuation_delimiter"])
	}
}

func TestParseYAMLScalars(tester *testing.T) {
	doc := "a: 1\nb: 2.5\nc: hello\nd: true\ne: null\n"
	root, err := ParseYAML(doc)
	if err != nil {
		tester.Fatal(err)
	}
	if root["a"] != int64(1) {
		tester.Fatalf("a = %v", root["a"])
	}
	if root["b"] != 2.5 {
		tester.Fatalf("b = %v", root["b"])
	}
	if root["c"] != "hello" {
		tester.Fatalf("c = %v", root["c"])
	}
	if root["d"] != true {
		tester.Fatalf("d = %v", root["d"])
	}
	if root["e"] != nil {
		tester.Fatalf("e = %v", root["e"])
	}
}
