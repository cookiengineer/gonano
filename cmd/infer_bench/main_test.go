package main

import "testing"

func TestHumanBytes(t *testing.T) {
	cases := []struct {
		bytes int64
		want  string
	}{
		{0, "0 B"},
		{512, "512 B"},
		{1024, "1.0 KB"},
		{204800, "200.0 KB"},
		{1744830464, "1.6 GB"},
	}
	for _, testCase := range cases {
		if got := humanBytes(testCase.bytes); got != testCase.want {
			t.Errorf("humanBytes(%d) = %q, want %q", testCase.bytes, got, testCase.want)
		}
	}
}

func TestParseBatchSizes(t *testing.T) {
	cases := []struct {
		spec string
		want []int
	}{
		{"1", []int{1}},
		{"1,4,16", []int{1, 4, 16}},
		{"8,64", []int{8, 64}},
	}
	for _, testCase := range cases {
		got := parseBatchSizes(testCase.spec)
		if len(got) != len(testCase.want) {
			t.Fatalf("parseBatchSizes(%q) length = %d, want %d", testCase.spec, len(got), len(testCase.want))
		}
		for index := range got {
			if got[index] != testCase.want[index] {
				t.Errorf("parseBatchSizes(%q)[%d] = %d, want %d", testCase.spec, index, got[index], testCase.want[index])
			}
		}
	}
}
