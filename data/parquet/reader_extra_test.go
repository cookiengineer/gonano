package parquet

import (
	"reflect"
	"testing"
)

func TestReaderCountsAndBytes(t *testing.T) {
	page := buildPlainPage([][]byte{[]byte("a"), []byte("bc")})
	path := writeParquet(t, page, nil, 2)

	reader, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer reader.Close()

	if got := reader.NumRows(); got != 2 {
		t.Fatalf("NumRows = %d, want 2", got)
	}
	if got := reader.RowGroupNumRows(0); got != 2 {
		t.Fatalf("RowGroupNumRows = %d, want 2", got)
	}
	values, err := reader.ReadColumnBytes(0, "text")
	if err != nil {
		t.Fatalf("ReadColumnBytes: %v", err)
	}
	if len(values) != 2 || string(values[0]) != "a" || string(values[1]) != "bc" {
		t.Fatalf("ReadColumnBytes = %q", values)
	}
	if !reflect.DeepEqual(reader.ColumnNames(), []string{"text"}) {
		t.Fatalf("ColumnNames = %v", reader.ColumnNames())
	}
}
