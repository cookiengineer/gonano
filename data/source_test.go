package data

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// A minimal Thrift compact writer, used only to build test Parquet files.
type compactWriter struct {
	buffer    []byte
	lastField int
}

func (writer *compactWriter) field(id int, typ byte) {
	delta := id - writer.lastField
	if delta > 0 && delta <= 15 {
		writer.buffer = append(writer.buffer, byte(delta<<4)|typ)
	} else {
		writer.buffer = append(writer.buffer, typ)
		writer.zigzagI16(int16(id))
	}
	writer.lastField = id
}
func (writer *compactWriter) stop() { writer.buffer = append(writer.buffer, 0x00) }
func (writer *compactWriter) uvarint(value uint64) {
	for value >= 0x80 {
		writer.buffer = append(writer.buffer, byte(value)|0x80)
		value >>= 7
	}
	writer.buffer = append(writer.buffer, byte(value))
}
func (writer *compactWriter) zigzag(value int64) {
	writer.uvarint(uint64((value << 1) ^ (value >> 63)))
}
func (writer *compactWriter) zigzagI16(value int16) {
	writer.uvarint(uint64((int32(value) << 1) ^ (int32(value) >> 31)))
}
func (writer *compactWriter) i32(value int32) { writer.zigzag(int64(value)) }
func (writer *compactWriter) i64(value int64) { writer.zigzag(value) }
func (writer *compactWriter) str(text string) {
	writer.uvarint(uint64(len(text)))
	writer.buffer = append(writer.buffer, text...)
}
func (writer *compactWriter) listHeader(size int, elemType byte) {
	if size < 15 {
		writer.buffer = append(writer.buffer, byte(size<<4)|elemType)
	} else {
		writer.buffer = append(writer.buffer, 0xF0|elemType)
		writer.uvarint(uint64(size))
	}
}

// buildParquetFile writes a parquet file with a single "text" column (PLAIN,
// uncompressed) and returns its path.
func buildParquetFile(tests *testing.T, docs []string) string {
	tests.Helper()

	// PLAIN byte-array data.
	var data []byte
	for _, document := range docs {
		data = append(data, byte(len(document)))
		data = append(data, document...)
	}

	var dp compactWriter
	dp.field(1, 0x05)
	dp.i32(int32(len(docs)))
	dp.field(2, 0x05)
	dp.i32(0) // PLAIN
	dp.field(3, 0x05)
	dp.i32(3) // def level RLE
	dp.field(4, 0x05)
	dp.i32(3) // rep level RLE
	dp.stop()

	var ph compactWriter
	ph.field(1, 0x05)
	ph.i32(0) // DATA_PAGE
	ph.field(2, 0x05)
	ph.i32(int32(len(data)))
	ph.field(3, 0x05)
	ph.i32(int32(len(data)))
	ph.field(5, 0x0C)
	ph.buffer = append(ph.buffer, dp.buffer...)
	ph.stop()

	dataPage := append(append([]byte(nil), ph.buffer...), data...)

	body := []byte("PAR1")
	dataOffset := int64(len(body))
	body = append(body, dataPage...)

	var cc compactWriter
	cc.field(2, 0x06)
	cc.i64(dataOffset)
	cc.field(3, 0x0C)
	var cm compactWriter
	cm.field(1, 0x05)
	cm.i32(6) // BYTE_ARRAY
	cm.field(2, 0x09)
	cm.listHeader(1, 0x05)
	cm.i32(0) // PLAIN
	cm.field(3, 0x09)
	cm.listHeader(1, 0x08)
	cm.str("text")
	cm.field(4, 0x05)
	cm.i32(0) // UNCOMPRESSED
	cm.field(5, 0x06)
	cm.i64(int64(len(docs)))
	cm.field(6, 0x06)
	cm.i64(int64(len(data)))
	cm.field(7, 0x06)
	cm.i64(int64(len(data)))
	cm.field(9, 0x06)
	cm.i64(dataOffset)
	cm.stop()
	cc.buffer = append(cc.buffer, cm.buffer...)
	cc.stop()

	var rg compactWriter
	rg.field(1, 0x09)
	rg.listHeader(1, 0x0C)
	rg.buffer = append(rg.buffer, cc.buffer...)
	rg.field(2, 0x06)
	rg.i64(0)
	rg.field(3, 0x06)
	rg.i64(int64(len(docs)))
	rg.stop()

	var root compactWriter
	root.field(4, 0x08)
	root.str("schema")
	root.field(5, 0x05)
	root.i32(1)
	root.stop()

	var leaf compactWriter
	leaf.field(1, 0x05)
	leaf.i32(6)
	leaf.field(4, 0x08)
	leaf.str("text")
	leaf.stop()

	var meta compactWriter
	meta.field(1, 0x05)
	meta.i32(1)
	meta.field(2, 0x09)
	meta.listHeader(2, 0x0C)
	meta.buffer = append(meta.buffer, root.buffer...)
	meta.buffer = append(meta.buffer, leaf.buffer...)
	meta.field(3, 0x06)
	meta.i64(int64(len(docs)))
	meta.field(4, 0x09)
	meta.listHeader(1, 0x0C)
	meta.buffer = append(meta.buffer, rg.buffer...)
	meta.stop()

	body = append(body, meta.buffer...)
	body = append(body, make([]byte, 4)...)
	binary.LittleEndian.PutUint32(body[len(body)-4:], uint32(len(meta.buffer)))
	body = append(body, "PAR1"...)

	path := filepath.Join(tests.TempDir(), "test.parquet")
	if err := os.WriteFile(path, body, 0o644); err != nil {
		tests.Fatalf("write: %v", err)
	}
	return path
}

func TestParquetSourceYieldsDocuments(tests *testing.T) {
	path := buildParquetFile(tests, []string{"doc1", "doc2", "doc3", "doc4", "doc5"})
	src := NewParquetSource([]string{path}, 2)

	var got []string
	for len(got) < 6 {
		batch, _ := src.Next()
		got = append(got, batch...)
	}
	got = got[:6]
	want := []string{"doc1", "doc2", "doc3", "doc4", "doc5", "doc1"}
	if !reflect.DeepEqual(got, want) {
		tests.Fatalf("got %v, want %v", got, want)
	}
}

func TestParquetSourceWrapsEpoch(tests *testing.T) {
	path := buildParquetFile(tests, []string{"a", "b"})
	src := NewParquetSource([]string{path}, 100)
	_, state := src.Next()
	if state.Epoch != 1 {
		tests.Fatalf("epoch = %d, want 1", state.Epoch)
	}
	// Exhaust the first epoch (already got both docs), then wrap.
	_, state = src.Next()
	if state.Epoch != 2 {
		tests.Fatalf("epoch = %d, want 2", state.Epoch)
	}
}

func TestListParquetFiles(tests *testing.T) {
	dir := tests.TempDir()
	os.WriteFile(filepath.Join(dir, "shard_00001.parquet"), nil, 0o644)
	os.WriteFile(filepath.Join(dir, "shard_00002.parquet"), nil, 0o644)
	os.WriteFile(filepath.Join(dir, "shard_00003.parquet.tmp"), nil, 0o644)
	os.WriteFile(filepath.Join(dir, "readme.txt"), nil, 0o644)

	got := ListParquetFiles(dir)
	if len(got) != 2 {
		tests.Fatalf("got %d files, want 2 (%v)", len(got), got)
	}
}
