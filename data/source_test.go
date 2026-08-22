package data

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// A minimal Thrift compact writer, used only to build test Parquet files.
type cw struct {
	b         []byte
	lastField int
}

func (w *cw) field(id int, typ byte) {
	delta := id - w.lastField
	if delta > 0 && delta <= 15 {
		w.b = append(w.b, byte(delta<<4)|typ)
	} else {
		w.b = append(w.b, typ)
		w.zigzagI16(int16(id))
	}
	w.lastField = id
}
func (w *cw) stop() { w.b = append(w.b, 0x00) }
func (w *cw) uvarint(u uint64) {
	for u >= 0x80 {
		w.b = append(w.b, byte(u)|0x80)
		u >>= 7
	}
	w.b = append(w.b, byte(u))
}
func (w *cw) zigzag(v int64) { w.uvarint(uint64((v << 1) ^ (v >> 63))) }
func (w *cw) zigzagI16(v int16) { w.uvarint(uint64((int32(v) << 1) ^ (int32(v) >> 31))) }
func (w *cw) i32(v int32) { w.zigzag(int64(v)) }
func (w *cw) i64(v int64) { w.zigzag(v) }
func (w *cw) str(s string) {
	w.uvarint(uint64(len(s)))
	w.b = append(w.b, s...)
}
func (w *cw) listHeader(size int, elemType byte) {
	if size < 15 {
		w.b = append(w.b, byte(size<<4)|elemType)
	} else {
		w.b = append(w.b, 0xF0|elemType)
		w.uvarint(uint64(size))
	}
}

// buildParquetFile writes a parquet file with a single "text" column (PLAIN,
// uncompressed) and returns its path.
func buildParquetFile(t *testing.T, docs []string) string {
	t.Helper()

	// PLAIN byte-array data.
	var data []byte
	for _, d := range docs {
		data = append(data, byte(len(d)))
		data = append(data, d...)
	}

	var dp cw
	dp.field(1, 0x05)
	dp.i32(int32(len(docs)))
	dp.field(2, 0x05)
	dp.i32(0) // PLAIN
	dp.field(3, 0x05)
	dp.i32(3) // def level RLE
	dp.field(4, 0x05)
	dp.i32(3) // rep level RLE
	dp.stop()

	var ph cw
	ph.field(1, 0x05)
	ph.i32(0) // DATA_PAGE
	ph.field(2, 0x05)
	ph.i32(int32(len(data)))
	ph.field(3, 0x05)
	ph.i32(int32(len(data)))
	ph.field(5, 0x0C)
	ph.b = append(ph.b, dp.b...)
	ph.stop()

	dataPage := append(append([]byte(nil), ph.b...), data...)

	body := []byte("PAR1")
	dataOffset := int64(len(body))
	body = append(body, dataPage...)

	var cc cw
	cc.field(2, 0x06)
	cc.i64(dataOffset)
	cc.field(3, 0x0C)
	var cm cw
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
	cc.b = append(cc.b, cm.b...)
	cc.stop()

	var rg cw
	rg.field(1, 0x09)
	rg.listHeader(1, 0x0C)
	rg.b = append(rg.b, cc.b...)
	rg.field(2, 0x06)
	rg.i64(0)
	rg.field(3, 0x06)
	rg.i64(int64(len(docs)))
	rg.stop()

	var root cw
	root.field(4, 0x08)
	root.str("schema")
	root.field(5, 0x05)
	root.i32(1)
	root.stop()

	var leaf cw
	leaf.field(1, 0x05)
	leaf.i32(6)
	leaf.field(4, 0x08)
	leaf.str("text")
	leaf.stop()

	var meta cw
	meta.field(1, 0x05)
	meta.i32(1)
	meta.field(2, 0x09)
	meta.listHeader(2, 0x0C)
	meta.b = append(meta.b, root.b...)
	meta.b = append(meta.b, leaf.b...)
	meta.field(3, 0x06)
	meta.i64(int64(len(docs)))
	meta.field(4, 0x09)
	meta.listHeader(1, 0x0C)
	meta.b = append(meta.b, rg.b...)
	meta.stop()

	body = append(body, meta.b...)
	body = append(body, make([]byte, 4)...)
	binary.LittleEndian.PutUint32(body[len(body)-4:], uint32(len(meta.b)))
	body = append(body, "PAR1"...)

	path := filepath.Join(t.TempDir(), "test.parquet")
	if err := os.WriteFile(path, body, 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	return path
}

func TestParquetSourceYieldsDocuments(t *testing.T) {
	path := buildParquetFile(t, []string{"doc1", "doc2", "doc3", "doc4", "doc5"})
	src := NewParquetSource([]string{path}, 2)

	var got []string
	for len(got) < 6 {
		batch, _ := src.Next()
		got = append(got, batch...)
	}
	got = got[:6]
	want := []string{"doc1", "doc2", "doc3", "doc4", "doc5", "doc1"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestParquetSourceWrapsEpoch(t *testing.T) {
	path := buildParquetFile(t, []string{"a", "b"})
	src := NewParquetSource([]string{path}, 100)
	_, state := src.Next()
	if state.Epoch != 1 {
		t.Fatalf("epoch = %d, want 1", state.Epoch)
	}
	// Exhaust the first epoch (already got both docs), then wrap.
	_, state = src.Next()
	if state.Epoch != 2 {
		t.Fatalf("epoch = %d, want 2", state.Epoch)
	}
}

func TestListParquetFiles(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "shard_00001.parquet"), nil, 0o644)
	os.WriteFile(filepath.Join(dir, "shard_00002.parquet"), nil, 0o644)
	os.WriteFile(filepath.Join(dir, "shard_00003.parquet.tmp"), nil, 0o644)
	os.WriteFile(filepath.Join(dir, "readme.txt"), nil, 0o644)

	got := ListParquetFiles(dir)
	if len(got) != 2 {
		t.Fatalf("got %d files, want 2 (%v)", len(got), got)
	}
}
