package parquet

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// compactWriter builds Thrift Compact Protocol bytes, used to construct valid
// test Parquet files.
type compactWriter struct {
	b         []byte
	lastField int
}

func (w *compactWriter) field(id int, typ byte) {
	delta := id - w.lastField
	if delta > 0 && delta <= 15 {
		w.b = append(w.b, byte(delta<<4)|typ)
	} else {
		w.b = append(w.b, typ)
		w.zigzagI16(int16(id))
	}
	w.lastField = id
}

func (w *compactWriter) stop() { w.b = append(w.b, 0x00) }

func (w *compactWriter) uvarint(u uint64) {
	for u >= 0x80 {
		w.b = append(w.b, byte(u)|0x80)
		u >>= 7
	}
	w.b = append(w.b, byte(u))
}

func (w *compactWriter) zigzag(v int64) {
	w.uvarint(uint64((v << 1) ^ (v >> 63)))
}

func (w *compactWriter) zigzagI16(v int16) {
	w.uvarint(uint64((int32(v) << 1) ^ (int32(v) >> 31)))
}

func (w *compactWriter) i32(v int32) { w.zigzag(int64(v)) }
func (w *compactWriter) i64(v int64) { w.zigzag(v) }

func (w *compactWriter) bin(s []byte) {
	w.uvarint(uint64(len(s)))
	w.b = append(w.b, s...)
}

func (w *compactWriter) str(s string) { w.bin([]byte(s)) }

func (w *compactWriter) listHeader(size int, elemType byte) {
	if size < 15 {
		w.b = append(w.b, byte(size<<4)|elemType)
	} else {
		w.b = append(w.b, 0xF0|elemType)
		w.uvarint(uint64(size))
	}
}

// buildPlainPage encodes a DATA_PAGE with PLAIN BYTE_ARRAY values.
func buildPlainPage(values [][]byte) []byte {
	var data []byte
	for _, v := range values {
		var u compactWriter
		u.uvarint(uint64(len(v)))
		data = append(data, u.b...)
		data = append(data, v...)
	}
	return buildDataPage(data, len(values), encPlain)
}

// buildDataPage wraps page data in a PageHeader + DataPageHeader.
func buildDataPage(data []byte, numValues int, encoding int32) []byte {
	var dp compactWriter
	dp.field(1, ctI32)
	dp.i32(int32(numValues))
	dp.field(2, ctI32)
	dp.i32(encoding)
	dp.field(3, ctI32)
	dp.i32(encRLE) // definition level encoding
	dp.field(4, ctI32)
	dp.i32(encRLE) // repetition level encoding
	dp.stop()

	var ph compactWriter
	ph.field(1, ctI32)
	ph.i32(pageData) // DATA_PAGE
	ph.field(2, ctI32)
	ph.i32(int32(len(data))) // uncompressed size
	ph.field(3, ctI32)
	ph.i32(int32(len(data))) // compressed size
	ph.field(5, ctStruct)
	ph.b = append(ph.b, dp.b...)
	ph.stop()

	out := append([]byte(nil), ph.b...)
	out = append(out, data...)
	return out
}

// buildDictionaryPage encodes a DICTIONARY_PAGE with PLAIN BYTE_ARRAY values.
func buildDictionaryPage(dict [][]byte) []byte {
	var data []byte
	for _, v := range dict {
		var u compactWriter
		u.uvarint(uint64(len(v)))
		data = append(data, u.b...)
		data = append(data, v...)
	}
	var dh compactWriter
	dh.field(1, ctI32)
	dh.i32(int32(len(dict)))
	dh.field(2, ctI32)
	dh.i32(encPlain)
	dh.stop()

	var ph compactWriter
	ph.field(1, ctI32)
	ph.i32(pageDictionary)
	ph.field(2, ctI32)
	ph.i32(int32(len(data)))
	ph.field(3, ctI32)
	ph.i32(int32(len(data)))
	ph.field(7, ctStruct)
	ph.b = append(ph.b, dh.b...)
	ph.stop()

	return append(append([]byte(nil), ph.b...), data...)
}

// buildRLEDictionaryIndexData builds an RLE/bit-packed encoded index buffer.
func buildRLEDictionaryIndexData(idxs []int32, bitWidth int) []byte {
	// Use a single RLE run of all indexes (only valid if they are identical);
	// for general indexes use bit-packed runs. Here we support both by building
	// bit-packed runs of 8 values.
	var body []byte
	for i := 0; i < len(idxs); i += 8 {
		n := len(idxs) - i
		if n > 8 {
			n = 8
		}
		// Bit-packed run header: 1 group (bit 0 set).
		var u compactWriter
		u.uvarint(uint64(1)<<1 | 1)
		body = append(body, u.b...)
		group := make([]byte, bitWidth)
		for v := 0; v < n; v++ {
			val := uint64(uint32(idxs[i+v]))
			for b := 0; b < bitWidth; b++ {
				if (val>>b)&1 == 1 {
					bitPos := v*bitWidth + b
					group[bitPos/8] |= 1 << (bitPos % 8)
				}
			}
		}
		body = append(body, group...)
	}
	// Prefix with 4-byte little-endian length.
	out := make([]byte, 4)
	binary.LittleEndian.PutUint32(out, uint32(len(body)))
	return append(out, body...)
}

// buildMetaClean builds FileMetaData compact bytes.
func buildMetaClean(columns []columnChunk, numRows int64) []byte {
	var root compactWriter
	root.field(4, ctBinary)
	root.str("schema")
	root.field(5, ctI32)
	root.i32(1) // num_children = 1
	root.stop()

	var leaf compactWriter
	leaf.field(1, ctI32)
	leaf.i32(typeByteArray)
	leaf.field(4, ctBinary)
	leaf.str("text")
	leaf.stop()

	// One row group with the given columns.
	var rg compactWriter
	rg.field(1, ctList)
	rg.listHeader(len(columns), ctStruct)
	for _, cc := range columns {
		var c compactWriter
		c.field(2, ctI64)
		c.i64(cc.fileOffset)
		if cc.meta != nil {
			c.field(3, ctStruct)
			var cm compactWriter
			cm.field(1, ctI32)
			cm.i32(cc.meta.typ)
			cm.field(2, ctList)
			cm.listHeader(len(cc.meta.encodings), ctI32)
			for _, e := range cc.meta.encodings {
				cm.i32(e)
			}
			cm.field(3, ctList)
			cm.listHeader(len(cc.meta.pathInSchema), ctBinary)
			for _, p := range cc.meta.pathInSchema {
				cm.str(p)
			}
			cm.field(4, ctI32)
			cm.i32(cc.meta.codec)
			cm.field(5, ctI64)
			cm.i64(cc.meta.numValues)
			cm.field(6, ctI64)
			cm.i64(cc.meta.totalUncompressedSize)
			cm.field(7, ctI64)
			cm.i64(cc.meta.totalCompressedSize)
			cm.field(9, ctI64)
			cm.i64(cc.meta.dataPageOffset)
			if cc.meta.dictionaryPageOffset > 0 {
				cm.field(11, ctI64)
				cm.i64(cc.meta.dictionaryPageOffset)
			}
			cm.stop()
			c.b = append(c.b, cm.b...)
		}
		c.stop()
		rg.b = append(rg.b, c.b...)
	}
	rg.field(2, ctI64)
	rg.i64(0) // total_byte_size
	rg.field(3, ctI64)
	rg.i64(numRows)
	rg.stop()

	var w compactWriter
	w.field(1, ctI32)
	w.i32(1)
	w.field(2, ctList)
	w.listHeader(2, ctStruct)
	w.b = append(w.b, root.b...)
	w.b = append(w.b, leaf.b...)
	w.field(3, ctI64)
	w.i64(numRows)
	w.field(4, ctList)
	w.listHeader(1, ctStruct)
	w.b = append(w.b, rg.b...)
	w.stop()
	return w.b
}

func writeParquet(t *testing.T, dataPage []byte, dictPage []byte, numRows int64) string {
	t.Helper()
	// Layout: "PAR1" | [dictionary page] | [data page] | FileMetaData | len | "PAR1"
	body := []byte("PAR1")
	dictOffset := int64(0)
	if len(dictPage) > 0 {
		dictOffset = int64(len(body))
		body = append(body, dictPage...)
	}
	dataOffset := int64(len(body))
	body = append(body, dataPage...)

	cc := columnChunk{
		fileOffset: dataOffset,
		meta: &columnMetaData{
			typ:                   typeByteArray,
			encodings:             []int32{encPlain},
			pathInSchema:          []string{"text"},
			codec:                 codecUncompressed,
			numValues:             numRows,
			dataPageOffset:        dataOffset,
			dictionaryPageOffset:  dictOffset,
		},
	}
	meta := buildMetaClean([]columnChunk{cc}, numRows)
	body = append(body, meta...)
	body = append(body, make([]byte, 4)...)
	binary.LittleEndian.PutUint32(body[len(body)-4:], uint32(len(meta)))
	body = append(body, "PAR1"...)

	path := filepath.Join(t.TempDir(), "test.parquet")
	if err := os.WriteFile(path, body, 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	return path
}

func TestReaderPlainStrings(t *testing.T) {
	values := [][]byte{[]byte("hello"), []byte("world"), []byte("foo bar")}
	page := buildPlainPage(values)
	path := writeParquet(t, page, nil, int64(len(values)))

	r, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer r.Close()
	if r.NumRowGroups() != 1 {
		t.Fatalf("row groups = %d, want 1", r.NumRowGroups())
	}
	got, err := r.ReadColumnStrings(0, "text")
	if err != nil {
		t.Fatalf("ReadColumnStrings: %v", err)
	}
	want := []string{"hello", "world", "foo bar"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestReaderRLEDictionary(t *testing.T) {
	dict := [][]byte{[]byte("alpha"), []byte("beta"), []byte("gamma")}
	dictPage := buildDictionaryPage(dict)
	idxs := []int32{0, 1, 2, 0, 1}
	bitWidth := bitWidthForDict(len(dict))
	indexData := buildRLEDictionaryIndexData(idxs, bitWidth)
	dataPage := buildDataPage(indexData, len(idxs), encRLE_Dictionary)

	path := writeParquet(t, dataPage, dictPage, int64(len(idxs)))
	r, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer r.Close()
	got, err := r.ReadColumnStrings(0, "text")
	if err != nil {
		t.Fatalf("ReadColumnStrings: %v", err)
	}
	want := []string{"alpha", "beta", "gamma", "alpha", "beta"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestReaderColumnNames(t *testing.T) {
	page := buildPlainPage([][]byte{[]byte("x")})
	path := writeParquet(t, page, nil, 1)
	r, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer r.Close()
	names := r.ColumnNames()
	if !reflect.DeepEqual(names, []string{"text"}) {
		t.Fatalf("names = %v, want [text]", names)
	}
}

func TestReaderMissingColumn(t *testing.T) {
	page := buildPlainPage([][]byte{[]byte("x")})
	path := writeParquet(t, page, nil, 1)
	r, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer r.Close()
	if _, err := r.ReadColumnStrings(0, "nope"); err == nil {
		t.Fatal("expected error for missing column")
	}
}

func TestReaderRejectsNonParquet(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.parquet")
	os.WriteFile(path, []byte("not a parquet file"), 0o644)
	if _, err := Open(path); err == nil {
		t.Fatal("expected error for non-parquet file")
	}
}
