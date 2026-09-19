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

func (writer *compactWriter) bin(data []byte) {
	writer.uvarint(uint64(len(data)))
	writer.buffer = append(writer.buffer, data...)
}

func (writer *compactWriter) str(text string) { writer.bin([]byte(text)) }

func (writer *compactWriter) listHeader(size int, elemType byte) {
	if size < 15 {
		writer.buffer = append(writer.buffer, byte(size<<4)|elemType)
	} else {
		writer.buffer = append(writer.buffer, 0xF0|elemType)
		writer.uvarint(uint64(size))
	}
}

// buildPlainPage encodes a DATA_PAGE with PLAIN BYTE_ARRAY values.
func buildPlainPage(values [][]byte) []byte {
	var data []byte
	for _, value := range values {
		var lengthWriter compactWriter
		lengthWriter.uvarint(uint64(len(value)))
		data = append(data, lengthWriter.buffer...)
		data = append(data, value...)
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
	ph.buffer = append(ph.buffer, dp.buffer...)
	ph.stop()

	out := append([]byte(nil), ph.buffer...)
	out = append(out, data...)
	return out
}

// buildDictionaryPage encodes a DICTIONARY_PAGE with PLAIN BYTE_ARRAY values.
func buildDictionaryPage(dict [][]byte) []byte {
	var data []byte
	for _, value := range dict {
		var lengthWriter compactWriter
		lengthWriter.uvarint(uint64(len(value)))
		data = append(data, lengthWriter.buffer...)
		data = append(data, value...)
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
	ph.buffer = append(ph.buffer, dh.buffer...)
	ph.stop()

	return append(append([]byte(nil), ph.buffer...), data...)
}

// buildRLEDictionaryIndexData builds an RLE/bit-packed encoded index buffer.
func buildRLEDictionaryIndexData(idxs []int32, bitWidth int) []byte {
	// Use a single RLE run of all indexes (only valid if they are identical);
	// for general indexes use bit-packed runs. Here we support both by building
	// bit-packed runs of 8 values.
	var body []byte
	for start := 0; start < len(idxs); start += 8 {
		remaining := len(idxs) - start
		if remaining > 8 {
			remaining = 8
		}
		// Bit-packed run header: 1 group (bit 0 set).
		var headerWriter compactWriter
		headerWriter.uvarint(uint64(1)<<1 | 1)
		body = append(body, headerWriter.buffer...)
		group := make([]byte, bitWidth)
		for valueIndex := 0; valueIndex < remaining; valueIndex++ {
			value := uint64(uint32(idxs[start+valueIndex]))
			for bit := 0; bit < bitWidth; bit++ {
				if (value>>bit)&1 == 1 {
					bitPos := valueIndex*bitWidth + bit
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
	for _, column := range columns {
		var chunkWriter compactWriter
		chunkWriter.field(2, ctI64)
		chunkWriter.i64(column.fileOffset)
		if column.meta != nil {
			chunkWriter.field(3, ctStruct)
			var columnMetaWriter compactWriter
			columnMetaWriter.field(1, ctI32)
			columnMetaWriter.i32(column.meta.typ)
			columnMetaWriter.field(2, ctList)
			columnMetaWriter.listHeader(len(column.meta.encodings), ctI32)
			for _, encoding := range column.meta.encodings {
				columnMetaWriter.i32(encoding)
			}
			columnMetaWriter.field(3, ctList)
			columnMetaWriter.listHeader(len(column.meta.pathInSchema), ctBinary)
			for _, path := range column.meta.pathInSchema {
				columnMetaWriter.str(path)
			}
			columnMetaWriter.field(4, ctI32)
			columnMetaWriter.i32(column.meta.codec)
			columnMetaWriter.field(5, ctI64)
			columnMetaWriter.i64(column.meta.numValues)
			columnMetaWriter.field(6, ctI64)
			columnMetaWriter.i64(column.meta.totalUncompressedSize)
			columnMetaWriter.field(7, ctI64)
			columnMetaWriter.i64(column.meta.totalCompressedSize)
			columnMetaWriter.field(9, ctI64)
			columnMetaWriter.i64(column.meta.dataPageOffset)
			if column.meta.dictionaryPageOffset > 0 {
				columnMetaWriter.field(11, ctI64)
				columnMetaWriter.i64(column.meta.dictionaryPageOffset)
			}
			columnMetaWriter.stop()
			chunkWriter.buffer = append(chunkWriter.buffer, columnMetaWriter.buffer...)
		}
		chunkWriter.stop()
		rg.buffer = append(rg.buffer, chunkWriter.buffer...)
	}
	rg.field(2, ctI64)
	rg.i64(0) // total_byte_size
	rg.field(3, ctI64)
	rg.i64(numRows)
	rg.stop()

	var metadataWriter compactWriter
	metadataWriter.field(1, ctI32)
	metadataWriter.i32(1)
	metadataWriter.field(2, ctList)
	metadataWriter.listHeader(2, ctStruct)
	metadataWriter.buffer = append(metadataWriter.buffer, root.buffer...)
	metadataWriter.buffer = append(metadataWriter.buffer, leaf.buffer...)
	metadataWriter.field(3, ctI64)
	metadataWriter.i64(numRows)
	metadataWriter.field(4, ctList)
	metadataWriter.listHeader(1, ctStruct)
	metadataWriter.buffer = append(metadataWriter.buffer, rg.buffer...)
	metadataWriter.stop()
	return metadataWriter.buffer
}

func writeParquet(tests *testing.T, dataPage []byte, dictPage []byte, numRows int64) string {
	tests.Helper()
	// Layout: "PAR1" | [dictionary page] | [data page] | FileMetaData | len | "PAR1"
	body := []byte("PAR1")
	dictOffset := int64(0)
	if len(dictPage) > 0 {
		dictOffset = int64(len(body))
		body = append(body, dictPage...)
	}
	dataOffset := int64(len(body))
	body = append(body, dataPage...)

	chunk := columnChunk{
		fileOffset: dataOffset,
		meta: &columnMetaData{
			typ:                  typeByteArray,
			encodings:            []int32{encPlain},
			pathInSchema:         []string{"text"},
			codec:                codecUncompressed,
			numValues:            numRows,
			dataPageOffset:       dataOffset,
			dictionaryPageOffset: dictOffset,
		},
	}
	metadata := buildMetaClean([]columnChunk{chunk}, numRows)
	body = append(body, metadata...)
	body = append(body, make([]byte, 4)...)
	binary.LittleEndian.PutUint32(body[len(body)-4:], uint32(len(metadata)))
	body = append(body, "PAR1"...)

	path := filepath.Join(tests.TempDir(), "test.parquet")
	if err := os.WriteFile(path, body, 0o644); err != nil {
		tests.Fatalf("write: %v", err)
	}
	return path
}

func TestReaderPlainStrings(tests *testing.T) {
	values := [][]byte{[]byte("hello"), []byte("world"), []byte("foo bar")}
	page := buildPlainPage(values)
	path := writeParquet(tests, page, nil, int64(len(values)))

	reader, err := Open(path)
	if err != nil {
		tests.Fatalf("Open: %v", err)
	}
	defer reader.Close()
	if reader.NumRowGroups() != 1 {
		tests.Fatalf("row groups = %d, want 1", reader.NumRowGroups())
	}
	got, err := reader.ReadColumnStrings(0, "text")
	if err != nil {
		tests.Fatalf("ReadColumnStrings: %v", err)
	}
	want := []string{"hello", "world", "foo bar"}
	if !reflect.DeepEqual(got, want) {
		tests.Fatalf("got %v, want %v", got, want)
	}
}

func TestReaderRLEDictionary(tests *testing.T) {
	dict := [][]byte{[]byte("alpha"), []byte("beta"), []byte("gamma")}
	dictPage := buildDictionaryPage(dict)
	idxs := []int32{0, 1, 2, 0, 1}
	bitWidth := computeDictBitWidth(len(dict))
	indexData := buildRLEDictionaryIndexData(idxs, bitWidth)
	dataPage := buildDataPage(indexData, len(idxs), encRLE_Dictionary)

	path := writeParquet(tests, dataPage, dictPage, int64(len(idxs)))
	reader, err := Open(path)
	if err != nil {
		tests.Fatalf("Open: %v", err)
	}
	defer reader.Close()
	got, err := reader.ReadColumnStrings(0, "text")
	if err != nil {
		tests.Fatalf("ReadColumnStrings: %v", err)
	}
	want := []string{"alpha", "beta", "gamma", "alpha", "beta"}
	if !reflect.DeepEqual(got, want) {
		tests.Fatalf("got %v, want %v", got, want)
	}
}

func TestReaderColumnNames(tests *testing.T) {
	page := buildPlainPage([][]byte{[]byte("x")})
	path := writeParquet(tests, page, nil, 1)
	reader, err := Open(path)
	if err != nil {
		tests.Fatalf("Open: %v", err)
	}
	defer reader.Close()
	names := reader.ColumnNames()
	if !reflect.DeepEqual(names, []string{"text"}) {
		tests.Fatalf("names = %v, want [text]", names)
	}
}

func TestReaderMissingColumn(tests *testing.T) {
	page := buildPlainPage([][]byte{[]byte("x")})
	path := writeParquet(tests, page, nil, 1)
	reader, err := Open(path)
	if err != nil {
		tests.Fatalf("Open: %v", err)
	}
	defer reader.Close()
	if _, err := reader.ReadColumnStrings(0, "nope"); err == nil {
		tests.Fatal("expected error for missing column")
	}
}

func TestReaderRejectsNonParquet(tests *testing.T) {
	path := filepath.Join(tests.TempDir(), "bad.parquet")
	os.WriteFile(path, []byte("not a parquet file"), 0o644)
	if _, err := Open(path); err == nil {
		tests.Fatal("expected error for non-parquet file")
	}
}
