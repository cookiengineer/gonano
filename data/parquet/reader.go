package parquet

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
)

var errBadParquet = errors.New("parquet: corrupt file")

// Reader reads flat Parquet columns. It supports PLAIN and RLE_DICTIONARY
// encodings for BYTE_ARRAY, INT32, and INT64 columns, with SNAPPY or no
// compression.
type Reader struct {
	file *os.File
	meta fileMetaData
}

// Open opens a Parquet file and parses its footer metadata.
func Open(path string) (*Reader, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	fileInfo, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, err
	}
	size := fileInfo.Size()
	if size < 12 {
		file.Close()
		return nil, errBadParquet
	}
	// Footer: 4-byte metadata length + "PAR1". Verify leading magic too.
	tail := make([]byte, 8)
	if _, err := file.ReadAt(tail, size-8); err != nil {
		file.Close()
		return nil, err
	}
	if string(tail[4:]) != "PAR1" {
		file.Close()
		return nil, errBadParquet
	}
	metaLen := int64(binary.LittleEndian.Uint32(tail[:4]))
	if metaLen <= 0 || metaLen > size-8 {
		file.Close()
		return nil, errBadParquet
	}
	metaBytes := make([]byte, metaLen)
	if _, err := file.ReadAt(metaBytes, size-8-metaLen); err != nil {
		file.Close()
		return nil, err
	}
	meta, err := parseFileMetaData(metaBytes)
	if err != nil {
		file.Close()
		return nil, err
	}
	return &Reader{file: file, meta: meta}, nil
}

// Close closes the underlying file.
func (reader *Reader) Close() error { return reader.file.Close() }

// NumRowGroups returns the number of row groups.
func (reader *Reader) NumRowGroups() int { return len(reader.meta.rowGroups) }

// NumRows returns the total number of rows in the file.
func (reader *Reader) NumRows() int64 { return reader.meta.numRows }

// RowGroupNumRows returns the number of rows in a row group.
func (reader *Reader) RowGroupNumRows(rowGroupIndex int) int64 {
	return reader.meta.rowGroups[rowGroupIndex].numRows
}

// ColumnNames returns the leaf column names (flat columns only).
func (reader *Reader) ColumnNames() []string {
	seen := map[string]bool{}
	var names []string
	for _, rowGroup := range reader.meta.rowGroups {
		for _, chunk := range rowGroup.columns {
			if chunk.meta == nil || len(chunk.meta.pathInSchema) == 0 {
				continue
			}
			name := chunk.meta.pathInSchema[len(chunk.meta.pathInSchema)-1]
			if !seen[name] {
				seen[name] = true
				names = append(names, name)
			}
		}
	}
	return names
}

// findChunk locates the column chunk for a named flat column in a row group.
func (reader *Reader) findChunk(rowGroupIndex int, name string) (columnChunk, bool) {
	for _, chunk := range reader.meta.rowGroups[rowGroupIndex].columns {
		if chunk.meta != nil && len(chunk.meta.pathInSchema) > 0 &&
			chunk.meta.pathInSchema[len(chunk.meta.pathInSchema)-1] == name {
			return chunk, true
		}
	}
	return columnChunk{}, false
}

// ReadColumnStrings reads a flat BYTE_ARRAY column for a row group as strings.
func (reader *Reader) ReadColumnStrings(rowGroupIndex int, name string) ([]string, error) {
	chunk, ok := reader.findChunk(rowGroupIndex, name)
	if !ok {
		return nil, fmt.Errorf("parquet: column %q not found", name)
	}
	values, err := reader.readColumnValues(chunk)
	if err != nil {
		return nil, err
	}
	out := make([]string, len(values))
	for index, value := range values {
		out[index] = string(value)
	}
	return out, nil
}

// ReadColumnBytes reads a flat BYTE_ARRAY column as raw bytes.
func (reader *Reader) ReadColumnBytes(rowGroupIndex int, name string) ([][]byte, error) {
	chunk, ok := reader.findChunk(rowGroupIndex, name)
	if !ok {
		return nil, fmt.Errorf("parquet: column %q not found", name)
	}
	return reader.readColumnValues(chunk)
}

// ReadColumnInt64 reads a flat INT32 or INT64 column as int64.
func (reader *Reader) ReadColumnInt64(rowGroupIndex int, name string) ([]int64, error) {
	chunk, ok := reader.findChunk(rowGroupIndex, name)
	if !ok {
		return nil, fmt.Errorf("parquet: column %q not found", name)
	}
	dict, pages, err := reader.readPages(chunk)
	if err != nil {
		return nil, err
	}
	var out []int64
	for _, page := range pages {
		encoding := page.encoding
		switch encoding {
		case encPlain:
			if chunk.meta.typ == typeInt32 {
				vals, err := decodePlainInt32(page.data, int(page.numValues))
				if err != nil {
					return nil, err
				}
				for _, value := range vals {
					out = append(out, int64(value))
				}
			} else {
				vals, err := decodePlainInt64(page.data, int(page.numValues))
				if err != nil {
					return nil, err
				}
				out = append(out, vals...)
			}
		case encRLE_Dictionary:
			bitWidth := computeDictBitWidth(len(dict))
			idxs, err := decodeRLEBitPacked(page.data, bitWidth, int(page.numValues))
			if err != nil {
				return nil, err
			}
			for _, dictIndex := range idxs {
				if dictIndex < 0 || int(dictIndex) >= len(dict) {
					return nil, errBadEncoding
				}
				out = append(out, int64(decodeDictInt(dict[dictIndex])))
			}
		default:
			return nil, fmt.Errorf("parquet: unsupported encoding %d for int column", encoding)
		}
	}
	return out, nil
}

func decodeDictInt(encoded []byte) int32 {
	if len(encoded) == 4 {
		return int32(binary.LittleEndian.Uint32(encoded))
	}
	if len(encoded) == 8 {
		return int32(binary.LittleEndian.Uint64(encoded))
	}
	// Fallback: decode as text? Not expected for int columns.
	return 0
}

// readColumnValues reads a BYTE_ARRAY column, handling PLAIN and
// RLE_DICTIONARY encodings.
func (reader *Reader) readColumnValues(chunk columnChunk) ([][]byte, error) {
	dict, pages, err := reader.readPages(chunk)
	if err != nil {
		return nil, err
	}
	var out [][]byte
	for _, page := range pages {
		switch page.encoding {
		case encPlain:
			vals, err := decodePlainByteArray(page.data, int(page.numValues))
			if err != nil {
				return nil, err
			}
			out = append(out, vals...)
		case encRLE_Dictionary:
			bitWidth := computeDictBitWidth(len(dict))
			idxs, err := decodeRLEBitPacked(page.data, bitWidth, int(page.numValues))
			if err != nil {
				return nil, err
			}
			for _, dictIndex := range idxs {
				if dictIndex < 0 || int(dictIndex) >= len(dict) {
					return nil, errBadEncoding
				}
				out = append(out, dict[dictIndex])
			}
		default:
			return nil, fmt.Errorf("parquet: unsupported encoding %d", page.encoding)
		}
	}
	return out, nil
}

type rawPage struct {
	numValues int32
	encoding  int32
	data      []byte
}

// readPages reads all pages of a column chunk, returning the dictionary
// entries (empty if no dictionary) and the data pages.
func (reader *Reader) readPages(chunk columnChunk) ([][]byte, []rawPage, error) {
	if chunk.meta == nil {
		return nil, nil, errBadParquet
	}
	var dict [][]byte
	var pages []rawPage

	if chunk.meta.dictionaryPageOffset > 0 {
		header, data, err := reader.readPageAt(chunk.meta.dictionaryPageOffset, chunk.meta.codec)
		if err != nil {
			return nil, nil, err
		}
		if header.dictionary == nil {
			return nil, nil, errBadParquet
		}
		dict, err = decodePlainByteArray(data, int(header.dictionary.numValues))
		if err != nil {
			return nil, nil, err
		}
	}

	offset := chunk.meta.dataPageOffset
	if offset == 0 {
		offset = chunk.fileOffset
	}
	var total int64
	for total < chunk.meta.numValues {
		header, data, err := reader.readPageAt(offset, chunk.meta.codec)
		if err != nil {
			return nil, nil, err
		}
		if header.typ == pageDictionary {
			offset += header.headerLen + int64(header.compressedSize)
			continue
		}
		if header.data == nil {
			return nil, nil, errBadParquet
		}
		pages = append(pages, rawPage{
			numValues: header.data.numValues,
			encoding:  header.data.encoding,
			data:      data,
		})
		total += int64(header.data.numValues)
		offset += header.headerLen + int64(header.compressedSize)
	}
	return dict, pages, nil
}

type rawPageHeader struct {
	typ              int32
	uncompressedSize int32
	compressedSize   int32
	headerLen        int64
	data             *dataPageHeader
	dictionary       *dictionaryPageHeader
}

// readPageAt reads and decompresses a single page starting at the given file
// offset.
func (reader *Reader) readPageAt(offset int64, codec int32) (rawPageHeader, []byte, error) {
	const headerWindow = 1 << 16
	window := make([]byte, headerWindow)
	bytesRead, err := reader.file.ReadAt(window, offset)
	if err != nil && err != io.EOF {
		return rawPageHeader{}, nil, err
	}
	window = window[:bytesRead]
	header, consumed, err := parsePageHeader(window)
	if err != nil {
		return rawPageHeader{}, nil, err
	}
	header.headerLen = int64(consumed)
	compressed := make([]byte, header.compressedSize)
	if _, err := reader.file.ReadAt(compressed, offset+int64(consumed)); err != nil {
		return rawPageHeader{}, nil, err
	}
	data, err := decompress(compressed, codec)
	if err != nil {
		return rawPageHeader{}, nil, err
	}
	return header, data, nil
}

func decompress(data []byte, codec int32) ([]byte, error) {
	switch codec {
	case codecUncompressed:
		return data, nil
	case codecSnappy:
		return snappyDecode(data)
	default:
		return nil, fmt.Errorf("parquet: unsupported compression codec %d", codec)
	}
}

func parsePageHeader(data []byte) (rawPageHeader, int, error) {
	reader := newCompactReader(data)
	var header rawPageHeader
	for {
		id, typ, ok, err := reader.readFieldHeader()
		if err != nil {
			return header, 0, err
		}
		if !ok {
			return header, reader.pos, nil
		}
		switch id {
		case 1:
			value, err := reader.zigzag()
			if err != nil {
				return header, 0, err
			}
			header.typ = int32(value)
		case 2:
			value, err := reader.zigzag()
			if err != nil {
				return header, 0, err
			}
			header.uncompressedSize = int32(value)
		case 3:
			value, err := reader.zigzag()
			if err != nil {
				return header, 0, err
			}
			header.compressedSize = int32(value)
		case 5:
			header.data, err = parseDataPageHeader(reader)
		case 7:
			header.dictionary, err = parseDictionaryPageHeader(reader)
		default:
			err = reader.skipValue(typ)
		}
		if err != nil {
			return header, 0, err
		}
	}
}

func parseDataPageHeader(reader *compactReader) (*dataPageHeader, error) {
	previousFieldID := reader.enterStruct()
	defer reader.exitStruct(previousFieldID)
	header := &dataPageHeader{}
	for {
		id, typ, ok, err := reader.readFieldHeader()
		if err != nil {
			return header, err
		}
		if !ok {
			return header, nil
		}
		switch id {
		case 1:
			value, err := reader.zigzag()
			if err != nil {
				return header, err
			}
			header.numValues = int32(value)
		case 2:
			value, err := reader.zigzag()
			if err != nil {
				return header, err
			}
			header.encoding = int32(value)
		case 3:
			value, err := reader.zigzag()
			if err != nil {
				return header, err
			}
			header.defLevelEncoding = int32(value)
		case 4:
			value, err := reader.zigzag()
			if err != nil {
				return header, err
			}
			header.repLevelEncoding = int32(value)
		default:
			err = reader.skipValue(typ)
		}
		if err != nil {
			return header, err
		}
	}
}

func parseDictionaryPageHeader(reader *compactReader) (*dictionaryPageHeader, error) {
	previousFieldID := reader.enterStruct()
	defer reader.exitStruct(previousFieldID)
	header := &dictionaryPageHeader{}
	for {
		id, typ, ok, err := reader.readFieldHeader()
		if err != nil {
			return header, err
		}
		if !ok {
			return header, nil
		}
		switch id {
		case 1:
			value, err := reader.zigzag()
			if err != nil {
				return header, err
			}
			header.numValues = int32(value)
		case 2:
			value, err := reader.zigzag()
			if err != nil {
				return header, err
			}
			header.encoding = int32(value)
		default:
			err = reader.skipValue(typ)
		}
		if err != nil {
			return header, err
		}
	}
}
