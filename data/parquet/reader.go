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
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	size := st.Size()
	if size < 12 {
		f.Close()
		return nil, errBadParquet
	}
	// Footer: 4-byte metadata length + "PAR1". Verify leading magic too.
	tail := make([]byte, 8)
	if _, err := f.ReadAt(tail, size-8); err != nil {
		f.Close()
		return nil, err
	}
	if string(tail[4:]) != "PAR1" {
		f.Close()
		return nil, errBadParquet
	}
	metaLen := int64(binary.LittleEndian.Uint32(tail[:4]))
	if metaLen <= 0 || metaLen > size-8 {
		f.Close()
		return nil, errBadParquet
	}
	metaBytes := make([]byte, metaLen)
	if _, err := f.ReadAt(metaBytes, size-8-metaLen); err != nil {
		f.Close()
		return nil, err
	}
	meta, err := parseFileMetaData(metaBytes)
	if err != nil {
		f.Close()
		return nil, err
	}
	return &Reader{file: f, meta: meta}, nil
}

// Close closes the underlying file.
func (r *Reader) Close() error { return r.file.Close() }

// NumRowGroups returns the number of row groups.
func (r *Reader) NumRowGroups() int { return len(r.meta.rowGroups) }

// NumRows returns the total number of rows in the file.
func (r *Reader) NumRows() int64 { return r.meta.numRows }

// RowGroupNumRows returns the number of rows in a row group.
func (r *Reader) RowGroupNumRows(i int) int64 { return r.meta.rowGroups[i].numRows }

// ColumnNames returns the leaf column names (flat columns only).
func (r *Reader) ColumnNames() []string {
	seen := map[string]bool{}
	var names []string
	for _, rg := range r.meta.rowGroups {
		for _, cc := range rg.columns {
			if cc.meta == nil || len(cc.meta.pathInSchema) == 0 {
				continue
			}
			name := cc.meta.pathInSchema[len(cc.meta.pathInSchema)-1]
			if !seen[name] {
				seen[name] = true
				names = append(names, name)
			}
		}
	}
	return names
}

// findChunk locates the column chunk for a named flat column in a row group.
func (r *Reader) findChunk(rgIdx int, name string) (columnChunk, bool) {
	for _, cc := range r.meta.rowGroups[rgIdx].columns {
		if cc.meta != nil && len(cc.meta.pathInSchema) > 0 &&
			cc.meta.pathInSchema[len(cc.meta.pathInSchema)-1] == name {
			return cc, true
		}
	}
	return columnChunk{}, false
}

// ReadColumnStrings reads a flat BYTE_ARRAY column for a row group as strings.
func (r *Reader) ReadColumnStrings(rgIdx int, name string) ([]string, error) {
	cc, ok := r.findChunk(rgIdx, name)
	if !ok {
		return nil, fmt.Errorf("parquet: column %q not found", name)
	}
	values, err := r.readColumnValues(cc)
	if err != nil {
		return nil, err
	}
	out := make([]string, len(values))
	for i, v := range values {
		out[i] = string(v)
	}
	return out, nil
}

// ReadColumnBytes reads a flat BYTE_ARRAY column as raw bytes.
func (r *Reader) ReadColumnBytes(rgIdx int, name string) ([][]byte, error) {
	cc, ok := r.findChunk(rgIdx, name)
	if !ok {
		return nil, fmt.Errorf("parquet: column %q not found", name)
	}
	return r.readColumnValues(cc)
}

// ReadColumnInt64 reads a flat INT32 or INT64 column as int64.
func (r *Reader) ReadColumnInt64(rgIdx int, name string) ([]int64, error) {
	cc, ok := r.findChunk(rgIdx, name)
	if !ok {
		return nil, fmt.Errorf("parquet: column %q not found", name)
	}
	dict, pages, err := r.readPages(cc)
	if err != nil {
		return nil, err
	}
	var out []int64
	for _, p := range pages {
		enc := p.encoding
		switch enc {
		case encPlain:
			if cc.meta.typ == typeInt32 {
				vals, err := decodePlainInt32(p.data, int(p.numValues))
				if err != nil {
					return nil, err
				}
				for _, v := range vals {
					out = append(out, int64(v))
				}
			} else {
				vals, err := decodePlainInt64(p.data, int(p.numValues))
				if err != nil {
					return nil, err
				}
				out = append(out, vals...)
			}
		case encRLE_Dictionary:
			bitWidth := bitWidthForDict(len(dict))
			idxs, err := decodeRLEBitPacked(p.data, bitWidth, int(p.numValues))
			if err != nil {
				return nil, err
			}
			for _, idx := range idxs {
				if idx < 0 || int(idx) >= len(dict) {
					return nil, errBadEncoding
				}
				out = append(out, int64(dictInt(dict[idx])))
			}
		default:
			return nil, fmt.Errorf("parquet: unsupported encoding %d for int column", enc)
		}
	}
	return out, nil
}

func dictInt(b []byte) int32 {
	if len(b) == 4 {
		return int32(binary.LittleEndian.Uint32(b))
	}
	if len(b) == 8 {
		return int32(binary.LittleEndian.Uint64(b))
	}
	// Fallback: decode as text? Not expected for int columns.
	return 0
}

// readColumnValues reads a BYTE_ARRAY column, handling PLAIN and
// RLE_DICTIONARY encodings.
func (r *Reader) readColumnValues(cc columnChunk) ([][]byte, error) {
	dict, pages, err := r.readPages(cc)
	if err != nil {
		return nil, err
	}
	var out [][]byte
	for _, p := range pages {
		switch p.encoding {
		case encPlain:
			vals, err := decodePlainByteArray(p.data, int(p.numValues))
			if err != nil {
				return nil, err
			}
			out = append(out, vals...)
		case encRLE_Dictionary:
			bitWidth := bitWidthForDict(len(dict))
			idxs, err := decodeRLEBitPacked(p.data, bitWidth, int(p.numValues))
			if err != nil {
				return nil, err
			}
			for _, idx := range idxs {
				if idx < 0 || int(idx) >= len(dict) {
					return nil, errBadEncoding
				}
				out = append(out, dict[idx])
			}
		default:
			return nil, fmt.Errorf("parquet: unsupported encoding %d", p.encoding)
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
func (r *Reader) readPages(cc columnChunk) ([][]byte, []rawPage, error) {
	if cc.meta == nil {
		return nil, nil, errBadParquet
	}
	var dict [][]byte
	var pages []rawPage

	if cc.meta.dictionaryPageOffset > 0 {
		hdr, data, err := r.readPageAt(cc.meta.dictionaryPageOffset, cc.meta.codec)
		if err != nil {
			return nil, nil, err
		}
		if hdr.dictionary == nil {
			return nil, nil, errBadParquet
		}
		dict, err = decodePlainByteArray(data, int(hdr.dictionary.numValues))
		if err != nil {
			return nil, nil, err
		}
	}

	offset := cc.meta.dataPageOffset
	if offset == 0 {
		offset = cc.fileOffset
	}
	var total int64
	for total < cc.meta.numValues {
		hdr, data, err := r.readPageAt(offset, cc.meta.codec)
		if err != nil {
			return nil, nil, err
		}
		if hdr.typ == pageDictionary {
			offset += hdr.headerLen + int64(hdr.compressedSize)
			continue
		}
		if hdr.data == nil {
			return nil, nil, errBadParquet
		}
		pages = append(pages, rawPage{
			numValues: hdr.data.numValues,
			encoding:  hdr.data.encoding,
			data:      data,
		})
		total += int64(hdr.data.numValues)
		offset += hdr.headerLen + int64(hdr.compressedSize)
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
func (r *Reader) readPageAt(offset int64, codec int32) (rawPageHeader, []byte, error) {
	const headerWindow = 1 << 16
	win := make([]byte, headerWindow)
	n, err := r.file.ReadAt(win, offset)
	if err != nil && err != io.EOF {
		return rawPageHeader{}, nil, err
	}
	win = win[:n]
	hdr, consumed, err := parsePageHeader(win)
	if err != nil {
		return rawPageHeader{}, nil, err
	}
	hdr.headerLen = int64(consumed)
	compressed := make([]byte, hdr.compressedSize)
	if _, err := r.file.ReadAt(compressed, offset+int64(consumed)); err != nil {
		return rawPageHeader{}, nil, err
	}
	data, err := decompress(compressed, codec)
	if err != nil {
		return rawPageHeader{}, nil, err
	}
	return hdr, data, nil
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
	r := newCompactReader(data)
	var h rawPageHeader
	for {
		id, typ, ok, err := r.fieldHeader()
		if err != nil {
			return h, 0, err
		}
		if !ok {
			return h, r.pos, nil
		}
		switch id {
		case 1:
			v, err := r.zigzag()
			if err != nil {
				return h, 0, err
			}
			h.typ = int32(v)
		case 2:
			v, err := r.zigzag()
			if err != nil {
				return h, 0, err
			}
			h.uncompressedSize = int32(v)
		case 3:
			v, err := r.zigzag()
			if err != nil {
				return h, 0, err
			}
			h.compressedSize = int32(v)
		case 5:
			h.data, err = parseDataPageHeader(r)
		case 7:
			h.dictionary, err = parseDictionaryPageHeader(r)
		default:
			err = r.skipValue(typ)
		}
		if err != nil {
			return h, 0, err
		}
	}
}

func parseDataPageHeader(r *compactReader) (*dataPageHeader, error) {
	last := r.enterStruct()
	defer r.exitStruct(last)
	h := &dataPageHeader{}
	for {
		id, typ, ok, err := r.fieldHeader()
		if err != nil {
			return h, err
		}
		if !ok {
			return h, nil
		}
		switch id {
		case 1:
			v, err := r.zigzag()
			if err != nil {
				return h, err
			}
			h.numValues = int32(v)
		case 2:
			v, err := r.zigzag()
			if err != nil {
				return h, err
			}
			h.encoding = int32(v)
		case 3:
			v, err := r.zigzag()
			if err != nil {
				return h, err
			}
			h.defLevelEncoding = int32(v)
		case 4:
			v, err := r.zigzag()
			if err != nil {
				return h, err
			}
			h.repLevelEncoding = int32(v)
		default:
			err = r.skipValue(typ)
		}
		if err != nil {
			return h, err
		}
	}
}

func parseDictionaryPageHeader(r *compactReader) (*dictionaryPageHeader, error) {
	last := r.enterStruct()
	defer r.exitStruct(last)
	h := &dictionaryPageHeader{}
	for {
		id, typ, ok, err := r.fieldHeader()
		if err != nil {
			return h, err
		}
		if !ok {
			return h, nil
		}
		switch id {
		case 1:
			v, err := r.zigzag()
			if err != nil {
				return h, err
			}
			h.numValues = int32(v)
		case 2:
			v, err := r.zigzag()
			if err != nil {
				return h, err
			}
			h.encoding = int32(v)
		default:
			err = r.skipValue(typ)
		}
		if err != nil {
			return h, err
		}
	}
}
