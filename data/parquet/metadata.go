package parquet

// Parquet Thrift metadata structures and their compact-protocol decoders.

// Type enum (parquet.Type).
const (
	typeBoolean          = 0
	typeInt32            = 1
	typeInt64            = 2
	typeInt96            = 3
	typeFloat            = 4
	typeDouble           = 5
	typeByteArray        = 6
	typeFixedLenByteArr  = 7
)

// Encoding enum (parquet.Encoding).
const (
	encPlain            = 0
	encPlainDictionary  = 2
	encRLE              = 3
	encBitPacked        = 4
	encRLE_Dictionary   = 8
)

// CompressionCodec enum (parquet.CompressionCodec).
const (
	codecUncompressed = 0
	codecSnappy       = 1
	codecGzip         = 2
	codecBrotli       = 4
	codecLz4          = 5
	codecZstd         = 6
	codecLz4Raw       = 7
)

// PageType enum (parquet.PageType).
const (
	pageData       = 0
	pageIndex      = 1
	pageDictionary = 2
	pageDataV2     = 3
)

type schemaElement struct {
	typ         int32
	typeLength  int32
	repetition  int32
	name        string
	numChildren int32
}

type columnMetaData struct {
	typ                    int32
	encodings              []int32
	pathInSchema           []string
	codec                  int32
	numValues              int64
	totalUncompressedSize  int64
	totalCompressedSize    int64
	dataPageOffset         int64
	dictionaryPageOffset   int64
}

type columnChunk struct {
	filePath   string
	fileOffset int64
	meta       *columnMetaData
}

type rowGroup struct {
	columns       []columnChunk
	totalByteSize int64
	numRows       int64
}

type fileMetaData struct {
	version   int32
	schema    []schemaElement
	numRows   int64
	rowGroups []rowGroup
}

type dataPageHeader struct {
	numValues         int32
	encoding          int32
	defLevelEncoding  int32
	repLevelEncoding  int32
}

type dictionaryPageHeader struct {
	numValues int32
	encoding  int32
}

type pageHeader struct {
	typ                 int32
	uncompressedSize    int32
	compressedSize      int32
	data                *dataPageHeader
	dictionary          *dictionaryPageHeader
}

// listHeader reads a compact-protocol list header, returning the element count
// and element type.
func (r *compactReader) listHeader() (int, byte, error) {
	b, err := r.byte()
	if err != nil {
		return 0, 0, err
	}
	size := int(b >> 4)
	elemType := b & 0x0F
	if size == 15 {
		u, err := r.uvarint()
		if err != nil {
			return 0, 0, err
		}
		size = int(u)
	}
	return size, elemType, nil
}

func (r *compactReader) readString() (string, error) {
	b, err := r.binary()
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func parseFileMetaData(data []byte) (fileMetaData, error) {
	r := newCompactReader(data)
	var meta fileMetaData
	for {
		id, typ, ok, err := r.fieldHeader()
		if err != nil {
			return meta, err
		}
		if !ok {
			break
		}
		switch id {
		case 1:
			v, err := r.zigzag()
			if err != nil {
				return meta, err
			}
			meta.version = int32(v)
		case 2:
			meta.schema, err = parseSchemaElementList(r)
		case 3:
			meta.numRows, err = r.zigzag()
		case 4:
			meta.rowGroups, err = parseRowGroupList(r)
		default:
			err = r.skipValue(typ)
		}
		if err != nil {
			return meta, err
		}
	}
	return meta, nil
}

func parseSchemaElementList(r *compactReader) ([]schemaElement, error) {
	size, elemType, err := r.listHeader()
	if err != nil {
		return nil, err
	}
	if elemType != ctStruct {
		return nil, errBadCompact
	}
	out := make([]schemaElement, size)
	for i := 0; i < size; i++ {
		out[i], err = parseSchemaElement(r)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func parseSchemaElement(r *compactReader) (schemaElement, error) {
	last := r.enterStruct()
	defer r.exitStruct(last)
	var e schemaElement
	for {
		id, typ, ok, err := r.fieldHeader()
		if err != nil {
			return e, err
		}
		if !ok {
			return e, nil
		}
		switch id {
		case 1:
			v, err := r.zigzag()
			if err != nil {
				return e, err
			}
			e.typ = int32(v)
		case 2:
			v, err := r.zigzag()
			if err != nil {
				return e, err
			}
			e.typeLength = int32(v)
		case 3:
			v, err := r.zigzag()
			if err != nil {
				return e, err
			}
			e.repetition = int32(v)
		case 4:
			e.name, err = r.readString()
		case 5:
			v, err := r.zigzag()
			if err != nil {
				return e, err
			}
			e.numChildren = int32(v)
		default:
			err = r.skipValue(typ)
		}
		if err != nil {
			return e, err
		}
	}
}

func parseRowGroupList(r *compactReader) ([]rowGroup, error) {
	size, elemType, err := r.listHeader()
	if err != nil {
		return nil, err
	}
	if elemType != ctStruct {
		return nil, errBadCompact
	}
	out := make([]rowGroup, size)
	for i := 0; i < size; i++ {
		out[i], err = parseRowGroup(r)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func parseRowGroup(r *compactReader) (rowGroup, error) {
	last := r.enterStruct()
	defer r.exitStruct(last)
	var g rowGroup
	for {
		id, typ, ok, err := r.fieldHeader()
		if err != nil {
			return g, err
		}
		if !ok {
			return g, nil
		}
		switch id {
		case 1:
			g.columns, err = parseColumnChunkList(r)
		case 2:
			g.totalByteSize, err = r.zigzag()
		case 3:
			g.numRows, err = r.zigzag()
		default:
			err = r.skipValue(typ)
		}
		if err != nil {
			return g, err
		}
	}
}

func parseColumnChunkList(r *compactReader) ([]columnChunk, error) {
	size, elemType, err := r.listHeader()
	if err != nil {
		return nil, err
	}
	if elemType != ctStruct {
		return nil, errBadCompact
	}
	out := make([]columnChunk, size)
	for i := 0; i < size; i++ {
		out[i], err = parseColumnChunk(r)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func parseColumnChunk(r *compactReader) (columnChunk, error) {
	last := r.enterStruct()
	defer r.exitStruct(last)
	var c columnChunk
	for {
		id, typ, ok, err := r.fieldHeader()
		if err != nil {
			return c, err
		}
		if !ok {
			return c, nil
		}
		switch id {
		case 1:
			c.filePath, err = r.readString()
		case 2:
			c.fileOffset, err = r.zigzag()
		case 3:
			if typ != ctStruct {
				err = r.skipValue(typ)
			} else {
				c.meta, err = parseColumnMetaData(r)
			}
		default:
			err = r.skipValue(typ)
		}
		if err != nil {
			return c, err
		}
	}
}

func parseColumnMetaData(r *compactReader) (*columnMetaData, error) {
	last := r.enterStruct()
	defer r.exitStruct(last)
	m := &columnMetaData{}
	for {
		id, typ, ok, err := r.fieldHeader()
		if err != nil {
			return m, err
		}
		if !ok {
			return m, nil
		}
		switch id {
		case 1:
			v, err := r.zigzag()
			if err != nil {
				return m, err
			}
			m.typ = int32(v)
		case 2:
			m.encodings, err = readI32List(r)
		case 3:
			m.pathInSchema, err = readStringList(r)
		case 4:
			v, err := r.zigzag()
			if err != nil {
				return m, err
			}
			m.codec = int32(v)
		case 5:
			m.numValues, err = r.zigzag()
		case 6:
			m.totalUncompressedSize, err = r.zigzag()
		case 7:
			m.totalCompressedSize, err = r.zigzag()
		case 9:
			m.dataPageOffset, err = r.zigzag()
		case 11:
			m.dictionaryPageOffset, err = r.zigzag()
		default:
			err = r.skipValue(typ)
		}
		if err != nil {
			return m, err
		}
	}
}

func readI32List(r *compactReader) ([]int32, error) {
	size, elemType, err := r.listHeader()
	if err != nil {
		return nil, err
	}
	out := make([]int32, size)
	for i := 0; i < size; i++ {
		switch elemType {
		case ctI16, ctI32, ctI64:
			v, err := r.zigzag()
			if err != nil {
				return nil, err
			}
			out[i] = int32(v)
		default:
			return nil, errBadCompact
		}
	}
	return out, nil
}

func readStringList(r *compactReader) ([]string, error) {
	size, elemType, err := r.listHeader()
	if err != nil {
		return nil, err
	}
	if elemType != ctBinary {
		return nil, errBadCompact
	}
	out := make([]string, size)
	for i := 0; i < size; i++ {
		out[i], err = r.readString()
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}
