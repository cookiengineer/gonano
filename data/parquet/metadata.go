package parquet

// Parquet Thrift metadata structures and their compact-protocol decoders.

// Type enum (parquet.Type).
const (
	typeBoolean         = 0
	typeInt32           = 1
	typeInt64           = 2
	typeInt96           = 3
	typeFloat           = 4
	typeDouble          = 5
	typeByteArray       = 6
	typeFixedLenByteArr = 7
)

// Encoding enum (parquet.Encoding).
const (
	encPlain           = 0
	encPlainDictionary = 2
	encRLE             = 3
	encBitPacked       = 4
	encRLE_Dictionary  = 8
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
	typ                   int32
	encodings             []int32
	pathInSchema          []string
	codec                 int32
	numValues             int64
	totalUncompressedSize int64
	totalCompressedSize   int64
	dataPageOffset        int64
	dictionaryPageOffset  int64
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
	numValues        int32
	encoding         int32
	defLevelEncoding int32
	repLevelEncoding int32
}

type dictionaryPageHeader struct {
	numValues int32
	encoding  int32
}

type pageHeader struct {
	typ              int32
	uncompressedSize int32
	compressedSize   int32
	data             *dataPageHeader
	dictionary       *dictionaryPageHeader
}

// readListHeader reads a compact-protocol list header, returning the element
// count and element type.
func (reader *compactReader) readListHeader() (int, byte, error) {
	header, err := reader.byte()
	if err != nil {
		return 0, 0, err
	}
	size := int(header >> 4)
	elemType := header & 0x0F
	if size == 15 {
		encoded, err := reader.uvarint()
		if err != nil {
			return 0, 0, err
		}
		size = int(encoded)
	}
	return size, elemType, nil
}

func (reader *compactReader) readString() (string, error) {
	raw, err := reader.binary()
	if err != nil {
		return "", err
	}
	return string(raw), nil
}

func parseFileMetaData(data []byte) (fileMetaData, error) {
	reader := newCompactReader(data)
	var meta fileMetaData
	for {
		id, typ, ok, err := reader.readFieldHeader()
		if err != nil {
			return meta, err
		}
		if !ok {
			break
		}
		switch id {
		case 1:
			value, err := reader.zigzag()
			if err != nil {
				return meta, err
			}
			meta.version = int32(value)
		case 2:
			meta.schema, err = parseSchemaElementList(reader)
		case 3:
			meta.numRows, err = reader.zigzag()
		case 4:
			meta.rowGroups, err = parseRowGroupList(reader)
		default:
			err = reader.skipValue(typ)
		}
		if err != nil {
			return meta, err
		}
	}
	return meta, nil
}

func parseSchemaElementList(reader *compactReader) ([]schemaElement, error) {
	size, elemType, err := reader.readListHeader()
	if err != nil {
		return nil, err
	}
	if elemType != ctStruct {
		return nil, errBadCompact
	}
	out := make([]schemaElement, size)
	for index := 0; index < size; index++ {
		out[index], err = parseSchemaElement(reader)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func parseSchemaElement(reader *compactReader) (schemaElement, error) {
	previousFieldID := reader.enterStruct()
	defer reader.exitStruct(previousFieldID)
	var element schemaElement
	for {
		id, typ, ok, err := reader.readFieldHeader()
		if err != nil {
			return element, err
		}
		if !ok {
			return element, nil
		}
		switch id {
		case 1:
			value, err := reader.zigzag()
			if err != nil {
				return element, err
			}
			element.typ = int32(value)
		case 2:
			value, err := reader.zigzag()
			if err != nil {
				return element, err
			}
			element.typeLength = int32(value)
		case 3:
			value, err := reader.zigzag()
			if err != nil {
				return element, err
			}
			element.repetition = int32(value)
		case 4:
			element.name, err = reader.readString()
		case 5:
			value, err := reader.zigzag()
			if err != nil {
				return element, err
			}
			element.numChildren = int32(value)
		default:
			err = reader.skipValue(typ)
		}
		if err != nil {
			return element, err
		}
	}
}

func parseRowGroupList(reader *compactReader) ([]rowGroup, error) {
	size, elemType, err := reader.readListHeader()
	if err != nil {
		return nil, err
	}
	if elemType != ctStruct {
		return nil, errBadCompact
	}
	out := make([]rowGroup, size)
	for index := 0; index < size; index++ {
		out[index], err = parseRowGroup(reader)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func parseRowGroup(reader *compactReader) (rowGroup, error) {
	previousFieldID := reader.enterStruct()
	defer reader.exitStruct(previousFieldID)
	var group rowGroup
	for {
		id, typ, ok, err := reader.readFieldHeader()
		if err != nil {
			return group, err
		}
		if !ok {
			return group, nil
		}
		switch id {
		case 1:
			group.columns, err = parseColumnChunkList(reader)
		case 2:
			group.totalByteSize, err = reader.zigzag()
		case 3:
			group.numRows, err = reader.zigzag()
		default:
			err = reader.skipValue(typ)
		}
		if err != nil {
			return group, err
		}
	}
}

func parseColumnChunkList(reader *compactReader) ([]columnChunk, error) {
	size, elemType, err := reader.readListHeader()
	if err != nil {
		return nil, err
	}
	if elemType != ctStruct {
		return nil, errBadCompact
	}
	out := make([]columnChunk, size)
	for index := 0; index < size; index++ {
		out[index], err = parseColumnChunk(reader)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func parseColumnChunk(reader *compactReader) (columnChunk, error) {
	previousFieldID := reader.enterStruct()
	defer reader.exitStruct(previousFieldID)
	var chunk columnChunk
	for {
		id, typ, ok, err := reader.readFieldHeader()
		if err != nil {
			return chunk, err
		}
		if !ok {
			return chunk, nil
		}
		switch id {
		case 1:
			chunk.filePath, err = reader.readString()
		case 2:
			chunk.fileOffset, err = reader.zigzag()
		case 3:
			if typ != ctStruct {
				err = reader.skipValue(typ)
			} else {
				chunk.meta, err = parseColumnMetaData(reader)
			}
		default:
			err = reader.skipValue(typ)
		}
		if err != nil {
			return chunk, err
		}
	}
}

func parseColumnMetaData(reader *compactReader) (*columnMetaData, error) {
	previousFieldID := reader.enterStruct()
	defer reader.exitStruct(previousFieldID)
	metadata := &columnMetaData{}
	for {
		id, typ, ok, err := reader.readFieldHeader()
		if err != nil {
			return metadata, err
		}
		if !ok {
			return metadata, nil
		}
		switch id {
		case 1:
			value, err := reader.zigzag()
			if err != nil {
				return metadata, err
			}
			metadata.typ = int32(value)
		case 2:
			metadata.encodings, err = readI32List(reader)
		case 3:
			metadata.pathInSchema, err = readStringList(reader)
		case 4:
			value, err := reader.zigzag()
			if err != nil {
				return metadata, err
			}
			metadata.codec = int32(value)
		case 5:
			metadata.numValues, err = reader.zigzag()
		case 6:
			metadata.totalUncompressedSize, err = reader.zigzag()
		case 7:
			metadata.totalCompressedSize, err = reader.zigzag()
		case 9:
			metadata.dataPageOffset, err = reader.zigzag()
		case 11:
			metadata.dictionaryPageOffset, err = reader.zigzag()
		default:
			err = reader.skipValue(typ)
		}
		if err != nil {
			return metadata, err
		}
	}
}

func readI32List(reader *compactReader) ([]int32, error) {
	size, elemType, err := reader.readListHeader()
	if err != nil {
		return nil, err
	}
	out := make([]int32, size)
	for index := 0; index < size; index++ {
		switch elemType {
		case ctI16, ctI32, ctI64:
			value, err := reader.zigzag()
			if err != nil {
				return nil, err
			}
			out[index] = int32(value)
		default:
			return nil, errBadCompact
		}
	}
	return out, nil
}

func readStringList(reader *compactReader) ([]string, error) {
	size, elemType, err := reader.readListHeader()
	if err != nil {
		return nil, err
	}
	if elemType != ctBinary {
		return nil, errBadCompact
	}
	out := make([]string, size)
	for index := 0; index < size; index++ {
		out[index], err = reader.readString()
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}
