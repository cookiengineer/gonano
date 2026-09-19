package parquet

import (
	"encoding/binary"
	"errors"
	"math"
)

// Thrift Compact Protocol types.
const (
	ctStop      = 0x00
	ctBoolTrue  = 0x01
	ctBoolFalse = 0x02
	ctByte      = 0x03
	ctI16       = 0x04
	ctI32       = 0x05
	ctI64       = 0x06
	ctDouble    = 0x07
	ctBinary    = 0x08
	ctList      = 0x09
	ctSet       = 0x0A
	ctMap       = 0x0B
	ctStruct    = 0x0C
)

var errBadCompact = errors.New("parquet: corrupt thrift compact data")

// compactReader reads Thrift Compact Protocol from a byte slice.
type compactReader struct {
	data        []byte
	pos         int
	lastFieldID int
}

func newCompactReader(data []byte) *compactReader {
	return &compactReader{data: data}
}

func (reader *compactReader) byte() (byte, error) {
	if reader.pos >= len(reader.data) {
		return 0, errBadCompact
	}
	value := reader.data[reader.pos]
	reader.pos++
	return value, nil
}

func (reader *compactReader) uvarint() (uint64, error) {
	var value uint64
	var shift uint
	for index := 0; ; index++ {
		if reader.pos >= len(reader.data) {
			return 0, errBadCompact
		}
		currentByte := reader.data[reader.pos]
		reader.pos++
		if currentByte < 0x80 {
			return value | uint64(currentByte)<<shift, nil
		}
		value |= uint64(currentByte&0x7f) << shift
		shift += 7
	}
}

func (reader *compactReader) zigzag() (int64, error) {
	encoded, err := reader.uvarint()
	if err != nil {
		return 0, err
	}
	// zigzag decode: (u >> 1) ^ -(u & 1)
	return int64(encoded>>1) ^ -int64(encoded&1), nil
}

func (reader *compactReader) binary() ([]byte, error) {
	size, err := reader.uvarint()
	if err != nil {
		return nil, err
	}
	if int(size) > len(reader.data)-reader.pos {
		return nil, errBadCompact
	}
	raw := reader.data[reader.pos : reader.pos+int(size)]
	reader.pos += int(size)
	return raw, nil
}

func (reader *compactReader) double() (float64, error) {
	if reader.pos+8 > len(reader.data) {
		return 0, errBadCompact
	}
	bits := binary.LittleEndian.Uint64(reader.data[reader.pos:])
	reader.pos += 8
	return math.Float64frombits(bits), nil
}

// readFieldHeader reads the next field header. It returns (fieldID, type, true),
// or (0, ctStop, false) at the stop field.
func (reader *compactReader) readFieldHeader() (int, byte, bool, error) {
	header, err := reader.byte()
	if err != nil {
		return 0, 0, false, err
	}
	if header == ctStop {
		return 0, ctStop, false, nil
	}
	typ := header & 0x0F
	delta := int(header >> 4)
	fieldID := 0
	if delta == 0 {
		// Long form: field id is a zigzag i16.
		encoded, err := reader.uvarint()
		if err != nil {
			return 0, 0, false, err
		}
		fieldID = int(int16(encoded>>1) ^ -int16(encoded&1))
	} else {
		fieldID = reader.lastFieldID + delta
	}
	reader.lastFieldID = fieldID
	return fieldID, typ, true, nil
}

// skipValue skips a value of the given compact type.
func (reader *compactReader) skipValue(typ byte) error {
	switch typ {
	case ctBoolTrue, ctBoolFalse:
		return nil
	case ctByte:
		_, err := reader.byte()
		return err
	case ctI16, ctI32, ctI64:
		_, err := reader.uvarint()
		return err
	case ctDouble:
		_, err := reader.double()
		return err
	case ctBinary:
		_, err := reader.binary()
		return err
	case ctList, ctSet:
		return reader.skipList()
	case ctStruct:
		return reader.skipStruct()
	case ctMap:
		return reader.skipMap()
	}
	return errBadCompact
}

func (reader *compactReader) skipList() error {
	header, err := reader.byte()
	if err != nil {
		return err
	}
	size := int(header >> 4)
	elemType := header & 0x0F
	if size == 15 {
		encoded, err := reader.uvarint()
		if err != nil {
			return err
		}
		size = int(encoded)
	}
	for index := 0; index < size; index++ {
		if err := reader.skipValue(elemType); err != nil {
			return err
		}
	}
	return nil
}

// enterStruct saves the current field-id context and resets it, because field
// ids restart from zero inside each nested struct.
func (reader *compactReader) enterStruct() int {
	previousFieldID := reader.lastFieldID
	reader.lastFieldID = 0
	return previousFieldID
}

func (reader *compactReader) exitStruct(previousFieldID int) {
	reader.lastFieldID = previousFieldID
}

func (reader *compactReader) skipStruct() error {
	previousFieldID := reader.enterStruct()
	defer reader.exitStruct(previousFieldID)
	for {
		_, typ, ok, err := reader.readFieldHeader()
		if err != nil {
			return err
		}
		if !ok {
			return nil
		}
		if err := reader.skipValue(typ); err != nil {
			return err
		}
	}
}

func (reader *compactReader) skipMap() error {
	size, err := reader.uvarint()
	if err != nil {
		return err
	}
	if size == 0 {
		return nil
	}
	typesHeader, err := reader.byte()
	if err != nil {
		return err
	}
	keyType := typesHeader >> 4
	valType := typesHeader & 0x0F
	for index := 0; index < int(size); index++ {
		if err := reader.skipValue(keyType); err != nil {
			return err
		}
		if err := reader.skipValue(valType); err != nil {
			return err
		}
	}
	return nil
}
