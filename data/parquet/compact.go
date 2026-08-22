package parquet

import (
	"encoding/binary"
	"errors"
	"math"
)

// Thrift Compact Protocol types.
const (
	ctStop         = 0x00
	ctBoolTrue     = 0x01
	ctBoolFalse    = 0x02
	ctByte         = 0x03
	ctI16          = 0x04
	ctI32          = 0x05
	ctI64          = 0x06
	ctDouble       = 0x07
	ctBinary       = 0x08
	ctList         = 0x09
	ctSet          = 0x0A
	ctMap          = 0x0B
	ctStruct       = 0x0C
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

func (r *compactReader) byte() (byte, error) {
	if r.pos >= len(r.data) {
		return 0, errBadCompact
	}
	b := r.data[r.pos]
	r.pos++
	return b, nil
}

func (r *compactReader) uvarint() (uint64, error) {
	var x uint64
	var s uint
	for i := 0; ; i++ {
		if r.pos >= len(r.data) {
			return 0, errBadCompact
		}
		b := r.data[r.pos]
		r.pos++
		if b < 0x80 {
			return x | uint64(b)<<s, nil
		}
		x |= uint64(b&0x7f) << s
		s += 7
	}
}

func (r *compactReader) zigzag() (int64, error) {
	u, err := r.uvarint()
	if err != nil {
		return 0, err
	}
	// zigzag decode: (u >> 1) ^ -(u & 1)
	return int64(u>>1) ^ -int64(u&1), nil
}

func (r *compactReader) binary() ([]byte, error) {
	n, err := r.uvarint()
	if err != nil {
		return nil, err
	}
	if int(n) > len(r.data)-r.pos {
		return nil, errBadCompact
	}
	b := r.data[r.pos : r.pos+int(n)]
	r.pos += int(n)
	return b, nil
}

func (r *compactReader) double() (float64, error) {
	if r.pos+8 > len(r.data) {
		return 0, errBadCompact
	}
	v := binary.LittleEndian.Uint64(r.data[r.pos:])
	r.pos += 8
	return math.Float64frombits(v), nil
}

// fieldHeader reads the next field header. It returns (fieldID, type, true),
// or (0, ctStop, false) at the stop field.
func (r *compactReader) fieldHeader() (int, byte, bool, error) {
	b, err := r.byte()
	if err != nil {
		return 0, 0, false, err
	}
	if b == ctStop {
		return 0, ctStop, false, nil
	}
	typ := b & 0x0F
	delta := int(b >> 4)
	fieldID := 0
	if delta == 0 {
		// Long form: field id is a zigzag i16.
		u, err := r.uvarint()
		if err != nil {
			return 0, 0, false, err
		}
		fieldID = int(int16(u>>1) ^ -int16(u&1))
	} else {
		fieldID = r.lastFieldID + delta
	}
	r.lastFieldID = fieldID
	return fieldID, typ, true, nil
}

// skipValue skips a value of the given compact type.
func (r *compactReader) skipValue(typ byte) error {
	switch typ {
	case ctBoolTrue, ctBoolFalse:
		return nil
	case ctByte:
		_, err := r.byte()
		return err
	case ctI16, ctI32, ctI64:
		_, err := r.uvarint()
		return err
	case ctDouble:
		_, err := r.double()
		return err
	case ctBinary:
		_, err := r.binary()
		return err
	case ctList, ctSet:
		return r.skipList()
	case ctStruct:
		return r.skipStruct()
	case ctMap:
		return r.skipMap()
	}
	return errBadCompact
}

func (r *compactReader) skipList() error {
	b, err := r.byte()
	if err != nil {
		return err
	}
	size := int(b >> 4)
	elemType := b & 0x0F
	if size == 15 {
		u, err := r.uvarint()
		if err != nil {
			return err
		}
		size = int(u)
	}
	for i := 0; i < size; i++ {
		if err := r.skipValue(elemType); err != nil {
			return err
		}
	}
	return nil
}

// enterStruct saves the current field-id context and resets it, because field
// ids restart from zero inside each nested struct.
func (r *compactReader) enterStruct() int {
	last := r.lastFieldID
	r.lastFieldID = 0
	return last
}

func (r *compactReader) exitStruct(last int) {
	r.lastFieldID = last
}

func (r *compactReader) skipStruct() error {
	last := r.enterStruct()
	defer r.exitStruct(last)
	for {
		_, typ, ok, err := r.fieldHeader()
		if err != nil {
			return err
		}
		if !ok {
			return nil
		}
		if err := r.skipValue(typ); err != nil {
			return err
		}
	}
}

func (r *compactReader) skipMap() error {
	size, err := r.uvarint()
	if err != nil {
		return err
	}
	if size == 0 {
		return nil
	}
	b, err := r.byte()
	if err != nil {
		return err
	}
	keyType := b >> 4
	valType := b & 0x0F
	for i := 0; i < int(size); i++ {
		if err := r.skipValue(keyType); err != nil {
			return err
		}
		if err := r.skipValue(valType); err != nil {
			return err
		}
	}
	return nil
}
