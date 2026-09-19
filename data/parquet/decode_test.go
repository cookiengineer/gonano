package parquet

import (
	"encoding/binary"
	"math"
	"reflect"
	"testing"
)

func TestDecodePlainIntegers(t *testing.T) {
	int32Data := make([]byte, 12)
	binary.LittleEndian.PutUint32(int32Data[0:], 1)
	binary.LittleEndian.PutUint32(int32Data[4:], 0xFFFFFFFF) // -1
	binary.LittleEndian.PutUint32(int32Data[8:], 42)
	values32, err := decodePlainInt32(int32Data, 3)
	if err != nil {
		t.Fatalf("decodePlainInt32: %v", err)
	}
	if !reflect.DeepEqual(values32, []int32{1, -1, 42}) {
		t.Fatalf("int32 values = %v", values32)
	}
	if _, err := decodePlainInt32(int32Data, 4); err == nil {
		t.Fatal("expected error for short int32 input")
	}

	int64Data := make([]byte, 16)
	binary.LittleEndian.PutUint64(int64Data[0:], 1<<40)
	binary.LittleEndian.PutUint64(int64Data[8:], 7)
	values64, err := decodePlainInt64(int64Data, 2)
	if err != nil {
		t.Fatalf("decodePlainInt64: %v", err)
	}
	if !reflect.DeepEqual(values64, []int64{1 << 40, 7}) {
		t.Fatalf("int64 values = %v", values64)
	}
	if _, err := decodePlainInt64(int64Data, 3); err == nil {
		t.Fatal("expected error for short int64 input")
	}
}

func TestDecodePlainByteArray(t *testing.T) {
	var writer compactWriter
	writer.uvarint(3)
	writer.buffer = append(writer.buffer, []byte("foo")...)
	writer.uvarint(2)
	writer.buffer = append(writer.buffer, []byte("hi")...)
	values, err := decodePlainByteArray(writer.buffer, 2)
	if err != nil {
		t.Fatalf("decodePlainByteArray: %v", err)
	}
	if string(values[0]) != "foo" || string(values[1]) != "hi" {
		t.Fatalf("values = %q", values)
	}
}

func TestComputeDictBitWidth(t *testing.T) {
	cases := map[int]int{0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 8: 3, 9: 4}
	for size, want := range cases {
		if got := computeDictBitWidth(size); got != want {
			t.Fatalf("computeDictBitWidth(%d) = %d, want %d", size, got, want)
		}
	}
}

func TestDecodeRLEBitPackedRLERun(t *testing.T) {
	// 4-byte length prefix, then an RLE run header (numValues<<1) and value.
	// With bitWidth 3 the value 3 stays positive (its sign bit is clear).
	data := []byte{0, 0, 0, 0, 4 << 1, 3}
	values, err := decodeRLEBitPacked(data, 3, 4)
	if err != nil {
		t.Fatalf("decodeRLEBitPacked: %v", err)
	}
	if !reflect.DeepEqual(values, []int32{3, 3, 3, 3}) {
		t.Fatalf("values = %v", values)
	}
}

func TestDecodeRLEBitPackedBitPackedRun(t *testing.T) {
	// One bit-packed group of 8 two-bit values: 0,1,2,3,1,0,2,3.
	group := []byte{0xE4, 0xE1}
	data := append([]byte{0, 0, 0, 0}, byte(1<<1|1))
	data = append(data, group...)
	values, err := decodeRLEBitPacked(data, 2, 8)
	if err != nil {
		t.Fatalf("decodeRLEBitPacked: %v", err)
	}
	if !reflect.DeepEqual(values, []int32{0, 1, 2, 3, 1, 0, 2, 3}) {
		t.Fatalf("values = %v", values)
	}
}

func TestReadBitsLE(t *testing.T) {
	buffer := []byte{0b10110100, 0b00000001}
	if got := readBitsLE(buffer, 0, 1); got != 0 {
		t.Fatalf("bit 0 = %d, want 0", got)
	}
	if got := readBitsLE(buffer, 2, 2); got != 1 {
		t.Fatalf("bits 2..3 = %d, want 1", got)
	}
	if got := readBitsLE(buffer, 8, 1); got != 1 {
		t.Fatalf("bit 8 = %d, want 1", got)
	}
}

func TestDecodeDictInt(t *testing.T) {
	if got := decodeDictInt([]byte{42, 0, 0, 0}); got != 42 {
		t.Fatalf("decodeDictInt int32 = %d", got)
	}
	if got := decodeDictInt([]byte{7, 0, 0, 0, 0, 0, 0, 0}); got != 7 {
		t.Fatalf("decodeDictInt int64 = %d", got)
	}
	if got := decodeDictInt([]byte{1, 2, 3}); got != 0 {
		t.Fatalf("decodeDictInt fallback = %d, want 0", got)
	}
}

func TestCompactReaderScalars(t *testing.T) {
	var writer compactWriter
	writer.uvarint(300)
	writer.i32(-7)
	reader := newCompactReader(writer.buffer)
	if value, err := reader.uvarint(); err != nil || value != 300 {
		t.Fatalf("uvarint = %d, err = %v", value, err)
	}
	if value, err := reader.zigzag(); err != nil || value != -7 {
		t.Fatalf("zigzag = %d, err = %v", value, err)
	}

	doubleBits := make([]byte, 8)
	binary.LittleEndian.PutUint64(doubleBits, math.Float64bits(3.5))
	if value, err := newCompactReader(doubleBits).double(); err != nil || value != 3.5 {
		t.Fatalf("double = %v, err = %v", value, err)
	}

	var binaryWriter compactWriter
	binaryWriter.str("hello")
	if value, err := newCompactReader(binaryWriter.buffer).binary(); err != nil || string(value) != "hello" {
		t.Fatalf("binary = %q, err = %v", value, err)
	}
}

func TestCompactReaderReadFieldHeaderLongForm(t *testing.T) {
	var writer compactWriter
	writer.field(20, ctI32)
	writer.i32(1)
	reader := newCompactReader(writer.buffer)
	fieldID, typ, ok, err := reader.readFieldHeader()
	if err != nil || !ok || fieldID != 20 || typ != ctI32 {
		t.Fatalf("field header = (%d,%d,%v,%v)", fieldID, typ, ok, err)
	}
}

func TestCompactReaderSkipValues(t *testing.T) {
	var i32Writer compactWriter
	i32Writer.i32(9)

	var binaryWriter compactWriter
	binaryWriter.str("hi")

	var listWriter compactWriter
	listWriter.listHeader(2, ctI32)
	listWriter.i32(1)
	listWriter.i32(2)

	var structWriter compactWriter
	structWriter.field(1, ctI32)
	structWriter.i32(5)
	structWriter.stop()

	var mapWriter compactWriter
	mapWriter.uvarint(1)
	mapWriter.buffer = append(mapWriter.buffer, byte(ctI32<<4)|ctI32)
	mapWriter.i32(1)
	mapWriter.i32(2)

	doubleBits := make([]byte, 8)
	binary.LittleEndian.PutUint64(doubleBits, math.Float64bits(1.5))

	cases := []struct {
		name string
		typ  byte
		data []byte
	}{
		{"bool", ctBoolTrue, nil},
		{"byte", ctByte, []byte{0x42}},
		{"i32", ctI32, i32Writer.buffer},
		{"double", ctDouble, doubleBits},
		{"binary", ctBinary, binaryWriter.buffer},
		{"list", ctList, listWriter.buffer},
		{"struct", ctStruct, structWriter.buffer},
		{"map", ctMap, mapWriter.buffer},
	}
	for _, testCase := range cases {
		reader := newCompactReader(testCase.data)
		if err := reader.skipValue(testCase.typ); err != nil {
			t.Fatalf("skipValue(%s): %v", testCase.name, err)
		}
		if reader.pos != len(testCase.data) {
			t.Fatalf("skipValue(%s) left %d bytes", testCase.name, len(testCase.data)-reader.pos)
		}
	}

	if err := newCompactReader(nil).skipValue(0x0F); err == nil {
		t.Fatal("expected error for unknown compact type")
	}
	if err := newCompactReader([]byte{0x01}).skipMap(); err == nil {
		t.Fatal("expected error for truncated map")
	}
}
