package parquet

import (
	"encoding/binary"
	"errors"
)

// Value decoding for the encodings gonano needs: PLAIN and RLE_DICTIONARY
// (with its RLE/bit-packed hybrid index encoding).

var errBadEncoding = errors.New("parquet: corrupt encoding")

// decodePlainByteArray decodes count PLAIN-encoded BYTE_ARRAY values.
func decodePlainByteArray(data []byte, count int) ([][]byte, error) {
	out := make([][]byte, count)
	for index := 0; index < count; index++ {
		size, consumed, err := uvarint(data)
		if err != nil {
			return nil, err
		}
		data = data[consumed:]
		if int(size) > len(data) {
			return nil, errBadEncoding
		}
		out[index] = data[:int(size)]
		data = data[int(size):]
	}
	return out, nil
}

// decodePlainInt32 decodes count PLAIN-encoded INT32 values (4-byte LE each).
func decodePlainInt32(data []byte, count int) ([]int32, error) {
	if len(data) < count*4 {
		return nil, errBadEncoding
	}
	out := make([]int32, count)
	for index := 0; index < count; index++ {
		out[index] = int32(binary.LittleEndian.Uint32(data[index*4:]))
	}
	return out, nil
}

// decodePlainInt64 decodes count PLAIN-encoded INT64 values.
func decodePlainInt64(data []byte, count int) ([]int64, error) {
	if len(data) < count*8 {
		return nil, errBadEncoding
	}
	out := make([]int64, count)
	for index := 0; index < count; index++ {
		out[index] = int64(binary.LittleEndian.Uint64(data[index*8:]))
	}
	return out, nil
}

// computeDictBitWidth computes the bit width required to index a dictionary of
// the given size (parquet requires a minimum of 1).
func computeDictBitWidth(dictSize int) int {
	if dictSize <= 1 {
		return 1
	}
	width := 0
	for (1 << uint(width)) < dictSize {
		width++
	}
	return width
}

// decodeRLEBitPacked decodes the RLE/bit-packed hybrid encoding of count int32
// values with the given bit width. The input begins with a 4-byte
// little-endian length prefix.
func decodeRLEBitPacked(data []byte, bitWidth, count int) ([]int32, error) {
	if len(data) < 4 {
		return nil, errBadEncoding
	}
	// The length prefix describes the encoded data that follows.
	data = data[4:]
	out := make([]int32, 0, count)
	byteWidth := (bitWidth + 7) / 8
	for len(out) < count {
		header, consumed, err := uvarint(data)
		if err != nil {
			return nil, err
		}
		data = data[consumed:]
		isBitPacked := header&1 == 1
		numValues := int(header >> 1)
		if numValues <= 0 {
			return nil, errBadEncoding
		}
		if isBitPacked {
			// A bit-packed run header stores the number of groups, each group
			// holding 8 values bit-packed into bitWidth bytes.
			numGroups := int(header >> 1)
			if len(data) < numGroups*bitWidth {
				return nil, errBadEncoding
			}
			for group := 0; group < numGroups; group++ {
				groupBytes := data[group*bitWidth : (group+1)*bitWidth]
				for valueIndex := 0; valueIndex < 8 && len(out) < count; valueIndex++ {
					out = append(out, int32(readBitsLE(groupBytes, valueIndex*bitWidth, bitWidth)))
				}
			}
			data = data[numGroups*bitWidth:]
		} else {
			// RLE run: one value repeated numValues times.
			if len(data) < byteWidth {
				return nil, errBadEncoding
			}
			var value int32
			for byteIndex := 0; byteIndex < byteWidth; byteIndex++ {
				value |= int32(data[byteIndex]) << (8 * byteIndex)
			}
			// Sign-extend for negative values.
			if bitWidth < 32 && (value&(1<<(bitWidth-1))) != 0 {
				value |= ^((1 << bitWidth) - 1)
			}
			data = data[byteWidth:]
			for index := 0; index < numValues && len(out) < count; index++ {
				out = append(out, value)
			}
		}
	}
	return out[:count], nil
}

// readBitsLE reads bitWidth bits starting at bitOffset (little-endian bit
// order) from buffer.
func readBitsLE(buffer []byte, bitOffset, bitWidth int) uint64 {
	var value uint64
	for bitIndex := 0; bitIndex < bitWidth; bitIndex++ {
		bit := bitOffset + bitIndex
		if (buffer[bit/8]>>(bit%8))&1 == 1 {
			value |= 1 << bitIndex
		}
	}
	return value
}
