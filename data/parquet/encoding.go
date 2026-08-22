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
	for i := 0; i < count; i++ {
		n, k, err := uvarint(data)
		if err != nil {
			return nil, err
		}
		data = data[k:]
		if int(n) > len(data) {
			return nil, errBadEncoding
		}
		out[i] = data[:int(n)]
		data = data[int(n):]
	}
	return out, nil
}

// decodePlainInt32 decodes count PLAIN-encoded INT32 values (4-byte LE each).
func decodePlainInt32(data []byte, count int) ([]int32, error) {
	if len(data) < count*4 {
		return nil, errBadEncoding
	}
	out := make([]int32, count)
	for i := 0; i < count; i++ {
		out[i] = int32(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out, nil
}

// decodePlainInt64 decodes count PLAIN-encoded INT64 values.
func decodePlainInt64(data []byte, count int) ([]int64, error) {
	if len(data) < count*8 {
		return nil, errBadEncoding
	}
	out := make([]int64, count)
	for i := 0; i < count; i++ {
		out[i] = int64(binary.LittleEndian.Uint64(data[i*8:]))
	}
	return out, nil
}

// bitWidthForDict computes the bit width required to index a dictionary of the
// given size (parquet requires a minimum of 1).
func bitWidthForDict(dictSize int) int {
	if dictSize <= 1 {
		return 1
	}
	w := 0
	for (1 << uint(w)) < dictSize {
		w++
	}
	return w
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
		header, k, err := uvarint(data)
		if err != nil {
			return nil, err
		}
		data = data[k:]
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
			for g := 0; g < numGroups; g++ {
				group := data[g*bitWidth : (g+1)*bitWidth]
				for v := 0; v < 8 && len(out) < count; v++ {
					out = append(out, int32(readBitsLE(group, v*bitWidth, bitWidth)))
				}
			}
			data = data[numGroups*bitWidth:]
		} else {
			// RLE run: one value repeated numValues times.
			if len(data) < byteWidth {
				return nil, errBadEncoding
			}
			var val int32
			for b := 0; b < byteWidth; b++ {
				val |= int32(data[b]) << (8 * b)
			}
			// Sign-extend for negative values.
			if bitWidth < 32 && (val&(1<<(bitWidth-1))) != 0 {
				val |= ^((1 << bitWidth) - 1)
			}
			data = data[byteWidth:]
			for i := 0; i < numValues && len(out) < count; i++ {
				out = append(out, val)
			}
		}
	}
	return out[:count], nil
}

// readBitsLE reads bitWidth bits starting at bitOffset (little-endian bit
// order) from b.
func readBitsLE(b []byte, bitOffset, bitWidth int) uint64 {
	var v uint64
	for i := 0; i < bitWidth; i++ {
		bit := bitOffset + i
		if (b[bit/8]>>(bit%8))&1 == 1 {
			v |= 1 << i
		}
	}
	return v
}
