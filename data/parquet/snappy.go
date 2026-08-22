// Package parquet implements a minimal, dependency-free reader for the Parquet
// columnar format, sufficient to read the text columns used by gonano's
// datasets (PLAIN and RLE_DICTIONARY BYTE_ARRAY encodings, with SNAPPY or no
// compression).
package parquet

import (
	"encoding/binary"
	"errors"
)

var errCorrupt = errors.New("parquet: corrupt snappy block")

// snappyDecode decompresses a raw (unframed) Snappy block as used inside
// Parquet data pages. It follows the format described in Google's snappy
// format_description.txt.
func snappyDecode(src []byte) ([]byte, error) {
	// Read the uncompressed length varint.
	length, n, err := uvarint(src)
	if err != nil {
		return nil, err
	}
	src = src[n:]
	dst := make([]byte, 0, length)
	pos := 0
	for len(src) > 0 {
		tag := src[0]
		src = src[1:]
		switch tag & 0x03 {
		case 0: // literal
			var litLen int
			switch tag >> 2 {
			case 60:
				if len(src) < 1 {
					return nil, errCorrupt
				}
				litLen = int(src[0]) + 1
				src = src[1:]
			case 61:
				if len(src) < 2 {
					return nil, errCorrupt
				}
				litLen = int(binary.LittleEndian.Uint16(src)) + 1
				src = src[2:]
			case 62:
				if len(src) < 3 {
					return nil, errCorrupt
				}
				litLen = int(loadUint24(src)) + 1
				src = src[3:]
			case 63:
				if len(src) < 4 {
					return nil, errCorrupt
				}
				litLen = int(binary.LittleEndian.Uint32(src)) + 1
				src = src[4:]
			default:
				litLen = int(tag>>2) + 1
			}
			if len(src) < litLen {
				return nil, errCorrupt
			}
			dst = append(dst, src[:litLen]...)
			src = src[litLen:]
			pos += litLen
		case 1: // copy, 1-byte offset
			if len(src) < 1 {
				return nil, errCorrupt
			}
			length := 4 + int((tag>>2)&0x07)
			offset := int((tag&0xE0)<<3) | int(src[0])
			src = src[1:]
			if offset <= 0 || offset > pos {
				return nil, errCorrupt
			}
			for i := 0; i < length; i++ {
				dst = append(dst, dst[pos-offset+i])
			}
			pos += length
		case 2: // copy, 2-byte offset
			if len(src) < 2 {
				return nil, errCorrupt
			}
			length := 1 + int(tag>>2)
			offset := int(binary.LittleEndian.Uint16(src))
			src = src[2:]
			if offset <= 0 || offset > pos {
				return nil, errCorrupt
			}
			for i := 0; i < length; i++ {
				dst = append(dst, dst[pos-offset+i])
			}
			pos += length
		case 3: // copy, 4-byte offset
			if len(src) < 4 {
				return nil, errCorrupt
			}
			length := 1 + int(tag>>2)
			offset := int(binary.LittleEndian.Uint32(src))
			src = src[4:]
			if offset <= 0 || offset > pos {
				return nil, errCorrupt
			}
			for i := 0; i < length; i++ {
				dst = append(dst, dst[pos-offset+i])
			}
			pos += length
		}
	}
	if pos != int(length) {
		return nil, errCorrupt
	}
	return dst, nil
}

func uvarint(src []byte) (uint64, int, error) {
	var x uint64
	var s uint
	for i := 0; i < len(src); i++ {
		b := src[i]
		if b < 0x80 {
			if i > 9 || (i == 9 && b > 1) {
				return 0, 0, errCorrupt
			}
			return x | uint64(b)<<s, i + 1, nil
		}
		x |= uint64(b&0x7f) << s
		s += 7
	}
	return 0, 0, errCorrupt
}

func loadUint24(b []byte) uint32 {
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16
}
