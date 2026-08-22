package parquet

import (
	"bytes"
	"testing"
)

func TestSnappyLiteral(t *testing.T) {
	// "hello": uncompressed len 5, literal tag (5-1)<<2 = 0x10.
	raw := []byte{0x05, 0x10, 'h', 'e', 'l', 'l', 'o'}
	out, err := snappyDecode(raw)
	if err != nil {
		t.Fatalf("snappyDecode: %v", err)
	}
	if string(out) != "hello" {
		t.Fatalf("got %q, want hello", out)
	}
}

func TestSnappyCopy1(t *testing.T) {
	// "aaaaa": literal 'a' then copy 4 bytes from offset 1.
	raw := []byte{0x05, 0x00, 'a', 0x01, 0x01}
	out, err := snappyDecode(raw)
	if err != nil {
		t.Fatalf("snappyDecode: %v", err)
	}
	if string(out) != "aaaaa" {
		t.Fatalf("got %q, want aaaaa", out)
	}
}

func TestSnappyLiteral60(t *testing.T) {
	// A literal of length 61 uses the 60-tag form: 1-byte length (60) + 1.
	lit := bytes.Repeat([]byte("x"), 61)
	raw := []byte{byte(len(lit))} // uncompressed length (61 < 128)
	raw = append(raw, 60<<2)      // tag: literal, length-in-next-byte
	raw = append(raw, byte(60))   // length-1 = 60
	raw = append(raw, lit...)
	out, err := snappyDecode(raw)
	if err != nil {
		t.Fatalf("snappyDecode: %v", err)
	}
	if !bytes.Equal(out, lit) {
		t.Fatalf("length %d, want %d", len(out), len(lit))
	}
}

func TestSnappyCopy2(t *testing.T) {
	// Literal of 12 "ab" patterns, then a copy with 2-byte offset.
	lit := []byte("abcdefghijkl")
	// First element: literal of length 12: tag = (12-1)<<2 = 0x2C.
	raw := []byte{byte(len(lit) * 2)} // uncompressed len 24
	raw = append(raw, 0x2C)           // literal tag
	raw = append(raw, lit...)
	// Copy 12 bytes from offset 12: type 2, length = 1 + (tag>>2). length=12 => tag>>2 = 11 => tag = 11<<2 | 2 = 0x2E.
	raw = append(raw, 0x2E)
	raw = append(raw, 12, 0) // offset = 12 (2-byte LE)
	out, err := snappyDecode(raw)
	if err != nil {
		t.Fatalf("snappyDecode: %v", err)
	}
	want := append(append([]byte(nil), lit...), lit...)
	if !bytes.Equal(out, want) {
		t.Fatalf("got %q, want %q", out, want)
	}
}

func TestSnappyRoundtrip(t *testing.T) {
	// A manually compressed stream: "the quick brown fox" with a trailing copy.
	lit := []byte("the quick brown fox")
	raw := []byte{byte(len(lit))}
	raw = append(raw, byte((len(lit)-1)<<2))
	raw = append(raw, lit...)
	out, err := snappyDecode(raw)
	if err != nil {
		t.Fatalf("snappyDecode: %v", err)
	}
	if !bytes.Equal(out, lit) {
		t.Fatalf("got %q, want %q", out, lit)
	}
}

func TestSnappyCorrupt(t *testing.T) {
	if _, err := snappyDecode([]byte{0x05, 0x10, 'h'}); err == nil {
		t.Fatal("expected error for truncated literal")
	}
	if _, err := snappyDecode([]byte{}); err == nil {
		t.Fatal("expected error for empty input")
	}
}
