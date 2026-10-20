package model

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/tensors"
)

// TestReplaySWARebuildsExactWhenWindowCoversPrefix verifies the bounded-replay
// primitive: when the replay window covers the whole cached prefix, replaying a
// stripped snapshot reproduces the original raw sliding-window keys exactly.
func TestReplaySWARebuildsExactWhenWindowCoversPrefix(t *testing.T) {
	config := Config{
		SequenceLen: 16, VocabSize: 16, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L", CompressionRatio: 2, SWAWindow: 8,
	}
	model := NewTransformer(config)
	model.InitWeights(tensors.NewRNG(42))
	perturb := tensors.NewRNG(123)
	for _, parameter := range model.Parameters() {
		for index := range parameter.Data {
			parameter.Data[index] += perturb.NormFloat32() * 0.1
		}
	}

	prompt := []int32{1, 5, 2, 8, 3, 7} // len 6 <= SWAWindow+1
	ids := tensors.NewInt32sWithData([]int{1, len(prompt)}, prompt)

	cache := NewKVBuffer(1, config.SequenceLen, config.NumLayer, config.NumKVHead, config.HeadDim())
	cache.EnableCompression(config.Compression(), config.EmbedDim, config.NumKVHead*config.HeadDim(), config.SequenceLen/config.Compression()+1)
	model.Forward(ids, cache)

	headDim := config.HeadDim()
	for layer := 0; layer < config.NumLayer; layer++ {
		for head := 0; head < config.NumKVHead; head++ {
			key, _ := cache.storage.readRaw(layer, 0, head, len(prompt))
			for element := 0; element < len(prompt)*headDim; element++ {
				if math.IsNaN(float64(key[element])) {
					t.Fatal("original key cache is non-finite")
				}
			}
		}
	}

	stripped := cache.Clone()
	stripped.StripRaw()
	stripped.SetPosition(0)
	stripped.SetPrevEmbedding(nil)
	model.ReplaySWA(ids, stripped)

	if stripped.Position() != len(prompt) {
		t.Fatalf("position after replay = %d, want %d", stripped.Position(), len(prompt))
	}
	for layer := 0; layer < config.NumLayer; layer++ {
		for head := 0; head < config.NumKVHead; head++ {
			gotKey, gotValue := stripped.storage.readRaw(layer, 0, head, len(prompt))
			wantKey, wantValue := cache.storage.readRaw(layer, 0, head, len(prompt))
			for element := 0; element < len(prompt)*headDim; element++ {
				if math.Abs(float64(gotKey[element]-wantKey[element])) > 1e-5 {
					t.Fatalf("replayed key layer %d head %d element %d = %v, want %v", layer, head, element, gotKey[element], wantKey[element])
				}
				if math.Abs(float64(gotValue[element]-wantValue[element])) > 1e-5 {
					t.Fatalf("replayed value layer %d head %d element %d = %v, want %v", layer, head, element, gotValue[element], wantValue[element])
				}
			}
		}
	}
}

// TestReplaySWARepopulatesTailWindow verifies that a window smaller than the
// prefix is rebuilt (the tail rows become non-zero and finite) and the cache is
// left at the full prefix length.
func TestReplaySWARepopulatesTailWindow(t *testing.T) {
	config := Config{
		SequenceLen: 32, VocabSize: 16, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L", CompressionRatio: 2, SWAWindow: 4,
	}
	model := NewTransformer(config)
	model.InitWeights(tensors.NewRNG(7))
	perturb := tensors.NewRNG(99)
	for _, parameter := range model.Parameters() {
		for index := range parameter.Data {
			parameter.Data[index] += perturb.NormFloat32() * 0.1
		}
	}

	prompt := []int32{1, 5, 2, 8, 3, 7, 4, 6, 2, 9, 1, 3}
	ids := tensors.NewInt32sWithData([]int{1, len(prompt)}, prompt)
	cache := NewKVBuffer(1, config.SequenceLen, config.NumLayer, config.NumKVHead, config.HeadDim())
	cache.EnableCompression(config.Compression(), config.EmbedDim, config.NumKVHead*config.HeadDim(), config.SequenceLen/config.Compression()+1)
	model.Forward(ids, cache)

	stripped := cache.Clone()
	stripped.StripRaw()

	window := config.SWAWindowSize()
	start := len(prompt) - window - 1
	if start < 0 {
		start = 0
	}
	stripped.SetPosition(start)
	model.ReplaySWA(tensors.NewInt32sWithData([]int{1, len(prompt) - start}, prompt[start:]), stripped)

	if stripped.Position() != len(prompt) {
		t.Fatalf("position after replay = %d, want %d", stripped.Position(), len(prompt))
	}
	headDim := config.HeadDim()
	for layer := 0; layer < config.NumLayer; layer++ {
		for head := 0; head < config.NumKVHead; head++ {
			key, _ := stripped.storage.readRaw(layer, 0, head, len(prompt))
			nonZero := false
			for position := len(prompt) - window; position < len(prompt); position++ {
				base := position * headDim
				for element := 0; element < headDim; element++ {
					value := key[base+element]
					if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
						t.Fatal("replayed window key is non-finite")
					}
					if value != 0 {
						nonZero = true
					}
				}
			}
			if !nonZero {
				t.Fatal("replayed window key is all zero")
			}
		}
	}
}
