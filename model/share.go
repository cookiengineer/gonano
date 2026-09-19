package model

import "github.com/cookiengineer/gonano/tensors"

// compressionShare is the cross-layer compressed-attention state shared by one
// group of layers (a producing full layer followed by reindex/reuse layers).
// The transformer forward loop allocates a fresh share at each full layer and
// passes it to the following reuse layers, so per-group state survives until
// the backward pass.
//
// In training, keyCompressed/valueCompressed alias the producing layer's
// compressed tensors and gradientKey/gradientValue accumulate the compressed
// key/value gradients contributed by every layer in the group. In inference
// keyCompressed/valueCompressed are nil and reuse layers read the producing
// layer's cache slots instead.
type compressionShare struct {
	producer        int
	keyCompressed   *tensors.Tensor // [B, Hkv, blocks, D] (training only)
	valueCompressed *tensors.Tensor // [B, Hkv, blocks, D] (training only)
	selection       [][][]int       // [B*Hkv][token][], most recently published
	gradientKey     *tensors.Tensor // [B, Hkv, blocks, D] (training only)
	gradientValue   *tensors.Tensor // [B, Hkv, blocks, D] (training only)
}
