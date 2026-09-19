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

	// encoderHidden is the CED encoder's final hidden state H_(d/2) [B, T, d].
	// Decoder full layers project their global compressed KV from it; it is nil
	// on encoder groups and when CED is disabled.
	encoderHidden *tensors.Tensor
	// gradientEncoder accumulates the gradient of the decoder's global KV
	// projections with respect to encoderHidden. It is injected into the
	// encoder output at the split boundary during the backward pass.
	gradientEncoder *tensors.Tensor

	// replay marks a CED decoder bounded-replay pass: the group's global
	// compressed cache has already been filled from the encoder hidden state,
	// so the layers must not buffer or compress it again.
	replay bool
}
