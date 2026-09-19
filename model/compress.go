package model

import (
	"math"

	"github.com/cookiengineer/gonano/model/layers"
	"github.com/cookiengineer/gonano/tensors"
)

// ChannelCompressor is the learned dense compressor used by HCA-style
// compressed attention. It reduces the sequence length of a [batch, sequence,
// channels] tensor by a fixed ratio: every `ratio` consecutive rows are merged
// into one, using per-channel softmax weights derived from the hidden state
// plus a learnable per-position bias (DeepSeek-V4 eqs. 20-23).
//
// Unlike the sparse CSA path, compression is dense: there is no indexer and no
// top-k selection, which keeps the CPU kernels branch-free and vectorizable.
type ChannelCompressor struct {
	// Ratio is the number of input rows merged into one output row.
	Ratio int
	// Channels is the width of the compressed representation.
	Channels int
	// logitWeight maps the hidden state to per-channel compression logits.
	logitWeight *layers.Linear
	// bias is the [ratio, channels] learnable positional bias added to the
	// logits before the per-block softmax.
	bias *tensors.Tensor
}

// NewChannelCompressor builds a compressor that maps a hidden state of width
// embeddingDimension to per-channel logits and merges `ratio` rows into one
// row of width channels.
func NewChannelCompressor(embeddingDimension, channels, ratio int) *ChannelCompressor {
	if ratio < 1 {
		panic("model: compression ratio must be >= 1")
	}
	return &ChannelCompressor{
		Ratio:       ratio,
		Channels:    channels,
		logitWeight: layers.NewLinear(embeddingDimension, channels),
		bias:        tensors.New(ratio, channels),
	}
}

// ZeroGrad zeroes the gradients of the compressor's parameters.
func (compressor *ChannelCompressor) ZeroGrad() {
	compressor.logitWeight.Weight.ZeroGrad()
	compressor.bias.ZeroGrad()
}

// compressorContext holds the activations needed to backpropagate the
// compressor.
type compressorContext struct {
	hidden        *tensors.Tensor // [B, T, d]
	value         *tensors.Tensor // [B, T, C] compressor input
	compressed    *tensors.Tensor // [B, blocks, C] compressor output
	probabilities *tensors.Tensor // [B, blocks, ratio, C]
	batchSize     int
	sequenceLen   int
	blockCount    int
}

// Forward compresses `value` [batch, sequence, channels] into
// [batch, ceil(sequence/ratio), channels]. `hidden` [batch, sequence, d] is the
// attention input used to derive the compression logits. A trailing partial
// block is compressed from the rows that exist; padded positions are excluded
// from the softmax.
func (compressor *ChannelCompressor) Forward(hidden, value *tensors.Tensor) (*tensors.Tensor, *compressorContext) {
	batchSize, sequenceLength := hidden.Shape[0], hidden.Shape[1]
	channels := compressor.Channels
	ratio := compressor.Ratio

	logits := compressor.logitWeight.Forward(hidden) // [B, T, C]
	logitData, biasData := logits.Data, compressor.bias.Data
	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		for position := 0; position < sequenceLength; position++ {
			blockPosition := position % ratio
			base := (batchIndex*sequenceLength + position) * channels
			biasBase := blockPosition * channels
			for channel := 0; channel < channels; channel++ {
				logitData[base+channel] += biasData[biasBase+channel]
			}
		}
	}

	blockCount := (sequenceLength + ratio - 1) / ratio
	probabilities := tensors.New(batchSize, blockCount, ratio, channels)
	compressed := tensors.New(batchSize, blockCount, channels)
	valueData := value.Data
	probabilityData := probabilities.Data
	compressedData := compressed.Data

	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		for block := 0; block < blockCount; block++ {
			for channel := 0; channel < channels; channel++ {
				// Numerically stable softmax over the rows of this block,
				// treating out-of-range positions as -Inf.
				maximum := float32(math.Inf(-1))
				for row := 0; row < ratio; row++ {
					position := block*ratio + row
					if position >= sequenceLength {
						continue
					}
					logit := logitData[(batchIndex*sequenceLength+position)*channels+channel]
					if logit > maximum {
						maximum = logit
					}
				}
				var total float64
				for row := 0; row < ratio; row++ {
					position := block*ratio + row
					probability := float32(0)
					if position < sequenceLength && maximum > float32(math.Inf(-1)) {
						probability = float32(math.Exp(float64(logitData[(batchIndex*sequenceLength+position)*channels+channel] - maximum)))
					}
					probabilityData[((batchIndex*blockCount+block)*ratio+row)*channels+channel] = probability
					total += float64(probability)
				}
				if total > 0 {
					inverse := float32(1.0 / total)
					var accumulator float32
					for row := 0; row < ratio; row++ {
						probabilityIndex := ((batchIndex*blockCount+block)*ratio + row) * channels
						probabilityData[probabilityIndex+channel] *= inverse
						position := block*ratio + row
						if position < sequenceLength {
							accumulator += probabilityData[probabilityIndex+channel] * valueData[(batchIndex*sequenceLength+position)*channels+channel]
						}
					}
					compressedData[(batchIndex*blockCount+block)*channels+channel] = accumulator
				}
			}
		}
	}

	context := &compressorContext{
		hidden:        hidden,
		value:         value,
		compressed:    compressed,
		probabilities: probabilities,
		batchSize:     batchSize,
		sequenceLen:   sequenceLength,
		blockCount:    blockCount,
	}
	return compressed, context
}

// Backward distributes the gradient of the compressed output back to the
// hidden state and the value tensor, and accumulates the gradients of the
// compressor's own parameters. Both returned tensors have the same shapes as
// the corresponding Forward inputs.
func (compressor *ChannelCompressor) Backward(gradientCompressed *tensors.Tensor, context *compressorContext) (gradientHidden, gradientValue *tensors.Tensor) {
	batchSize := context.batchSize
	sequenceLength := context.sequenceLen
	blockCount := context.blockCount
	ratio := compressor.Ratio
	channels := compressor.Channels

	gradientLogits := tensors.New(batchSize, sequenceLength, channels)
	gradientValue = tensors.New(batchSize, sequenceLength, channels)
	probabilityData := context.probabilities.Data
	valueData := context.value.Data
	compressedData := context.compressed.Data
	gradientCompressedData := gradientCompressed.Data
	gradientLogitsData := gradientLogits.Data
	gradientValueData := gradientValue.Data

	// Backward through the per-channel block softmax. For
	// compressed = sum_j p_j * value_j the chain rule gives
	// dlogit_j = p_j * g * (value_j - compressed).
	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		for block := 0; block < blockCount; block++ {
			for channel := 0; channel < channels; channel++ {
				gradient := gradientCompressedData[(batchIndex*blockCount+block)*channels+channel]
				compressedValue := compressedData[(batchIndex*blockCount+block)*channels+channel]
				for row := 0; row < ratio; row++ {
					position := block*ratio + row
					if position >= sequenceLength {
						continue
					}
					probability := probabilityData[((batchIndex*blockCount+block)*ratio+row)*channels+channel]
					value := valueData[(batchIndex*sequenceLength+position)*channels+channel]
					gradientLogitsData[(batchIndex*sequenceLength+position)*channels+channel] += probability * gradient * (value - compressedValue)
					gradientValueData[(batchIndex*sequenceLength+position)*channels+channel] += probability * gradient
				}
			}
		}
	}

	// Logit projection and positional bias.
	gradientHidden = compressor.logitWeight.Backward(context.hidden, gradientLogits)
	compressor.bias.EnsureGrad()
	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		for position := 0; position < sequenceLength; position++ {
			blockPosition := position % ratio
			for channel := 0; channel < channels; channel++ {
				compressor.bias.Grad[blockPosition*channels+channel] += gradientLogitsData[(batchIndex*sequenceLength+position)*channels+channel]
			}
		}
	}
	return gradientHidden, gradientValue
}
