package model

import "testing"

// TestTotalParamsForConfigMatchesBuilt checks the analytic count matches a
// built dense model exactly (the dense config has no compressor/indexer/router
// tensors that the estimate deliberately omits).
func TestTotalParamsForConfigMatchesBuilt(t *testing.T) {
	config := testConfig()
	built := NewTransformer(config).TotalParams()
	estimated := TotalParamsForConfig(config)
	if int64(built) != estimated {
		t.Fatalf("TotalParamsForConfig = %d, want built TotalParams %d", estimated, built)
	}
}

// TestTotalParamsForConfigMoE checks the estimate tracks the MoE flash preset
// and is within a couple of percent of the built model.
func TestTotalParamsForConfigMoE(t *testing.T) {
	flash, _ := ParsePreset("flash")
	config := ConfigForPreset(flash, 4, 4096, 64, 64, 128, "SSSL")
	built := int64(NewTransformer(config).TotalParams())
	estimated := TotalParamsForConfig(config)
	delta := built - estimated
	if delta < 0 {
		delta = -delta
	}
	if float64(delta)/float64(built) > 0.02 {
		t.Fatalf("estimate %d deviates from built %d by more than 2%%", estimated, built)
	}
}

// TestEstimatedTrainingMemoryBytes checks the estimate dominates the raw
// weight footprint and grows with batch and sequence length.
func TestEstimatedTrainingMemoryBytes(t *testing.T) {
	config := testConfig()
	parameters := TotalParamsForConfig(config)
	weights := parameters * 4
	estimate := EstimatedTrainingMemoryBytes(config, 1, config.SequenceLen)
	if estimate <= weights {
		t.Fatalf("estimate %d should exceed weights %d", estimate, weights)
	}
	larger := EstimatedTrainingMemoryBytes(config, 4, config.SequenceLen*2)
	if larger <= estimate {
		t.Fatalf("estimate should grow with batch/sequence: %d vs %d", larger, estimate)
	}
	if EstimatedTrainingMemoryBytes(config, 0, 0) <= weights {
		t.Fatal("zero batch/sequence should still include the training state")
	}
}
