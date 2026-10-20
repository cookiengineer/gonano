package checkpoint

import (
	"path/filepath"
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
)

func TestDFlashRoundTrip(t *testing.T) {
	config := model.ConfigForDepth(2, 16, 16, 32, 32, "SSSL")
	config.ApplyPreset(model.PresetFlash)
	config.NumNextNPredictLayers = 1
	config.DFlashBlockSize = 3
	config.DFlashMarkovRank = 4
	config.DFlashTargetLayerIDs = []int{config.NumLayer - 1}
	config.Validate()
	dflash := model.NewDFlashFromConfig(config)
	dflash.InitWeights(tensors.NewRNG(42))
	path := filepath.Join(t.TempDir(), "dflash.gn")
	meta := Meta{Step: 1, ModelConfig: config, UserConfig: map[string]any{"dflash": true}}
	if err := Save(path, meta, dflash.NamedParameters()); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loadedMeta, loadedParams, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !IsDFlash(loadedMeta) {
		t.Fatal("checkpoint should be detected as DFlash")
	}
	loaded := LoadDFlash(loadedMeta, loadedParams)
	original := dflash.NamedParameters()
	rebuilt := loaded.NamedParameters()
	for name, parameter := range original {
		target, ok := rebuilt[name]
		if !ok {
			t.Fatalf("missing parameter %s", name)
		}
		for index := range parameter.Data {
			if parameter.Data[index] != target.Data[index] {
				t.Fatalf("parameter %s differs at %d", name, index)
			}
		}
	}
}
