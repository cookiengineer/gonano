// Package checkpoint saves and loads model checkpoints: parameters, optimizer
// state, and training metadata, in a compact versioned binary format.
package checkpoint

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
)

// magic identifies a gonano checkpoint file.
const magic = "GONANO\x00\x01"

// ParamMeta describes one parameter in the metadata header.
type ParamMeta struct {
	Name  string `json:"name"`
	Shape []int  `json:"shape"`
}

// Meta is the JSON metadata stored in a checkpoint.
type Meta struct {
	Step            int            `json:"step"`
	ValBPB          *float32       `json:"val_bpb,omitempty"`
	ModelConfig     model.Config   `json:"model_config"`
	UserConfig      map[string]any `json:"user_config,omitempty"`
	LoopState       map[string]any `json:"loop_state,omitempty"`
	DataloaderState map[string]any `json:"dataloader_state,omitempty"`
	Params          []ParamMeta    `json:"params"`
}

// Save writes the model parameters and metadata to a single checkpoint file.
func Save(path string, meta Meta, params map[string]*tensors.Tensor) error {
	names := make([]string, 0, len(params))
	for name := range params {
		names = append(names, name)
	}
	sort.Strings(names)
	meta.Params = make([]ParamMeta, len(names))
	for index, name := range names {
		meta.Params[index] = ParamMeta{Name: name, Shape: params[name].Shape}
	}

	metaBytes, err := json.Marshal(meta)
	if err != nil {
		return err
	}

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	if _, err := file.WriteString(magic); err != nil {
		return err
	}
	var lengthBuffer [4]byte
	binary.LittleEndian.PutUint32(lengthBuffer[:], uint32(len(metaBytes)))
	if _, err := file.Write(lengthBuffer[:]); err != nil {
		return err
	}
	if _, err := file.Write(metaBytes); err != nil {
		return err
	}

	buffer := make([]byte, 0, 1<<20)
	for _, name := range names {
		paramData := params[name].Data
		buffer = buffer[:0]
		buffer = appendFloat32sAsBytes(paramData, buffer)
		if _, err := file.Write(buffer); err != nil {
			return err
		}
	}
	return nil
}

// Load reads a checkpoint file, returning the metadata and parameters.
func Load(path string) (Meta, map[string]*tensors.Tensor, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Meta{}, nil, err
	}
	if len(data) < len(magic)+4 || string(data[:len(magic)]) != magic {
		return Meta{}, nil, fmt.Errorf("checkpoint: invalid magic in %s", path)
	}
	position := len(magic)
	metaLength := int(binary.LittleEndian.Uint32(data[position:]))
	position += 4
	if position+metaLength > len(data) {
		return Meta{}, nil, fmt.Errorf("checkpoint: truncated metadata in %s", path)
	}
	var meta Meta
	if err := json.Unmarshal(data[position:position+metaLength], &meta); err != nil {
		return Meta{}, nil, err
	}
	position += metaLength

	params := make(map[string]*tensors.Tensor, len(meta.Params))
	for _, paramMeta := range meta.Params {
		numElements := 1
		for _, dimension := range paramMeta.Shape {
			numElements *= dimension
		}
		neededBytes := numElements * 4
		if position+neededBytes > len(data) {
			return Meta{}, nil, fmt.Errorf("checkpoint: truncated params in %s", path)
		}
		tensor := tensors.New(paramMeta.Shape...)
		decodeFloat32sFromBytes(data[position:position+neededBytes], tensor.Data)
		position += neededBytes
		params[paramMeta.Name] = tensor
	}
	return meta, params, nil
}

func appendFloat32sAsBytes(source []float32, destination []byte) []byte {
	for _, value := range source {
		destination = binary.LittleEndian.AppendUint32(destination, math.Float32bits(value))
	}
	return destination
}

func decodeFloat32sFromBytes(source []byte, destination []float32) {
	for index := range destination {
		destination[index] = math.Float32frombits(binary.LittleEndian.Uint32(source[index*4:]))
	}
}

// IsDSpark reports whether a checkpoint holds a DSpark drafter rather than a
// plain transformer.
func IsDSpark(meta Meta) bool {
	dspark, _ := meta.UserConfig["dspark"].(bool)
	return dspark
}

// LoadModel reconstructs a Transformer from a checkpoint's metadata and params.
func LoadModel(meta Meta, params map[string]*tensors.Tensor) *model.Transformer {
	transformer := model.NewTransformer(meta.ModelConfig)
	parametersByName := transformer.NamedParameters()
	for name, param := range params {
		if target, ok := parametersByName[name]; ok {
			copy(target.Data, param.Data)
		}
	}
	return transformer
}

// findLastStep returns the largest step number among model_<step>.gn files in
// dir.
func findLastStep(dir string) (int, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0, err
	}
	pattern := regexp.MustCompile(`model_(\d+)\.gn$`)
	lastStep := -1
	for _, entry := range entries {
		matches := pattern.FindStringSubmatch(entry.Name())
		if matches != nil {
			var step int
			fmt.Sscanf(matches[1], "%d", &step)
			if step > lastStep {
				lastStep = step
			}
		}
	}
	if lastStep < 0 {
		return 0, fmt.Errorf("checkpoint: no checkpoints in %s", dir)
	}
	return lastStep, nil
}

// FindLastStep returns the largest checkpoint step in dir.
func FindLastStep(dir string) (int, error) { return findLastStep(dir) }

// ModelPath returns the checkpoint file path for a directory and step.
func ModelPath(dir string, step int) string {
	return filepath.Join(dir, fmt.Sprintf("model_%06d.gn", step))
}
