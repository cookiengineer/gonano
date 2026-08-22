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
	"github.com/cookiengineer/gonano/tensor"
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
	Step            int                    `json:"step"`
	ValBPB          *float32               `json:"val_bpb,omitempty"`
	ModelConfig     model.Config           `json:"model_config"`
	UserConfig      map[string]any         `json:"user_config,omitempty"`
	LoopState       map[string]any         `json:"loop_state,omitempty"`
	DataloaderState map[string]any         `json:"dataloader_state,omitempty"`
	Params          []ParamMeta            `json:"params"`
}

// Save writes the model parameters and metadata to a single checkpoint file.
func Save(path string, meta Meta, params map[string]*tensor.Tensor) error {
	names := make([]string, 0, len(params))
	for name := range params {
		names = append(names, name)
	}
	sort.Strings(names)
	meta.Params = make([]ParamMeta, len(names))
	for i, name := range names {
		meta.Params[i] = ParamMeta{Name: name, Shape: params[name].Shape}
	}

	metaBytes, err := json.Marshal(meta)
	if err != nil {
		return err
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	if _, err := f.WriteString(magic); err != nil {
		return err
	}
	var lenBuf [4]byte
	binary.LittleEndian.PutUint32(lenBuf[:], uint32(len(metaBytes)))
	if _, err := f.Write(lenBuf[:]); err != nil {
		return err
	}
	if _, err := f.Write(metaBytes); err != nil {
		return err
	}

	buf := make([]byte, 0, 1<<20)
	for _, name := range names {
		data := params[name].Data
		buf = buf[:0]
		buf = float32sToBytes(data, buf)
		if _, err := f.Write(buf); err != nil {
			return err
		}
	}
	return nil
}

// Load reads a checkpoint file, returning the metadata and parameters.
func Load(path string) (Meta, map[string]*tensor.Tensor, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Meta{}, nil, err
	}
	if len(data) < len(magic)+4 || string(data[:len(magic)]) != magic {
		return Meta{}, nil, fmt.Errorf("checkpoint: invalid magic in %s", path)
	}
	pos := len(magic)
	metaLen := int(binary.LittleEndian.Uint32(data[pos:]))
	pos += 4
	if pos+metaLen > len(data) {
		return Meta{}, nil, fmt.Errorf("checkpoint: truncated metadata in %s", path)
	}
	var meta Meta
	if err := json.Unmarshal(data[pos:pos+metaLen], &meta); err != nil {
		return Meta{}, nil, err
	}
	pos += metaLen

	params := make(map[string]*tensor.Tensor, len(meta.Params))
	for _, pm := range meta.Params {
		n := 1
		for _, d := range pm.Shape {
			n *= d
		}
		need := n * 4
		if pos+need > len(data) {
			return Meta{}, nil, fmt.Errorf("checkpoint: truncated params in %s", path)
		}
		t := tensor.New(pm.Shape...)
		bytesToFloat32s(data[pos:pos+need], t.Data)
		pos += need
		params[pm.Name] = t
	}
	return meta, params, nil
}

func float32sToBytes(src []float32, dst []byte) []byte {
	for _, v := range src {
		dst = binary.LittleEndian.AppendUint32(dst, math.Float32bits(v))
	}
	return dst
}

func bytesToFloat32s(src []byte, dst []float32) {
	for i := range dst {
		dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:]))
	}
}

// LoadModel reconstructs a Transformer from a checkpoint's metadata and params.
func LoadModel(meta Meta, params map[string]*tensor.Tensor) *model.Transformer {
	m := model.NewTransformer(meta.ModelConfig)
	byName := m.NamedParameters()
	for name, p := range params {
		if target, ok := byName[name]; ok {
			copy(target.Data, p.Data)
		}
	}
	return m
}

// findLastStep returns the largest step number among model_<step>.gn files in
// dir.
func findLastStep(dir string) (int, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0, err
	}
	re := regexp.MustCompile(`model_(\d+)\.gn$`)
	last := -1
	for _, e := range entries {
		m := re.FindStringSubmatch(e.Name())
		if m != nil {
			var step int
			fmt.Sscanf(m[1], "%d", &step)
			if step > last {
				last = step
			}
		}
	}
	if last < 0 {
		return 0, fmt.Errorf("checkpoint: no checkpoints in %s", dir)
	}
	return last, nil
}

// FindLastStep returns the largest checkpoint step in dir.
func FindLastStep(dir string) (int, error) { return findLastStep(dir) }

// ModelPath returns the checkpoint file path for a directory and step.
func ModelPath(dir string, step int) string {
	return filepath.Join(dir, fmt.Sprintf("model_%06d.gn", step))
}
