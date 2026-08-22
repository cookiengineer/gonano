# gonano — Implementation Plan

Reimplementation of [nanochat](https://github.com/karpathy/nanochat) in pure Go (1.27) using the `simd` standard-library package. This is a reusable library and framework: no `internal/` package paths, every package is importable, and `cmd/*` are thin executables that compose the library.

## Locked decisions

| Decision | Choice |
|---|---|
| Scope | Full nanochat parity (tokenizer, pretrain, SFT, RL/GRPO, CORE/BPB/ChatCORE eval, chat CLI, infer benchmark) |
| Dependencies | Zero third-party deps — pure stdlib (hand-written BPE, parquet reader, HF-Hub HTTP, minimal YAML) |
| Numeric precision | `float32` everywhere (no bf16, fp8, int8) |
| Parallelism | Goroutine-per-op data parallel via a shared `parallel.Pool` sized to `runtime.NumCPU()` |
| Compute | `simd.Float32s` (AVX/AVX2/AVX-512, auto-selected, `GODEBUG=simd=512` forces 512-bit) |

## Build prerequisite

The `simd` package is experimental and tagged `//go:build goexperiment.simd`. Every build, test, and run must set:

```bash
GOEXPERIMENT=simd go build ./...
GOEXPERIMENT=simd go test ./...
```

`simd.VectorBitSize()` is 128/256/512 depending on CPU; code is written width-agnostic via `Len()` / `LoadFloat32sPart` / `StorePart`, with scalar tails. `simd.Emulated()` reports pure-Go emulation (tests must pass in both modes).

## Guiding principles

- Three logical domains: **training** (`train`), **inference** (`infer`), and the **execution substrate** (`tensor` + `nn` + `model`). A small `exec` package handles sandboxed code execution.
- Every SIMD kernel has a scalar twin and a parity test (`scalar == simd` within tolerance). A feature is only "done" when its unit tests pass.
- float32 everywhere — master weights, activations, gradients, and optimizer state.
- Expressive naming mirrors the reference (`Transformer`, `CausalSelfAttention`, `BitsPerByte`, `KVBuffer`, `MuonAdamW`, ...).

## Package layout

```
gonano/
├── go.mod                          # module github.com/cookiengineer/gonano, go 1.27.0
├── parallel/                       # goroutine worker-pool
│   ├── pool.go                     # Pool, ParallelFor, ParallelReduce
├── tensor/                         # EXECUTION SUBSTRATE: tensors + SIMD kernels
│   ├── tensor.go                   # Tensor { Shape []int; Data []float32 } row-major
│   ├── shape.go                    # shape/strides/2D/3D/4D views, Reshape
│   ├── matmul.go                   # tiled GEMM (SIMD MulAdd inner loop, op-parallel)
│   ├── elementwise.go              # Add/Sub/Mul/Div/Neg/Abs/Scale/Relu2 (simd)
│   ├── reduce.go                   # Sum/Max/ArgMax/Mean over an axis (simd)
│   ├── softmax.go                  # stable softmax (max-subtract; scalar exp)
│   ├── rmsnorm.go                  # RMSNorm
│   ├── math.go                     # Sigmoid/Tanh/Softcap/Rsqrt
│   ├── random.go                   # seeded RNG (PCG), Normal/Uniform fill
│   └── ids.go                      # Int32s { Shape; Data []int32 } for token ids/masks
├── nn/                             # layer primitives
│   ├── linear.go                   # Linear{Weight [out,in] float32; Bias bool}
│   ├── embedding.go                # Embedding lookup
│   ├── activation.go               # ReLU2, Sigmoid, Tanh, Softcap
│   ├── norm.go                     # RMSNorm (stateless)
│   └── init.go                     # initNormal/initUniform/initZeros (nanochat stds)
├── model/                          # the GPT transformer
│   ├── config.go                   # Config + Depth→width/heads derivation
│   ├── transformer.go              # Transformer, Forward (train+infer), InitWeights, SetupOptimizer
│   ├── attention.go                # CausalSelfAttention (GQA, QK-norm, rotary, value-emb gate)
│   ├── mlp.go                      # MLP (c_fc 4x → relu² → c_proj)
│   ├── block.go                    # Block = attn + mlp
│   ├── rotary.go                   # RoPE precompute + ApplyRotary
│   └── flops.go                    # FLOPs/param/scaling accounting
├── optim/                          # optimizers
│   ├── optimizer.go                # ParamGroup, Optimizer interface
│   ├── adamw.go                    # fused AdamW step (SIMD)
│   ├── muon.go                     # momentum→MuonEq→PolarExpress→Muon+→var-reduction→cautious wd
│   └── muon_adamw.go               # MuonAdamW: group routing + state mgmt
├── tokenizer/                      # BPE
│   ├── tokenizer.go                # Tokenizer: Encode/Decode/Batch/special tokens
│   ├── splitter.go                 # hand-written GPT-4 split-pattern matcher
│   ├── bpe.go                      # TrainFromIterator (byte-level BPE)
│   ├── render.go                   # RenderConversation, RenderForCompletion
│   ├── special.go                  # 9 special tokens
│   └── bytes.go                    # DecodeSingleTokenBytes, TokenBytes
├── data/                           # datasets + dataloaders
│   ├── dataset.go                  # shard download, file lock, list_parquet_files
│   ├── parquet/                    # minimal parquet reader
│   │   ├── reader.go               # metadata (Thrift compact), column chunks
│   │   ├── snappy.go               # Snappy block decoder
│   │   └── encoding.go             # PLAIN + RLE_DICTIONARY for BYTE_ARRAY
│   ├── hub.go                      # HF Hub API → parquet shard URLs + manifest
│   ├── pretrain.go                 # BOS-aligned best-fit dataloader
│   └── sft.go                      # SFT best-fit packing loader (loss masks, -1 padding)
├── train/                          # TRAINING
│   ├── scaling.go                  # depth→tokens/batch/LR/wd scaling laws
│   ├── scheduler.go                # LR/Muon-momentum/wd schedules
│   ├── loop.go                     # gradient-accumulation training loop
│   ├── base.go                     # base_train orchestration
│   ├── sft.go                      # SFT loop (dataset-driven stopping)
│   ├── rl.go                       # GRPO/REINFORCE on GSM8K
│   └── metrics.go                  # EMA loss, tok/sec, MFU, ETA
├── infer/                          # INFERENCE
│   ├── kvcache.go                  # KVBuffer
│   ├── sampler.go                  # SampleNextToken (temp, top-k, argmax)
│   ├── engine.go                   # Engine.Generate (prefill→replicate→decode)
│   ├── tooluse.go                  # calculator state machine (safe Go evaluator)
│   └── bench.go                    # TTFT/TPOT/MBU/MFU
├── eval/                           # evaluation
│   ├── bpb.go                      # BitsPerByte
│   ├── core.go                     # CORE metric
│   ├── chatcore.go                 # ChatCORE + categorical/generative loops
│   ├── yaml.go                     # minimal YAML subset for core.yaml
│   └── tasks/                      # Task interface + datasets
│       ├── task.go  mmlu.go  gsm8k.go  arc.go  humaneval.go  smoltalk.go
├── exec/                           # sandboxed code execution (HumanEval/tool use)
│   └── exec.go                     # python subprocess w/ rlimits + scrubbed env + timeout
├── checkpoint/                     # persistence
│   ├── format.go                   # versioned binary: magic + JSON meta + raw float32 blobs
│   ├── checkpoint.go               # Save/Load model+optim+meta
│   └── layout.go                   # dirs, FindLargestModel, FindLastStep
├── device/                         # hardware abstraction
│   ├── device.go                   # NumCPU, simd.VectorBitSize(), Emulated()
│   └── roofline.go                 # CPU peak-flops/bw estimate
├── logging/                        # slog setup, banner, Metrics sink
└── cmd/
    ├── tok_train/  tok_eval/
    ├── base_train/  base_eval/
    ├── chat_sft/  chat_rl/  chat_eval/
    └── chat_cli/  infer_bench/
```

Reference→Go traceability: `gpt.py→model/`, `engine.py→infer/`, `optim.py→optim/`, `tokenizer.py→tokenizer/`, `dataloader.py→data/pretrain.go`, `dataset.py→data/`, `loss_eval.py→eval/bpb.go`, `core_eval.py→eval/core.go`, `flash_attention.py→model/attention.go` (CPU attention), `fp8.py→dropped`, `execution.py→exec/`, `checkpoint_manager.py→checkpoint/`, `tasks/*→eval/tasks/`, `scripts/*→cmd/*`.

## Core data model

- `tensor.Tensor`: `Shape []int` + contiguous row-major `Data []float32`. 4D tensors use `(B,T,H,D)`. Weights are `[out,in]`.
- `tensor.Int32s`: `[]int32` for token ids/loss masks.
- `optim.ParamGroup` carries `Kind` + hyperparameters (adamw/muon) mirroring `setup_optimizer`.
- `checkpoint` serializes `map[string]*Tensor` + JSON metadata (config, step, val_bpb, loop_state, dataloader_state) + raw little-endian float32 payloads, with a version field.

## SIMD kernel design

| Kernel | SIMD usage | Notes |
|---|---|---|
| GEMM | `Float32s.MulAdd` inner K loop, tiled M×N, `parallel.Pool` over tiles | contiguous weight loads along `in` dim |
| RMSNorm | `Mul` squares, `Sqrt`+`Div` rsqrt, `Mul` scale | |
| Softmax | `Max` reduce (SIMD), scalar `math.Exp` | simd has no exp; minor cost |
| ReLU² | `Max(0,x)` then `Mul` | |
| Sigmoid/Tanh/Softcap | scalar `math.Exp` | tiny |
| Sum/Max/ArgMax reduce | SIMD combine + horizontal fold | |
| Embedding gather | index math + copy | |
| Rotary apply | elementwise `Mul`/`Add` on dim pairs | |

## Parallelism (goroutine-per-op data parallel)

- `parallel.Pool` sized `runtime.NumCPU()` (respecting `GOMAXPROCS`), one per process.
- Matmul/reduce fan out over output tiles; tokenization fans out per document.
- Inference: `Engine` batches `num_samples`; the `(B,T,H,D)` matmuls are op-parallel.
- Training: single model copy, gradient accumulation; optimizer steps op-parallel.
- Multi-process scale-out is out of scope for v1; optimizer exposes an `AllReduce` hook.

## Key risks & mitigations

1. **Parquet reader (largest item)** — hand-written Thrift compact metadata, Snappy, PLAIN + RLE_DICTIONARY BYTE_ARRAY. Scope to the known climbmix/HF schema; golden fixtures; fail loudly on unsupported encodings.
2. **GPT-4 split regex** — Go `regexp` (RE2) rejects possessive quantifiers and lookahead. Hand-written splitter validated against the Python reference.
3. **No `exp`/`tanh` in simd** — scalar `math.Exp` first; vectorized fast-exp later if benchmarks justify.
4. **`core.yaml`** — minimal parser for the known `icl_tasks` structure.
5. **BPE memory over 2B chars** — streaming, `doc_cap`, compact pair-count map.
6. **No numerical parity with Python (bf16 vs fp32)** — target self-consistency + sane loss/bpb, not bit-parity.

## Milestones (each ends with `go vet` + `GOEXPERIMENT=simd go test -race ./...` green)

- **P0 Scaffold** — GOEXPERIMENT build setup, `logging`, `parallel.Pool`, `device`.
- **P1 Tensor** — tensor/ids, elementwise, reduce, matmul, softmax, rmsnorm, rng (scalar-vs-SIMD parity tests).
- **P2 Model** — `nn` + `model` + `flops`; forward shape/numeric tests on tiny configs; `InitWeights` checks.
- **P3 Tokenizer** — splitter + BPE train + encode/decode + render + serialization (round-trip + golden tests).
- **P4 Data** — parquet + snappy + hub download + pretrain/SFT dataloaders (fixture tests).
- **P5 Optim** — adamw + muon + muonadamw (unit tests on synthetic grads).
- **P6 Train + checkpoint** — scaling laws, schedulers, loop, checkpoint (depth-2 overfit smoke test).
- **P7 Infer** — kvcache + sampler + engine + tool-use (engine == naive generate parity test).
- **P8 Eval** — bpb + core + tasks + chatcore (smoke on tiny slices).
- **P9 SFT + RL** — chat_sft, chat_rl loops.
- **P10 Harden** — cmd mains, benchmarks, `runs/`-equivalent configs.

## Testing strategy

- Scalar-vs-SIMD parity tests for every kernel (fp32 relative tolerance ~1e-4).
- Tokenizer round-trip + special-token + merge-order + reference-splitter golden tests.
- Parquet golden fixtures (hand-built + real climbmix shard).
- End-to-end: depth-2 overfit, checkpoint round-trip, Engine==naive, BPB/CORE smoke.

## Definition of "done"

A feature is **done** only when its dedicated unit tests pass (`GOEXPERIMENT=simd go test -race`), covering both the SIMD and scalar/emulated paths where applicable. Proof-of-concept or untested code is not "done".
