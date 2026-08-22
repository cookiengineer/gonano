# gonano — TODO

Legend: `[ ]` pending, `[x]` done (unit tests pass). A feature is done only when its tests pass.

## P0 — Scaffold
- [x] `go.mod` build setup documented (GOEXPERIMENT=simd)
- [x] `logging`: slog setup, banner, Metrics sink
- [x] `parallel`: Pool, ParallelFor, ParallelReduce
- [x] `device`: NumCPU, simd.VectorBitSize(), Emulated()

## P1 — tensor (execution substrate)
- [x] `tensor.Tensor` + `Int32s` (shape, strides, views, reshape)
- [x] elementwise: Add/Sub/Mul/Div/Neg/Abs/Scale/Relu2
- [x] reduce: Sum/Max/ArgMax/Mean over axis
- [x] matmul: tiled SIMD GEMM (op-parallel)
- [x] softmax, rmsnorm, math (Sigmoid/Tanh/Softcap/Rsqrt)
- [x] random: seeded RNG + Normal/Uniform fill

## P2 — nn + model
- [x] `nn`: Linear, Embedding, activations, RMSNorm, init
- [x] `model.Config` + depth→width/heads derivation
- [x] `model`: Transformer, CausalSelfAttention (GQA/QK-norm/rotary/value-emb), MLP, Block
- [x] rotary precompute + apply
- [x] flops: EstimateFlopsPerToken, NumMatmulParams, NumScalingParams, Decode/Prefill, KVBytes
- [x] InitWeights (nanochat stds)

## P3 — tokenizer
- [x] special tokens
- [x] splitter (GPT-4 pattern)
- [x] BPE training (byte-level, heap merges)
- [x] encode/decode/batch + DecodeSingleTokenBytes + TokenBytes
- [x] RenderConversation / RenderForCompletion
- [x] serialization (save/load)

## P4 — data
- [x] Snappy block decoder
- [x] parquet reader (metadata + PLAIN + RLE_DICTIONARY)
- [x] HF Hub download + manifest
- [x] dataset: shard download, list_parquet_files
- [x] pretrain dataloader (BOS best-fit)
- [x] SFT dataloader (best-fit pad, loss masks)

## P5 — optim
- [x] AdamW (fused, SIMD)
- [x] Muon (PolarExpress + MuonEq + Muon+ + variance reduction)
- [x] MuonAdamW (group routing + state)

## P6 — train + checkpoint
- [x] scaling laws (batch/LR/wd)
- [x] schedulers (LR warmup/warmdown, Muon momentum, wd cosine)
- [x] training loop (grad accumulation, metrics)
- [x] checkpoint format + save/load + layout

## P7 — infer
- [x] KVBuffer
- [x] sampler (temp, top-k, argmax, multinomial)
- [x] Engine (prefill→replicate→decode, tool-use state machine)
- [x] bench (TTFT/TPOT/MBU/MFU)

## P8 — eval
- [x] BitsPerByte
- [x] CORE metric (+ minimal YAML)
- [x] tasks: MMLU/GSM8K/ARC/HumanEval/SmolTalk + TaskMixture
- [x] ChatCORE (categorical + generative)

## P9 — SFT + RL
- [x] SFT training loop
- [x] RL (GRPO/REINFORCE) loop

## P10 — hardening
- [x] cmd/ mains (tok_train, tok_eval, base_train, base_eval, chat_sft, chat_rl, chat_eval, chat_cli, infer_bench)
- [x] benchmarks + `runs/`-equivalent configs
- [x] `go vet` + `GOEXPERIMENT=simd go test -race ./...` clean
