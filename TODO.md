# TODO

The verified baseline, key files, tests and caveats are documented in
`HANDOVER.md`. Only outstanding work remains here.

# Remaining V4.1 serving work

- [ ] EnsembleScheduler preemption and replay across member caches
- [ ] Broader JSON Schema coverage: anyOf/oneOf/allOf, patternProperties, type unions, structural enums, numeric bounds and string formats
- [ ] Token-trie mask construction to avoid full-vocabulary scans on new constraint states
- [ ] Latency metrics: TTFT, tokens/second, queue wait time and per-step latency
- [ ] Mixed prefill/decode throughput: batched prefill with padding/query masks instead of T=1 prefill while decoding
- [ ] Batched CED, batched speculative decoding and batched Ensemble `n > 1` paths
- [ ] Scheduler prefix-cache persistence to disk; currently scheduler mode disables the engine disk cache
- [ ] Append-log speculative rollback instead of replaying the accepted prefix
- [ ] Tool-call argument validation against client-supplied JSON Schemas
- [ ] Parallel merge of paged-attention page results (page computation is already parallel)
