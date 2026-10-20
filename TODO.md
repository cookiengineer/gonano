# TODO

The verified baseline, key files, tests and caveats are documented in
`HANDOVER.md`. Only outstanding work remains here.

# Remaining V4.1 serving work

- [ ] Broader JSON Schema coverage: anyOf/oneOf/allOf, patternProperties, type unions, structural enums, numeric bounds and string formats
- [ ] Scheduler prefix-cache persistence to disk; currently scheduler mode disables the engine disk cache
- [ ] Append-log speculative rollback instead of replaying the accepted prefix
- [ ] Tool-call argument validation against client-supplied JSON Schemas
