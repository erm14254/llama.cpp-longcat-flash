# MiniMax-M3 upstream reconstruction notes

## Approved refs

- Approved v2 base: `67b9b0e7f6ce45d929a4411907d3c48ec719e81c`
- Frozen v1 reference: `74c6fa3a2409bee89cb0f5ba7fadd533642a9ac1`
- Model/MSA source: `#24908` at `da2c16850307dfe595c8d5d21c3a7257916cd017`
- Vision source: `#25113` at `e56d2f029621e657ffcbfa2c4eb6bb2e432746dd`
- Parser source: `#24523` at `66f43aa655a07999c7746fe9ff5ede94835e921e`

## Component ownership

The model/MSA import owns the text conversion, GGUF metadata, architecture, graph,
hyperparameter, KV-cache, model, vocabulary, and architecture-test changes needed
for MiniMax-M3 text inference and sparse attention.

The vision overlay owns only the conversion/GGUF vision additions and the mtmd
MiniMax-M3 projector implementation. It is intentionally restricted to:

- `conversion/__init__.py`
- `conversion/minimax.py`
- `gguf-py/gguf/constants.py`
- `gguf-py/gguf/tensor_mapping.py`
- `tools/mtmd/CMakeLists.txt`
- `tools/mtmd/clip-impl.h`
- `tools/mtmd/clip-model.h`
- `tools/mtmd/clip.cpp`
- `tools/mtmd/models/minimax-m3.cpp`
- `tools/mtmd/models/models.h`
- `tools/mtmd/mtmd.cpp`

The parser import owns only `common/chat.cpp`.

## Corrected applicability check

The approved no-working-tree applicability check for the model/MSA layer was:

```sh
git merge-tree --write-tree \
  --merge-base=67776eaee549be9e1e0359726c13c399b9224d2e \
  67b9b0e7f6ce45d929a4411907d3c48ec719e81c \
  da2c16850307dfe595c8d5d21c3a7257916cd017
```

It exited with status 0 and produced tree:

```text
6cdf753da3ff91e802886216a61b8b4dcc4b2701
```

The import was mechanically three-way applicable, but enum, tensor-map,
registration, graph, and cache files still required source inspection because
those files are order-sensitive or behavior-sensitive.

## Current-master preservation decisions

Current master added Laguna support after the original phase-1 base. The
reconstruction preserves the Laguna conversion registration, tokenizer pre-type,
GGUF architecture and tensors, tensor mappings, model registration, and
architecture tests.

Current master also changed the DeepSeek4 APE tensor operation classification in
`src/llama-arch.cpp`; that correction is preserved.

`src/llama-context.cpp` preserves all current graph-heavy architecture cases and
adds MiniMax-M3 to the same larger graph-node budget path.

## Vision divergence and stale-MSA exclusion

The pinned `#25113` head contains both vision work and stale `src/` files. The
reconstruction imports only the approved vision overlay paths and does not copy
any `#25113` `src/` file. The text/model/MSA implementation remains the
`#24908` version after the vision overlay.

## Parser isolation

The parser import is isolated to the pinned `common/chat.cpp` delta. Changes to
chat-auto-parser files from other histories do not authorize importing additional
parser or conversion files.

## Known limitations

- This reconstruction does not include model-weight tests and does not download
  GGUF or Hugging Face model files.
- MSA runtime behavior is compile-tested and source-audited here; full numerical
  validation requires MiniMax-M3 weights.
- Vision conversion and mtmd behavior are compile-tested and source-audited here;
  full image-path validation requires a compatible MiniMax-M3 vision model.

## Future update procedure

1. Re-read protected refs and pinned source refs before changing the branch.
2. If the base moves, perform a focused drift review before importing changes.
3. Re-run the model/MSA merge-tree check with the pinned `#24908` merge base.
4. Re-apply source layers in order: model/MSA, vision-only overlay, parser-only.
5. Keep newly authored tests separate from imported-source commits.
6. Re-run conversion imports, CPU build, focused tests, and full available ctest.
7. Compare the final tree with the frozen v1 reference and classify every diff.
