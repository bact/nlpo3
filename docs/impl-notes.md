---
SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
SPDX-License-Identifier: CC0-1.0
---

# Implementation notes

This document stores implementation details and design decisions that are not
intended for end-user release notes.

## API redesign summary (v2 line)

- The tokenizer API was normalized around explicit tokenizer objects.
- Core trait methods were simplified to text-only calls for default behavior.
- Advanced controls were moved to explicit option-bearing methods.
- Python and Node.js bindings migrated from global registry helpers to
  class-based APIs.

## Tokenizer families

- NewMM (Trie backend): default dictionary tokenizer, optimized for speed.
- NewMM (FST backend): same algorithm with reduced dictionary memory.
- Deepcut: neural tokenizer backed by ONNX inference.

## Parallel segmentation design

- Parallel controls use chunk-based processing.
- User-facing explicit control is exposed through `parallel_chunk_size`.
- Ergonomic `segment_parallel(...)` methods compute chunk size automatically
  and call the same underlying options path.

### Auto chunk-size heuristic

Current heuristic inputs:

- Input byte length.
- Runtime available parallelism from `std::thread::available_parallelism()`.

Current safeguards:

- Bound target chunk size to a min/max range.
- Limit target chunk fan-out.
- Disable parallel mode for small input.

## One-chunk rule

- If total input length is below `MIN_CHUNK_SIZE * 2`, processing remains
  single-chunk.
- This rule applies in auto mode and in parallelization gating logic.

Rationale:

- Splitting small text usually adds overhead without throughput benefit.
- It reduces extra allocations and scheduling overhead.

## UTF-8 chunk safety

- Chunk search windows are aligned to UTF-8 character boundaries before slicing.
- Boundary correction uses `is_char_boundary()` checks.
- This prevents slicing in the middle of a multi-byte character.

## Split-point strategy

Chunk boundaries prefer:

1. Sentence-ending punctuation.
2. Whitespace.
3. Valid Thai Character Cluster boundaries.

Fallback:

- Use nearest safe boundary to preserve forward progress.

## Chunk boundary behavior

When `parallel_chunk_size` is set, text is split into chunks before
tokenization. Token sequences near chunk boundaries can differ from
full-text tokenization.

Reasons for the divergence:

- **NewMM:** dictionary matching at the boundary may prefer different token
  paths when the surrounding context changes between full-text and chunked
  processing.
- **Deepcut:** the CNN model uses a fixed-width context window (21 characters).
  Characters near chunk boundaries have fewer adjacent context characters from
  the neighboring chunk, which shifts model predictions.

The chunked output is acceptable for tasks that treat text holistically, such
as text classification and word embedding. It may not be suitable for tasks
that require precise linguistic unit identification.

### Future direction: fine-grained chunk merging

Smooth boundary stitching is a potential future improvement. For
dictionary-based tokenizers (NewMM), a partial implementation exists in the
safe-mode boundary-scanning logic (`TEXT_SCAN_LEFT`/`TEXT_SCAN_RIGHT` window),
but it trades performance for accuracy, and enabling it on the chunk boundary
alone may degrade overall throughput. The boundary window size and the chunk
size ratio that yield negligible accuracy loss have not been established. A
future implementation would need to measure this trade-off to find practical
default settings.

## NewMM safety and ambiguity behavior

- Safe mode remains available for highly ambiguous text.
- Previous BFS path explosion risk was mitigated with visited-set controls.

## Concurrency model

- Tokenizer instances are designed for read-heavy concurrent usage.
- Dictionary structures are shared where possible.
- Mutation methods use copy-on-write behavior when shared ownership exists.

## Why user docs are concise

- README and CHANGELOG files target average library users.
- Internal architecture details, heuristics, and trade-offs are documented here.

## Proposed split packaging for bindings (design only)

Status: proposed architecture. This section documents a packaging direction and
release strategy. It is not implemented yet.

### Goals

- Keep the base package small for common dictionary-based tokenization.
- Avoid shipping heavy model/runtime dependencies by default.
- Keep user-facing API stable where possible.
- Let users install optional assets/features explicitly.

### Python packaging proposal

Proposed packages:

- `nlpo3` (base):
  - Includes NewMM tokenizers and core Rust extension.
  - Does not bundle default dictionary.
  - Does not include Deepcut model or ONNX-heavy dependency path.
  - Users pass their own dictionary path.
- `nlpo3-dict` (optional data package):
  - Provides default `words_th.txt` and dictionary access helpers.
  - No ONNX/model dependency.
- `nlpo3-deepcut` (optional model package):
  - Provides Deepcut model assets and Deepcut-enabled extension/runtime.
  - Pulls required model/runtime dependencies.

Expected install flows:

- `pip install nlpo3` for lightweight dictionary-based tokenization with
  user-supplied dictionary.
- `pip install nlpo3-dict` to add default dictionary data.
- `pip install nlpo3-deepcut` to add `DeepcutTokenizer` support.

API behavior proposal:

- Base package exports NewMM classes always.
- Deepcut import is conditional:
  - If deepcut package is present, `DeepcutTokenizer` is available.
  - If absent, import or construction returns a clear guidance error message.

### Node.js packaging proposal

Proposed packages:

- `nlpo3` (base):
  - Includes dictionary-based tokenizers only.
  - No bundled dictionary by default.
  - No Deepcut model/runtime dependency.
- `nlpo3-dict` (optional data package):
  - Ships `words_th.txt` and helper utilities for locating/loading it.
- `nlpo3-deepcut` (optional model package):
  - Ships Deepcut model assets and Deepcut-enabled native addon/runtime.

Expected install flows:

- `npm install nlpo3` for lightweight default usage.
- `npm install nlpo3-dict` to add default dictionary assets.
- `npm install nlpo3-deepcut` to add deepcut tokenizer support.

API behavior proposal:

- Base package exports NewMM tokenizers always.
- Deepcut symbol is optional and loaded only when deepcut package is installed.
- Missing deepcut package should produce deterministic runtime guidance.

### Build and release model

- Maintain separate release pipelines per package.
- Keep core tokenizer code in one source tree; package-specific wrappers select
  enabled features and bundled assets.
- Use compatibility constraints so `-dict` and `-deepcut` packages track the
  same major/minor line as base.

### CI and test strategy

- Base package CI validates:
  - NewMM tokenizers, dictionary-from-user-path flows, no deepcut dependency.
- Dict package CI validates:
  - Data packaging, default dictionary loading behavior, version sync.
- Deepcut package CI validates:
  - Model availability, deepcut inference path, platform wheel/addon coverage.
- Add compatibility tests for mixed installs (base + dict, base + deepcut,
  base + dict + deepcut).

### Migration and compatibility notes

- Keep current monolithic behavior for one transition window.
- Provide deprecation notes and migration examples in README/CHANGELOG.
- Preserve class names and method signatures where possible.
- Document package-selection matrix for common use cases (small install,
  dictionary convenience, neural tokenizer support).
