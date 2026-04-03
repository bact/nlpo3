---
SPDX-FileCopyrightText: 2026 PyThaiNLP Project
SPDX-License-Identifier: CC0-1.0
---

# Design notes

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

### npm packaging proposal

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
