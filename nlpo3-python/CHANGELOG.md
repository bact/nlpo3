---
SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
SPDX-License-Identifier: Apache-2.0
---

# Changelog

All notable changes to the nlpo3 Python package are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-04-01

### Added

- `NewmmTokenizer` class exposing the Rust `NewmmTokenizer` struct, which
  implements the `Tokenizer` trait. Replace `load_dict(path, name)` +
  `segment(text, name)` with `NewmmTokenizer(path).segment(text)`.
- `NewmmFstTokenizer` class exposing the Rust `NewmmFstTokenizer` struct.
  Uses a finite-state automaton (FST) dictionary back-end that is ~49×
  more memory-efficient than the default TrieChar back-end at the cost of
  slower per-lookup speed.
- `DeepcutTokenizer` class exposing the Rust `DeepcutTokenizer` struct,
  which implements the `Tokenizer` trait and is enabled by the `deepcut`
  Cargo feature. Performs Thai word tokenization using a CNN (deepcut
  model) via ONNX inference (`tract-onnx`). The model is bundled with
  the package and runs with no additional Python runtime dependencies.
  Custom ONNX model paths are supported via `DeepcutTokenizer(model_path=...)`.
  The deepcut model and its ONNX port originate from
  [Deepcut](https://github.com/rkcosmos/deepcut) and
  [LEKCut](https://github.com/PyThaiNLP/LEKCut).
- All three tokenizer classes are exposed at the top-level `nlpo3` namespace.
- All classes are `frozen` in PyO3 (immutable Python objects), enabling
  lock-free concurrent use from multiple threads, including free-threaded
  CPython (PEP 703).
- `segment()` on dictionary-based tokenizers now raises `RuntimeError` on
  tokenization failure instead of panicking.

### Changed

- **Breaking**: version bumped to 2.0.0.
- Rust crate edition updated from 2018 to 2024;
  `rust-version` set to `"1.88.0"`.

### Removed

- **Breaking**: `load_dict()`, `segment()`, and `segment_deepcut()` free
  functions removed. Use `NewmmTokenizer`, `NewmmFstTokenizer`, and
  `DeepcutTokenizer` classes instead.

### Migration

```python
# Before (v1.x)
from nlpo3 import load_dict, segment, segment_deepcut
load_dict("path/to/dict.txt", "mydict")
tokens = segment("สวัสดีครับ", "mydict")
tokens = segment_deepcut("สวัสดีครับ")

# After (v2.0)
from nlpo3 import NewmmTokenizer, NewmmFstTokenizer, DeepcutTokenizer
tok = NewmmTokenizer("path/to/dict.txt")
tokens = tok.segment("สวัสดีครับ")
tokens = DeepcutTokenizer().segment("สวัสดีครับ")
```

[Unreleased]: https://github.com/PyThaiNLP/nlpo3/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/PyThaiNLP/nlpo3/compare/v1.4.0...v2.0.0
