---
SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
SPDX-License-Identifier: Apache-2.0
---

# Changelog

All notable changes to the nlpo3 Python package are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-03-31

### Added

- `segment_deepcut()` function for Thai word tokenization using the deepcut
  deep learning model (CNN) via ONNX inference. The model is bundled with
  the package and runs via the pure-Rust `tract-onnx` engine with no
  additional Python runtime dependencies. Custom ONNX model paths are
  supported via the `DeepcutTokenizer(model_path=...)` API.
  The deepcut model and its ONNX port originate from
  [Deepcut](https://github.com/rkcosmos/deepcut) and
  [LEKCut](https://github.com/PyThaiNLP/LEKCut).
- `DeepcutTokenizer` class exposing the Rust `DeepcutTokenizer` struct,
  which implements the `Tokenizer` trait and is enabled by the `deepcut`
  Cargo feature.

### Changed

- **Breaking**: version bumped to 2.0.0.
- Rust crate edition updated from 2018 to 2024; `rust-version` set to `"1.88.0"`.
- Rust 2024 edition features applied throughout the core crate:
  - `let` chains: `while let … && condition` in the main tokenization loop;
    `if let … && let …` in trie prefix lookup.
  - Reference patterns: `&position`, `&token`, `&(s, idx)`, `&(group, type_idx)`
    and `|(_, ch)|` patterns replace explicit `*deref` operators.
  - `Copy` types passed by value: `TrieNode` helper methods now take `char`
    instead of `&char`.
  - `entry().or_default()` / `entry().or_insert_with()` replace duplicated
    `match` / `if let … else` graph-update blocks.
  - Removed `unsafe { from_utf8_unchecked }` in favor of safe
    `from_utf8().expect(…)`.
  - `rfind_space_char_index` rewritten as a functional iterator chain.

[Unreleased]: https://github.com/PyThaiNLP/nlpo3/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/PyThaiNLP/nlpo3/compare/v1.4.0...v2.0.0
