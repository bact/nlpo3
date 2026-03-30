---
SPDX-FileCopyrightText: 2024 PyThaiNLP Project
SPDX-License-Identifier: Apache-2.0
---

# Changelog

All notable changes to the nlpo3 Python package are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `segment_deepcut()` function for Thai word tokenization using the deepcut
  deep learning model (CNN) via ONNX inference.  The model is bundled with
  the package and runs via the pure-Rust `tract-onnx` engine with no
  additional Python runtime dependencies.  Custom ONNX model paths are
  supported via the `DeepcutTokenizer(model_path=...)` API in the core Rust
  crate and exposed through the `segment_deepcut()` Python wrapper.
  The deepcut model and its ONNX port originate from
  [Deepcut](https://github.com/rkcosmos/deepcut) and
  [LEKCut](https://github.com/PyThaiNLP/LEKCut).
- `DeepcutTokenizer` struct in the core `nlpo3` Rust crate
  (`src/tokenizer/deepcut`), implementing the `Tokenizer` trait.
  Enabled with the `deepcut` Cargo feature.
