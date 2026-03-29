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
  deep learning model (CNN) via ONNX inference. The model is bundled with
  the package. Optional dependencies `numpy` and `onnxruntime` are required
  and can be installed with `pip install nlpo3[deepcut]`.
  The deepcut model and its ONNX port originate from
  [Deepcut](https://github.com/rkcosmos/deepcut) and
  [LEKCut](https://github.com/PyThaiNLP/LEKCut).
