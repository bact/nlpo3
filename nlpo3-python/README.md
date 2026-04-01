---
SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
SPDX-License-Identifier: Apache-2.0
---

# nlpO3 Python binding

[![PyPI](https://img.shields.io/pypi/v/nlpo3.svg "PyPI")](https://pypi.python.org/pypi/nlpo3)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg "Python 3.9")](https://www.python.org/downloads/)
[![Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg "Apache-2.0")](https://opensource.org/license/apache-2-0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14082448.svg)](https://doi.org/10.5281/zenodo.14082448)

Python binding for nlpO3,
a Thai natural language processing library written in Rust.

To install:

```bash
pip install nlpo3
```

## Table of Contents

- [Features](#features)
- [Use](#use)
  - [Dictionary](#dictionary)
- [Build](#build)
- [Issues](#issues)
- [License](#license)
- [Binary wheels](#binary-wheels)

## Features

Three interchangeable Thai word tokenizers:

- `NewmmTokenizer` — dictionary-based maximal-matching (TrieChar backend,
  fastest, ~43 MB for 62k words). [2.5× faster][benchmark] than PyThaiNLP's newmm.
- `NewmmFstTokenizer` — same algorithm with FST backend (~49× less memory)
- `DeepcutTokenizer` — neural CNN model (no dictionary needed).
  Based on [deepcut][deepcut] via [LEKCut][lekcut].

All three implement the same `segment()` interface and are interchangeable.
Each instance is **read-only after construction** and safe to reuse across
many concurrent calls or Python threads.

[benchmark]: ./notebooks/nlpo3_segment_benchmarks.ipynb
[deepcut]: https://github.com/rkcosmos/deepcut
[lekcut]: https://github.com/PyThaiNLP/LEKCut

## Use

### Dictionary-based tokenizer (newmm, TrieChar backend)

```python
from nlpo3 import NewmmTokenizer

# Create once — the dictionary is loaded only once.
# Reuse the same instance for every call; no dictionary copying occurs.
tok = NewmmTokenizer("path/to/dict.txt")

# Basic tokenization
tok.segment("สวัสดีครับ")
# => ["สวัสดี", "ครับ"]

# Safe mode — avoids long run times on ambiguous input
tok.segment("สวัสดีครับ", safe=True)

# Parallel mode — multi-threaded processing (higher memory use)
tok.segment("สวัสดีครับ", parallel=True)
```

### Dictionary-based tokenizer (nf, FST backend)

Same API as `NewmmTokenizer`, but uses ~49× less memory:

```python
from nlpo3 import NewmmFstTokenizer

tok = NewmmFstTokenizer("path/to/dict.txt")
tok.segment("สวัสดีครับ")
```

### Neural tokenizer (deepcut)

```python
from nlpo3 import DeepcutTokenizer

# No dictionary needed. The model is compiled once and cached in the instance.
tok = DeepcutTokenizer()
tok.segment("สวัสดีครับ")
# => ["สวัสดี", "ครับ"]

# Custom ONNX model
tok = DeepcutTokenizer(model_path="/path/to/custom.onnx")
tok.segment("สวัสดีครับ")
```

### Parallel and distributed environments

Each tokenizer instance is **read-only after construction** and safe to
call from multiple Python threads concurrently — no locking is needed:

```python
import concurrent.futures
from nlpo3 import NewmmTokenizer

# Create once, share across all workers.
tok = NewmmTokenizer("path/to/dict.txt")

texts = ["สวัสดีครับ", "การตัดคำ", "ภาษาไทย"]

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(tok.segment, texts))
```

For multiprocessing (separate processes), each process creates its own
instance. Loading a 62k-word dictionary typically takes less than 200 ms.

### Dictionary

To keep the library small, nlpO3 does not include a dictionary.
For dictionary-based tokenizers you must provide one.

Recommended dictionaries:

- [words_th.txt][dict-pythainlp] from [PyThaiNLP][pythainlp]
  (~62,000 words, CC0-1.0)
- [word break dictionary][dict-libthai] from [libthai][libthai]
  (categories, LGPL-2.1)

[pythainlp]: https://github.com/PyThaiNLP/pythainlp
[libthai]: https://github.com/tlwg/libthai/
[dict-pythainlp]: https://github.com/PyThaiNLP/pythainlp/blob/dev/pythainlp/corpus/words_th.txt
[dict-libthai]: https://github.com/tlwg/libthai/tree/master/data

## Build

### Requirements

- [Rust 2024 Edition](https://www.rust-lang.org/tools/install)
- Python 3.9 or newer
- Python Development Headers
  - Ubuntu: `sudo apt-get install python3-dev`
  - macOS: No action needed
- [PyO3](https://github.com/PyO3/pyo3) — already included in `Cargo.toml`
- [setuptools-rust](https://github.com/PyO3/setuptools-rust)

### Steps

```bash
python -m pip install --upgrade build
python -m build
```

This generates a wheel file in `dist/`, installable with pip:

```bash
pip install dist/nlpo3-2.0.0-cp311-cp311-macosx_12_0_x86_64.whl
```

### Test

```bash
cd tests
python -m unittest
```

## Issues

Please report issues at <https://github.com/PyThaiNLP/nlpo3/issues>

## License

nlpO3 Python binding is copyrighted by its authors
and licensed under terms of the Apache Software License 2.0 (Apache-2.0).
See file [LICENSE](./LICENSE) for details.

## Binary wheels

Pre-built binary packages for CPython, GraalPy, and PyPy are available
on [PyPI][pypi] for the platforms listed below.
Versions with a "t" suffix indicate CPython with free threading.

[pypi]: https://pypi.org/project/nlpo3/

| Python       | OS        | Architecture | Binary wheel  |
| ------------ | --------- | ------------ | ------------- |
| 3.14         | Windows   | x86          | ✓             |
|              |           | AMD64        | ✓             |
|              | macOS     | x86_64       | ✓             |
|              |           | arm64        | ✓             |
|              | manylinux | x86_64       | ✓             |
|              |           | i686         | ✓             |
|              | musllinux | x86_64       | ✓             |
| 3.14t        | Windows   | x86          | ✓             |
|              |           | AMD64        | ✓             |
|              | macOS     | x86_64       | ✓             |
|              |           | arm64        | ✓             |
|              | manylinux | x86_64       | ✓             |
|              |           | i686         | ✓             |
|              | musllinux | x86_64       | ✓             |
| 3.13         | Windows   | x86          | ✓             |
|              |           | AMD64        | ✓             |
|              | macOS     | x86_64       | ✓             |
|              |           | arm64        | ✓             |
|              | manylinux | x86_64       | ✓             |
|              |           | i686         | ✓             |
|              | musllinux | x86_64       | ✓             |
| 3.12         | Windows   | x86          | ✓             |
|              |           | AMD64        | ✓             |
|              | macOS     | x86_64       | ✓             |
|              |           | arm64        | ✓             |
|              | manylinux | x86_64       | ✓             |
|              |           | i686         | ✓             |
|              | musllinux | x86_64       | ✓             |
| 3.11         | Windows   | x86          | ✓             |
|              |           | AMD64        | ✓             |
|              | macOS     | x86_64       | ✓             |
|              |           | arm64        | ✓             |
|              | manylinux | x86_64       | ✓             |
|              |           | i686         | ✓             |
|              | musllinux | x86_64       | ✓             |
| 3.10         | Windows   | x86          | ✓             |
|              |           | AMD64        | ✓             |
|              | macOS     | x86_64       | ✓             |
|              |           | arm64        | ✓             |
|              | manylinux | x86_64       | ✓             |
|              |           | i686         | ✓             |
|              | musllinux | x86_64       | ✓             |
| 3.9          | Windows   | x86          | ✓             |
|              |           | AMD64        | ✓             |
|              | macOS     | x86_64       | ✓             |
|              |           | arm64        | ✓             |
|              | manylinux | x86_64       | ✓             |
|              |           | i686         | ✓             |
|              | musllinux | x86_64       | ✓             |
