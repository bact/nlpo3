---
SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
SPDX-License-Identifier: Apache-2.0
---

# nlpO3

[![crates.io](https://img.shields.io/crates/v/nlpo3.svg "crates.io")](https://crates.io/crates/nlpo3/)
[![Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg "Apache-2.0")](https://opensource.org/license/apache-2-0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14082448.svg)](https://doi.org/10.5281/zenodo.14082448)

A Thai natural language processing library in Rust, with Python and Node.js
bindings.

Formerly known as `oxidized-thainlp`, it was originally developed by
Thanathip Suntorntip.

## Overview

- Three tokenizer choices with a consistent API:
  - `NewmmTokenizer` (dictionary, fastest)
  - `NewmmFstTokenizer` (dictionary, lower memory)
  - `DeepcutTokenizer` (neural model)
- Thread-safe tokenizer instances for concurrent use.
- Improved handling for ambiguous and large input.

## Install

Rust:

```shell
cargo add nlpo3
```

Python:

```shell
pip install nlpo3
```

Node.js:

```shell
npm install nlpo3-nodejs
```

CLI:

```shell
cargo install nlpo3-cli
```

## Requirements

- Rust Edition 2024
- rustc 1.88.0 or newer

## Quick start

Dictionary tokenizer:

```rust
use nlpo3::tokenizer::newmm::NewmmTokenizer;

fn main() {
  let tok = NewmmTokenizer::new("words_th.txt").expect("dictionary load failed");
  let tokens = tok.segment("สวัสดีครับ").expect("segmentation failed");
  println!("{:?}", tokens);
}
```

Neural tokenizer:

```rust
use nlpo3::tokenizer::deepcut::DeepcutTokenizer;

fn main() {
  let tok = DeepcutTokenizer::new().expect("model load failed");
  let tokens = tok.segment("สวัสดีครับ").expect("segmentation failed");
  println!("{:?}", tokens);
}
```

## Segment options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `safe` | `bool` | `false` | For `NewmmTokenizer` and `NewmmFstTokenizer`: avoid long run times on highly ambiguous input |
| `parallel_chunk_size` | `Option<usize>` | `None` | Enable chunked parallel processing for larger text; `None`, `0`, or too-small values disable parallel mode |

Auto-parallel helpers are available via `segment_parallel(...)` on tokenizers.

For technical implementation and design notes, see
[docs/impl-notes.md](./docs/impl-notes.md).

## Bindings

- Python: [nlpo3-python](./nlpo3-python/)
- Node.js: [nlpo3-nodejs](./nlpo3-nodejs/)
- CLI: [nlpo3-cli](./nlpo3-cli/)

## Dictionary

`nlpO3` does not bundle a dictionary for dictionary-based tokenizers.

Recommended sources:

- [words_th.txt][dict-pythainlp] from [PyThaiNLP][pythainlp]
- [word break dictionary][dict-libthai] from [libthai][libthai]

[tcc]: https://dl.acm.org/doi/10.1145/355214.355225
[pythainlp]: https://github.com/PyThaiNLP/pythainlp
[libthai]: https://github.com/tlwg/libthai/
[dict-pythainlp]: https://github.com/PyThaiNLP/pythainlp/blob/dev/pythainlp/corpus/words_th.txt
[dict-libthai]: https://github.com/tlwg/libthai/tree/master/data

## Support

- Issues: <https://github.com/PyThaiNLP/nlpo3/issues>

## License

nlpO3 is copyrighted by its authors
and licensed under terms of the Apache Software License 2.0 (Apache-2.0).
See file [LICENSE](./LICENSE) for details.
