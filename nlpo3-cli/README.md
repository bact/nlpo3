---
SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
SPDX-License-Identifier: Apache-2.0
---

# nlpo3-cli

[![crates.io](https://img.shields.io/crates/v/nlpo3-cli.svg "crates.io")](https://crates.io/crates/nlpo3-cli/)
[![Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg "Apache-2.0")](https://opensource.org/licenses/Apache-2.0)

Command line interface for nlpO3, a Thai natural language processing library.
Originally developed by Vee Satayamas.

## Install

```bash
cargo install nlpo3-cli
```

## Usage

```bash
nlpo3 --help
nlpo3 segment --help
```

## Tokenizers

| Tokenizer | Flag | Description |
|-----------|------|-------------|
| `newmm` | *(default)* | Dictionary-based maximal-matching (TrieChar backend, fastest) |
| `nf` | `--tokenizer nf` | Same algorithm with FST backend (~49× less memory) |
| `deepcut` | `--tokenizer deepcut` | Neural CNN model (no dictionary needed) |

## Examples

Tokenize from standard input using the built-in dictionary:

```bash
echo "ฉันกินข้าว" | nlpo3 segment
```

Select a tokenizer:

```bash
# FST backend (less memory)
echo "ฉันกินข้าว" | nlpo3 segment --tokenizer nf

# Short form
echo "ฉันกินข้าว" | nlpo3 segment -t nf

# Neural CNN tokenizer (no dictionary needed)
echo "ฉันกินข้าว" | nlpo3 segment --tokenizer deepcut
```

Use a custom dictionary file:

```bash
echo "ฉันกินข้าว" | nlpo3 segment --dict-path /path/to/dict.txt
echo "ฉันกินข้าว" | nlpo3 segment -t nf --dict-path /path/to/dict.txt
```

Use a custom word delimiter:

```bash
echo "ฉันกินข้าว" | nlpo3 segment --word-delimiter " "
```

Enable safe mode (avoids long run times on ambiguous input):

```bash
echo "ฉันกินข้าว" | nlpo3 segment --safe
```

Enable parallel processing:

```bash
echo "ฉันกินข้าว" | nlpo3 segment --parallel
```

## License

nlpo3-cli is copyrighted by its authors
and licensed under terms of the Apache Software License 2.0 (Apache-2.0).
See file [LICENSE](./LICENSE) for details.
