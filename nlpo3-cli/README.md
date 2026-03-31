---
SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
SPDX-License-Identifier: Apache-2.0
---

# nlpo3-cli

[![crates.io](https://img.shields.io/crates/v/nlpo3-cli.svg "crates.io")](https://crates.io/crates/nlpo3-cli/)
[![Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg "Apache-2.0")](https://opensource.org/licenses/Apache-2.0)

Command line interface for nlpO3, a Thai natural language processing library.

## Install

```bash
cargo install nlpo3-cli
```

## Usage

```bash
nlpo3 --help
nlpo3 segment --help
```

## Example

Tokenize from standard input using the built-in dictionary:

```bash
echo "ฉันกินข้าว" | nlpo3 segment
```

Tokenize using a custom dictionary file:

```bash
echo "ฉันกินข้าว" | nlpo3 segment --dict-path /path/to/dict.txt
```

Use a custom word delimiter:

```bash
echo "ฉันกินข้าว" | nlpo3 segment --word-delimiter " "
```

Enable safe mode (avoids long run times on ambiguous input):

```bash
echo "ฉันกินข้าว" | nlpo3 segment --safe
```

## License

nlpo3-cli is copyrighted by its authors
and licensed under terms of the Apache Software License 2.0 (Apache-2.0).
See file [LICENSE](./LICENSE) for details.
