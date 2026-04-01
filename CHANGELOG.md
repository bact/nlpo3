# Changelog

All notable changes to this project are documented in this file.

This file follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
conventions. Version numbers follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [2.0.0] - 2026-04-01

### Added

- `src/char_string.rs`: `CharString` — a native UTF-8 string view with a
  precomputed byte-position index that gives O(1) character access and O(1)
  zero-copy `as_str()` slicing.
- `src/tokenizer/dict_backend.rs`: `DictBackend` trait that decouples the
  string representation (`CharString`) from the dictionary backend. Implemented
  by both `TrieChar` and `FstDict`.
- `src/tokenizer/fst_dict.rs`: `FstDict` — a memory-efficient dictionary
  backed by `fst::Set` (minimized finite-state automaton). Stores the 62 k-word
  Thai dictionary in ~0.85 MB (≈14 bytes/word) compared with ~43 MB
  (≈699 bytes/word) for `TrieChar`.
- `src/tokenizer/newmm.rs`: `NewmmFstTokenizer` — a concrete tokenizer struct
  (CharString + FstDict backend) with the same API as `NewmmTokenizer`.
- `src/tokenizer/deepcut.rs`: `DeepcutTokenizer` — CNN-based Thai word tokenizer
  using ONNX inference via `tract-onnx`. Based on
  [LEKCut](https://github.com/PyThaiNLP/LEKCut) and
  [deepcut](https://github.com/rkcosmos/deepcut).
  Enabled by the `deepcut` Cargo feature.
- Python binding: `DeepcutTokenizer` class and `segment_deepcut()` function.
- `BENCHMARK_RESULTS.md`: benchmark measurements for all three tokenizers.

### Changed

- **Breaking**: all `Cargo.toml` editions updated from 2018 to 2024;
  `rust-version` set to `"1.88.0"` across all packages.
- **Breaking**: version bumped to 2.0.0 across all packages (nlpo3, nlpo3-cli,
  nlpo3-python, nlpo3-nodejs).
- `src/tokenizer/fst_dict.rs`: restored `unsafe { from_utf8_unchecked(key) }`
  with a `// SAFETY:` comment in `prefix_lengths` hot loop. The FST is built
  exclusively from valid UTF-8 strings in `from_words`, so all keys are
  guaranteed valid UTF-8; the safe `from_utf8().expect()` was re-validating
  the same bytes on every call in a hot path.
- `nlpo3-cli/src/main.rs`: `main()` now returns `Result<(), Box<dyn Error>>`
  and propagates stdin I/O errors with `?` instead of `panic!`. On error Rust
  prints the message to stderr and exits with code 1, which is script-friendly.

### Removed

- Custom four-byte string implementation removed.
  Use `CharString` instead.

### Performance (from `BENCHMARK_RESULTS.md`)

All three tokenizers use the same `Tokenizer` trait and are interchangeable:

| Tokenizer | short (28 ch) | long (937 ch) | Dict memory |
| --------- | ------------- | ------------- | ----------- |
| `NewmmTokenizer` | **2.63 µs** | **165 µs** | ~43 MB |
| `NewmmFstTokenizer` | 29.5 µs | 2 225 µs | **~0.85 MB** |
| `DeepcutTokenizer` | - | - | ~3.9 MB model |

## [1.4.0] - 2024-11-09

### Changed

- Improve karan handling (#61)

## [1.1.2] - 2021-07-21

### Changed

- Python binding published on PyPI under new name as `nlpo3`.

## [0.2.0-beta] - 2021-05-09

### Added

- Newmm word segmentation.
- Custom dictionary support.
- Python binding published on PyPI as `pythainlp-rust-modules`.

[Unreleased]: https://github.com/PyThaiNLP/nlpo3/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/PyThaiNLP/nlpo3/compare/v1.4.0...v2.0.0
[1.4.0]: https://github.com/PyThaiNLP/nlpo3/releases/tag/v1.4.0
[1.1.2]: https://github.com/PyThaiNLP/nlpo3/releases/tag/v1.1.2
[0.2.0-beta]: https://github.com/PyThaiNLP/nlpo3/releases/tag/v0.2.0-beta
