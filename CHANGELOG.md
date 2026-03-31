# Changelog

All notable changes to this project are documented in this file.

This file follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
conventions. Version numbers follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `src/tokenizer/dict_backend.rs`: `DictBackend` trait that decouples the
  string representation (`CharString`) from the dictionary backend. Implemented
  by both `TrieChar` and `FstDictionary`.
- `src/char_string.rs`: `CharString` — a native UTF-8 string view with a
  precomputed byte-position index that gives O(1) character access and O(1)
  zero-copy `as_str()` slicing.
- `src/tokenizer/fst_dict.rs`: `FstDictionary` — a memory-efficient dictionary
  backed by `fst::Set` (minimized finite-state automaton). Stores the 62 k-word
  Thai dictionary in ~0.85 MB (≈14 bytes/word) compared with ~43 MB (≈699
  bytes/word) for `TrieChar`.
- `src/tokenizer/newmm.rs`: `NewmmFstTokenizer` — a concrete tokenizer struct
  (CharString + FstDictionary backend) with the same `new()` / `from_word_list()`
  API as `NewmmTokenizer`. Implements `Tokenizer`.
- `benches/tokenizer.rs`: updated Criterion benchmark suite covering all three
  tokenizers (`NewmmTokenizer`, `NewmmFstTokenizer`, `DeepcutTokenizer`).
  `DeepcutTokenizer` benchmarks are feature-gated (`--features deepcut`).
- `BENCHMARK_RESULTS.md`: benchmark measurements for all three tokenizers.

### Changed

- `src/tokenizer/newmm.rs`:
  - `NewmmTokenizer<D: DictBackend = TrieChar>` is generic over the backend.
    The default (no angle brackets) is unchanged: `CharString + TrieChar`.
  - `NewmmFstTokenizer` replaces the old `NewmmTokenizerFst` type alias.
    It is now a standalone concrete struct with clean `new()` / `from_word_list()`
    constructors. All three tokenizers implement `Tokenizer` and are
    interchangeable via `Box<dyn Tokenizer>`.
- `src/tokenizer/tcc/tcc_rules.rs`: TCC regex patterns use `regex::Regex`
  (Unicode mode) on plain UTF-8 strings.
- `src/tokenizer/tcc/tcc_tokenizer.rs`: `tcc_pos()` accepts `&str` directly.
- `src/tokenizer/trie_char.rs`: stores words as `String` (UTF-8), navigates
  trie nodes by `char`. Implements `DictBackend`.
- `src/tokenizer/fst_dict.rs`: implements `DictBackend`; adds `Debug` impl.
- `src/tokenizer/dict_reader.rs`: updated to use `CharString`; added
  `create_dict_fst`.
- `src/tokenizer.rs`: exports `dict_backend` module.
- `Cargo.toml`: removed `bytecount` and `regex-syntax`; added `fst = "0.4.7"`.

### Removed

- `src/four_bytes_str/custom_string.rs` — four-byte string encoding.
- `src/four_bytes_str/custom_regex.rs` — regex-pattern transformation.
- `src/four_bytes_str.rs` — module declaration.
- `benches/tokenizer.rs`: removed `old_impl` module and the four benchmark
  groups that compared old 4-byte encoding vs new `CharString`
  (`string_construction`, `char_access`, `tcc_pos`, `encode_plus_tcc`).

### Performance (from `BENCHMARK_RESULTS.md`)

All three tokenizers use the same `Tokenizer` trait and are interchangeable:

| Tokenizer | short (28 ch) | long (937 ch) | Dict memory |
|-----------|--------------|---------------|-------------|
| `NewmmTokenizer` | **2.63 µs** | **165 µs** | ~43 MB |
| `NewmmFstTokenizer` | 29.5 µs | 2 225 µs | **~0.85 MB** |
| `DeepcutTokenizer` | ONNX-based | ONNX-based | ~3.9 MB model |

[Unreleased]: https://github.com/PyThaiNLP/nlpo3/compare/main...HEAD
