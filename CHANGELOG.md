# Changelog

All notable changes to this project are documented in this file.

This file follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
conventions. Version numbers follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [2.0.0] - 2026-03-31

### Added

- `src/tokenizer/dict_backend.rs`: `DictBackend` trait that decouples the
  string representation (`CharString`) from the dictionary backend. Implemented
  by both `TrieChar` and `FstDictionary`.
- `src/char_string.rs`: `CharString` — a native UTF-8 string view with a
  precomputed byte-position index that gives O(1) character access and O(1)
  zero-copy `as_str()` slicing.
- `src/tokenizer/fst_dict.rs`: `FstDictionary` — a memory-efficient dictionary
  backed by `fst::Set` (minimized finite-state automaton). Stores the 62 k-word
  Thai dictionary in ~0.85 MB (≈14 bytes/word) compared with ~43 MB
  (≈699 bytes/word) for `TrieChar`.
- `src/tokenizer/newmm.rs`: `NewmmFstTokenizer` — a concrete tokenizer struct
  (CharString + FstDictionary backend) with the same `new()` / `from_word_list()`
  API as `NewmmTokenizer`. Implements `Tokenizer`.
- `src/tokenizer/deepcut.rs`: `DeepcutTokenizer` — CNN-based Thai word tokenizer
  using ONNX inference via `tract-onnx`. Based on
  [LEKCut](https://github.com/PyThaiNLP/LEKCut) and
  [deepcut](https://github.com/rkcosmos/deepcut).
  Enabled by the `deepcut` Cargo feature.
- Python binding: `DeepcutTokenizer` class and `segment_deepcut()` function,
  backed by a process-level `OnceLock` singleton for the default model.
- `benches/tokenizer.rs`: updated Criterion benchmark suite covering all three
  tokenizers (`NewmmTokenizer`, `NewmmFstTokenizer`, `DeepcutTokenizer`).
  `DeepcutTokenizer` benchmarks are feature-gated (`--features deepcut`).
- `BENCHMARK_RESULTS.md`: benchmark measurements for all three tokenizers.

### Changed

- **Breaking**: all `Cargo.toml` editions updated from 2018 to 2024;
  `rust-version` set to `"1.88.0"` across all packages.
- **Breaking**: version bumped to 2.0.0 across all packages (nlpo3, nlpo3-cli,
  nlpo3-python, nlpo3-nodejs).
- `src/tokenizer/newmm.rs`:
  - `NewmmTokenizer<D: DictBackend = TrieChar>` is now generic over the backend.
    The default (no angle brackets) is unchanged: `CharString + TrieChar`.
  - `NewmmFstTokenizer` replaces the old `NewmmTokenizerFst` type alias.
    It is now a standalone concrete struct with clean `new()` / `from_word_list()`
    constructors. All three tokenizers implement `Tokenizer` and are
    interchangeable via `Box<dyn Tokenizer>`.
  - `one_cut`: replaced `while match { … true/false }` with a Rust 2024
    `while let … && condition` let chain, eliminating the unreachable arm.
  - `one_cut`: collapsed duplicated `match get_mut` / `insert` and
    `if let … else` graph-update blocks into `entry().or_default()` and
    `entry().or_insert_with()`.
  - `one_cut` / `bfs_paths_graph`: replaced explicit `*deref` operators with
    Rust 2024 reference patterns — `for &position in idk`,
    `if let Some(&first) = peek()`, `for (idx, &token)`, `for &token`.
- `src/tokenizer/tcc/tcc_rules.rs`: TCC regex patterns use `regex::Regex`
  (Unicode mode) on plain UTF-8 strings.
- `src/tokenizer/tcc/tcc_tokenizer.rs`: `tcc_pos()` accepts `&str` directly.
- `src/tokenizer/trie_char.rs`:
  - Stores words as `String` (UTF-8), navigates trie nodes by `char`.
    Implements `DictBackend`.
  - `find_child`, `find_mut_child`, `remove_child`: now take `char` by value
    (a `Copy` type) instead of `&char`; all callers updated.
  - `prefix_ref`: replaced double-nested `match current_node { … match
    find_child { … } }` with a Rust 2024 let chain
    (`if let Some(node) = … && let Some(child) = …`).
  - `TrieChar::new`: `for word in words` (slice `IntoIterator`) instead of
    `words.iter()`.
- `src/tokenizer/fst_dict.rs`:
  - Implements `DictBackend`; adds `Debug` impl.
  - Replaced `unsafe { std::str::from_utf8_unchecked(key) }` with safe
    `std::str::from_utf8(key).expect(…)`.
- `src/tokenizer/deepcut.rs`:
  - `build_chars_map`: `|(s, idx)|` closure + `*idx` deref replaced by
    `|&(s, idx)|` reference pattern.
  - `build_char_type_map`: `for (group, type_idx)` + `*type_idx` deref
    replaced by `for &(group, type_idx)` reference pattern.
- `src/char_string.rs`:
  - `rfind_space_char_index`: replaced manual `for` loop with an iterator
    chain using a Rust 2024 reference pattern —
    `.filter(|&(_, ch)| ch == ' ').last().map(|(i, _)| i)`.
- `src/tokenizer/dict_reader.rs`: updated to use `CharString`; added
  `create_dict_fst`.
- `src/tokenizer.rs`: exports `dict_backend` module.
- `Cargo.toml`: removed `bytecount` and `regex-syntax`; added `fst = "0.4.7"`.
- `nlpo3-cli`: updated `clap` from 3.0.0-beta.2 to 4.x; migrated from
  `clap::Clap` derive macro to `clap::Parser`; improved help text and error
  messages.
- `nlpo3-nodejs/src/lib.rs`: fixed `if let Some(_)` → `contains_key()`;
  updated SPDX headers; added `Arthit Suriyawongkul` as contributor in
  `Cargo.toml`; fixed typo in `package.json` contributor name.
- `nlpo3-python/src/lib.rs`: updated SPDX headers and author attribution.
- `CITATION.cff` and `nlpo3-python/CITATION.cff`: updated version and
  release date.

### Removed

- **Breaking**: `src/four_bytes_str/custom_string.rs` — four-byte string
  encoding removed. Use `CharString` instead.
- **Breaking**: `src/four_bytes_str/custom_regex.rs` — regex-pattern
  transformation removed.
- **Breaking**: `src/four_bytes_str.rs` — module declaration removed.
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

[Unreleased]: https://github.com/PyThaiNLP/nlpo3/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/PyThaiNLP/nlpo3/compare/v1.4.0...v2.0.0
