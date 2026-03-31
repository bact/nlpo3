---
SPDX-FileContributor: Thanathip Suntorntip Gorlph
SPDX-FileContributor: PyThaiNLP Project
SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
SPDX-License-Identifier: Apache-2.0
---

# String Representation in the Tokenizer

## Previous design: custom four-byte encoding

Rust `String` (and `&str`) is a slice of valid UTF-8 bytes, which are
variable-length. Accessing the *n*-th character by index takes O(n) time,
making any algorithm that relies on character positions very slow.

The original implementation by Thanathip Suntorntip worked around this by
converting every character to a fixed 4-byte slot padded with leading zeros
(`fixed_bytes_str`):

- 1-byte ASCII: `[0, 0, 0, byte]`
- 2-byte UTF-8: `[0, 0, b1, b2]`
- 3-byte UTF-8 (Thai): `[0, b1, b2, b3]`
- 4-byte UTF-8: `[b1, b2, b3, b4]`

This gave O(1) character indexing (`bytes[i*4 .. (i+1)*4]`) and direct
`regex::bytes::Regex` matching, but required:

- A complex 4-byte encoding/decoding layer.
- A separate `custom_regex.rs` module that rewrote human-readable Unicode
  regex patterns into 4-byte padded patterns.
- Approximately 8 bytes per character (4-byte buffer + `Vec<char>` cache).

## Current design: native UTF-8 with byte-position index

The `CharString` type in `src/char_string.rs` stores:

- `Arc<String>` — the original UTF-8 source (shared across substrings).
- `Arc<Vec<u32>>` — a byte-position table: `byte_positions[i]` is the byte
  offset of character `i` in the source string.
- `start` and `end` — character indices defining the current view.

Key properties:

| Operation | Complexity |
|---|---|
| Character access by index | O(1) — one table lookup + ≤4 bytes decode |
| Character count | O(1) — `end - start` |
| Substring (view) | O(1) — copy Arc pointers + new start/end |
| `as_str()` for regex matching | O(1) — one table lookup, no allocation |

Memory footprint: ~3 bytes/char (UTF-8 Thai) + 4 bytes/char (u32 position)
= ~7 bytes/char, down from ~8 bytes/char in the old scheme. The real gain is
in code simplicity and the elimination of the encoding layer.

## Regex: native Unicode

The TCC rules (`tcc_rules.rs`) and non-Thai patterns (`newmm.rs`) are now
compiled with `regex::Regex` (Unicode mode) using plain UTF-8 patterns such
as `r"^[ก-ฮ]"`. The `custom_regex.rs` module and its `\x00` padding approach
have been removed.

> **Note on verbose mode (`(?x)`):** In Rust's `regex` crate, the `(?x)` flag
> ignores whitespace *including inside character classes*. For example,
> `(?x)^[ \t]+` would drop the literal space; use `(?x)^[\ \t]+` instead, or
> omit `(?x)` when the pattern contains meaningful whitespace.

## Dictionary: FST-backed (optional)

The `FstDictionary` type in `src/tokenizer/fst_dict.rs` provides a
memory-efficient alternative to `TrieChar` for storing the word list. It uses
the `fst` crate (finite state transducers) which stores sorted string sets in
a minimized DFA, achieving a few bytes per entry versus tens of bytes per
trie node.

`FstDictionary` supports:
- O(k) prefix lookup via `prefix_lengths(text)` (where k = text length).
- Dynamic add/remove operations via small delta `HashSet`s.

## References

- [Rust String indexing](https://doc.rust-lang.org/book/ch08-02-strings.html#indexing-into-strings)
- [UTF-8 on Wikipedia](https://en.wikipedia.org/wiki/UTF-8)
- [`fst` crate documentation](https://docs.rs/fst/)
