---
SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
SPDX-License-Identifier: Apache-2.0
---

# Benchmark results: old vs new implementation, and dictionary backend comparison

This document records the results of running `cargo bench` after:
1. Replacing the custom four-byte string encoding (`four_bytes_str`) with
   native Rust types (`CharString`, `regex::Regex`).
2. Adding `FstDictionary` as an alternative dictionary backend alongside the
   existing `TrieChar`.
3. Confirming that string representation and dictionary backend are
   **fully orthogonal** through the `DictBackend` trait, and benchmarking all
   relevant combinations.

Measurements were collected with [Criterion.rs](https://github.com/bheisler/criterion.rs)
on a single-threaded workload. Each timing is the **mean** of 100 samples
(10 for slow groups such as dictionary construction).

## Benchmark environment

- Rust: stable (release profile, `lto = true`, `codegen-units = 1`)
- Dictionary: `words_th.txt`, 62 018 words
- Text sizes:
  - **short** – 28 Unicode characters, Thai-only
  - **medium** – 219 characters, mixed Thai / Latin / CJK / digits
  - **long** – 937 characters, mixed

---

## Design axes: are they orthogonal?

The tokenizer has two independent design choices:

| Axis | Option A | Option B |
|------|----------|----------|
| String representation | Old: 4-byte-per-char custom encoding | New: `CharString` (native UTF-8) |
| Dictionary backend | `TrieChar` (fast lookup, ~43 MB) | `FstDictionary` (compact, ~0.85 MB) |

**Yes, they are orthogonal.** The `DictBackend` trait (`src/tokenizer/dict_backend.rs`)
decouples the two:

- `TrieChar` implements `DictBackend` via `prefix_lengths_of(prefix: &CharString)`
- `FstDictionary` implements `DictBackend` via `self.prefix_lengths(prefix.as_str())`
- `NewmmTokenizer<D: DictBackend>` accepts either

The default `NewmmTokenizer` uses `CharString + TrieChar` (fastest). The
`NewmmTokenizerFst` type alias uses `CharString + FstDictionary` (compact).

---

## 1. String construction

Converting a raw UTF-8 string into the representation used by the tokenizer.

| Implementation | 28 chars | 219 chars | 937 chars |
|---|---:|---:|---:|
| **Old** — `to_four_bytes` + `Vec<char>` | 228 ns | 1 045 ns | 4 739 ns |
| **New** — `CharString::new` | **81 ns** | **308 ns** | **1 152 ns** |
| Speed-up | **2.8×** | **3.4×** | **4.1×** |

The old approach made two passes over the string (one to build the four-byte
buffer, one to collect a `Vec<char>`). The new `CharString` makes a single
pass to build the byte-position table from `str::char_indices()`.

---

## 2. Character access (single character, by index)

| Implementation | Middle of 219-char string |
|---|---:|
| Baseline — `str.chars().nth(n)` (O(n)) | 58 ns |
| **Old** — `char_vec[i]` (O(1), `Vec<char>`) | **0.86 ns** |
| **New** — `CharString::get_char_at(i)` (O(1), position table) | **2.74 ns** |

The old `Vec<char>` cache had slightly lower raw access latency because it is
a direct pointer offset. The new position-table lookup requires one additional
array dereference to compute the byte offset, then a short UTF-8 decode
(≤ 4 bytes). Both are constant-time and far faster than the O(n) baseline.

Sequential scan of all 219 characters:

| Implementation | All-chars scan |
|---|---:|
| **Old** — `Vec<char>` sequential | 90 ns |
| **New** — `CharString` sequential | 615 ns |

The sequential scan of the new implementation is slower because it decodes
UTF-8 for each character. If only sequential iteration is needed, calling
`char_string.as_str().chars()` (native UTF-8 iteration) is faster and should
be preferred over repeated `get_char_at` calls.

---

## 3. TCC boundary detection (`tcc_pos`)

This is the inner hot loop of the tokenizer, called once per segment.
Encoding cost is **excluded** (measured separately in section 4).

| Implementation | short (28 chars) | medium (219 chars) | long (937 chars) |
|---|---:|---:|---:|
| **Old** — `bytes::Regex` on 4-byte text | 705 ns | 6 217 ns | 27 455 ns |
| **New** — `Regex` on UTF-8 | 733 ns | 6 654 ns | 28 381 ns |
| Ratio | 0.96× | 0.93× | 0.97× |

The two approaches have virtually identical TCC throughput. The Unicode
`regex::Regex` engine is on-par with the `bytes::Regex` engine for this
workload. Simplicity was gained without any speed regression.

---

## 4. Encode + TCC pipeline (encoding included)

This reflects the real cost per `segment()` call in the old code: encode text
to four bytes, then run TCC. In the new code: create `CharString`, then run TCC.

| Implementation | short | medium | long |
|---|---:|---:|---:|
| **Old** — `to_four_bytes` + `tcc_pos` | 841 ns | 7 019 ns | 31 193 ns |
| **New** — `CharString::new` + `tcc_pos` | **831 ns** | **7 030 ns** | **29 785 ns** |
| Speed-up | **1.01×** | **1.0×** | **1.05×** |

End-to-end performance is equivalent or marginally faster for the new code,
because the construction saving (section 1) fully offsets the slightly slower
sequential character access.

---

## 5. Dictionary construction

| Implementation | Time for 62 018 words |
|---|---:|
| `TrieChar::new` | 65.3 ms |
| `FstDictionary::from_words` | 62.2 ms |

Construction time is similar. FST construction sorts the input and builds a
minimized automaton; `TrieChar` inserts into a `HashMap`-based trie.

---

## 6. Dictionary prefix lookup

Finding all dictionary entries that are prefixes of a query string.

| Implementation | short_thai (5 chars) | mixed (7 chars) | medium_thai (14 chars) |
|---|---:|---:|---:|
| `TrieChar::prefix_ref` | **62 ns** | **72 ns** | **86 ns** |
| `FstDictionary::prefix_lengths` | 3 741 ns | 2 041 ns | 4 393 ns |
| Ratio | **60× faster** | **28× faster** | **51× faster** |

`TrieChar` wins on lookup speed because it navigates a `HashMap` per character
(O(k) pointer chasing, but very cache-friendly for short words). `FstDictionary`
runs a streaming FST search which has higher per-call overhead.

**Recommendation:** use `TrieChar` for the hot tokenization path (where prefix
lookup is called thousands of times per segment); use `FstDictionary` when
memory is constrained and lookups are infrequent.

---

## 7. Full end-to-end tokenization: `CharString + TrieChar` vs `CharString + FstDictionary`

This benchmark directly answers the question "can we use native string with
TrieChar to get maximum speed?" by measuring both combinations.

Both backends use the same `CharString` string representation. The difference
is solely due to the dictionary prefix-lookup hot path.

| Backend | short (28 chars) | medium (219 chars) | long (937 chars) |
|---|---:|---:|---:|
| `CharString + TrieChar` (safe=false) | **2.40 µs** | **34.2 µs** | **140 µs** |
| `CharString + TrieChar` (safe=true) | **2.61 µs** | **36.0 µs** | **186 µs** |
| `CharString + FstDictionary` (safe=false) | 25.7 µs | 244 µs | 1 843 µs |
| `CharString + FstDictionary` (safe=true) | 25.6 µs | 218 µs | 1 411 µs |
| Speed ratio (TrieChar vs FstDict, safe=false) | **11× faster** | **7× faster** | **13× faster** |

**Conclusion:** `CharString + TrieChar` is the fastest combination for
end-to-end tokenization. The `FstDictionary` backend is 7–13× slower due to
the high per-lookup overhead of the streaming FST automaton compared with the
O(k) pointer-chasing trie.

Use `NewmmTokenizer` (default, `CharString + TrieChar`) for maximum speed.
Use `NewmmTokenizerFst` (`CharString + FstDictionary`) when the 50-fold memory
saving is more important than throughput.

---

## 8. Memory footprint

### Per-character string storage

| Representation | Heap bytes per character |
|---|---:|
| **Old** — 4-byte buffer + `Vec<char>` | **8.0 bytes/char** |
| **New** — UTF-8 source + `u32` position table | **6.3 bytes/char** |
| Saving | **~21%** |

Thai characters are 3-byte UTF-8 sequences; storing them as UTF-8 (3 bytes)
plus one `u32` position entry (4 bytes) = 7 bytes/char is less than the old
4-byte fixed buffer (4 bytes) plus `Vec<char>` entry (4 bytes) = 8 bytes/char.
The per-character overhead measurement above reflects a real mixed text string.

### Dictionary storage

| Structure | Total bytes | Per word |
|---|---:|---:|
| `FstDictionary` (FST automaton) | 891 464 bytes (0.85 MB) | **14.4 bytes** |
| `TrieChar` (trie with HashMap nodes, estimated) | ~43 MB | **~699 bytes** |
| Reduction | | **~49× smaller** |

The FST stores the full 62 018-word Thai dictionary in under 1 MB. The
`TrieChar` trie stores roughly 80 bytes per character edge in `HashMap`
entries. For memory-constrained deployments (embedded, mobile), `FstDictionary`
is the strongly preferred choice.

---

## Summary

### Old vs new

| Metric | Old | New | Change |
|---|---|---|---|
| String construction | 228 – 4 739 ns | 81 – 1 152 ns | **2.8 – 4.1× faster** |
| Single-char access (random) | 0.86 ns | 2.74 ns | 3× slower (still O(1)) |
| TCC boundary detection | 705 – 27 455 ns | 733 – 28 381 ns | ≈ same |
| Encode + TCC total | 841 – 31 193 ns | 831 – 29 785 ns | ≈ same |
| Per-char heap | 8.0 bytes/char | 6.3 bytes/char | **21% smaller** |
| Code removed | – | 580 lines | Custom codec gone |

### Dictionary backend comparison (both use `CharString`)

| Backend | Tokenization speed | Dictionary size | Suitable for |
|---|---|---|---|
| `TrieChar` (default) | **2.4 – 140 µs** per call | ~43 MB | Maximum throughput |
| `FstDictionary` | 25 – 1 843 µs per call | **0.85 MB** | Memory-constrained deployments |
| Ratio | **TrieChar is 7 – 13× faster** | **FstDict is 49× smaller** | — |

The two design dimensions are independent: `CharString` is the string
representation and `DictBackend` is the dictionary interface. Use
`NewmmTokenizer` (`CharString + TrieChar`) for maximum tokenization speed.
Use `NewmmTokenizerFst` (`CharString + FstDictionary`) when memory matters more
than throughput.
