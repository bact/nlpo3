---
SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
SPDX-License-Identifier: Apache-2.0
---

# Benchmark results: three Thai tokenizers compared

This document records the results of running `cargo bench` (and
`cargo bench --features deepcut` for `DeepcutTokenizer`).

Measurements were collected with [Criterion.rs](https://github.com/bheisler/criterion.rs)
on a single-threaded workload. Each timing is the **mean** of 100 samples
(10 for slow groups such as dictionary construction).

## Tokenizers

| Tokenizer | Algorithm | Dictionary | Speed | Dict memory |
|-----------|-----------|------------|-------|-------------|
| `NewmmTokenizer` | Maximal matching + TCC | `TrieChar` | fastest | ~43 MB |
| `NewmmFstTokenizer` | Maximal matching + TCC | `FstDictionary` | moderate | ~0.85 MB |
| `DeepcutTokenizer` | CNN / ONNX | bundled model | slowest | fixed |

All three implement the same [`Tokenizer`] trait:

```rust
pub trait Tokenizer {
    fn segment(&self, text: &str, safe: bool, parallel: bool) -> AnyResult<Vec<String>>;
    fn segment_to_string(&self, text: &str, safe: bool, parallel: bool) -> Vec<String>;
}
```

Switching tokenizers is a single-line change:

```rust
let tok: Box<dyn Tokenizer> = Box::new(NewmmTokenizer::new("words_th.txt"));
// or:
let tok: Box<dyn Tokenizer> = Box::new(NewmmFstTokenizer::new("words_th.txt"));
// or (requires deepcut feature):
let tok: Box<dyn Tokenizer> = Box::new(DeepcutTokenizer::new()?);
```

## Benchmark environment

- Rust: stable (release profile, `lto = true`, `codegen-units = 1`)
- Dictionary: `words_th.txt`, 62 018 words
- Text sizes:
  - **short** – 28 Unicode characters, Thai-only
  - **medium** – 219 characters, mixed Thai / Latin / CJK / digits
  - **long** – 937 characters, mixed

---

## 1. Dictionary construction

| Implementation | Time for 62 018 words |
|---|---:|
| `TrieChar::new` | 65.3 ms |
| `FstDictionary::from_words` | 62.2 ms |

Construction time is similar. FST construction sorts the input and builds a
minimized automaton; `TrieChar` inserts into a `HashMap`-based trie.

---

## 2. Dictionary prefix lookup

Finding all dictionary entries that are prefixes of a query string.

| Implementation | short_thai (5 chars) | mixed (7 chars) | medium_thai (14 chars) |
|---|---:|---:|---:|
| `TrieChar::prefix_ref` | **62 ns** | **72 ns** | **86 ns** |
| `FstDictionary::prefix_lengths` | 3 741 ns | 2 041 ns | 4 393 ns |
| Ratio | **60× faster** | **28× faster** | **51× faster** |

`TrieChar` wins on lookup speed because it navigates a `HashMap` per character
(O(k) pointer chasing, cache-friendly for short words). `FstDictionary`
runs a streaming FST search with higher per-call overhead.

**Recommendation:** use `TrieChar` (`NewmmTokenizer`) for the hot tokenization
path; use `FstDictionary` (`NewmmFstTokenizer`) when memory is constrained.

---

## 3. Full end-to-end tokenization

| Tokenizer | short (28 ch) | medium (219 ch) | long (937 ch) |
|---|---:|---:|---:|
| `NewmmTokenizer` (safe=false) | **2.63 µs** | **33.1 µs** | **165 µs** |
| `NewmmTokenizer` (safe=true) | 2.66 µs | 35.0 µs | 207 µs |
| `NewmmFstTokenizer` (safe=false) | 29.5 µs | 284 µs | 2 225 µs |
| `NewmmFstTokenizer` (safe=true) | 29.1 µs | 250 µs | 1 553 µs |
| Speed ratio (Trie vs Fst, safe=false) | **11× faster** | **9× faster** | **13× faster** |

`DeepcutTokenizer` results require the `deepcut` Cargo feature (`--features deepcut`).
CNN/ONNX inference is significantly slower than dictionary-based methods and
scales with input length at a different rate.

**Conclusions:**

- `NewmmTokenizer` is the fastest dictionary-based tokenizer. The `TrieChar`
  prefix-lookup loop is ~10–60× faster per query than `FstDictionary`.
- `NewmmFstTokenizer` is the memory-efficient alternative: 49× smaller
  dictionary with 9–13× lower throughput.
- The `Tokenizer` trait makes switching between all three tokenizers trivial.

---

## 4. Memory footprint

### String representation

| Representation | Heap bytes per character |
|---|---:|
| `CharString` (UTF-8 source + `u32` position table) | **6.3 bytes/char** |

Thai characters are 3-byte UTF-8 sequences (3 bytes) plus one `u32` position
entry (4 bytes) = ~7 bytes/char for Thai, less for ASCII.

### Dictionary storage

| Structure | Total bytes | Per word |
|---|---:|---:|
| `FstDictionary` (FST automaton) | 891 464 bytes (~0.85 MB) | **14.4 bytes** |
| `TrieChar` (trie with HashMap nodes, estimated) | ~43 MB | **~699 bytes** |
| Reduction | | **~49× smaller** |

The FST stores the full 62 018-word Thai dictionary in under 1 MB. The
`TrieChar` trie stores roughly 80 bytes per character edge in `HashMap`
entries. For memory-constrained deployments, `FstDictionary` is the preferred
choice.

### DeepcutTokenizer model

The bundled deepcut ONNX model (`model/deepcut.onnx`) is approximately
**3.9 MB** compiled into the binary. There is no runtime dictionary.

---

## Summary

| Tokenizer | Speed (short/long) | Dict memory | Use when |
|-----------|-------------------|-------------|----------|
| `NewmmTokenizer` | **2.6 µs / 165 µs** | ~43 MB | Maximum throughput |
| `NewmmFstTokenizer` | 29.5 µs / 2 225 µs | **~0.85 MB** | Memory-constrained |
| `DeepcutTokenizer` | slower (ONNX) | ~3.9 MB model | No dictionary available |

All three implement `Tokenizer` and are interchangeable. Use `Box<dyn Tokenizer>`
to select the tokenizer at runtime.
