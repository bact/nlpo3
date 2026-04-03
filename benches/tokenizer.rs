// SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
// SPDX-License-Identifier: Apache-2.0

//! Performance and memory benchmarks for all three Thai tokenizers.
//!
//! Tokenizers covered:
//!
//! | Tokenizer | Algorithm | Dictionary | Speed | Memory |
//! |-----------|-----------|------------|-------|--------|
//! | [`NewmmTokenizer`] | Maximal matching + TCC | `TrieChar` | fastest | ~43 MB |
//! | [`NewmmFstTokenizer`] | Maximal matching + TCC | `FstDict` | moderate | ~0.85 MB |
//! | [`DeepcutTokenizer`] | CNN (ONNX) | bundled model | slowest | fixed |
//!
//! All three implement [`Tokenizer`] and are interchangeable at call sites that
//! accept `&dyn Tokenizer`.
//!
//! # Benchmark groups
//!
//! | Group | What is measured |
//! |-------|-----------------|
//! | `dict_construction` | Build time for `TrieChar` vs `FstDict` |
//! | `prefix_lookup` | Per-query prefix scan: `TrieChar` vs `FstDict` |
//! | `full_tokenization` | End-to-end segment() for all three tokenizers |
//! | `memory_footprint` | Heap-size estimates printed to stderr |
//!
//! Run with:
//! ```sh
//! cargo bench
//! # all three tokenizers:
//! cargo bench --features deepcut
//! # specific group only:
//! cargo bench -- full_tokenization
//! ```
//! HTML reports land in `target/criterion/`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use nlpo3::{
    char_string::CharString,
    tokenizer::{
        fst_dict::FstDict,
        newmm::{NewmmFstTokenizer, NewmmTokenizer},
        tokenizer_trait::Tokenizer,
        trie_char::TrieChar,
    },
};
use std::hint::black_box;

// ---------------------------------------------------------------------------
// Shared test fixtures
// ---------------------------------------------------------------------------

const SHORT_TEXT: &str = "พิสูจน์ได้ค่ะสวัสดีประเทศไทย";

const MEDIUM_TEXT: &str = "\
ไต้หวัน (แป่ะเอ๋ยี้: Tâi-oân; ไต่อวัน) หรือ ไถวาน \
(อักษรโรมัน: Taiwan; จีนตัวย่อ: 台湾; จีนตัวเต็ม: 臺灣/台灣; พินอิน: \
Táiwān; ไถวาน) หรือชื่อทางการว่า สาธารณรัฐจีน (จีนตัวย่อ: 中华民国; \
จีนตัวเต็ม: 中華民國; พินอิน: Zhōnghuá Mínguó)";

const LONG_TEXT: &str = "\
ไต้หวัน (แป่ะเอ๋ยี้: Tâi-oân; ไต่อวัน) หรือ ไถวาน \
(อักษรโรมัน: Taiwan; จีนตัวย่อ: 台湾; จีนตัวเต็ม: 臺灣/台灣; พินอิน: \
Táiwān; ไถวาน) หรือชื่อทางการว่า สาธารณรัฐจีน (จีนตัวย่อ: 中华民国; \
จีนตัวเต็ม: 中華民國; พินอิน: Zhōnghuá \
Mínguó) เป็นรัฐในทวีปเอเชียตะวันออก[7][8][9] ปัจจุบันประกอบด้วย\
เกาะใหญ่ 5 แห่ง คือ จินเหมิน (金門), ไต้หวัน, เผิงหู (澎湖), หมาจู่ \
(馬祖), และอูชิว (烏坵) กับทั้งเกาะเล็กเกาะน้อยอีกจำนวนหนึ่ง \
ท้องที่ดังกล่าวเรียกรวมกันว่า \"พื้นที่ไต้หวัน\" (臺灣地區)\n\
ไต้หวันด้านตะวันตกติดกับจีนแผ่นดินใหญ่ ด้านตะวันออกและตะวันออก\
เฉียงเหนือติดกับญี่ปุ่น และด้านใต้ติดกับฟิลิปปินส์ กรุงไทเปเป็น\
เมืองหลวง ส่วนไทเปใหม่เป็นเขตปกครองที่จัดตั้งขึ้นใหม่ กินพื้นที่\
กรุงไทเปและเป็นเขตซึ่งประชากรหนาแน่นที่สุดในเวลานี้\n\
เกาะไต้หวันเดิมเป็นที่อยู่ของชนพื้นเมือง และมีชาวจีนจากแผ่นดิน\
ใหญ่เข้ามาอาศัยร่วมด้วย จนกระทั่งชาววิลันดาและสเปนเดินทางเข้า\
มาในยุคสำรวจเมื่อศตวรรษที่ 17 และมาตั้งบ้านเรือนกลายเป็นนิคม\
ใหญ่โต ต่อมาปี 1662 ราชวงศ์หมิงในแผ่นดินใหญ่ถูกราชวงศ์ชิงแทนที่";

const DICT_PATH: &str = env!("CARGO_MANIFEST_DIR");

fn dict_path() -> String {
    format!("{}/words_th.txt", DICT_PATH)
}

fn load_word_list() -> Vec<String> {
    std::fs::read_to_string(dict_path())
        .expect("words_th.txt not found")
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

// ===========================================================================
// 1. Dictionary construction — TrieChar vs FstDict
// ===========================================================================

fn bench_dict_construction(c: &mut Criterion) {
    let word_list = load_word_list();
    let char_words: Vec<CharString> = word_list.iter().map(|w| CharString::new(w)).collect();
    let word_strs: Vec<&str> = word_list.iter().map(|s| s.as_str()).collect();

    let mut group = c.benchmark_group("dict_construction");
    group.sample_size(10);

    group.bench_function("TrieChar::new", |b| {
        b.iter(|| black_box(TrieChar::new(black_box(&char_words))))
    });

    group.bench_function("FstDict::from_words", |b| {
        b.iter(|| black_box(FstDict::from_words(black_box(word_strs.iter().copied())).unwrap()))
    });

    group.finish();
}

// ===========================================================================
// 2. Dictionary prefix lookup — TrieChar vs FstDict
// ===========================================================================

fn bench_prefix_lookup(c: &mut Criterion) {
    let word_list = load_word_list();
    let char_words: Vec<CharString> = word_list.iter().map(|w| CharString::new(w)).collect();
    let trie = TrieChar::new(&char_words);
    let fst_dict = FstDict::from_words(word_list.iter().map(|s| s.as_str())).unwrap();

    let mut group = c.benchmark_group("prefix_lookup");

    for (label, text) in &[
        ("short_thai", "สวัสดีครับ"),
        ("mixed", "ไต้หวัน1984"),
        ("medium_thai", "อาชญากรรมทางการแพทย์"),
    ] {
        let cs = CharString::new(text);

        group.bench_with_input(
            BenchmarkId::new("TrieChar::prefix_ref", label),
            &cs,
            |b, input| b.iter(|| black_box(TrieChar::prefix_ref(black_box(input), &trie))),
        );

        group.bench_with_input(
            BenchmarkId::new("FstDict::prefix_lengths", label),
            text,
            |b, t| b.iter(|| black_box(fst_dict.prefix_lengths(black_box(t)))),
        );
    }
    group.finish();
}

// ===========================================================================
// 3. End-to-end tokenization — all three tokenizers
//
// NewmmTokenizer    = CharString + TrieChar        (fastest, ~43 MB dict)
// NewmmFstTokenizer = CharString + FstDict   (moderate, ~0.85 MB dict)
// DeepcutTokenizer  = CNN/ONNX                     (requires --features deepcut)
// ===========================================================================

fn bench_full_tokenization(c: &mut Criterion) {
    let path = dict_path();
    let tok_trie = NewmmTokenizer::new(&path).unwrap();
    let tok_fst = NewmmFstTokenizer::new(&path).unwrap();

    #[cfg(feature = "deepcut")]
    let tok_deepcut = nlpo3::tokenizer::deepcut::DeepcutTokenizer::new()
        .expect("deepcut: ONNX model failed to load");

    let mut group = c.benchmark_group("full_tokenization");

    for (label, text) in &[
        ("short", SHORT_TEXT),
        ("medium", MEDIUM_TEXT),
        ("long", LONG_TEXT),
    ] {
        group.throughput(Throughput::Bytes(text.len() as u64));

        // NewmmTokenizer — CharString + TrieChar (fastest dict-based tokenizer)
        group.bench_with_input(
            BenchmarkId::new("NewmmTokenizer/safe=false", label),
            text,
            |b, t| b.iter(|| black_box(tok_trie.segment(black_box(t)).unwrap())),
        );
        group.bench_with_input(
            BenchmarkId::new("NewmmTokenizer/safe=true", label),
            text,
            |b, t| {
                b.iter(|| {
                    black_box(
                        tok_trie
                            .segment_with_options(black_box(t), true, None)
                            .unwrap(),
                    )
                })
            },
        );

        // NewmmFstTokenizer — CharString + FstDict (memory-efficient)
        group.bench_with_input(
            BenchmarkId::new("NewmmFstTokenizer/safe=false", label),
            text,
            |b, t| b.iter(|| black_box(tok_fst.segment(black_box(t)).unwrap())),
        );
        group.bench_with_input(
            BenchmarkId::new("NewmmFstTokenizer/safe=true", label),
            text,
            |b, t| {
                b.iter(|| {
                    black_box(
                        tok_fst
                            .segment_with_options(black_box(t), true, None)
                            .unwrap(),
                    )
                })
            },
        );

        // DeepcutTokenizer — CNN/ONNX (only with --features deepcut)
        #[cfg(feature = "deepcut")]
        group.bench_with_input(BenchmarkId::new("DeepcutTokenizer", label), text, |b, t| {
            b.iter(|| black_box(tok_deepcut.segment(black_box(t)).unwrap()))
        });
    }
    group.finish();
}

// ===========================================================================
// 4. Memory footprint — printed to stderr during benchmark run
// ===========================================================================

fn bench_memory_footprint(c: &mut Criterion) {
    use std::mem;

    let word_list = load_word_list();
    let n_words = word_list.len();
    let fst_dict = FstDict::from_words(word_list.iter().map(|s| s.as_str())).unwrap();

    eprintln!("\n╔══════════════════════════════════════════════════════╗");
    eprintln!("║          Memory footprint analysis                   ║");
    eprintln!("╚══════════════════════════════════════════════════════╝");

    // --- struct stack sizes ---
    eprintln!("Stack sizes:");
    eprintln!("  CharString: {} bytes", mem::size_of::<CharString>());

    // --- per-character heap usage ---
    let text = MEDIUM_TEXT;
    let n_chars = text.chars().count();
    let utf8_bytes = text.len();

    // UTF-8 bytes + u32 positions table
    let heap_per_char =
        (utf8_bytes + (n_chars + 1) * mem::size_of::<u32>()) as f64 / n_chars as f64;

    eprintln!(
        "\nPer-character heap ({} chars, mixed Thai/Latin/digits):",
        n_chars
    );
    eprintln!(
        "  CharString (UTF-8 source + u32 pos table): {:.1} bytes/char",
        heap_per_char
    );

    // --- dictionary memory ---
    let fst_bytes = fst_dict.fst_size_bytes();
    let total_chars: usize = word_list.iter().map(|w| w.chars().count()).sum();
    // 48 bytes per String: 24 bytes stack (ptr+len+cap) + ~24 bytes heap overhead
    // (allocation header; varies by allocator — jemalloc, glibc, etc.).
    let words_set_bytes: usize = word_list.iter().map(|w| w.len() + 48).sum();
    // ~80 bytes per trie edge: 24-byte TrieNode stack + HashMap bucket (~56 bytes for char+ptr entry).
    let trie_estimate = words_set_bytes + total_chars * 80;

    eprintln!("\nDictionary ({} words):", n_words);
    eprintln!(
        "  FstDict base FST:    {:>8} bytes  ({:.1} bytes/word)",
        fst_bytes,
        fst_bytes as f64 / n_words as f64
    );
    eprintln!(
        "  TrieChar (rough estimate): ~{:>7} MB  (~{:.0} bytes/word)",
        trie_estimate / 1_000_000,
        trie_estimate as f64 / n_words as f64
    );
    eprintln!(
        "  → FstDict is ~{:.0}× smaller than TrieChar",
        trie_estimate as f64 / fst_bytes as f64
    );

    #[cfg(feature = "deepcut")]
    {
        // The model size is constant; read at runtime to avoid embedding the
        // full ONNX blob just for the size query.
        let model_path = format!("{}/model/deepcut.onnx", DICT_PATH);
        let model_bytes = std::fs::metadata(&model_path).map(|m| m.len()).unwrap_or(0);
        eprintln!("\nDeepcutTokenizer:");
        eprintln!("  ONNX model (bundled): {} bytes", model_bytes);
        eprintln!("  No dictionary — model weights are fixed-size.");
    }
    eprintln!();

    let mut group = c.benchmark_group("memory_footprint");
    group.bench_function("CharString::new/overhead", |b| {
        b.iter(|| {
            let cs = CharString::new(black_box(MEDIUM_TEXT));
            black_box(mem::size_of::<CharString>() + cs.as_str().len() + (cs.chars_len() + 1) * 4)
        })
    });
    group.finish();
}

// ===========================================================================
// 5. Clone cost — Arc-backed dicts make clone O(1)
//
// After the 2.0 Arc migration, cloning a tokenizer is a single atomic
// reference-count increment rather than a full dictionary copy.  These
// benchmarks verify the O(1) claim and help catch regressions.
// ===========================================================================

fn bench_clone_cost(c: &mut Criterion) {
    let path = dict_path();
    let tok_trie = NewmmTokenizer::new(&path).unwrap();
    let tok_fst = NewmmFstTokenizer::new(&path).unwrap();

    let mut group = c.benchmark_group("clone_cost");
    group.sample_size(200);

    // Cloning a NewmmTokenizer<TrieChar> is O(1) — Arc reference-count bump.
    group.bench_function("NewmmTokenizer::clone", |b| {
        b.iter(|| black_box(tok_trie.clone()))
    });

    // Cloning a NewmmFstTokenizer is O(1) — Arc reference-count bump.
    group.bench_function("NewmmFstTokenizer::clone", |b| {
        b.iter(|| black_box(tok_fst.clone()))
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Register all benchmark groups
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_dict_construction,
    bench_prefix_lookup,
    bench_full_tokenization,
    bench_memory_footprint,
    bench_clone_cost,
);
criterion_main!(benches);
