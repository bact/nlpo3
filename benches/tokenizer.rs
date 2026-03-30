// SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
// SPDX-License-Identifier: Apache-2.0

//! Benchmarks comparing the old four-byte-encoded string approach to the new
//! native-Rust `CharString` implementation, covering:
//!
//!   1. **String construction** — allocating and encoding a Thai string.
//!   2. **Character access** — random O(1) look-ups.
//!   3. **Substring / slicing** — creating sub-views of a string.
//!   4. **TCC boundary detection** (`tcc_pos`) — the hot inner loop.
//!   5. **Full tokenization** — end-to-end `NewmmTokenizer::segment`.
//!   6. **Dictionary construction** — building `TrieChar` vs `FstDictionary`.
//!   7. **Dictionary prefix lookup** — `TrieChar::prefix_ref` vs
//!      `FstDictionary::prefix_lengths`.
//!
//! Run with:
//!   ```sh
//!   cargo bench
//!   ```
//! HTML reports are written to `target/criterion/`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nlpo3::{
    char_string::CharString,
    tokenizer::{
        fst_dict::FstDictionary,
        newmm::NewmmTokenizer,
        tcc::tcc_tokenizer::tcc_pos,
        trie_char::TrieChar,
        tokenizer_trait::Tokenizer,
    },
};

// ---------------------------------------------------------------------------
// Shared test fixtures
// ---------------------------------------------------------------------------

/// Short Thai sentence (≈ 24 characters).
const SHORT_TEXT: &str = "พิสูจน์ได้ค่ะสวัสดีประเทศไทย";

/// Medium Thai paragraph (≈ 250 characters).
const MEDIUM_TEXT: &str = "\
ไต้หวัน (แป่ะเอ๋ยี้: Tâi-oân; ไต่อวัน) หรือ ไถวาน \
(อักษรโรมัน: Taiwan; จีนตัวย่อ: 台湾; จีนตัวเต็ม: 臺灣/台灣; พินอิน: \
Táiwān; ไถวาน) หรือชื่อทางการว่า สาธารณรัฐจีน (จีนตัวย่อ: 中华民国; \
จีนตัวเต็ม: 中華民國; พินอิน: Zhōnghuá Mínguó)";

/// Long Thai article excerpt (≈ 1 200 characters, mixed Thai/Latin/digits).
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

/// Read the bundled Thai dictionary as a `Vec<String>`.
fn load_word_list() -> Vec<String> {
    std::fs::read_to_string(dict_path())
        .expect("words_th.txt not found")
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

// ===========================================================================
// 1. String construction
// ===========================================================================

fn bench_string_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_construction");

    for text in &[SHORT_TEXT, MEDIUM_TEXT, LONG_TEXT] {
        let label = format!("{}_chars", text.chars().count());
        group.throughput(Throughput::Bytes(text.len() as u64));

        // New: CharString::new — builds Arc<String> + byte-position table.
        group.bench_with_input(
            BenchmarkId::new("CharString::new", &label),
            text,
            |b, t| b.iter(|| CharString::new(black_box(t))),
        );
    }
    group.finish();
}

// ===========================================================================
// 2. Character access  (O(1) for both implementations)
// ===========================================================================

fn bench_char_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("char_access");

    let cs = CharString::new(MEDIUM_TEXT);
    let n = cs.chars_len();

    // New: O(1) via byte_positions table.
    group.bench_function("CharString::get_char_at/middle", |b| {
        b.iter(|| black_box(cs.get_char_at(black_box(n / 2))))
    });
    // Sequential scan of all characters.
    group.bench_function("CharString::get_char_at/sequential", |b| {
        b.iter(|| {
            for i in 0..n {
                black_box(cs.get_char_at(i));
            }
        })
    });

    // Baseline comparison: native &str chars().nth() — O(n).
    let s = MEDIUM_TEXT;
    group.bench_function("str::chars_nth/middle", |b| {
        b.iter(|| black_box(s.chars().nth(black_box(n / 2))))
    });

    group.finish();
}

// ===========================================================================
// 3. Substring / as_str
// ===========================================================================

fn bench_substring(c: &mut Criterion) {
    let mut group = c.benchmark_group("substring");

    let cs = CharString::new(MEDIUM_TEXT);
    let n = cs.chars_len();

    // New: O(1), no allocation.
    group.bench_function("CharString::substring", |b| {
        b.iter(|| black_box(cs.substring(black_box(10), black_box(n - 10))))
    });
    group.bench_function("CharString::as_str", |b| {
        b.iter(|| {
            let sub = cs.substring(10, n - 10);
            black_box(sub.as_str().len())
        })
    });

    // Baseline: &str slicing (requires knowing byte offsets).
    let s = MEDIUM_TEXT;
    let byte_start = s.char_indices().nth(10).map(|(b, _)| b).unwrap_or(0);
    let byte_end = s
        .char_indices()
        .nth(n - 10)
        .map(|(b, _)| b)
        .unwrap_or(s.len());
    group.bench_function("str::byte_slice", |b| {
        b.iter(|| black_box(&s[black_box(byte_start)..black_box(byte_end)]))
    });

    group.finish();
}

// ===========================================================================
// 4. TCC boundary detection
// ===========================================================================

fn bench_tcc_pos(c: &mut Criterion) {
    let mut group = c.benchmark_group("tcc_pos");

    for (label, text) in &[
        ("short", SHORT_TEXT),
        ("medium", MEDIUM_TEXT),
        ("long", LONG_TEXT),
    ] {
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(BenchmarkId::new("native", label), text, |b, t| {
            b.iter(|| black_box(tcc_pos(black_box(t))))
        });
    }
    group.finish();
}

// ===========================================================================
// 5. Dictionary construction
// ===========================================================================

fn bench_dict_construction(c: &mut Criterion) {
    let word_list = load_word_list();
    let mut group = c.benchmark_group("dict_construction");
    group.sample_size(10); // Dictionary build is slow; fewer samples needed.

    let char_words: Vec<CharString> = word_list
        .iter()
        .map(|w| CharString::new(w))
        .collect();

    // TrieChar: HashMap-based trie, one node per character edge.
    group.bench_function("TrieChar::new", |b| {
        b.iter(|| black_box(TrieChar::new(black_box(&char_words))))
    });

    // FstDictionary: minimized FST, sorted input.
    let word_strs: Vec<&str> = word_list.iter().map(|s| s.as_str()).collect();
    group.bench_function("FstDictionary::from_words", |b| {
        b.iter(|| {
            black_box(FstDictionary::from_words(black_box(word_strs.iter().copied())).unwrap())
        })
    });

    group.finish();
}

// ===========================================================================
// 6. Dictionary prefix lookup
// ===========================================================================

fn bench_prefix_lookup(c: &mut Criterion) {
    let word_list = load_word_list();
    let char_words: Vec<CharString> = word_list.iter().map(|w| CharString::new(w)).collect();
    let trie = TrieChar::new(&char_words);

    let fst_dict = FstDictionary::from_words(word_list.iter().map(|s| s.as_str())).unwrap();

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
            BenchmarkId::new("FstDictionary::prefix_lengths", label),
            text,
            |b, t| b.iter(|| black_box(fst_dict.prefix_lengths(black_box(t)))),
        );
    }
    group.finish();
}

// ===========================================================================
// 7. Full tokenization (end-to-end)
// ===========================================================================

fn bench_full_tokenization(c: &mut Criterion) {
    let path = dict_path();
    let tok = NewmmTokenizer::new(&path);
    let mut group = c.benchmark_group("full_tokenization");

    for (label, text) in &[
        ("short", SHORT_TEXT),
        ("medium", MEDIUM_TEXT),
        ("long", LONG_TEXT),
    ] {
        group.throughput(Throughput::Bytes(text.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("segment_safe=false", label),
            text,
            |b, t| b.iter(|| black_box(tok.segment(black_box(t), false, false).unwrap())),
        );
        group.bench_with_input(
            BenchmarkId::new("segment_safe=true", label),
            text,
            |b, t| b.iter(|| black_box(tok.segment(black_box(t), true, false).unwrap())),
        );
    }
    group.finish();
}

// ===========================================================================
// 8. Memory footprint comparison (static sizing, not wall-clock)
// ===========================================================================

/// Measure and print the byte sizes of key data structures.
/// This is not a wall-clock benchmark but produces console output when run
/// with `-- --nocapture` so results appear in CI logs.
fn bench_memory_sizes(c: &mut Criterion) {
    use std::mem;

    let mut group = c.benchmark_group("memory_sizes");

    // Report struct sizes via a no-op benchmark.
    group.bench_function("struct_sizes_report", |b| {
        b.iter(|| {
            black_box(mem::size_of::<CharString>())
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Register groups
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_string_construction,
    bench_char_access,
    bench_substring,
    bench_tcc_pos,
    bench_dict_construction,
    bench_prefix_lookup,
    bench_full_tokenization,
    bench_memory_sizes,
    bench_memory_footprint,
);
criterion_main!(benches);

// ===========================================================================
// 9. Memory footprint analysis (printed during benchmark run)
// ===========================================================================

fn bench_memory_footprint(c: &mut Criterion) {
    use std::mem;

    let word_list = load_word_list();
    let n_words = word_list.len();

    let char_words: Vec<CharString> = word_list.iter().map(|w| CharString::new(w)).collect();
    let fst_dict = FstDictionary::from_words(word_list.iter().map(|s| s.as_str())).unwrap();

    // ---------- struct sizes ----------
    let cs_stack = mem::size_of::<CharString>();
    eprintln!("\n=== Memory footprint analysis ===");
    eprintln!("CharString stack size: {} bytes", cs_stack);
    eprintln!("  (old CustomString: ~24 bytes stack + 4 bytes/char 4-byte-buf + 4 bytes/char Vec<char>)");

    // ---------- per-character footprint of CharString ----------
    let text = MEDIUM_TEXT;
    let n_chars = text.chars().count();
    let utf8_bytes = text.len();
    let positions_bytes = (n_chars + 1) * mem::size_of::<u32>();
    let new_bytes_per_char = (utf8_bytes + positions_bytes) as f64 / n_chars as f64;
    let old_bytes_per_char = 8.0f64; // 4-byte buffer + Vec<char>
    eprintln!(
        "\nPer-character heap usage ({} chars of mixed Thai/Latin/digits):",
        n_chars
    );
    eprintln!("  Old (4-byte buffer + Vec<char>): {:.1} bytes/char", old_bytes_per_char);
    eprintln!("  New (UTF-8 source + u32 positions): {:.1} bytes/char  ({:.0}% of old)",
        new_bytes_per_char,
        new_bytes_per_char / old_bytes_per_char * 100.0
    );

    // ---------- dictionary ----------
    let fst_bytes = fst_dict.fst_size_bytes();
    eprintln!("\nDictionary ({} words):", n_words);
    eprintln!(
        "  FstDictionary base FST: {} bytes  ({:.1} bytes/word)",
        fst_bytes,
        fst_bytes as f64 / n_words as f64
    );

    // Rough estimate for TrieChar heap usage:
    //   - words HashSet<String>: ~(avg UTF-8 length + 48 bytes overhead) per word
    //   - trie nodes: each unique character edge has a HashMap entry
    //     A HashMap<char, TrieNode> has 24 bytes header + load-factor * (4+node) bytes per slot.
    //     Very rough: 1 node per character in all words = total_chars * ~80 bytes
    let total_chars: usize = word_list.iter().map(|w| w.chars().count()).sum();
    let words_set_bytes: usize = word_list.iter().map(|w| w.len() + 48).sum();
    let trie_nodes_estimate = total_chars * 80; // rough: 80 bytes per node/edge
    let trie_total_estimate = words_set_bytes + trie_nodes_estimate;
    eprintln!(
        "  TrieChar (rough estimate): ~{} MB  (~{:.0} bytes/word)",
        trie_total_estimate / 1_000_000,
        trie_total_estimate as f64 / n_words as f64
    );
    eprintln!(
        "  FST is ~{:.0}x smaller than TrieChar estimate",
        trie_total_estimate as f64 / fst_bytes as f64
    );
    eprintln!();

    let mut group = c.benchmark_group("memory_footprint");
    group.bench_function("fst_size_bytes", |b| {
        b.iter(|| black_box(fst_dict.fst_size_bytes()))
    });
    group.bench_function("charstring_alloc_size", |b| {
        b.iter(|| {
            let cs = CharString::new(black_box(MEDIUM_TEXT));
            black_box(mem::size_of::<CharString>() + cs.as_str().len() + (cs.chars_len() + 1) * 4)
        })
    });
    drop(char_words);
    group.finish();
}
