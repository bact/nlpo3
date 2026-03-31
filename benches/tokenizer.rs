// SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
// SPDX-License-Identifier: Apache-2.0

//! Benchmarks comparing the **old** four-byte-encoded string approach to the
//! **new** native-Rust `CharString` implementation, and comparing the two
//! independent axes of the new design:
//!
//! - **String representation**: old 4-byte encoding vs new `CharString` (native UTF-8)
//! - **Dictionary backend**: `TrieChar` (fast lookup) vs `FstDictionary` (compact)
//!
//! Because `CharString` and `DictBackend` are orthogonal, all four combinations
//! are valid. This file benchmarks three end-to-end combinations:
//!
//! - **`CharString + TrieChar`** (`NewmmTokenizer`) — maximum speed (new default)
//! - **`FourByteStr + TrieChar`** — old 4-byte TCC overhead with same dict; isolate
//!   the string-encoding cost
//! - **`CharString + FstDictionary`** (`NewmmTokenizerFst`) — compact memory
//!
//! # Structure
//!
//! | Group | Benchmarks |
//! |-------|-----------|
//! | `string_construction` | old 4-byte+char_vec vs `CharString::new` |
//! | `char_access` | `Vec<char>` vs position-table vs `str::chars().nth` |
//! | `tcc_pos` | `bytes::Regex` on 4-byte vs `Regex` on UTF-8 |
//! | `encode_plus_tcc` | full old/new pipeline |
//! | `dict_construction` | `TrieChar::new` vs `FstDictionary::from_words` |
//! | `prefix_lookup` | `TrieChar::prefix_ref` vs `FstDictionary::prefix_lengths` |
//! | `full_tokenization` | `CharString+TrieChar` vs `FourByteStr+TrieChar` vs `CharString+FstDict` |
//! | `memory_footprint` | heap-size estimates printed to stderr |
//!
//! Run with:
//! ```sh
//! cargo bench
//! # or just a specific group:
//! cargo bench -- full_tokenization
//! ```
//! HTML reports land in `target/criterion/`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nlpo3::{
    char_string::CharString,
    tokenizer::{
        fst_dict::FstDictionary,
        newmm::{NewmmTokenizer, NewmmTokenizerFst},
        tcc::tcc_tokenizer::tcc_pos,
        trie_char::TrieChar,
        tokenizer_trait::Tokenizer,
    },
};

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
// Old implementation — reconstructed inline for comparison
//
// Sources: src/four_bytes_str/custom_string.rs and
//          src/tokenizer/tcc/tcc_tokenizer.rs (before the rewrite).
// ===========================================================================

mod old_impl {
    use regex::bytes::Regex as BytesRegex;
    use rustc_hash::FxHashSet as HashSet;

    pub const BYTES_PER_CHAR: usize = 4;

    // -----------------------------------------------------------------------
    // String encoding: UTF-8 → 4-byte left-zero-padded representation
    // -----------------------------------------------------------------------

    /// Encode a UTF-8 string into the old four-byte-per-character format.
    ///
    /// Each Unicode scalar value is encoded into exactly 4 bytes, padded with
    /// leading zeros so that:
    ///   - ASCII  (1-byte UTF-8): `[0, 0, 0, b1]`
    ///   - 2-byte UTF-8:          `[0, 0, b1, b2]`
    ///   - 3-byte UTF-8 (Thai):   `[0, b1, b2, b3]`
    ///   - 4-byte UTF-8:          `[b1, b2, b3, b4]`
    pub fn to_four_bytes(input: &str) -> Vec<u8> {
        let n_chars = input.chars().count();
        let mut out: Vec<u8> = Vec::with_capacity(n_chars * BYTES_PER_CHAR);
        for ch in input.chars() {
            let mut buf = [0u8; 4];
            ch.encode_utf8(&mut buf);
            let arranged = match buf {
                [a, 0, 0, 0] => [0, 0, 0, a],
                [a, b, 0, 0] => [0, 0, a, b],
                [a, b, c, 0] => [0, a, b, c],
                _ => buf,
            };
            out.extend_from_slice(&arranged);
        }
        out
    }

    /// Build the `Vec<char>` cache that `CustomString` kept alongside the
    /// four-byte buffer (used for `get_char_at`).
    pub fn build_char_vec(input: &str) -> Vec<char> {
        input.chars().collect()
    }

    /// O(1) character access via the `Vec<char>` cache (old approach).
    #[inline]
    pub fn get_char_at(char_vec: &[char], index: usize) -> char {
        char_vec[index]
    }

    // -----------------------------------------------------------------------
    // TCC boundary detection using regex::bytes::Regex on 4-byte text
    //
    // These pattern strings are the exact outputs of the old
    // `regex_pattern_to_custom_pattern()` function, taken verbatim from the
    // test assertions in src/tokenizer/tcc/tcc_rules.rs.
    // -----------------------------------------------------------------------

    // Shared suffix for optional final consonant group (old `k` expansion):
    //   k = (cc?[dิ]?[์])?  after c=[ก-ฮ], d=[ุู]
    //   4-byte form: (\x00[ก-ฮ](\x00[ก-ฮ])?(\x00[ิุ-ู])?\x00[์])?
    const K: &str = r"(\x00[ก-ฮ](\x00[ก-ฮ])?(\x00[ิุ-ู])?\x00[์])?";

    // The full combined pattern for NON_LOOKAHEAD_TCC, mirroring the old
    // lazy_static in tcc_rules.rs.  Each sub-pattern is the exact output of
    // the old `regex_pattern_to_custom_pattern(replace_tcc_symbol(p))`
    // function, taken verbatim from the test assertions in the old
    // src/tokenizer/tcc/tcc_rules.rs (removed in the rewrite commit).
    fn build_old_non_lookahead_pattern() -> String {
        let k = K;
        // Patterns 1-24 + special chars + lookahead captures
        // (derived from test assertions in the old tcc_rules.rs)
        let parts: &[&str] = &[
            // 1: ^เc็ck
            &format!(r"^\x00เ\x00[ก-ฮ]\x00็\x00[ก-ฮ]{}", k),
            // 2: ^เcctาะk
            &format!(r"^\x00เ\x00[ก-ฮ]\x00[ก-ฮ](\x00[่-๋])?\x00า\x00ะ{}", k),
            // 3: ^เccีtยะk
            &format!(r"^\x00เ\x00[ก-ฮ]\x00[ก-ฮ]\x00ี(\x00[่-๋])?\x00ย\x00ะ{}", k),
            // 4: ^เcc็ck
            &format!(r"^\x00เ\x00[ก-ฮ]\x00[ก-ฮ]\x00็\x00[ก-ฮ]{}", k),
            // 5: ^เcิc์ck
            &format!(r"^\x00เ\x00[ก-ฮ]\x00ิ\x00[ก-ฮ]\x00์\x00[ก-ฮ]{}", k),
            // 6: ^เcิtck
            &format!(r"^\x00เ\x00[ก-ฮ]\x00ิ(\x00[่-๋])?\x00[ก-ฮ]{}", k),
            // 7: ^เcีtยะ?k
            &format!(r"^\x00เ\x00[ก-ฮ]\x00ี(\x00[่-๋])?\x00ย(\x00ะ)?{}", k),
            // 8: ^เcืtอะ?k
            &format!(r"^\x00เ\x00[ก-ฮ]\x00ื(\x00[่-๋])?\x00อ(\x00ะ)?{}", k),
            // 9: ^เctา?ะ?k
            &format!(r"^\x00เ\x00[ก-ฮ](\x00[่-๋])?(\x00า)?(\x00ะ)?{}", k),
            // 10: ^cัtวะk
            &format!(r"^\x00[ก-ฮ]\x00ั(\x00[่-๋])?\x00ว\x00ะ{}", k),
            // 11: ^c[ัื]tc[ุิะ]?k
            &format!(r"^\x00[ก-ฮ]\x00[ัื](\x00[่-๋])?\x00[ก-ฮ](\x00[ะิุ])?{}", k),
            // 12: ^c[ิุู]์k
            &format!(r"^\x00[ก-ฮ]\x00[ิุ-ู]\x00์{}", k),
            // 13: ^c[ะ-ู]tk
            &format!(r"^\x00[ก-ฮ]\x00[ะ-ู](\x00[่-๋])?{}", k),
            // 14: ^cรรc์ (no k)
            r"^\x00[ก-ฮ]\x00ร\x00ร\x00[ก-ฮ]\x00์",
            // 15: ^c็
            r"^\x00[ก-ฮ]\x00็",
            // 16: ^ct[ะาำ]?k
            &format!(r"^\x00[ก-ฮ](\x00[่-๋])?(\x00[ะา-ำ])?{}", k),
            // 17: ^ck
            &format!(r"^\x00[ก-ฮ]{}", k),
            // 18: ^แc็c
            r"^\x00แ\x00[ก-ฮ]\x00็\x00[ก-ฮ]",
            // 19: ^แcc์
            r"^\x00แ\x00[ก-ฮ]\x00[ก-ฮ]\x00์",
            // 20: ^แctะ
            r"^\x00แ\x00[ก-ฮ](\x00[่-๋])?\x00ะ",
            // 21: ^แcc็c
            r"^\x00แ\x00[ก-ฮ]\x00[ก-ฮ]\x00็\x00[ก-ฮ]",
            // 22: ^แccc์
            r"^\x00แ\x00[ก-ฮ]\x00[ก-ฮ]\x00[ก-ฮ]\x00์",
            // 23: ^โctะ
            r"^\x00โ\x00[ก-ฮ](\x00[่-๋])?\x00ะ",
            // 24: ^[เ-ไ]ct
            r"^\x00[เ-ไ]\x00[ก-ฮ](\x00[่-๋])?",
            // Special Thai chars
            r"^\x00ก\x00็",
            r"^\x00อ\x00ึ",
            r"^\x00ห\x00ึ",
            // Look-ahead captures (included verbatim; trimmed by LOOKAHEAD_TCC post-match)
            &format!(
                r"^(\x00เ\x00[ก-ฮ]\x00[ก-ฮ]\x00ี(\x00[่-๋])?\x00ย)\x00[ก-ฮเ-ไ]{}", k
            ),
            &format!(
                r"^(\x00เ\x00[ก-ฮ]\x00[ิ-ีุ-ู](\x00[่-๋])?\x00ย)\x00[ก-ฮเ-ไ]{}", k
            ),
        ];
        parts.join("|")
    }

    fn build_old_lookahead_pattern() -> String {
        let k = K;
        [
            format!(r"^(\x00เ\x00[ก-ฮ]\x00[ก-ฮ]\x00ี(\x00[่-๋])?\x00ย)\x00[ก-ฮเ-ไ]{}", k),
            format!(r"^(\x00เ\x00[ก-ฮ]\x00[ิ-ีุ-ู](\x00[่-๋])?\x00ย)\x00[ก-ฮเ-ไ]{}", k),
        ]
        .join("|")
    }

    lazy_static::lazy_static! {
        pub static ref OLD_NON_LOOKAHEAD_TCC: BytesRegex =
            BytesRegex::new(&build_old_non_lookahead_pattern()).unwrap();
        pub static ref OLD_LOOKAHEAD_TCC: BytesRegex =
            BytesRegex::new(&build_old_lookahead_pattern()).unwrap();
    }

    // -----------------------------------------------------------------------
    // Old TCC boundary detection
    // Mirrors the old src/tokenizer/tcc/tcc_tokenizer.rs::tcc_pos()
    // -----------------------------------------------------------------------

    pub fn tcc_pos(four_byte_text: &[u8]) -> HashSet<usize> {
        let mut set: HashSet<usize> = HashSet::default();
        let total = four_byte_text.len() / BYTES_PER_CHAR;
        set.reserve(total / 4 + 1);

        let mut txt = four_byte_text;
        let mut position: usize = 0;

        while !txt.is_empty() {
            if let Some(m) = OLD_NON_LOOKAHEAD_TCC.find(txt) {
                let matched = &txt[..m.end()];
                let match_char_len = m.end() / BYTES_PER_CHAR;

                if OLD_LOOKAHEAD_TCC.is_match(matched) {
                    let seg = match_char_len - 1;
                    position += seg;
                    set.insert(position);
                    txt = &txt[seg * BYTES_PER_CHAR..];
                } else {
                    position += match_char_len;
                    set.insert(position);
                    txt = &txt[m.end()..];
                }
            } else {
                // Non-Thai: one character
                position += 1;
                set.insert(position);
                txt = &txt[BYTES_PER_CHAR..];
            }
        }
        set
    }

    // -----------------------------------------------------------------------
    // Full tokenizer using 4-byte string (for TCC) + TrieChar (for dict)
    //
    // This reconstruction mirrors NewmmTokenizer::one_cut from newmm.rs,
    // replacing the single difference: `valid_position` is computed from
    // the old 4-byte TCC path instead of the new Unicode Regex path.
    //
    // Both string representations then use CharString for TrieChar prefix
    // lookups (the current TrieChar API requires &CharString), so the
    // measured delta is exactly:
    //   to_four_bytes() + old_tcc_pos()  vs.  CharString::new() + new_tcc_pos()
    // -----------------------------------------------------------------------

    use binary_heap_plus::{BinaryHeap, MinComparator};
    use nlpo3::{char_string::CharString, tokenizer::trie_char::TrieChar};
    use regex::Regex;
    use rustc_hash::FxHashMap as HashMap;
    use std::collections::VecDeque;

    const MAX_GRAPH_SIZE: usize = 50;
    const USE_MT_THRESHOLD: usize = 10_000;

    const NON_THAI_READABLE: &[&str; 5] = &[
        r"^[-a-zA-Z]+",
        r"^[0-9]+([,\.][0-9]+)*",
        r"^[๐-๙]+([,\.][๐-๙]+)*",
        r"^[ \t]+",
        r"^\r?\n",
    ];

    lazy_static::lazy_static! {
        static ref NON_THAI: Regex =
            Regex::new(&NON_THAI_READABLE.join("|")).unwrap();
        static ref THAI_TWOCHARS: Regex =
            Regex::new(r"^[ก-ฮ]{0,2}$").unwrap();
    }

    fn bfs_paths(
        graph: &HashMap<usize, Vec<usize>>,
        start: usize,
        goal: usize,
        queue: &mut VecDeque<(usize, Vec<usize>)>,
    ) -> Vec<usize> {
        queue.clear();
        let mut init = Vec::with_capacity(goal.saturating_sub(start));
        init.push(start);
        queue.push_back((start, init));
        while let Some((v, path)) = queue.pop_front() {
            if let Some(nexts) = graph.get(&v) {
                for &next in nexts {
                    if next != goal {
                        let mut p = path.clone();
                        p.push(next);
                        queue.push_back((next, p));
                    } else {
                        let mut p = path;
                        p.push(next);
                        return p;
                    }
                }
            }
        }
        // Fallback: return direct edge (should not happen with valid TCC positions).
        vec![start, goal]
    }

    /// Segment `text` using the **old** 4-byte TCC path with **TrieChar** for
    /// dictionary lookups.
    ///
    /// The overhead vs `NewmmTokenizer` (`CharString + TrieChar`) is:
    ///   `to_four_bytes(text)` + `old_tcc_pos(&four_bytes)` on top of the
    ///   identical `CharString::new(text)` + TrieChar prefix-lookup work.
    ///
    /// Safe mode (long-text splitting) is omitted; `safe=false` is sufficient
    /// to isolate the string-encoding overhead for the benchmark.
    pub fn segment_four_bytes_trie(text: &str, dict: &TrieChar) -> Vec<String> {
        if text.is_empty() {
            return vec![];
        }

        // OLD path: encode to 4-byte and run old bytes::Regex TCC.
        let four_bytes = to_four_bytes(text);
        let valid_position = tcc_pos(&four_bytes);

        // Build CharString once — needed for O(1) substring views used by TrieChar.
        let text_cs = CharString::new(text);
        let text_length = text_cs.chars_len();

        let mut reused_queue: VecDeque<(usize, Vec<usize>)> = VecDeque::with_capacity(10);
        let mut graph_size: usize = 0;
        let mut graph: HashMap<usize, Vec<usize>> = HashMap::default();
        graph.reserve(text_length / 10);
        let mut result: Vec<&str> = Vec::with_capacity(text_length / 10);

        let mut position_list: BinaryHeap<usize, MinComparator> = BinaryHeap::new_min();
        let mut existing: HashSet<usize> = HashSet::default();
        existing.reserve(text_length / 10);
        position_list.push(0);
        existing.insert(0);
        let mut end_pos: usize = 0;

        while position_list.peek().map_or(false, |&p| p < text_length) {
            if let Some(begin_pos) = position_list.pop() {
                let sub = text_cs.substring(begin_pos, text_cs.chars_len());
                let prefixes = TrieChar::prefix_ref(&sub, dict);
                for wlen in prefixes {
                    let cand = begin_pos + wlen;
                    if valid_position.contains(&cand) {
                        match graph.get_mut(&begin_pos) {
                            Some(v) => v.push(cand),
                            None => {
                                graph.insert(begin_pos, vec![cand]);
                            }
                        }
                        graph_size += 1;
                        if existing.insert(cand) {
                            position_list.push(cand);
                        }
                        if graph_size > MAX_GRAPH_SIZE {
                            break;
                        }
                    }
                }

                let pl_len = position_list.len();
                if pl_len == 1 {
                    if let Some(&first) = position_list.peek() {
                        let path = bfs_paths(&graph, end_pos, first, &mut reused_queue);
                        graph_size = 0;
                        for &pos in path.iter().skip(1) {
                            result.push(text_cs.substring_as_str(end_pos, pos));
                            end_pos = pos;
                        }
                    }
                } else if pl_len == 0 {
                    let sub_str = sub.as_str();
                    match NON_THAI.find(sub_str) {
                        Some(m) => {
                            let chars = sub_str[..m.end()].chars().count();
                            end_pos = begin_pos + chars;
                        }
                        None => {
                            let mut done = false;
                            for pos in begin_pos + 1..text_length {
                                if valid_position.contains(&pos) {
                                    let prefix = text_cs.substring(pos, text_length);
                                    let prefixes2 = TrieChar::prefix_ref(&prefix, dict);
                                    let valid: Vec<usize> = if prefixes2.len()
                                        >= USE_MT_THRESHOLD
                                    {
                                        prefixes2
                                            .into_iter()
                                            .filter(|&wl| {
                                                let np = pos + wl;
                                                valid_position.contains(&np)
                                                    && !THAI_TWOCHARS.is_match(
                                                        text_cs.substring_as_str(pos, pos + wl),
                                                    )
                                            })
                                            .collect()
                                    } else {
                                        prefixes2
                                            .into_iter()
                                            .filter(|&wl| {
                                                let np = pos + wl;
                                                valid_position.contains(&np)
                                                    && !THAI_TWOCHARS.is_match(
                                                        text_cs.substring_as_str(pos, pos + wl),
                                                    )
                                            })
                                            .collect()
                                    };
                                    if !valid.is_empty() {
                                        end_pos = pos;
                                        done = true;
                                        break;
                                    }
                                    if NON_THAI.is_match(prefix.as_str()) {
                                        end_pos = pos;
                                        done = true;
                                        break;
                                    }
                                }
                            }
                            if !done {
                                end_pos = text_length;
                            }
                        }
                    }

                    let tok = text_cs.substring_as_str(begin_pos, end_pos);
                    match graph.get_mut(&begin_pos) {
                        Some(v) => v.push(end_pos),
                        None => {
                            graph.insert(begin_pos, vec![end_pos]);
                        }
                    }
                    graph_size += 1;
                    result.push(tok);
                    if existing.insert(end_pos) {
                        position_list.push(end_pos);
                    }
                }
            }
        }
        result.into_iter().map(|s| s.to_string()).collect()
    }
}

// ===========================================================================
// 1. String construction — old vs new
// ===========================================================================

fn bench_string_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_construction");

    for text in &[SHORT_TEXT, MEDIUM_TEXT, LONG_TEXT] {
        let label = format!("{}_chars", text.chars().count());
        group.throughput(Throughput::Bytes(text.len() as u64));

        // Old: encode to 4-byte-per-char buffer + collect Vec<char>
        group.bench_with_input(
            BenchmarkId::new("old/to_four_bytes+char_vec", &label),
            text,
            |b, t| {
                b.iter(|| {
                    let bytes = old_impl::to_four_bytes(black_box(t));
                    let chars = old_impl::build_char_vec(black_box(t));
                    black_box((bytes, chars))
                })
            },
        );

        // New: build CharString (Arc<String> + byte-position table)
        group.bench_with_input(
            BenchmarkId::new("new/CharString::new", &label),
            text,
            |b, t| b.iter(|| black_box(CharString::new(black_box(t)))),
        );
    }
    group.finish();
}

// ===========================================================================
// 2. Character access — old vs new
// ===========================================================================

fn bench_char_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("char_access");

    let old_chars = old_impl::build_char_vec(MEDIUM_TEXT);
    let new_cs = CharString::new(MEDIUM_TEXT);
    let n = new_cs.chars_len();
    let mid = n / 2;

    // Old: index into Vec<char>
    group.bench_function("old/get_char_at/middle", |b| {
        b.iter(|| black_box(old_impl::get_char_at(&old_chars, black_box(mid))))
    });
    group.bench_function("old/get_char_at/sequential", |b| {
        b.iter(|| {
            for i in 0..n {
                black_box(old_impl::get_char_at(&old_chars, i));
            }
        })
    });

    // New: O(1) byte-position table look-up
    group.bench_function("new/CharString::get_char_at/middle", |b| {
        b.iter(|| black_box(new_cs.get_char_at(black_box(mid))))
    });
    group.bench_function("new/CharString::get_char_at/sequential", |b| {
        b.iter(|| {
            for i in 0..n {
                black_box(new_cs.get_char_at(i));
            }
        })
    });

    // Baseline: &str linear scan
    let s = MEDIUM_TEXT;
    group.bench_function("baseline/str::chars_nth/middle", |b| {
        b.iter(|| black_box(s.chars().nth(black_box(mid))))
    });

    group.finish();
}

// ===========================================================================
// 3. TCC boundary detection — old vs new
// ===========================================================================

fn bench_tcc_pos(c: &mut Criterion) {
    let mut group = c.benchmark_group("tcc_pos");

    for (label, text) in &[
        ("short", SHORT_TEXT),
        ("medium", MEDIUM_TEXT),
        ("long", LONG_TEXT),
    ] {
        group.throughput(Throughput::Bytes(text.len() as u64));

        // Pre-encode once; encoding cost is measured separately.
        let four_bytes = old_impl::to_four_bytes(text);

        // Old: bytes::Regex on 4-byte encoded text
        group.bench_with_input(
            BenchmarkId::new("old/bytes_regex_on_4byte", label),
            &four_bytes,
            |b, fb| b.iter(|| black_box(old_impl::tcc_pos(black_box(fb)))),
        );

        // New: unicode Regex on UTF-8
        group.bench_with_input(
            BenchmarkId::new("new/unicode_regex_on_utf8", label),
            text,
            |b, t| b.iter(|| black_box(tcc_pos(black_box(t)))),
        );
    }
    group.finish();
}

// ===========================================================================
// 4. Full encode+TCC pipeline — old vs new  (encoding included)
// ===========================================================================

fn bench_encode_plus_tcc(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_plus_tcc");

    for (label, text) in &[
        ("short", SHORT_TEXT),
        ("medium", MEDIUM_TEXT),
        ("long", LONG_TEXT),
    ] {
        group.throughput(Throughput::Bytes(text.len() as u64));

        // Old: encode then TCC  (what happened inside every segment() call)
        group.bench_with_input(
            BenchmarkId::new("old/to_four_bytes+tcc_pos", label),
            text,
            |b, t| {
                b.iter(|| {
                    let fb = old_impl::to_four_bytes(black_box(t));
                    black_box(old_impl::tcc_pos(&fb))
                })
            },
        );

        // New: CharString::new then TCC
        group.bench_with_input(
            BenchmarkId::new("new/CharString_new+tcc_pos", label),
            text,
            |b, t| {
                b.iter(|| {
                    let cs = CharString::new(black_box(t));
                    black_box(tcc_pos(cs.as_str()))
                })
            },
        );
    }
    group.finish();
}

// ===========================================================================
// 5. Dictionary construction — TrieChar vs FstDictionary
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

    group.bench_function("FstDictionary::from_words", |b| {
        b.iter(|| {
            black_box(FstDictionary::from_words(black_box(word_strs.iter().copied())).unwrap())
        })
    });

    group.finish();
}

// ===========================================================================
// 6. Dictionary prefix lookup — TrieChar vs FstDictionary
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
// 7. End-to-end tokenization: three combinations compared
//
// All three use TrieChar for dict lookups. The difference between the first
// two is the string representation; the third adds FstDictionary for memory
// comparison.
//
// `CharString + TrieChar`    = new tokenizer (NewmmTokenizer default)
// `FourByteStr + TrieChar`   = old 4-byte TCC + CharString for TrieChar; measures
//                              the encoding overhead of the 4-byte approach
// `CharString + FstDict`     = new tokenizer with compact dictionary backend
// ===========================================================================

fn bench_full_tokenization(c: &mut Criterion) {
    let path = dict_path();
    // Default backend: CharString + TrieChar
    let tok_trie = NewmmTokenizer::new(&path);
    // Memory-efficient backend: CharString + FstDictionary
    let tok_fst = NewmmTokenizerFst::new_fst(&path);

    // Build a separate TrieChar for the 4-byte benchmark
    let word_list = load_word_list();
    let char_words: Vec<CharString> = word_list.iter().map(|w| CharString::new(w)).collect();
    let trie = TrieChar::new(&char_words);

    let mut group = c.benchmark_group("full_tokenization");

    for (label, text) in &[
        ("short", SHORT_TEXT),
        ("medium", MEDIUM_TEXT),
        ("long", LONG_TEXT),
    ] {
        group.throughput(Throughput::Bytes(text.len() as u64));

        // CharString + TrieChar (new default — maximum speed)
        group.bench_with_input(
            BenchmarkId::new("CharString+TrieChar/safe=false", label),
            text,
            |b, t| b.iter(|| black_box(tok_trie.segment(black_box(t), false, false).unwrap())),
        );
        group.bench_with_input(
            BenchmarkId::new("CharString+TrieChar/safe=true", label),
            text,
            |b, t| b.iter(|| black_box(tok_trie.segment(black_box(t), true, false).unwrap())),
        );

        // FourByteStr + TrieChar (old 4-byte TCC + TrieChar dict, safe=false only)
        //
        // Extra overhead vs CharString+TrieChar: `to_four_bytes()` + old
        // `bytes::Regex` TCC on top of the identical CharString+TrieChar work.
        group.bench_with_input(
            BenchmarkId::new("FourByteStr+TrieChar/safe=false", label),
            text,
            |b, t| {
                b.iter(|| {
                    black_box(old_impl::segment_four_bytes_trie(black_box(t), &trie))
                })
            },
        );

        // CharString + FstDictionary (memory-efficient combination)
        group.bench_with_input(
            BenchmarkId::new("CharString+FstDict/safe=false", label),
            text,
            |b, t| b.iter(|| black_box(tok_fst.segment(black_box(t), false, false).unwrap())),
        );
        group.bench_with_input(
            BenchmarkId::new("CharString+FstDict/safe=true", label),
            text,
            |b, t| b.iter(|| black_box(tok_fst.segment(black_box(t), true, false).unwrap())),
        );
    }
    group.finish();
}

// ===========================================================================
// 8. Memory footprint — printed to stderr during benchmark run
// ===========================================================================

fn bench_memory_footprint(c: &mut Criterion) {
    use std::mem;

    let word_list = load_word_list();
    let n_words = word_list.len();
    let fst_dict = FstDictionary::from_words(word_list.iter().map(|s| s.as_str())).unwrap();

    eprintln!("\n╔══════════════════════════════════════════════════════╗");
    eprintln!("║          Memory footprint analysis                   ║");
    eprintln!("╚══════════════════════════════════════════════════════╝");

    // --- struct stack sizes ---
    eprintln!("Stack sizes:");
    eprintln!("  CharString (new):   {} bytes", mem::size_of::<CharString>());
    eprintln!("  CustomString (old): 24 bytes  (Arc×2 + 2×usize)");

    // --- per-character heap usage ---
    let text = MEDIUM_TEXT;
    let n_chars = text.chars().count();
    let utf8_bytes = text.len(); // UTF-8 source bytes

    // Old: 4 bytes/char (4-byte buf) + 4 bytes/char (Vec<char>)
    let old_heap_per_char = 8.0f64;
    // New: UTF-8 bytes + u32 positions table
    let new_heap_per_char =
        (utf8_bytes + (n_chars + 1) * mem::size_of::<u32>()) as f64 / n_chars as f64;

    eprintln!(
        "\nPer-character heap ({} chars, mixed Thai/Latin/digits):",
        n_chars
    );
    eprintln!(
        "  Old (4-byte buf + Vec<char>):          {:.1} bytes/char",
        old_heap_per_char
    );
    eprintln!(
        "  New (UTF-8 source + u32 pos table):    {:.1} bytes/char  ({:.0}% of old)",
        new_heap_per_char,
        new_heap_per_char / old_heap_per_char * 100.0
    );

    // --- dictionary ---
    let fst_bytes = fst_dict.fst_size_bytes();
    let total_chars: usize = word_list.iter().map(|w| w.chars().count()).sum();
    // 48 bytes per String: 24 bytes stack (ptr+len+cap) + ~24 bytes heap overhead
    // (allocation header; varies by allocator — jemalloc, glibc, etc.).
    let words_set_bytes: usize = word_list.iter().map(|w| w.len() + 48).sum();
    // ~80 bytes per trie edge: 24-byte TrieNode stack + HashMap bucket (~56 bytes for char+ptr entry).
    let trie_estimate = words_set_bytes + total_chars * 80;

    eprintln!("\nDictionary ({} words):", n_words);
    eprintln!(
        "  FstDictionary base FST:   {:>8} bytes  ({:.1} bytes/word)",
        fst_bytes,
        fst_bytes as f64 / n_words as f64
    );
    eprintln!(
        "  TrieChar (rough estimate):~{:>7} MB  (~{:.0} bytes/word)",
        trie_estimate / 1_000_000,
        trie_estimate as f64 / n_words as f64
    );
    eprintln!(
        "  → FST is ~{:.0}× smaller than TrieChar",
        trie_estimate as f64 / fst_bytes as f64
    );
    eprintln!();

    let mut group = c.benchmark_group("memory_footprint");
    group.bench_function("CharString::new/overhead", |b| {
        b.iter(|| {
            let cs = CharString::new(black_box(MEDIUM_TEXT));
            black_box(
                mem::size_of::<CharString>() + cs.as_str().len() + (cs.chars_len() + 1) * 4,
            )
        })
    });
    group.bench_function("old::to_four_bytes/overhead", |b| {
        b.iter(|| {
            let fb = old_impl::to_four_bytes(black_box(MEDIUM_TEXT));
            let cv = old_impl::build_char_vec(black_box(MEDIUM_TEXT));
            black_box(24 + fb.len() + cv.len() * 4) // 24 = old CustomString stack
        })
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Register all groups
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_string_construction,
    bench_char_access,
    bench_tcc_pos,
    bench_encode_plus_tcc,
    bench_dict_construction,
    bench_prefix_lookup,
    bench_full_tokenization,
    bench_memory_footprint,
);
criterion_main!(benches);
