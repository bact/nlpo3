// SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
// SPDX-License-Identifier: Apache-2.0

/**
 * Native Unicode string type for the word tokenizer.
 *
 * Replaces the legacy four-byte padded representation with a design
 * based on Rust's standard `String` (UTF-8) plus a precomputed byte-position
 * index. This gives:
 *
 * - **O(1)** character access by index (via the position table).
 * - **O(1)** substring as `&str` (zero-copy slice of the source string).
 * - **Native Unicode regex** matching – no `\x00` padding required.
 * - **Smaller memory footprint** – roughly 7 bytes per character (UTF-8
 *   source ≈ 3 bytes for Thai + u32 position entry = 4 bytes) instead of the
 *   previous 8 bytes per character (4-byte padded buffer + Vec<char>).
 *
 * The byte-position table stores one `u32` per character (plus a sentinel
 * at the end). Strings longer than 4 GiB are not supported; Thai text in
 * practice is always many orders of magnitude below that limit.
 */
use std::{fmt, sync::Arc};

/// Shared, slice-able Unicode string with O(1) character access.
#[derive(Clone, Debug)]
pub struct CharString {
    /// Immutable UTF-8 source text, shared across substrings.
    source: Arc<String>,
    /// `byte_positions[i]` is the byte offset of character `i` in `source`.
    /// Length is `total_chars + 1`; the last entry equals `source.len()`.
    byte_positions: Arc<Vec<u32>>,
    /// Inclusive start character index in `source`.
    start: usize,
    /// Exclusive end character index in `source`.
    end: usize,
}

impl CharString {
    /// Create a new `CharString` from a UTF-8 string slice.
    pub fn new(s: &str) -> Self {
        let source = Arc::new(s.to_string());
        let mut positions: Vec<u32> = Vec::with_capacity(s.len() + 1);
        for (byte_pos, _ch) in s.char_indices() {
            positions.push(byte_pos as u32);
        }
        positions.push(s.len() as u32);
        let len = positions.len() - 1; // number of characters
        Self {
            source,
            byte_positions: Arc::new(positions),
            start: 0,
            end: len,
        }
    }

    /// Number of Unicode characters in this view.
    #[inline]
    pub fn chars_len(&self) -> usize {
        self.end - self.start
    }

    /// Returns `true` if this view contains no characters.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// O(1) character access by index relative to this view.
    ///
    /// # Panics
    /// Panics if `index >= self.chars_len()`.
    #[inline]
    pub fn get_char_at(&self, index: usize) -> char {
        let abs = self.start + index;
        let byte_start = self.byte_positions[abs] as usize;
        let byte_end = self.byte_positions[abs + 1] as usize;
        // SAFETY: byte_positions was built from char_indices, so the slice
        // is always on a valid UTF-8 character boundary.
        self.source[byte_start..byte_end]
            .chars()
            .next()
            .expect("byte_positions table is consistent")
    }

    /// Return all characters in this view as a `char` iterator.
    #[allow(dead_code)]
    pub fn chars(&self) -> impl Iterator<Item = char> + '_ {
        let byte_start = self.byte_positions[self.start] as usize;
        let byte_end = self.byte_positions[self.end] as usize;
        self.source[byte_start..byte_end].chars()
    }

    /// Return this view as a `&str` slice — O(1), no allocation.
    #[inline]
    pub fn as_str(&self) -> &str {
        let byte_start = self.byte_positions[self.start] as usize;
        let byte_end = self.byte_positions[self.end] as usize;
        &self.source[byte_start..byte_end]
    }

    /// Return a sub-view — O(1), shares the same Arc backing store.
    ///
    /// `start` and `end` are character indices relative to `self`.
    ///
    /// # Panics
    /// Panics if `start > end` or `end > self.chars_len()`.
    #[inline]
    pub fn substring(&self, start: usize, end: usize) -> Self {
        debug_assert!(start <= end, "substring: start > end");
        debug_assert!(end <= self.chars_len(), "substring: end out of bounds");
        Self {
            source: Arc::clone(&self.source),
            byte_positions: Arc::clone(&self.byte_positions),
            start: self.start + start,
            end: self.start + end,
        }
    }

    /// Return a `&str` slice for a sub-range given as character indices
    /// relative to `self` — O(1), no allocation.
    #[inline]
    pub fn substring_as_str(&self, char_start: usize, char_end: usize) -> &str {
        let abs_start = self.start + char_start;
        let abs_end = self.start + char_end;
        let byte_start = self.byte_positions[abs_start] as usize;
        let byte_end = self.byte_positions[abs_end] as usize;
        &self.source[byte_start..byte_end]
    }

    /// Trim leading and trailing Unicode whitespace.
    pub fn trim(&self) -> Self {
        let s = self.as_str();
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return Self::new("");
        }
        // Rebuild from the trimmed slice to get a fresh CharString.
        // This path is only called during dictionary loading, not in the hot
        // tokenization loop, so the allocation is acceptable.
        Self::new(trimmed)
    }
}

impl fmt::Display for CharString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Find the last space character and return its character index within the
/// provided `&str`. Returns `None` if no space is found.
pub fn rfind_space_char_index(text: &str) -> Option<usize> {
    let mut last_space: Option<usize> = None;
    for (char_idx, ch) in text.chars().enumerate() {
        if ch == ' ' {
            last_space = Some(char_idx);
        }
    }
    last_space
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ascii() {
        let s = CharString::new("hello");
        assert_eq!(s.chars_len(), 5);
        assert_eq!(s.get_char_at(0), 'h');
        assert_eq!(s.get_char_at(4), 'o');
        assert_eq!(s.as_str(), "hello");
    }

    #[test]
    fn test_thai_chars() {
        let s = CharString::new("กข");
        assert_eq!(s.chars_len(), 2);
        assert_eq!(s.get_char_at(0), 'ก');
        assert_eq!(s.get_char_at(1), 'ข');
        assert_eq!(s.as_str(), "กข");
    }

    #[test]
    fn test_substring() {
        let s = CharString::new("abcde");
        let sub = s.substring(1, 4);
        assert_eq!(sub.chars_len(), 3);
        assert_eq!(sub.as_str(), "bcd");
    }

    #[test]
    fn test_substring_thai() {
        let s = CharString::new("กขคงจ");
        let sub = s.substring(1, 3);
        assert_eq!(sub.chars_len(), 2);
        assert_eq!(sub.as_str(), "ขค");
    }

    #[test]
    fn test_is_empty() {
        assert!(CharString::new("").is_empty());
        assert!(!CharString::new("ก").is_empty());
    }

    #[test]
    fn test_trim() {
        assert!(CharString::new("  ").trim().is_empty());
        assert_eq!(CharString::new(" abc ").trim().as_str(), "abc");
        assert_eq!(CharString::new(" กข ").trim().as_str(), "กข");
    }

    #[test]
    fn test_long_thai_text() {
        let text = [
            "ไต้หวัน (แป่ะเอ๋ยี้: Tâi-oân; ไต่อวัน) หรือ ไถวาน ",
            "(อักษรโรมัน: Taiwan; จีนตัวย่อ: 台湾; จีนตัวเต็ม: 臺灣/台灣; พินอิน: ",
            "Táiwān; ไถวาน) หรือชื่อทางการว่า สาธารณรัฐจีน (จีนตัวย่อ: 中华民国; ",
        ]
        .join("");
        let cs = CharString::new(&text);
        // Round-trip: converting back should give the original string.
        assert_eq!(cs.as_str(), text.as_str());
        // All substrings should round-trip as well.
        for i in 0..cs.chars_len() {
            let sub = cs.substring(i, cs.chars_len());
            assert_eq!(sub.as_str(), &text[cs.byte_positions[i] as usize..]);
        }
    }

    #[test]
    fn test_rfind_space_char_index() {
        assert_eq!(rfind_space_char_index("hello world"), Some(5));
        assert_eq!(rfind_space_char_index("no_space"), None);
        assert_eq!(rfind_space_char_index("ไต้หวัน เป็น"), Some(7));
    }
}
