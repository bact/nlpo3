// SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
// SPDX-License-Identifier: Apache-2.0

//! Helper utilities for parallel tokenization.
//!
//! Text chunks are split at boundaries that respect Thai Character Cluster (TCC)
//! boundaries and prefer natural punctuation breaks. Never breaks inside a TCC cluster.

use rayon::prelude::*;
use rustc_hash::FxHashSet;

/// Context window to search for good break points around the target split location.
const BREAK_SEARCH_WINDOW: usize = 100;

/// Split text into chunks for parallel processing.
///
/// Chunks respect Thai Character Cluster (TCC) boundaries and prefer breaking at:
/// 1. Sentence-ending punctuation (。!?！？)
/// 2. Whitespace boundaries
/// 3. TCC cluster boundaries (never breaks inside a cluster)
///
/// # Arguments
/// - `text`: The text to chunk
/// - `target_chunk_size`: Approximate target chunk size
/// - `valid_break_points`: Set of valid character indices where chunking can occur
pub fn split_text_into_chunks<'a>(
    text: &'a str,
    target_chunk_size: usize,
    valid_break_points: &FxHashSet<usize>,
) -> Vec<&'a str> {
    if text.len() <= target_chunk_size {
        return vec![text];
    }
    let char_count = text.chars().count();
    let mut chunks = Vec::new();
    let mut start: usize = 0; // character index
    while start < char_count {
        let target_end = start + (target_chunk_size / 4); // rough estimate: ~4 bytes per Thai char
        let target_end = target_end.min(char_count);
        // Find the best break point near the target
        let break_pos =
            find_best_break_point(text, start, target_end, valid_break_points, char_count);
        // Extract chunk from start to break_pos (in byte offsets)
        if break_pos > start {
            let start_byte = char_index_to_byte_offset(text, start);
            let break_byte = char_index_to_byte_offset(text, break_pos);
            chunks.push(&text[start_byte..break_byte]);
            start = break_pos;
        } else {
            // Fallback: if no valid break point found, move forward at least 1 char
            start += 1.max((target_chunk_size / 4).max(1));
        }
    }
    chunks
}

/// Find the best break point in the range [start, end] that respects TCC boundaries.
///
/// Preference order:
/// 1. Sentence-ending punctuation with space after (ideal case)
/// 2. Other whitespace at valid TCC positions
/// 3. Valid TCC positions without whitespace
/// 4. Fallback: the target position if no better option
fn find_best_break_point(
    text: &str,
    start: usize,
    target_end: usize,
    tcc_positions: &FxHashSet<usize>,
    _char_count: usize,
) -> usize {
    let start_byte = char_index_to_byte_offset(text, start);
    let target_byte = char_index_to_byte_offset(text, target_end.min(text.chars().count()));

    // Define the search range: around the target position
    let raw_search_start_byte = if target_byte > BREAK_SEARCH_WINDOW {
        target_byte - BREAK_SEARCH_WINDOW
    } else {
        start_byte
    };
    let raw_search_end_byte = (target_byte + BREAK_SEARCH_WINDOW).min(text.len());
    let search_start_byte = clamp_to_char_boundary_left(text, raw_search_start_byte);
    let search_end_byte = clamp_to_char_boundary_right(text, raw_search_end_byte);

    if search_start_byte >= search_end_byte {
        return target_end;
    }

    let search_range = &text[search_start_byte..search_end_byte];

    let best_pos = target_end; // fallback

    // Look for punctuation breaks (priority: highest)
    if let Some(idx) = search_range.rfind(|c: char| matches!(c, '。' | '!' | '?' | '！' | '？'))
    {
        let punct_pos = search_start_byte + idx;
        // Prefer a space after punctuation if available
        if let Some(after) = punct_pos.checked_add(get_char_at_offset(text, punct_pos).len_utf8()) {
            if after < text.len() && text[after..].starts_with(' ') {
                if let Some(char_idx) = byte_offset_to_char_index(text, after + 1) {
                    if tcc_positions.contains(&char_idx) {
                        return char_idx;
                    }
                }
            }
        }
        // Fallback to just after the punctuation
        if let Some(char_idx) = byte_offset_to_char_index(text, punct_pos) {
            if let Some(next_idx) = char_idx.checked_add(1) {
                if tcc_positions.contains(&next_idx) {
                    return next_idx;
                }
            }
        }
    }

    // Look for whitespace breaks (priority: high)
    if let Some(idx) = search_range.rfind(|c: char| c.is_whitespace()) {
        let space_byte = search_start_byte + idx;
        if let Some(char_idx) = byte_offset_to_char_index(
            text,
            space_byte.saturating_add(get_char_at_offset(text, space_byte).len_utf8()),
        ) {
            if tcc_positions.contains(&char_idx) {
                return char_idx;
            }
        }
    }

    // Fall back to nearest TCC position within search range
    for pos in (start..=target_end.min(text.chars().count())).rev() {
        if tcc_positions.contains(&pos) {
            return pos;
        }
    }

    best_pos
}

/// Convert a character index to a byte offset in the text.
fn char_index_to_byte_offset(text: &str, char_index: usize) -> usize {
    text.chars().take(char_index).map(|c| c.len_utf8()).sum()
}

/// Convert a byte offset to a character index, if valid.
fn byte_offset_to_char_index(text: &str, byte_offset: usize) -> Option<usize> {
    if byte_offset > text.len() {
        return None;
    }
    Some(text[..byte_offset].chars().count())
}

/// Get the character at a byte offset.
fn get_char_at_offset(text: &str, byte_offset: usize) -> char {
    text[byte_offset..].chars().next().unwrap_or('\0')
}

/// Move offset left to the nearest UTF-8 char boundary.
fn clamp_to_char_boundary_left(text: &str, mut offset: usize) -> usize {
    offset = offset.min(text.len());
    while offset > 0 && !text.is_char_boundary(offset) {
        offset -= 1;
    }
    offset
}

/// Move offset right to the nearest UTF-8 char boundary.
fn clamp_to_char_boundary_right(text: &str, mut offset: usize) -> usize {
    offset = offset.min(text.len());
    while offset < text.len() && !text.is_char_boundary(offset) {
        offset += 1;
    }
    offset
}

/// Apply a segment function to chunks with optional parallelization.
///
/// - If `parallel=true`, uses Rayon to process chunks in parallel
/// - Otherwise, processes sequentially
pub fn tokenize_chunks<F>(
    chunks: Vec<&str>,
    parallel: bool,
    segment_fn: F,
) -> anyhow::Result<Vec<Vec<String>>>
where
    F: Fn(&str) -> anyhow::Result<Vec<String>> + Send + Sync,
{
    if parallel {
        chunks.into_par_iter().map(segment_fn).collect()
    } else {
        chunks.into_iter().map(segment_fn).collect()
    }
}

/// Flatten a vector of token vectors into a single vector.
pub fn flatten_tokens(token_vecs: Vec<Vec<String>>) -> Vec<String> {
    token_vecs.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp_to_char_boundary_handles_mid_byte_offsets() {
        let text = "AกB";
        let thai_start = text
            .char_indices()
            .find(|(_, ch)| *ch == 'ก')
            .map(|(idx, _)| idx)
            .unwrap();
        let thai_end = thai_start + 'ก'.len_utf8();
        let mid = thai_start + 1;

        assert!(!text.is_char_boundary(mid));
        assert_eq!(clamp_to_char_boundary_left(text, mid), thai_start);
        assert_eq!(clamp_to_char_boundary_right(text, mid), thai_end);
    }

    #[test]
    fn test_split_text_into_chunks_preserves_text_roundtrip() {
        let text = "ภาษาไทยภาษาไทยภาษาไทยABC";
        let char_count = text.chars().count();
        let valid_break_points: FxHashSet<usize> = (0..=char_count).collect();

        let chunks = split_text_into_chunks(text, 8, &valid_break_points);
        assert!(chunks.len() > 1);

        let rebuilt = chunks.concat();
        assert_eq!(rebuilt, text);
    }
}
