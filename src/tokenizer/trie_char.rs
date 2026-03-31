// SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
// SPDX-License-Identifier: Apache-2.0

/**
 * Character-based trie for dictionary prefix lookup.
 *
 * The trie nodes branch on `char` values (Rust's native Unicode scalar),
 * so the structure works directly on UTF-8 text without any custom encoding.
 * Words are stored as `String` keys for O(k) membership tests.
 *
 * For basic information on tries, see:
 *   https://en.wikipedia.org/wiki/Trie
 */
use crate::char_string::CharString;
use crate::tokenizer::dict_backend::DictBackend;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

#[derive(Debug)]
struct TrieNode {
    children: HashMap<char, Self>,
    end: bool,
}

impl Default for TrieNode {
    fn default() -> Self {
        Self::new()
    }
}

impl TrieNode {
    pub fn new() -> Self {
        Self {
            children: HashMap::default(),
            end: false,
        }
    }

    fn find_child(&self, ch: &char) -> Option<&Self> {
        self.children.get(ch)
    }

    fn find_mut_child(&mut self, ch: &char) -> Option<&mut Self> {
        self.children.get_mut(ch)
    }

    fn remove_child(&mut self, ch: &char) {
        self.children.remove(ch);
    }

    fn set_not_end(&mut self) {
        self.end = false;
    }

    fn add_word(&mut self, chars: &[char]) {
        if chars.is_empty() {
            self.end = true;
            return;
        }
        self.children
            .entry(chars[0])
            .or_default()
            .add_word(&chars[1..]);
    }

    fn remove_word(&mut self, chars: &[char]) {
        if chars.is_empty() {
            return;
        }
        let ch = chars[0];
        if let Some(child) = self.find_mut_child(&ch) {
            if chars.len() == 1 {
                child.set_not_end();
            } else {
                child.remove_word(&chars[1..]);
            }
            if !child.end && child.children.is_empty() {
                self.remove_child(&ch);
            }
        }
    }
}

/// Character-based trie storing a set of words.
///
/// Words are decoded to Unicode scalar values (`char`) on insertion, and the
/// trie branches on those `char` values.  This allows the trie to work
/// directly with Rust's standard UTF-8 string types without any intermediate
/// encoding: each `&str` is decoded once on the way in, and lookups compare
/// decoded `char` values.
#[derive(Debug)]
pub struct TrieChar {
    words: HashSet<String>,
    root: TrieNode,
}

impl TrieChar {
    pub fn new(words: &[CharString]) -> Self {
        let mut instance = Self {
            words: HashSet::default(),
            root: TrieNode::new(),
        };
        for word in words.iter() {
            instance.add(word);
        }
        instance
    }

    pub fn add(&mut self, word: &CharString) {
        let stripped = word.trim();
        if !stripped.is_empty() {
            let key = stripped.as_str().to_string();
            let chars: Vec<char> = key.chars().collect();
            self.words.insert(key);
            self.root.add_word(&chars);
        }
    }

    pub fn remove(&mut self, word: &CharString) {
        let stripped = word.trim();
        if !stripped.is_empty() {
            let key = stripped.as_str();
            if self.words.contains(key) {
                let chars: Vec<char> = key.chars().collect();
                self.words.remove(key);
                self.root.remove_word(&chars);
            }
        }
    }

    #[allow(dead_code)]
    pub fn contain(&self, word: &CharString) -> bool {
        self.words.contains(word.as_str())
    }

    #[allow(dead_code)]
    pub fn iterate(&self) -> std::collections::hash_set::Iter<'_, String> {
        self.words.iter()
    }

    #[allow(dead_code)]
    pub fn amount_of_words(&self) -> usize {
        self.words.len()
    }

    /// Return character lengths of all dictionary entries that are prefixes
    /// of `prefix`.
    ///
    /// For example, if the dictionary contains "กข" and "กขค" and `prefix`
    /// starts with "กขคงจ", the result is `[2, 3]`.
    pub fn prefix_ref(prefix: &CharString, dict_trie: &Self) -> Vec<usize> {
        let mut result: Vec<usize> = Vec::new();
        let mut current_node = Some(&dict_trie.root);
        let n = prefix.chars_len();

        for i in 0..n {
            let ch = prefix.get_char_at(i);
            match current_node {
                Some(node) => match node.find_child(&ch) {
                    Some(child) => {
                        if child.end {
                            result.push(i + 1); // word of length i+1 chars
                        }
                        current_node = Some(child);
                    }
                    None => break,
                },
                None => break,
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// DictBackend implementation
// ---------------------------------------------------------------------------

impl DictBackend for TrieChar {
    fn prefix_lengths_of(&self, prefix: &CharString) -> Vec<usize> {
        TrieChar::prefix_ref(prefix, self)
    }

    fn add_word(&mut self, word: &CharString) {
        self.add(word);
    }

    fn remove_word(&mut self, word: &CharString) {
        self.remove(word);
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_remove_word() {
        let mut trie = TrieChar::new(&[CharString::new("ศาล")]);
        assert_eq!(trie.amount_of_words(), 1);
        trie.add(&CharString::new("ศาล"));
        assert_eq!(trie.amount_of_words(), 1);
        trie.add(&CharString::new("  ศาล "));
        assert_eq!(trie.amount_of_words(), 1);
        trie.add(&CharString::new("ศาลา"));
        assert_eq!(trie.amount_of_words(), 2);
        trie.remove(&CharString::new("ศาลา"));
        assert_eq!(trie.amount_of_words(), 1);
        trie.remove(&CharString::new("ลา"));
        assert_eq!(trie.amount_of_words(), 1);
        trie.remove(&CharString::new("ศาล"));
        assert_eq!(trie.amount_of_words(), 0);
        trie.remove(&CharString::new(""));
        assert_eq!(trie.amount_of_words(), 0);
    }

    #[test]
    fn test_prefix_ref() {
        let words = vec![
            CharString::new("ก"),
            CharString::new("กข"),
            CharString::new("กขค"),
            CharString::new("คง"),
        ];
        let trie = TrieChar::new(&words);
        let input = CharString::new("กขคงจ");
        let lengths = TrieChar::prefix_ref(&input, &trie);
        assert!(lengths.contains(&1));
        assert!(lengths.contains(&2));
        assert!(lengths.contains(&3));
        assert_eq!(lengths.len(), 3);
    }
}
