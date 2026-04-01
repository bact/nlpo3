// SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
// SPDX-License-Identifier: Apache-2.0

/**
 * Dictionary reader.
 */
use crate::char_string::CharString;

use super::fst_dict::FstDict;
use super::trie_char::TrieChar as Trie;
use std::io::BufReader;
use std::{error::Error, io::prelude::*};
use std::{fs::File, path::PathBuf};

pub enum DictSource {
    FilePath(PathBuf),
    WordList(Vec<String>),
}

pub fn create_dict_trie(source: DictSource) -> Result<Trie, Box<dyn Error>> {
    match source {
        DictSource::FilePath(file_path) => {
            let file_reader = File::open(file_path.as_path());
            match file_reader {
                Ok(file) => {
                    let mut reader = BufReader::new(file);
                    let mut line = String::with_capacity(50);
                    let mut dict: Vec<CharString> = Vec::with_capacity(600);
                    while reader.read_line(&mut line).unwrap() != 0 {
                        dict.push(CharString::new(&line));
                        line.clear();
                    }
                    dict.shrink_to_fit();
                    Ok(Trie::new(&dict))
                }
                Err(error) => Err(Box::from(error)),
            }
        }
        DictSource::WordList(word_list) => {
            let char_word_list: Vec<CharString> =
                word_list.into_iter().map(|w| CharString::new(&w)).collect();
            Ok(Trie::new(&char_word_list))
        }
    }
}

pub fn create_dict_fst(source: DictSource) -> Result<FstDict, Box<dyn Error>> {
    match source {
        DictSource::FilePath(file_path) => {
            let content = std::fs::read_to_string(file_path)?;
            FstDict::from_words(content.lines().map(|l| l.trim()))
        }
        DictSource::WordList(word_list) => {
            FstDict::from_words(word_list.iter().map(|s| s.as_str()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trie() {
        let test_word_list = vec![
            "กากบาท".to_string(),
            "กาแฟ".to_string(),
            "กรรม".to_string(),
            "42".to_string(),
            "aง|.%".to_string(),
        ];
        let trie = create_dict_trie(DictSource::WordList(test_word_list)).unwrap();
        assert!(trie.contain(&CharString::new("กาแฟ")));
        assert_eq!(trie.amount_of_words(), 5);
    }
}
