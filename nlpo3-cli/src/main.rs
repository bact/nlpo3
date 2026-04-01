// SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
// SPDX-License-Identifier: Apache-2.0

use std::io;
use std::io::BufRead;

use clap::{Parser, Subcommand, ValueEnum};
use nlpo3::tokenizer::tokenizer_trait::Tokenizer;

#[cfg(feature = "deepcut")]
use nlpo3::tokenizer::deepcut::DeepcutTokenizer;
use nlpo3::tokenizer::newmm::{NewmmFstTokenizer, NewmmTokenizer};

const DEFAULT_DICT: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../words_th.txt"));

#[derive(Parser, Debug)]
#[command(name = "nlpo3", about = "Thai natural language processing CLI")]
struct App {
    #[command(subcommand)]
    subcommand: SubCommand,
}

#[derive(Subcommand, Debug)]
enum SubCommand {
    /// Tokenize text into words, reading from standard input line by line.
    Segment(SegmentOpts),
}

/// Tokenizer algorithm to use.
#[derive(ValueEnum, Clone, Debug, Default)]
enum TokenizerKind {
    /// Dictionary-based maximal-matching (default, fastest).
    #[default]
    Newmm,
    /// Dictionary-based maximal-matching with FST backend (lower memory use).
    Nf,
    /// Neural CNN-based tokenizer (deepcut).
    #[cfg(feature = "deepcut")]
    Deepcut,
}

#[derive(Parser, Debug)]
struct SegmentOpts {
    /// Tokenizer to use: newmm (default), nf, deepcut.
    #[arg(short = 't', long, default_value = "newmm")]
    tokenizer: TokenizerKind,

    /// Path to a dictionary file (one word per line).
    /// Use "default" or omit to use the built-in dictionary.
    /// Ignored when --tokenizer deepcut is selected.
    #[arg(short = 'd', long, default_value = "default")]
    dict_path: String,

    /// Token delimiter printed between words.
    #[arg(short = 's', long, default_value = "|")]
    word_delimiter: String,

    /// Enable safe mode to avoid long run times on inputs with many
    /// ambiguous word boundaries.
    #[arg(short = 'z', long)]
    safe: bool,

    /// Enable multithread mode (uses more memory).
    #[arg(short = 'p', long)]
    parallel: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = App::parse();

    let SubCommand::Segment(segment_opts) = opt.subcommand;

    let tokenizer: Box<dyn Tokenizer> = match segment_opts.tokenizer {
        TokenizerKind::Newmm => {
            let dict_path = match segment_opts.dict_path.as_str() {
                "default" => None,
                dict_name => Some(dict_name),
            };
            match dict_path {
                None => Box::new(NewmmTokenizer::from_word_list(
                    DEFAULT_DICT.lines().map(|s| s.to_owned()).collect(),
                )),
                Some(path) => Box::new(NewmmTokenizer::new(path)?),
            }
        }
        TokenizerKind::Nf => {
            let dict_path = match segment_opts.dict_path.as_str() {
                "default" => None,
                dict_name => Some(dict_name),
            };
            match dict_path {
                None => Box::new(NewmmFstTokenizer::from_word_list(
                    DEFAULT_DICT.lines().map(|s| s.to_owned()).collect(),
                )?),
                Some(path) => Box::new(NewmmFstTokenizer::new(path)?),
            }
        }
        #[cfg(feature = "deepcut")]
        TokenizerKind::Deepcut => Box::new(DeepcutTokenizer::new()?),
    };

    for line_opt in io::stdin().lock().lines() {
        let line = line_opt?;
        let cleaned_line = line.trim_end_matches('\n');
        let toks = tokenizer.segment_to_string(
            cleaned_line,
            segment_opts.safe,
            segment_opts.parallel,
        );
        println!("{}", toks.join(segment_opts.word_delimiter.as_str()));
    }
    Ok(())
}
