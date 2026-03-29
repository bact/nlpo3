// SPDX-FileCopyrightText: 2024 PyThaiNLP Project
// SPDX-License-Identifier: Apache-2.0

//! Deepcut Thai word tokenizer — Rust implementation using ONNX inference
//! via the pure-Rust `tract-onnx` engine.
//!
//! Based on LEKCut <https://github.com/PyThaiNLP/LEKCut>
//! and Deepcut <https://github.com/rkcosmos/deepcut>.
//!
//! Original deepcut authors:
//! Rakpong Kittinaradorn, Titipat Achakulvisut, Korakot Chaovavanich,
//! Kittinan Srithaworn, Pattarawat Chormai, Chanwit Kaewkasi,
//! Tulakan Ruangrong, Krichkorn Oparad.
//! Original deepcut license: MIT License.

use std::collections::HashMap;
use std::io::Cursor;

use lazy_static::lazy_static;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::infer::Factoid;

/// Bundled deepcut ONNX model bytes.
const MODEL_BYTES: &[u8] = include_bytes!("../nlpo3/model/deepcut.onnx");

/// Context window size (characters each side + current = 21).
const N_PAD: usize = 21;
/// Half the context window: (21 - 1) / 2 = 10.
const N_PAD_2: usize = (N_PAD - 1) / 2;
/// Vocabulary index for unknown characters ("other" sentinel).
const CHAR_INDEX_OTHER: u32 = 80;
/// Type index for unknown character types ("o" = other).
const CHAR_TYPE_INDEX_OTHER: u32 = 4;
/// Probability threshold for word-boundary classification.
const WORD_BOUNDARY_THRESHOLD: f32 = 0.5;

type DeepCutModel =
    RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

fn build_chars_map() -> HashMap<char, u32> {
    // Pairs of (single-char string, vocabulary index).
    // Index 80 is the "other" sentinel (multi-char key, so not included here).
    let entries: &[(&str, u32)] = &[
        ("\n", 0),
        (" ", 1),
        ("!", 2),
        ("\"", 3),
        ("#", 4),
        ("$", 5),
        ("%", 6),
        ("&", 7),
        ("'", 8),
        ("(", 9),
        (")", 10),
        ("*", 11),
        ("+", 12),
        (",", 13),
        ("-", 14),
        (".", 15),
        ("/", 16),
        ("0", 17),
        ("1", 18),
        ("2", 19),
        ("3", 20),
        ("4", 21),
        ("5", 22),
        ("6", 23),
        ("7", 24),
        ("8", 25),
        ("9", 26),
        (":", 27),
        (";", 28),
        ("<", 29),
        ("=", 30),
        (">", 31),
        ("?", 32),
        ("@", 33),
        ("A", 34),
        ("B", 35),
        ("C", 36),
        ("D", 37),
        ("E", 38),
        ("F", 39),
        ("G", 40),
        ("H", 41),
        ("I", 42),
        ("J", 43),
        ("K", 44),
        ("L", 45),
        ("M", 46),
        ("N", 47),
        ("O", 48),
        ("P", 49),
        ("Q", 50),
        ("R", 51),
        ("S", 52),
        ("T", 53),
        ("U", 54),
        ("V", 55),
        ("W", 56),
        ("X", 57),
        ("Y", 58),
        ("Z", 59),
        ("[", 60),
        ("\\", 61),
        ("]", 62),
        ("^", 63),
        ("_", 64),
        ("a", 65),
        ("b", 66),
        ("c", 67),
        ("d", 68),
        ("e", 69),
        ("f", 70),
        ("g", 71),
        ("h", 72),
        ("i", 73),
        ("j", 74),
        ("k", 75),
        ("l", 76),
        ("m", 77),
        ("n", 78),
        ("o", 79),
        // 80 = "other" sentinel — not a single char, so omitted from this map
        ("p", 81),
        ("q", 82),
        ("r", 83),
        ("s", 84),
        ("t", 85),
        ("u", 86),
        ("v", 87),
        ("w", 88),
        ("x", 89),
        ("y", 90),
        ("z", 91),
        ("}", 92),
        ("~", 93),
        ("ก", 94),
        ("ข", 95),
        ("ฃ", 96),
        ("ค", 97),
        ("ฅ", 98),
        ("ฆ", 99),
        ("ง", 100),
        ("จ", 101),
        ("ฉ", 102),
        ("ช", 103),
        ("ซ", 104),
        ("ฌ", 105),
        ("ญ", 106),
        ("ฎ", 107),
        ("ฏ", 108),
        ("ฐ", 109),
        ("ฑ", 110),
        ("ฒ", 111),
        ("ณ", 112),
        ("ด", 113),
        ("ต", 114),
        ("ถ", 115),
        ("ท", 116),
        ("ธ", 117),
        ("น", 118),
        ("บ", 119),
        ("ป", 120),
        ("ผ", 121),
        ("ฝ", 122),
        ("พ", 123),
        ("ฟ", 124),
        ("ภ", 125),
        ("ม", 126),
        ("ย", 127),
        ("ร", 128),
        ("ฤ", 129),
        ("ล", 130),
        ("ว", 131),
        ("ศ", 132),
        ("ษ", 133),
        ("ส", 134),
        ("ห", 135),
        ("ฬ", 136),
        ("อ", 137),
        ("ฮ", 138),
        ("ฯ", 139),
        ("ะ", 140),
        ("\u{0E31}", 141), // ั (sara a above, U+0E31)
        ("า", 142),
        ("ำ", 143),
        ("ิ", 144),
        ("ี", 145),
        ("ึ", 146),
        ("ื", 147),
        ("ุ", 148),
        ("ู", 149),
        ("ฺ", 150),
        ("เ", 151),
        ("แ", 152),
        ("โ", 153),
        ("ใ", 154),
        ("ไ", 155),
        ("ๅ", 156),
        ("ๆ", 157),
        ("็", 158),
        ("่", 159),
        ("้", 160),
        ("๊", 161),
        ("๋", 162),
        ("์", 163),
        ("ํ", 164),
        ("๐", 165),
        ("๑", 166),
        ("๒", 167),
        ("๓", 168),
        ("๔", 169),
        ("๕", 170),
        ("๖", 171),
        ("๗", 172),
        ("๘", 173),
        ("๙", 174),
        ("\u{2018}", 175), // ' LEFT SINGLE QUOTATION MARK
        ("\u{2019}", 176), // ' RIGHT SINGLE QUOTATION MARK
        ("\u{FEFF}", 177), // BOM
    ];
    entries
        .iter()
        .filter_map(|(s, idx)| {
            let mut chars = s.chars();
            let ch = chars.next()?;
            // Keep only single-character keys.
            if chars.next().is_none() {
                Some((ch, *idx))
            } else {
                None
            }
        })
        .collect()
}

fn build_char_type_map() -> HashMap<char, u32> {
    // (character group, type index)
    // Type indices: b_e=0, c=1, d=2, n=3, o=4(fallback), p=5, q=6, s=7, s_e=8, t=9, v=10, w=11
    let type_groups: &[(&str, u32)] = &[
        ("กขฃคฆงจชซญฎฏฐฑฒณดตถทธนบปพฟภมยรลวศษสฬอ", 1), // c
        ("ฅฉผฟฌหฮ", 3),                                    // n
        ("ะาำิีืึุู", 10),                                // v
        ("เแโใไ", 11),                                     // w
        ("่้๊๋", 9),                                       // t
        ("์ๆฯ.", 7),                                       // s
        ("0123456789๑๒๓๔๕๖๗๘๙", 2),                       // d
        ("\"\u{2018}\u{2019}'", 6),                        // q (quotes)
        (" ", 5),                                           // p (space)
        ("abcdefghijklmnopqrstuvwxyz", 8),                 // s_e
        ("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 0),                 // b_e
    ];
    let mut map = HashMap::new();
    for (group, type_idx) in type_groups {
        for ch in group.chars() {
            map.insert(ch, *type_idx);
        }
    }
    map
}

/// Load and compile the deepcut ONNX model using the same symbolic batch
/// dimension for both inputs so the internal concat constraint is satisfied.
fn load_model() -> anyhow::Result<DeepCutModel> {
    let inf_model = tract_onnx::onnx()
        .model_for_read(&mut Cursor::new(MODEL_BYTES))
        .map_err(|e| anyhow::anyhow!("deepcut: failed to read ONNX model: {}", e))?;

    // The model's two inputs carry different symbolic batch dimensions.
    // Unify them by assigning the same TDim (taken from input 0) to both.
    let dims0: Vec<_> = inf_model
        .input_fact(0)
        .map_err(|e| anyhow::anyhow!("deepcut: failed to get input fact: {}", e))?
        .shape
        .dims()
        .collect();
    let batch_tdim: TDim = dims0[0]
        .concretize()
        .ok_or_else(|| anyhow::anyhow!("deepcut: batch dimension is not symbolic"))?;
    let shared_shape: TVec<TDim> = tvec![batch_tdim, 21usize.into()];

    inf_model
        .with_input_fact(
            0,
            InferenceFact::dt_shape(f32::datum_type(), shared_shape.clone()),
        )
        .map_err(|e| anyhow::anyhow!("deepcut: failed to set input fact 0: {}", e))?
        .with_input_fact(
            1,
            InferenceFact::dt_shape(f32::datum_type(), shared_shape),
        )
        .map_err(|e| anyhow::anyhow!("deepcut: failed to set input fact 1: {}", e))?
        .into_optimized()
        .map_err(|e| anyhow::anyhow!("deepcut: failed to optimize ONNX model: {}", e))?
        .into_runnable()
        .map_err(|e| anyhow::anyhow!("deepcut: failed to compile ONNX model: {}", e))
}

lazy_static! {
    static ref CHARS_MAP: HashMap<char, u32> = build_chars_map();
    static ref CHAR_TYPE_MAP: HashMap<char, u32> = build_char_type_map();
    static ref MODEL: DeepCutModel =
        load_model().expect("deepcut: ONNX model failed to initialize");
}

/// Build the character-index and character-type feature arrays.
///
/// For each position `i` in the text the feature vector is:
/// - 10 forward-context characters (i+1 … i+10)
/// - 10 backward-context characters (i-1 … i-10, i.e. reversed)
/// - 1 current character
/// Total: 21 features per position.
fn build_features(text_chars: &[char]) -> (Vec<f32>, Vec<f32>) {
    let n = text_chars.len();

    // Padded sequence: N_PAD_2 spaces + text + N_PAD_2 spaces.
    let mut padded: Vec<char> = Vec::with_capacity(n + 2 * N_PAD_2);
    padded.extend(std::iter::repeat(' ').take(N_PAD_2));
    padded.extend_from_slice(text_chars);
    padded.extend(std::iter::repeat(' ').take(N_PAD_2));

    let mut x_char = Vec::with_capacity(n * N_PAD);
    let mut x_type = Vec::with_capacity(n * N_PAD);

    for i in N_PAD_2..(N_PAD_2 + n) {
        // Forward context: padded[i+1 .. i+N_PAD_2+1]
        for &ch in &padded[i + 1..i + N_PAD_2 + 1] {
            x_char.push(*CHARS_MAP.get(&ch).unwrap_or(&CHAR_INDEX_OTHER) as f32);
            x_type.push(*CHAR_TYPE_MAP.get(&ch).unwrap_or(&CHAR_TYPE_INDEX_OTHER) as f32);
        }
        // Backward context (reversed): padded[i-N_PAD_2 .. i]
        for &ch in padded[i - N_PAD_2..i].iter().rev() {
            x_char.push(*CHARS_MAP.get(&ch).unwrap_or(&CHAR_INDEX_OTHER) as f32);
            x_type.push(*CHAR_TYPE_MAP.get(&ch).unwrap_or(&CHAR_TYPE_INDEX_OTHER) as f32);
        }
        // Current character
        x_char.push(*CHARS_MAP.get(&padded[i]).unwrap_or(&CHAR_INDEX_OTHER) as f32);
        x_type.push(*CHAR_TYPE_MAP.get(&padded[i]).unwrap_or(&CHAR_TYPE_INDEX_OTHER) as f32);
    }

    (x_char, x_type)
}

/// Tokenize Thai text using the bundled deepcut ONNX model.
///
/// Returns `Err` if ONNX inference fails.  Returns an empty `Vec` for empty
/// or whitespace-only input.
pub fn tokenize(text: &str) -> anyhow::Result<Vec<String>> {
    if text.is_empty() {
        return Ok(vec![]);
    }

    let text_chars: Vec<char> = text.chars().collect();
    let n = text_chars.len();

    let (x_char_flat, x_type_flat) = build_features(&text_chars);

    let x_char =
        tract_ndarray::Array2::<f32>::from_shape_vec((n, N_PAD), x_char_flat)
            .map_err(|e| anyhow::anyhow!("deepcut: failed to build x_char array: {}", e))?;
    let x_type =
        tract_ndarray::Array2::<f32>::from_shape_vec((n, N_PAD), x_type_flat)
            .map_err(|e| anyhow::anyhow!("deepcut: failed to build x_type array: {}", e))?;

    let outputs = MODEL
        .run(tvec![
            x_char.into_tensor().into(),
            x_type.into_tensor().into(),
        ])
        .map_err(|e| anyhow::anyhow!("deepcut: ONNX inference failed: {}", e))?;

    let probs: Vec<f32> = outputs[0]
        .to_array_view::<f32>()
        .map_err(|e| anyhow::anyhow!("deepcut: failed to extract output: {}", e))?
        .iter()
        .copied()
        .collect();

    // Mirror the Python logic:
    //   y_predict = (probs > threshold).astype(int)
    //   word_end  = y_predict[1:].tolist() + [1]
    let word_end: Vec<bool> = probs[1..]
        .iter()
        .map(|&p| p > WORD_BOUNDARY_THRESHOLD)
        .chain(std::iter::once(true))
        .collect();

    let mut tokens = Vec::new();
    let mut current = String::new();
    for (&ch, &is_end) in text_chars.iter().zip(word_end.iter()) {
        current.push(ch);
        if is_end {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }

    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        assert_eq!(tokenize("").unwrap(), Vec::<String>::new());
    }

    #[test]
    fn test_basic_thai() {
        let tokens = tokenize("ทดสอบการตัดคำ").unwrap();
        assert!(!tokens.is_empty());
        assert_eq!(tokens.join(""), "ทดสอบการตัดคำ");
    }

    #[test]
    fn test_mixed_thai_latin() {
        let text = "หมอนทองตากลมหูว์MBK39";
        let tokens = tokenize(text).unwrap();
        assert_eq!(tokens.join(""), text);
    }
}
