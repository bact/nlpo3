// SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
// SPDX-License-Identifier: Apache-2.0

use std::sync::{LazyLock, Mutex, OnceLock};

use ahash::AHashMap as HashMap;
use neon::prelude::*;
use nlpo3::tokenizer::{
    newmm::{NewmmFstTokenizer, NewmmTokenizer},
    tokenizer_trait::Tokenizer,
};

#[cfg(feature = "deepcut")]
use nlpo3::tokenizer::deepcut::DeepcutTokenizer;

/// Global tokenizer registry: maps dictionary name to a loaded tokenizer.
static TOKENIZER_COLLECTION: LazyLock<
    Mutex<HashMap<String, Box<dyn Tokenizer + Send + Sync>>>,
> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// Lazily-initialized deepcut singleton.
#[cfg(feature = "deepcut")]
static DEEPCUT: OnceLock<DeepcutTokenizer> = OnceLock::new();

// Load a dictionary file to a tokenizer and add it to the registry.
//
// Arguments:
//   0: file_path (string) — absolute path to a one-word-per-line dictionary.
//   1: dict_name (string) — registry key for the loaded tokenizer.
//   2: tokenizer (string, optional) — "newmm" (default) or "nf".
//
// Returns a result string.
fn load_dict(mut cx: FunctionContext) -> JsResult<JsString> {
    let file_path = cx.argument::<JsString>(0)?.value(&mut cx);
    let dict_name = cx.argument::<JsString>(1)?.value(&mut cx);
    let tokenizer_kind = cx
        .argument_opt(2)
        .and_then(|v| v.downcast::<JsString, _>(&mut cx).ok())
        .map(|s| s.value(&mut cx))
        .unwrap_or_else(|| "newmm".to_string());

    let mut col = TOKENIZER_COLLECTION.lock().unwrap();
    if col.contains_key(&dict_name) {
        return Ok(cx.string(format!(
            "error: dictionary '{}' already exists",
            dict_name
        )));
    }

    let tokenizer: Box<dyn Tokenizer + Send + Sync> = match tokenizer_kind.as_str() {
        "newmm" => Box::new(NewmmTokenizer::new(&file_path)),
        "nf" => Box::new(NewmmFstTokenizer::new(&file_path)),
        other => {
            return cx.throw_error(format!("error: unknown tokenizer '{}'", other));
        }
    };

    col.insert(dict_name.clone(), tokenizer);
    Ok(cx.string(format!(
        "ok: loaded '{}' (tokenizer: {})",
        dict_name, tokenizer_kind
    )))
}

// Break text into tokens using a previously loaded dictionary tokenizer.
//
// Arguments:
//   0: text (string)
//   1: dict_name (string) — registry key of the tokenizer to use.
//   2: safe (boolean) — enable safe mode for ambiguous inputs.
//   3: parallel (boolean) — enable parallel processing.
//
// Returns an array of token strings.
fn segment(mut cx: FunctionContext) -> JsResult<JsArray> {
    let text = cx.argument::<JsString>(0)?.value(&mut cx);
    let dict_name = cx.argument::<JsString>(1)?.value(&mut cx);
    let safe = cx.argument::<JsBoolean>(2)?.value(&mut cx);
    let parallel = cx.argument::<JsBoolean>(3)?.value(&mut cx);

    let col = TOKENIZER_COLLECTION.lock().unwrap();
    let tokenizer = match col.get(&dict_name) {
        Some(t) => t,
        None => {
            return cx
                .throw_error(format!("error: dictionary '{}' not found", dict_name));
        }
    };

    let result = tokenizer.segment_to_string(&text, safe, parallel);
    let js_array = JsArray::new(&mut cx, result.len());
    for (i, s) in result.iter().enumerate() {
        let js_str = cx.string(s);
        js_array.set(&mut cx, i as u32, js_str).unwrap();
    }
    Ok(js_array)
}

// Tokenize text using the deepcut neural CNN model.
//
// Arguments:
//   0: text (string)
//
// Returns an array of token strings.
#[cfg(feature = "deepcut")]
fn segment_deepcut(mut cx: FunctionContext) -> JsResult<JsArray> {
    let text = cx.argument::<JsString>(0)?.value(&mut cx);

    let tokenizer = DEEPCUT.get_or_init(|| {
        DeepcutTokenizer::new().expect("deepcut: ONNX model initialization failed")
    });

    let result = tokenizer.segment_to_string(&text, false, false);
    let js_array = JsArray::new(&mut cx, result.len());
    for (i, s) in result.iter().enumerate() {
        let js_str = cx.string(s);
        js_array.set(&mut cx, i as u32, js_str).unwrap();
    }
    Ok(js_array)
}

#[neon::main]
fn main(mut cx: ModuleContext) -> NeonResult<()> {
    cx.export_function("loadDict", load_dict)?;
    cx.export_function("segment", segment)?;
    #[cfg(feature = "deepcut")]
    cx.export_function("segmentDeepcut", segment_deepcut)?;
    Ok(())
}
