// SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
// SPDX-License-Identifier: Apache-2.0

use neon::{prelude::*, types::Finalize};
use nlpo3::tokenizer::{
    newmm::{NewmmFstTokenizer, NewmmTokenizer},
    tokenizer_trait::Tokenizer,
};

#[cfg(feature = "deepcut")]
use nlpo3::tokenizer::deepcut::DeepcutTokenizer;

// ---------------------------------------------------------------------------
// Opaque wrapper — holds any tokenizer behind a trait object.
//
// A single TokenizerWrapper can serve many concurrent segment() calls because
// segment_to_string() takes &self (immutable read).  Create one wrapper,
// hold its JsBox handle, and reuse it for every call — the dictionary is
// never copied or reloaded.
// ---------------------------------------------------------------------------
struct TokenizerWrapper {
    inner: Box<dyn Tokenizer>,
}

impl Finalize for TokenizerWrapper {}

// ---------------------------------------------------------------------------
// Constructor: NewmmTokenizer
//
// Args:
//   0: dict_path (string) — path to a one-word-per-line dictionary file.
// Returns:
//   JsBox<TokenizerWrapper>
// ---------------------------------------------------------------------------
fn newmm_tokenizer_new(mut cx: FunctionContext) -> JsResult<JsBox<TokenizerWrapper>> {
    let dict_path = cx.argument::<JsString>(0)?.value(&mut cx);
    let tok = NewmmTokenizer::new(&dict_path)
        .or_else(|e| cx.throw_error(format!("error: failed to load dictionary: {}", e)))?;
    Ok(cx.boxed(TokenizerWrapper {
        inner: Box::new(tok),
    }))
}

// ---------------------------------------------------------------------------
// Constructor: NewmmFstTokenizer
//
// Args:
//   0: dict_path (string) — path to a one-word-per-line dictionary file.
// Returns:
//   JsBox<TokenizerWrapper>
// ---------------------------------------------------------------------------
fn newmm_fst_tokenizer_new(mut cx: FunctionContext) -> JsResult<JsBox<TokenizerWrapper>> {
    let dict_path = cx.argument::<JsString>(0)?.value(&mut cx);
    let tok = NewmmFstTokenizer::new(&dict_path)
        .or_else(|e| cx.throw_error(format!("error: failed to load dictionary: {}", e)))?;
    Ok(cx.boxed(TokenizerWrapper {
        inner: Box::new(tok),
    }))
}

// ---------------------------------------------------------------------------
// Constructor: DeepcutTokenizer
//
// No arguments.  Uses the bundled ONNX model.
// Returns:
//   JsBox<TokenizerWrapper>
// ---------------------------------------------------------------------------
#[cfg(feature = "deepcut")]
fn deepcut_tokenizer_new(mut cx: FunctionContext) -> JsResult<JsBox<TokenizerWrapper>> {
    let tok = DeepcutTokenizer::new()
        .or_else(|e| cx.throw_error(format!("deepcut: failed to load model: {}", e)))?;
    Ok(cx.boxed(TokenizerWrapper {
        inner: Box::new(tok),
    }))
}

// ---------------------------------------------------------------------------
// Shared segment function — works with any TokenizerWrapper handle.
//
// Because the tokenizer is read-only after construction, the same handle can
// be reused for every call — no dictionary copying occurs.
//
// Args:
//   0: handle   (JsBox<TokenizerWrapper>) — from one of the constructor fns.
//   1: text     (string)
//   2: safe     (boolean) — enable safe mode (ignored by DeepcutTokenizer).
//   3: parallel (boolean) — enable parallel mode (ignored by DeepcutTokenizer).
// Returns:
//   string[]
// ---------------------------------------------------------------------------
fn tokenizer_segment(mut cx: FunctionContext) -> JsResult<JsArray> {
    let wrapper = cx.argument::<JsBox<TokenizerWrapper>>(0)?;
    let text = cx.argument::<JsString>(1)?.value(&mut cx);
    let safe = cx.argument::<JsBoolean>(2)?.value(&mut cx);
    let parallel = cx.argument::<JsBoolean>(3)?.value(&mut cx);

    let result = wrapper.inner.segment_to_string(&text, safe, parallel);
    let js_array = JsArray::new(&mut cx, result.len());
    for (i, s) in result.iter().enumerate() {
        let js_str = cx.string(s);
        js_array.set(&mut cx, i as u32, js_str).unwrap();
    }
    Ok(js_array)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------
#[neon::main]
fn main(mut cx: ModuleContext) -> NeonResult<()> {
    cx.export_function("newmmTokenizerNew", newmm_tokenizer_new)?;
    cx.export_function("newmmFstTokenizerNew", newmm_fst_tokenizer_new)?;
    #[cfg(feature = "deepcut")]
    cx.export_function("deepcutTokenizerNew", deepcut_tokenizer_new)?;
    cx.export_function("tokenizerSegment", tokenizer_segment)?;
    Ok(())
}
