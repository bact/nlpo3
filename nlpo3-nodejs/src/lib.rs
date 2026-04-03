// SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
// SPDX-License-Identifier: Apache-2.0

use neon::{prelude::*, types::Finalize};
use nlpo3::tokenizer::newmm::{NewmmFstTokenizer, NewmmTokenizer};

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
enum TokenizerInner {
    Newmm(NewmmTokenizer),
    NewmmFst(NewmmFstTokenizer),
    #[cfg(feature = "deepcut")]
    Deepcut(DeepcutTokenizer),
}

struct TokenizerWrapper {
    inner: TokenizerInner,
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
        inner: TokenizerInner::Newmm(tok),
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
        inner: TokenizerInner::NewmmFst(tok),
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
        inner: TokenizerInner::Deepcut(tok),
    }))
}

// ---------------------------------------------------------------------------
// Segment: NewmmTokenizer
//
// Args:
//   0: handle               (JsBox<TokenizerWrapper>)
//   1: text                 (string)
//   2: safe                 (boolean)
//   3: parallel_chunk_size  (number | null | undefined)
// Returns:
//   string[]
// ---------------------------------------------------------------------------
fn newmm_tokenizer_segment(mut cx: FunctionContext) -> JsResult<JsArray> {
    let wrapper = cx.argument::<JsBox<TokenizerWrapper>>(0)?;
    let text = cx.argument::<JsString>(1)?.value(&mut cx);
    let safe = cx.argument::<JsBoolean>(2)?.value(&mut cx);
    let parallel_chunk_size = if cx.argument::<JsValue>(3)?.is_a::<JsNull, _>(&mut cx)
        || cx.argument::<JsValue>(3)?.is_a::<JsUndefined, _>(&mut cx)
    {
        None
    } else {
        let value = cx.argument::<JsNumber>(3)?.value(&mut cx);
        if value < 0.0 {
            return cx.throw_error("parallelChunkSize must be non-negative");
        }
        Some(value as usize)
    };

    let tok = match &wrapper.inner {
        TokenizerInner::Newmm(tok) => tok,
        _ => return cx.throw_error("invalid tokenizer handle for NewmmTokenizer"),
    };

    let segments = match tok.segment_with_options(&text, safe, parallel_chunk_size) {
        Ok(tokens) => tokens,
        Err(e) => {
            return cx.throw_error(format!("tokenizer: failed to segment input: {}", e));
        }
    };

    let js_array = JsArray::new(&mut cx, segments.len());
    for (i, s) in segments.iter().enumerate() {
        let js_str = cx.string(s);
        js_array.set(&mut cx, i as u32, js_str)?;
    }
    Ok(js_array)
}

// ---------------------------------------------------------------------------
// Segment: NewmmFstTokenizer
//
// Args:
//   0: handle               (JsBox<TokenizerWrapper>)
//   1: text                 (string)
//   2: safe                 (boolean)
//   3: parallel_chunk_size  (number | null | undefined)
// Returns:
//   string[]
// ---------------------------------------------------------------------------
fn newmm_fst_tokenizer_segment(mut cx: FunctionContext) -> JsResult<JsArray> {
    let wrapper = cx.argument::<JsBox<TokenizerWrapper>>(0)?;
    let text = cx.argument::<JsString>(1)?.value(&mut cx);
    let safe = cx.argument::<JsBoolean>(2)?.value(&mut cx);
    let parallel_chunk_size = if cx.argument::<JsValue>(3)?.is_a::<JsNull, _>(&mut cx)
        || cx.argument::<JsValue>(3)?.is_a::<JsUndefined, _>(&mut cx)
    {
        None
    } else {
        let value = cx.argument::<JsNumber>(3)?.value(&mut cx);
        if value < 0.0 {
            return cx.throw_error("parallelChunkSize must be non-negative");
        }
        Some(value as usize)
    };

    let tok = match &wrapper.inner {
        TokenizerInner::NewmmFst(tok) => tok,
        _ => return cx.throw_error("invalid tokenizer handle for NewmmFstTokenizer"),
    };

    let segments = match tok.segment_with_options(&text, safe, parallel_chunk_size) {
        Ok(tokens) => tokens,
        Err(e) => {
            return cx.throw_error(format!("tokenizer: failed to segment input: {}", e));
        }
    };

    let js_array = JsArray::new(&mut cx, segments.len());
    for (i, s) in segments.iter().enumerate() {
        let js_str = cx.string(s);
        js_array.set(&mut cx, i as u32, js_str)?;
    }
    Ok(js_array)
}

// ---------------------------------------------------------------------------
// Segment: DeepcutTokenizer
//
// Args:
//   0: handle               (JsBox<TokenizerWrapper>)
//   1: text                 (string)
//   2: parallel_chunk_size  (number | null | undefined)
// Returns:
//   string[]
// ---------------------------------------------------------------------------
fn deepcut_tokenizer_segment(mut cx: FunctionContext) -> JsResult<JsArray> {
    let wrapper = cx.argument::<JsBox<TokenizerWrapper>>(0)?;
    let text = cx.argument::<JsString>(1)?.value(&mut cx);
    let parallel_chunk_size = if cx.argument::<JsValue>(2)?.is_a::<JsNull, _>(&mut cx)
        || cx.argument::<JsValue>(2)?.is_a::<JsUndefined, _>(&mut cx)
    {
        None
    } else {
        let value = cx.argument::<JsNumber>(2)?.value(&mut cx);
        if value < 0.0 {
            return cx.throw_error("parallelChunkSize must be non-negative");
        }
        Some(value as usize)
    };

    let tok = match &wrapper.inner {
        #[cfg(feature = "deepcut")]
        TokenizerInner::Deepcut(tok) => tok,
        _ => return cx.throw_error("invalid tokenizer handle for DeepcutTokenizer"),
    };

    let segments = match tok.segment_with_options(&text, parallel_chunk_size) {
        Ok(tokens) => tokens,
        Err(e) => {
            return cx.throw_error(format!("tokenizer: failed to segment input: {}", e));
        }
    };

    let js_array = JsArray::new(&mut cx, segments.len());
    for (i, s) in segments.iter().enumerate() {
        let js_str = cx.string(s);
        js_array.set(&mut cx, i as u32, js_str)?;
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
    cx.export_function("newmmTokenizerSegment", newmm_tokenizer_segment)?;
    cx.export_function("newmmFstTokenizerSegment", newmm_fst_tokenizer_segment)?;
    #[cfg(feature = "deepcut")]
    cx.export_function("deepcutTokenizerSegment", deepcut_tokenizer_segment)?;
    Ok(())
}
