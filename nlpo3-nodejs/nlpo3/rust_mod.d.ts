// SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
// SPDX-License-Identifier: Apache-2.0

/**
 * Opaque handle to a Rust tokenizer instance.
 *
 * Returned by the constructor functions below and consumed by
 * {@link tokenizerSegment}.  Because the underlying tokenizer is read-only
 * after construction, the same handle may be passed to {@link tokenizerSegment}
 * any number of times — the dictionary is never reloaded or copied.
 */
export interface TokenizerHandle {
    readonly __type: unique symbol;
}

/**
 * Create a NewmmTokenizer backed by a TrieChar dictionary.
 *
 * @param dictPath  Path to a one-word-per-line dictionary file.
 */
export function newmmTokenizerNew(dictPath: string): TokenizerHandle;

/**
 * Create a NewmmFstTokenizer backed by a finite-state automaton dictionary.
 *
 * Uses ~49× less memory than {@link newmmTokenizerNew} at the cost of slower
 * per-lookup speed.
 *
 * @param dictPath  Path to a one-word-per-line dictionary file.
 */
export function newmmFstTokenizerNew(dictPath: string): TokenizerHandle;

/**
 * Create a DeepcutTokenizer using the bundled ONNX model.
 *
 * The model is compiled on construction.  Internally the model is
 * reference-counted, so cloning the handle is O(1).
 *
 * @throws Error if the ONNX model cannot be loaded.
 */
export function deepcutTokenizerNew(): TokenizerHandle;

/**
 * Tokenize `text` using a previously created tokenizer handle.
 *
 * The tokenizer is read-only — reusing the same handle for every call is both
 * safe and efficient (the dictionary is loaded only once).
 *
 * @param handle    Handle returned by one of the constructor functions.
 * @param text      Input text to tokenize.
 * @param safe      Enable safe mode (avoids long run times on ambiguous input).
 *                  Ignored by DeepcutTokenizer.
 * @param parallel  Enable parallel processing (uses more memory).
 *                  Ignored by DeepcutTokenizer.
 * @returns         Array of word tokens.
 */
export function tokenizerSegment(
    handle: TokenizerHandle,
    text: string,
    safe: boolean,
    parallel: boolean,
): string[];
