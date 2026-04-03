// SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
// SPDX-License-Identifier: Apache-2.0

/**
 * Opaque handle to a Rust tokenizer instance.
 *
 * Returned by the constructor functions below and consumed by
 * tokenizer-specific segment functions. Because the underlying tokenizer is read-only
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
 * Tokenize `text` using a NewmmTokenizer handle.
 *
 * The tokenizer is read-only — reusing the same handle for every call is both
 * safe and efficient (the dictionary is loaded only once).
 *
 * @param handle    Handle returned by {@link newmmTokenizerNew}.
 * @param text      Input text to tokenize.
 * @param safe      Enable safe mode (avoids long run times on ambiguous input).
 * @param parallelChunkSize Target chunk size in bytes for parallel mode.
 *                  `null`, `undefined`, `0`, or values below minimum disable parallel mode.
 * @returns         Array of word tokens.
 */
export function newmmTokenizerSegment(
    handle: TokenizerHandle,
    text: string,
    safe: boolean,
    parallelChunkSize?: number | null,
): string[];

/**
 * Tokenize `text` using a NewmmFstTokenizer handle.
 *
 * @param handle    Handle returned by {@link newmmFstTokenizerNew}.
 * @param text      Input text to tokenize.
 * @param safe      Enable safe mode (avoids long run times on ambiguous input).
 * @param parallelChunkSize Target chunk size in bytes for parallel mode.
 *                  `null`, `undefined`, `0`, or values below minimum disable parallel mode.
 * @returns         Array of word tokens.
 */
export function newmmFstTokenizerSegment(
    handle: TokenizerHandle,
    text: string,
    safe: boolean,
    parallelChunkSize?: number | null,
): string[];

/**
 * Tokenize `text` using a DeepcutTokenizer handle.
 *
 * @param handle    Handle returned by {@link deepcutTokenizerNew}.
 * @param text      Input text to tokenize.
 * @param parallelChunkSize Target chunk size in bytes for parallel mode.
 *                  `null`, `undefined`, `0`, or values below minimum disable parallel mode.
 * @returns         Array of word tokens.
 */
export function deepcutTokenizerSegment(
    handle: TokenizerHandle,
    text: string,
    parallelChunkSize?: number | null,
): string[];
