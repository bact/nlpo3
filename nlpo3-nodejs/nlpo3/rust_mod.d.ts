// SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
// SPDX-License-Identifier: Apache-2.0

/**
 * Load a dictionary file and register it under `dictName`.
 *
 * @param filePath  Absolute path to a one-word-per-line dictionary file.
 * @param dictName  Registry key used to reference the tokenizer later.
 * @param tokenizer Tokenizer backend: `"newmm"` (default) or `"nf"`.
 * @returns         A result string starting with `"ok:"` on success or
 *                  `"error:"` on failure.
 */
export function loadDict(
    filePath: string,
    dictName: string,
    tokenizer?: "newmm" | "nf",
): string;

/**
 * Tokenize `text` using the tokenizer registered under `dictName`.
 *
 * @param text      Input text to tokenize.
 * @param dictName  Registry key of a previously loaded dictionary tokenizer.
 * @param safe      Enable safe mode (avoids long run times on ambiguous input).
 * @param parallel  Enable parallel processing (uses more memory).
 * @returns         Array of word tokens.
 */
export function segment(
    text: string,
    dictName: string,
    safe: boolean,
    parallel: boolean,
): string[];

/**
 * Tokenize `text` using the deepcut neural CNN model.
 *
 * No dictionary is required. The model is bundled with the package.
 * The model is loaded on first call and reused for subsequent calls.
 *
 * @param text  Input text to tokenize.
 * @returns     Array of word tokens.
 */
export function segmentDeepcut(text: string): string[];
