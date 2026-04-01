// SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
// SPDX-License-Identifier: Apache-2.0

import * as native from "./rust_mod";

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/**
 * Options for dictionary-based tokenizer segment calls.
 *
 * Both flags are accepted for interface consistency but are ignored by
 * {@link DeepcutTokenizer}, which uses a fixed neural-network inference path.
 */
export interface SegmentOptions {
    /**
     * Enable safe mode to avoid long run times on inputs with many ambiguous
     * word boundaries.
     * Default: `false`.
     */
    safe?: boolean;
    /**
     * Enable parallel (multi-threaded) processing.  Uses more memory; benefits
     * long texts on multi-core hosts.
     * Default: `false`.
     */
    parallel?: boolean;
}

// ---------------------------------------------------------------------------
// NewmmTokenizer
// ---------------------------------------------------------------------------

/**
 * Dictionary-based maximal-matching Thai word tokenizer (TrieChar backend).
 *
 * Create **one instance** and reuse it for every segment call — the dictionary
 * is loaded only once and the same instance is safe to use for all concurrent
 * async operations on the Node.js event loop.
 *
 * For Node.js worker threads, create one instance per worker; there is no
 * shared-memory mechanism for passing a tokenizer handle across thread
 * boundaries.
 *
 * @example
 * ```typescript
 * import { NewmmTokenizer } from "nlpo3";
 *
 * const tok = new NewmmTokenizer("/path/to/dict.txt");
 *
 * // Reuse tok for every request — the dictionary is never reloaded.
 * const tokens = tok.segment("สวัสดีครับ");
 * const tokensParallel = tok.segment("สวัสดีครับ", { parallel: true });
 * ```
 */
export class NewmmTokenizer {
    private readonly _handle: native.TokenizerHandle;

    /**
     * @param dictPath  Path to a one-word-per-line dictionary file.
     */
    constructor(dictPath: string) {
        this._handle = native.newmmTokenizerNew(dictPath);
    }

    /**
     * Tokenize `text` into words.
     *
     * @param text     Input text.
     * @param options  Optional flags: `safe` and `parallel`.
     * @returns        Array of word tokens.
     */
    segment(text: string, options: SegmentOptions = {}): string[] {
        const { safe = false, parallel = false } = options;
        return native.tokenizerSegment(this._handle, text, safe, parallel);
    }
}

// ---------------------------------------------------------------------------
// NewmmFstTokenizer
// ---------------------------------------------------------------------------

/**
 * Dictionary-based maximal-matching Thai word tokenizer (FST backend).
 *
 * Uses the same algorithm as {@link NewmmTokenizer} but stores the dictionary
 * in a minimized finite-state automaton, reducing memory use by ~49× at the
 * cost of slower per-lookup speed.
 *
 * Create **one instance** and reuse it for every segment call.
 *
 * @example
 * ```typescript
 * import { NewmmFstTokenizer } from "nlpo3";
 *
 * const tok = new NewmmFstTokenizer("/path/to/dict.txt");
 * const tokens = tok.segment("สวัสดีครับ");
 * ```
 */
export class NewmmFstTokenizer {
    private readonly _handle: native.TokenizerHandle;

    /**
     * @param dictPath  Path to a one-word-per-line dictionary file.
     */
    constructor(dictPath: string) {
        this._handle = native.newmmFstTokenizerNew(dictPath);
    }

    /**
     * Tokenize `text` into words.
     *
     * @param text     Input text.
     * @param options  Optional flags: `safe` and `parallel`.
     * @returns        Array of word tokens.
     */
    segment(text: string, options: SegmentOptions = {}): string[] {
        const { safe = false, parallel = false } = options;
        return native.tokenizerSegment(this._handle, text, safe, parallel);
    }
}

// ---------------------------------------------------------------------------
// DeepcutTokenizer
// ---------------------------------------------------------------------------

/**
 * Neural CNN-based Thai word tokenizer (deepcut).
 *
 * Uses a bundled ONNX model for inference.  The model is compiled on
 * construction.  Internally the model is reference-counted, so construction
 * after the first time is O(1).
 *
 * Create **one instance** and reuse it for every segment call — the model is
 * compiled only once.
 *
 * @example
 * ```typescript
 * import { DeepcutTokenizer } from "nlpo3";
 *
 * const tok = new DeepcutTokenizer();
 * const tokens = tok.segment("สวัสดีครับ");
 * ```
 */
export class DeepcutTokenizer {
    private readonly _handle: native.TokenizerHandle;

    constructor() {
        this._handle = native.deepcutTokenizerNew();
    }

    /**
     * Tokenize `text` using the deepcut CNN model.
     *
     * @param text  Input text.
     * @returns     Array of word tokens.
     */
    segment(text: string): string[] {
        return native.tokenizerSegment(this._handle, text, false, false);
    }
}
