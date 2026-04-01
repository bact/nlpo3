// SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result as AnyResult;

/// Shared interface for all Thai word tokenizers.
///
/// All implementations are **read-only after construction** and safe to call
/// from multiple threads concurrently.  The `Send + Sync` supertraits are
/// required so that `Box<dyn Tokenizer>` and `Arc<dyn Tokenizer>` can be
/// shared across threads without additional bounds at call sites.
pub trait Tokenizer: Send + Sync {
    fn segment(&self, text: &str, safe: bool, parallel: bool) -> AnyResult<Vec<String>>;

    fn segment_to_string(&self, text: &str, safe: bool, parallel: bool) -> Vec<String>;
}
