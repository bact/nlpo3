// SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result as AnyResult;

/// Shared interface for all Thai word tokenizers.
///
/// The `segment` and `segment_to_string` methods are read-only operations and
/// are safe to call from multiple threads concurrently.  The `Send + Sync`
/// supertraits are required so that `Box<dyn Tokenizer>` and
/// `Arc<dyn Tokenizer>` can be shared across threads without additional
/// bounds at call sites.  Implementations may expose additional mutation
/// methods (for example, to add or remove words) that require `&mut self`
/// and may use copy-on-write internally (such as cloning an underlying
/// dictionary).
///
/// Each concrete tokenizer implementation may support additional options
/// via specialized methods. See the concrete type's documentation for details.
pub trait Tokenizer: Send + Sync {
    /// Segment text into words.
    fn segment(&self, text: &str) -> AnyResult<Vec<String>>;

    /// Segment text into words, returning a `Result<Vec<String>, E>`.
    ///
    /// This is a fallible entry point equivalent to [`Tokenizer::segment`].
    /// It is kept for API naming parity across bindings and tokenizers.
    /// Callers decide whether to propagate, recover, or panic.
    fn segment_to_string(&self, text: &str) -> AnyResult<Vec<String>>;
}
