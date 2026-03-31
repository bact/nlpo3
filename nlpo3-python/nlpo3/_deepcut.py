# SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
# SPDX-License-Identifier: Apache-2.0

"""Deepcut tokenizer — Python wrappers around the Rust backend."""

from __future__ import annotations

from typing import List, Optional

from ._nlpo3_python_backend import DeepcutTokenizer
from ._nlpo3_python_backend import segment_deepcut as _rust_segment_deepcut

__all__ = ["DeepcutTokenizer", "segment_deepcut"]


def segment_deepcut(
    text: Optional[str],
    model_path: Optional[str] = None,
) -> List[str]:
    """Break text into tokens using the deepcut CNN model.

    Uses a deep learning model (CNN) originally from the deepcut library,
    ported to ONNX format via LEKCut.  The model runs via the Rust ONNX
    engine (tract) with no extra Python dependencies.

    For distributed or parallel workloads, prefer :class:`DeepcutTokenizer`
    so that each worker process owns and controls its own model instance.

    :param text: Input text. Returns an empty list for ``None`` or empty input.
    :type text: str, optional
    :param model_path: Path to a custom deepcut ONNX model file.
        Uses the bundled default model if not specified.
    :type model_path: str, optional
    :return: List of tokens
    :rtype: List[str]
    """
    if not text or not isinstance(text, str):
        return []

    if model_path is not None:
        return DeepcutTokenizer(model_path=model_path).segment(text)

    return _rust_segment_deepcut(text)
