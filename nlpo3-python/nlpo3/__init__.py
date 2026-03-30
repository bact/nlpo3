# SPDX-FileCopyrightText: 2024 PyThaiNLP Project
# SPDX-License-Identifier: Apache-2.0

# Python-binding for nlpO3, an natural language process library.
#
# Provides a tokenizer.
#
# Authors:
# Thanathip Suntorntip
# Arthit Suriyawongkul

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

# Imports from the compiled Rust extension module.
# pylint: disable=import-error  # _nlpo3_python_backend requires a build step
from ._nlpo3_python_backend import DeepCutTokenizer
from ._nlpo3_python_backend import load_dict as rust_load_dict
from ._nlpo3_python_backend import segment as rust_segment
from ._nlpo3_python_backend import segment_deepcut as rust_segment_deepcut
# pylint: enable=import-error

__all__ = ["DeepCutTokenizer", "load_dict", "segment", "segment_deepcut"]

# TODO: load_dict from in-memory list of words


def load_dict(file_path: str, dict_name: str) -> Tuple[str, bool]:
    """Load dictionary from a file.

    Load a dictionary file into an in-memory dictionary collection,
    and assigned dict_name to it.
    *** This function does not override an existing dict name. ***

    :param file_path: Path to a dictionary file
    :type file_path: str
    :param dict_name: A unique dictionary name, use for reference.
    :type dict_name: str
    :return tuple[human_readable_result_str, bool]
    """
    path = Path(file_path).resolve()

    return rust_load_dict(str(path), dict_name)


def segment(
    text: str,
    dict_name: str,
    safe: bool = False,
    parallel: bool = False,
) -> List[str]:
    """Break text into tokens.

    This method is an implementation of newmm segmentaion.
    Support multithread mode - set by parallel flag.

    :param text: Input text
    :type text: str
    :param dict_name: Dictionary name, as assigned in load_dict()
    :type dict_name: str
    :param safe: Use safe mode to avoid long waiting time in
        a text with lots of ambiguous word boundaries,
        defaults to False
    :type safe: bool, optional
    :param parallel: Use multithread mode, defaults to False
    :type parallel: bool, optional
    :return: List of tokens
    :rtype: List[str]
    """
    if not text or not isinstance(text, str):
        return []

    result = rust_segment(text, dict_name, safe, parallel)

    return result


def segment_deepcut(
    text: str,
    model_path: Optional[str] = None,
    providers: Optional[List[str]] = None,
) -> List[str]:
    """Break text into tokens using the deepcut CNN model.

    Uses a deep learning model (CNN) originally from the deepcut library,
    ported to ONNX format via LEKCut.  The model runs via the Rust
    ONNX engine (tract) with no extra Python dependencies.

    ``providers`` enables custom ONNX Runtime execution providers (e.g.
    GPU).  When ``providers`` is set, the Python onnxruntime path is used
    (requires ``pip install nlpo3[deepcut]``).

    For distributed or parallel workloads, prefer :class:`DeepCutTokenizer`
    so that each worker process owns and controls its own model instance.

    :param text: Input text
    :type text: str
    :param model_path: Path to a custom deepcut ONNX model file.
        Uses the bundled default model if not specified.
    :type model_path: str, optional
    :param providers: ONNX Runtime execution providers.
        When set, the Python/onnxruntime path is used.
        Defaults to ``["CPUExecutionProvider"]``.
        Pass ``["CUDAExecutionProvider", "CPUExecutionProvider"]``
        for GPU acceleration (requires ``onnxruntime-gpu``).
    :type providers: list[str], optional
    :return: List of tokens
    :rtype: List[str]
    """
    if not text or not isinstance(text, str):
        return []

    if providers is not None:
        # Fall back to onnxruntime for custom execution providers (e.g. GPU).
        # Lazy import so numpy/onnxruntime are only loaded when needed.
        from .deepcut import segment_deepcut as _py_segment_deepcut

        kwargs: dict = {}
        if model_path is not None:
            kwargs["model_path"] = model_path
        kwargs["providers"] = providers
        return _py_segment_deepcut(text, **kwargs)

    if model_path is not None:
        # Custom model path: use Rust backend directly.
        return DeepCutTokenizer(model_path=model_path).segment(text)

    return rust_segment_deepcut(text)
