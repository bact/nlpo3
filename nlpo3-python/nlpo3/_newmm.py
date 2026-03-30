# SPDX-FileCopyrightText: 2024 PyThaiNLP Project
# SPDX-License-Identifier: Apache-2.0

"""Newmm tokenizer — Python wrappers around the Rust backend."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from ._nlpo3_python_backend import load_dict as _rust_load_dict
from ._nlpo3_python_backend import segment as _rust_segment

__all__ = ["load_dict", "segment"]


def load_dict(file_path: str, dict_name: str) -> Tuple[str, bool]:
    """Load dictionary from a file.

    Load a dictionary file into an in-memory dictionary collection,
    and assign dict_name to it.
    This function does not override an existing dict name.

    :param file_path: Path to a dictionary file
    :type file_path: str
    :param dict_name: A unique dictionary name, use for reference.
    :type dict_name: str
    :return: Tuple of (human_readable_result_str, success_bool)
    :rtype: tuple[str, bool]
    """
    path = Path(file_path).resolve()
    return _rust_load_dict(str(path), dict_name)


def segment(
    text: Optional[str],
    dict_name: str,
    safe: bool = False,
    parallel: bool = False,
) -> List[str]:
    """Break text into tokens using the newmm algorithm.

    Supports multithread mode via the parallel flag.

    :param text: Input text. Returns an empty list for ``None`` or empty input.
    :type text: str, optional
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
    return _rust_segment(text, dict_name, safe, parallel)
