# SPDX-FileCopyrightText: 2024 PyThaiNLP Project
# SPDX-License-Identifier: Apache-2.0

# Python binding for nlpO3, a natural language processing library.
#
# Authors:
# Thanathip Suntorntip
# Arthit Suriyawongkul

from __future__ import annotations

from ._deepcut import DeepCutTokenizer, segment_deepcut
from ._newmm import load_dict, segment

__all__ = ["DeepCutTokenizer", "load_dict", "segment", "segment_deepcut"]

# TODO: load_dict from in-memory list of words
