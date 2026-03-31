# SPDX-FileCopyrightText: 2024-2026 PyThaiNLP Project
# SPDX-License-Identifier: Apache-2.0

# Python binding for nlpO3, a natural language processing library.
#
# Authors:
# Thanathip Suntorntip
# Arthit Suriyawongkul

from __future__ import annotations

from ._deepcut import DeepcutTokenizer, segment_deepcut
from ._newmm import load_dict, segment

__all__ = ["DeepcutTokenizer", "load_dict", "segment", "segment_deepcut"]

# TODO: load_dict from in-memory list of words
