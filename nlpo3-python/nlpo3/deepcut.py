# SPDX-FileCopyrightText: 2024 PyThaiNLP Project
# SPDX-License-Identifier: Apache-2.0

# Deepcut word tokenizer using ONNX inference.
#
# Based on LEKCut (https://github.com/PyThaiNLP/LEKCut)
# and Deepcut (https://github.com/rkcosmos/deepcut).
#
# Original deepcut authors:
# Rakpong Kittinaradorn, Titipat Achakulvisut, Korakot Chaovavanich,
# Kittinan Srithaworn, Pattarawat Chormai, Chanwit Kaewkasi,
# Tulakan Ruangrong, Krichkorn Oparad.
#
# Original deepcut license: MIT License

from __future__ import annotations

import os
from typing import List, Optional, Tuple

try:
    import numpy as np
    import onnxruntime as ort

    _DEEPCUT_DEPS_AVAILABLE = True
except ImportError:
    _DEEPCUT_DEPS_AVAILABLE = False

_MODULE_DIR = os.path.dirname(__file__)
_DEFAULT_MODEL_PATH = os.path.join(_MODULE_DIR, "model", "deepcut.onnx")

# Character type map (from deepcut)
_CHAR_TYPE: dict = {
    "กขฃคฆงจชซญฎฏฐฑฒณดตถทธนบปพฟภมยรลวศษสฬอ": "c",
    "ฅฉผฟฌหฮ": "n",
    "ะาำิีืึุู": "v",
    "เแโใไ": "w",
    "่้๊๋": "t",
    "์ๆฯ.": "s",
    "0123456789๑๒๓๔๕๖๗๘๙": "d",
    '"': "q",
    "'": "q",
    "\u2018": "q",
    "\u2019": "q",
    " ": "p",
    "abcdefghijklmnopqrstuvwxyz": "s_e",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ": "b_e",
}

_CHAR_TYPE_FLATTEN: dict = {}
for _ks, _v in _CHAR_TYPE.items():
    for _k in _ks:
        _CHAR_TYPE_FLATTEN[_k] = _v

# Character vocabulary (from deepcut)
_CHARS: List[str] = [
    "\n",
    " ",
    "!",
    '"',
    "#",
    "$",
    "%",
    "&",
    "'",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    ";",
    "<",
    "=",
    ">",
    "?",
    "@",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "[",
    "\\",
    "]",
    "^",
    "_",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "other",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "}",
    "~",
    "ก",
    "ข",
    "ฃ",
    "ค",
    "ฅ",
    "ฆ",
    "ง",
    "จ",
    "ฉ",
    "ช",
    "ซ",
    "ฌ",
    "ญ",
    "ฎ",
    "ฏ",
    "ฐ",
    "ฑ",
    "ฒ",
    "ณ",
    "ด",
    "ต",
    "ถ",
    "ท",
    "ธ",
    "น",
    "บ",
    "ป",
    "ผ",
    "ฝ",
    "พ",
    "ฟ",
    "ภ",
    "ม",
    "ย",
    "ร",
    "ฤ",
    "ล",
    "ว",
    "ศ",
    "ษ",
    "ส",
    "ห",
    "ฬ",
    "อ",
    "ฮ",
    "ฯ",
    "ะ",
    "ั",
    "า",
    "ำ",
    "ิ",
    "ี",
    "ึ",
    "ื",
    "ุ",
    "ู",
    "ฺ",
    "เ",
    "แ",
    "โ",
    "ใ",
    "ไ",
    "ๅ",
    "ๆ",
    "็",
    "่",
    "้",
    "๊",
    "๋",
    "์",
    "ํ",
    "๐",
    "๑",
    "๒",
    "๓",
    "๔",
    "๕",
    "๖",
    "๗",
    "๘",
    "๙",
    "\u2018",
    "\u2019",
    "\ufeff",
]
_CHARS_MAP: dict = {v: k for k, v in enumerate(_CHARS)}
# Index of the "other" sentinel in _CHARS
_CHAR_INDEX_OTHER: int = _CHARS_MAP["other"]

_CHAR_TYPES: List[str] = [
    "b_e",
    "c",
    "d",
    "n",
    "o",
    "p",
    "q",
    "s",
    "s_e",
    "t",
    "v",
    "w",
]
_CHAR_TYPES_MAP: dict = {v: k for k, v in enumerate(_CHAR_TYPES)}
# Index of the "o" (other) type in _CHAR_TYPES
_CHAR_TYPE_INDEX_OTHER: int = _CHAR_TYPES_MAP["o"]

_N_PAD = 21
# Threshold for classifying a character as a word boundary
_WORD_BOUNDARY_THRESHOLD = 0.5


def _check_deps() -> None:
    """Raise ImportError if the deepcut optional dependencies are missing."""
    if not _DEEPCUT_DEPS_AVAILABLE:
        raise ImportError(
            "segment_deepcut requires numpy and onnxruntime. "
            "Install them with: pip install nlpo3[deepcut]"
        )


def _create_feature_array(
    text: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create input feature arrays for the deepcut ONNX model.

    :param text: Input text
    :return: Tuple of character and character-type feature arrays
    """
    n = len(text)
    n_pad_2 = (_N_PAD - 1) // 2
    text_pad = [" "] * n_pad_2 + list(text) + [" "] * n_pad_2
    x_char = []
    x_type = []
    for i in range(n_pad_2, n_pad_2 + n):
        char_list = (
            text_pad[i + 1 : i + n_pad_2 + 1]
            + list(reversed(text_pad[i - n_pad_2 : i]))
            + [text_pad[i]]
        )
        char_map = [_CHARS_MAP.get(c, _CHAR_INDEX_OTHER) for c in char_list]
        char_type = [
            _CHAR_TYPES_MAP.get(_CHAR_TYPE_FLATTEN.get(c, "o"), _CHAR_TYPE_INDEX_OTHER)
            for c in char_list
        ]
        x_char.append(char_map)
        x_type.append(char_type)
    return (
        np.array(x_char, dtype=np.float32),
        np.array(x_type, dtype=np.float32),
    )


class _DeepCutTokenizer:
    """Deepcut ONNX-based Thai word tokenizer."""

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        providers: Optional[List[str]] = None,
    ) -> None:
        self.model_path = model_path
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(model_path, providers=providers)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize Thai text using the deepcut ONNX model.

        :param text: Input text
        :return: List of tokens
        """
        if not text:
            return []
        x_char, x_type = _create_feature_array(text)
        outputs = self._session.run(
            None,
            {"input_1": x_char, "input_2": x_type},
        )
        y_predict = (outputs[0].ravel() > _WORD_BOUNDARY_THRESHOLD).astype(int)
        word_end = y_predict[1:].tolist() + [1]
        tokens = []
        word = ""
        for char, w_e in zip(text, word_end):
            word += char
            if w_e:
                tokens.append(word)
                word = ""
        return tokens


_DEFAULT_TOKENIZER: Optional[_DeepCutTokenizer] = None


def segment_deepcut(
    text: str,
    model_path: str = _DEFAULT_MODEL_PATH,
    providers: Optional[List[str]] = None,
) -> List[str]:
    """Break text into tokens using the deepcut ONNX model.

    Uses a deep learning model (CNN) originally from the deepcut library,
    ported to ONNX format via LEKCut.

    :param text: Input text
    :type text: str
    :param model_path: Path to a custom deepcut ONNX model file.
        Uses the bundled default model if not specified.
    :type model_path: str, optional
    :param providers: ONNX Runtime execution providers.
        Defaults to ``["CPUExecutionProvider"]``.
        Pass ``["CUDAExecutionProvider", "CPUExecutionProvider"]``
        for GPU acceleration (requires ``onnxruntime-gpu``).
    :type providers: list[str], optional
    :return: List of tokens
    :rtype: List[str]
    """
    global _DEFAULT_TOKENIZER

    if not text or not isinstance(text, str):
        return []

    _check_deps()

    if providers is None and model_path == _DEFAULT_MODEL_PATH:
        # Use the cached tokenizer for the default model.
        if _DEFAULT_TOKENIZER is None:
            _DEFAULT_TOKENIZER = _DeepCutTokenizer()
        return _DEFAULT_TOKENIZER.tokenize(text)

    return _DeepCutTokenizer(model_path=model_path, providers=providers).tokenize(text)
