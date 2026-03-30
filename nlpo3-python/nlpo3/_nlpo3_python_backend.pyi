# SPDX-FileCopyrightText: 2026 PyThaiNLP Project
# SPDX-License-Identifier: Apache-2.0

"""Type stubs for _nlpo3_python_backend Rust extension module."""

from typing import List, Optional, Tuple

class DeepCutTokenizer:
    """Deepcut CNN-based Thai word tokenizer (Rust backend).

    Each instance compiles and owns the ONNX model.  Cloning is cheap
    (the compiled model is reference-counted).  The same instance is safe
    to call from multiple threads simultaneously.

    For distributed or parallel workloads, create one instance per worker
    process to avoid sharing state across process boundaries.

    Example::

        from nlpo3 import DeepCutTokenizer

        # Bundled model
        tokenizer = DeepCutTokenizer()
        tokens = tokenizer.segment("ทดสอบการตัดคำ")

        # Custom model from disk
        tokenizer = DeepCutTokenizer(model_path="/path/to/custom.onnx")
    """

    def __new__(cls, model_path: Optional[str] = None) -> "DeepCutTokenizer": ...
    def segment(self, text: str) -> List[str]:
        """Break text into tokens using the deepcut CNN model.

        Thread-safe: multiple threads may call this on the same instance.

        Args:
            text: Input text to segment

        Returns:
            List of tokens
        """
        ...

def load_dict(file_path: str, dict_name: str) -> Tuple[str, bool]:
    """Load a dictionary file to a tokenizer.

    Load a dictionary file into an in-memory dictionary collection,
    and assign dict_name to it. This function does not override an
    existing dict name.

    Args:
        file_path: Path to a dictionary file (one word per line)
        dict_name: A unique dictionary name, used for reference

    Returns:
        A tuple of (human_readable_result_str, success_bool)
    """
    ...

def segment(
    text: str,
    dict_name: str,
    safe: bool = False,
    parallel: bool = False,
) -> List[str]:
    """Break text into tokens using newmm algorithm.

    Args:
        text: Input text to segment
        dict_name: Dictionary name, as assigned in load_dict()
        safe: Use safe mode to avoid long waiting time in a text with
              lots of ambiguous word boundaries (default: False)
        parallel: Use multithread mode (default: False)

    Returns:
        List of tokens

    Raises:
        RuntimeError: If dictionary name does not exist
    """
    ...

def segment_deepcut(text: str) -> List[str]:
    """Break text into tokens using the deepcut CNN model.

    Uses a process-level lazy singleton for the bundled model.  The
    singleton is initialised once on first call; a model-load failure
    raises RuntimeError instead of panicking.  For distributed or
    parallel workloads, use DeepCutTokenizer directly.

    Args:
        text: Input text to segment

    Returns:
        List of tokens

    Raises:
        RuntimeError: If model load or ONNX inference fails
    """
    ...
