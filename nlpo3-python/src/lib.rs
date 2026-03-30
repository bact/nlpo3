// SPDX-FileCopyrightText: 2024 PyThaiNLP Project
// SPDX-License-Identifier: Apache-2.0

/**
 * Python-binding for nlpO3, an natural language process library.
 *
 * Provides a tokenizer.
 *
 * Authors:
 * Thanathip Suntorntip
 * Arthit Suriyawongkul
 */
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use ahash::AHashMap as HashMap;
use lazy_static::lazy_static;
use nlpo3::tokenizer::deepcut::DeepCutTokenizer;
use nlpo3::tokenizer::newmm::NewmmTokenizer;
use nlpo3::tokenizer::tokenizer_trait::Tokenizer;
use pyo3::prelude::*;
use pyo3::types::PyString;
use pyo3::{exceptions, wrap_pyfunction};

lazy_static! {
    static ref TOKENIZER_COLLECTION: Mutex<HashMap<String, Box<NewmmTokenizer>>> =
        Mutex::new(HashMap::new());
}

/// Process-level lazy singleton used by the `segment_deepcut()` convenience
/// function.  Stored as `Result` so that a model-load failure yields a Python
/// `RuntimeError` rather than a panic.
static DEEPCUT_SINGLETON: OnceLock<Result<DeepCutTokenizer, String>> = OnceLock::new();

fn get_deepcut_singleton() -> PyResult<&'static DeepCutTokenizer> {
    DEEPCUT_SINGLETON
        .get_or_init(|| DeepCutTokenizer::new().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| {
            exceptions::PyRuntimeError::new_err(format!("deepcut: model load failed: {}", e))
        })
}

/// Load a dictionary file to a tokenizer,
/// and add that tokenizer to the tokenizer collection.
///
/// Dictionary file must one word per line.
/// If successful, will insert a NewmmTokenizer to TOKENIZER_COLLECTION.
/// returns a tuple of string of loading result and a boolean
///
/// signature: (file_path: str, dict_name: str) -> (str, boolean)
#[pyfunction]
#[pyo3(signature = (file_path, dict_name))]
fn load_dict(file_path: &str, dict_name: &str) -> PyResult<(String, bool)> {
    let mut tokenizer_col_lock = TOKENIZER_COLLECTION.lock().unwrap();
    if tokenizer_col_lock.get(dict_name).is_some() {
        Ok((
            format!(
                "Failed: dictionary name {} already exists, please use another name.",
                dict_name
            ),
            false,
        ))
    } else {
        let tokenizer = NewmmTokenizer::new(file_path);
        tokenizer_col_lock.insert(dict_name.to_owned(), Box::new(tokenizer));

        Ok((
            format!(
                "Successful: file {} has been successfully loaded to dictionary name {}.",
                file_path, dict_name
            ),
            true,
        ))
    }
}

/// Break text into tokens.
/// Use newmm algorithm.
/// Can use multithreading, but takes a lot of memory.
/// returns list of valid utf-8 bytes list
///
/// signature: (text: str, dict_name: str, safe: boolean = false, parallel: boolean = false) -> List[List[u8]]
///
#[pyfunction]
#[pyo3(signature = (text, dict_name, safe=false, parallel=false))]
fn segment(
    text: &Bound<'_, PyString>,
    dict_name: &str,
    safe: bool,
    parallel: bool,
) -> PyResult<Vec<String>> {
    if let Some(loaded_tokenizer) = TOKENIZER_COLLECTION.lock().unwrap().get(dict_name) {
        let result = loaded_tokenizer.segment_to_string(text.to_str()?, safe, parallel);
        Ok(result)
    } else {
        Err(exceptions::PyRuntimeError::new_err(format!(
            "Dictionary name {} does not exist.",
            dict_name
        )))
    }
}

/*
/// Add words to existing dictionary
#[pyfunction]
fn add_word(dict_name: &str, words: Vec<&str>) -> PyResult<(String, bool)> {
    let mut tokenizer_col_lock = TOKENIZER_COLLECTION.lock().unwrap();
    if let Some(newmm_dict) = tokenizer_col_lock.get(dict_name) {
        newmm_dict.add_word(&words);
        Ok((format!("Add new word(s) successfully."), true))
    } else {
        Ok((
            format!(
                "Cannot add new word(s) - dictionary instance named '{}' does not exist.",
                dict_name
            ),
            false,
        ))
    }
}

/// Remove words from existing dictionary
#[pyfunction]
fn remove_word(dict_name: &str, words: Vec<&str>) -> PyResult<(String, bool)> {
    let mut tokenizer_col_lock = TOKENIZER_COLLECTION.lock().unwrap();
    if let Some(newmm_dict) = tokenizer_col_lock.get(dict_name) {
        newmm_dict.remove_word(&words);
        Ok((format!("Remove word(s) successfully."), true))
    } else {
        Ok((
            format!(
                "Cannot remove word(s) - dictionary instance named '{}' does not exist.",
                dict_name
            ),
            false,
        ))
    }
}
*/

// ---------------------------------------------------------------------------
// Deepcut Python class
// ---------------------------------------------------------------------------

/// Deepcut CNN-based Thai word tokenizer (Python class).
///
/// Each instance compiles and owns the ONNX model.  Cloning is cheap
/// (the compiled model is reference-counted).  The same instance is safe to
/// call from multiple threads simultaneously.
///
/// For distributed or parallel workloads, create one `DeepCutTokenizer` per
/// worker process to avoid sharing state across process boundaries.
///
/// ```python
/// from nlpo3 import DeepCutTokenizer
///
/// tokenizer = DeepCutTokenizer()
/// tokens = tokenizer.segment("ทดสอบการตัดคำ")
///
/// # Load a custom ONNX model from disk
/// tokenizer = DeepCutTokenizer(model_path="/path/to/custom.onnx")
/// ```
#[pyclass(name = "DeepCutTokenizer")]
struct PyDeepCutTokenizer {
    inner: DeepCutTokenizer,
}

#[pymethods]
impl PyDeepCutTokenizer {
    /// Create a new DeepCutTokenizer.
    ///
    /// If ``model_path`` is ``None`` (the default), the bundled ONNX model is
    /// used.  Pass a filesystem path to load a custom compatible model.
    #[new]
    #[pyo3(signature = (model_path=None))]
    fn new(model_path: Option<&str>) -> PyResult<Self> {
        let result = match model_path {
            None => DeepCutTokenizer::new(),
            Some(p) => DeepCutTokenizer::from_path(Path::new(p)),
        };
        result
            .map(|inner| PyDeepCutTokenizer { inner })
            .map_err(|e| {
                exceptions::PyRuntimeError::new_err(format!(
                    "deepcut: failed to create tokenizer: {}",
                    e
                ))
            })
    }

    /// Break text into tokens using the deepcut CNN model.
    ///
    /// Thread-safe: multiple threads may call this on the same instance.
    fn segment(&self, text: &str) -> PyResult<Vec<String>> {
        if text.is_empty() {
            return Ok(vec![]);
        }
        self.inner.tokenize(text).map_err(|e| {
            exceptions::PyRuntimeError::new_err(format!("deepcut inference failed: {}", e))
        })
    }
}

// ---------------------------------------------------------------------------
// Deepcut convenience function
// ---------------------------------------------------------------------------

/// Break text into tokens using the deepcut CNN model (ONNX inference).
///
/// Uses a process-level lazy singleton for the bundled model.  The singleton
/// is initialised once on the first call and is safe for concurrent use.
/// A model-load failure returns a `RuntimeError` instead of panicking.
///
/// For distributed or parallel workloads where each worker should own its
/// model, use `DeepCutTokenizer` directly instead.
///
/// signature: (text: str) -> List[str]
#[pyfunction]
#[pyo3(signature = (text))]
fn segment_deepcut(text: &str) -> PyResult<Vec<String>> {
    if text.is_empty() {
        return Ok(vec![]);
    }
    get_deepcut_singleton()?
        .tokenize(text)
        .map_err(|e| {
            exceptions::PyRuntimeError::new_err(format!("deepcut inference failed: {}", e))
        })
}

#[pymodule]
fn _nlpo3_python_backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_dict, m)?)?;
    m.add_function(wrap_pyfunction!(segment, m)?)?;
    m.add_function(wrap_pyfunction!(segment_deepcut, m)?)?;
    m.add_class::<PyDeepCutTokenizer>()?;
    Ok(())
}
