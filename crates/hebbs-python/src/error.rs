use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use hebbs_core::error::HebbsError;

/// Convert a `HebbsError` into a Python exception.
///
/// Maps each HebbsError variant to the most appropriate Python
/// exception type. Structured context (memory_id, operation, etc.)
/// is included in the error message for actionability.
pub fn to_py_err(e: HebbsError) -> PyErr {
    match e {
        HebbsError::InvalidInput { operation, message } => {
            PyValueError::new_err(format!("invalid input for {}: {}", operation, message))
        }
        HebbsError::MemoryNotFound { memory_id } => {
            PyValueError::new_err(format!("memory not found: {}", memory_id))
        }
        HebbsError::Storage(e) => PyRuntimeError::new_err(format!("storage error: {}", e)),
        HebbsError::Embedding(e) => PyRuntimeError::new_err(format!("embedding error: {}", e)),
        HebbsError::Index(e) => PyRuntimeError::new_err(format!("index error: {}", e)),
        HebbsError::Reflect(e) => PyRuntimeError::new_err(format!("reflect error: {}", e)),
        HebbsError::Serialization { message } => {
            PyRuntimeError::new_err(format!("serialization error: {}", message))
        }
        HebbsError::Internal { operation, message } => {
            PyRuntimeError::new_err(format!("internal error in {}: {}", operation, message))
        }
        _ => PyRuntimeError::new_err(format!("hebbs error: {}", e)),
    }
}
