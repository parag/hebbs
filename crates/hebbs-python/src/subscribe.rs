use std::sync::Mutex;
use std::time::Duration;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use hebbs_core::subscribe::{SubscribePush, SubscriptionHandle};

use crate::convert::memory_to_py;
use crate::error::to_py_err;

/// Python wrapper for a HEBBS subscription.
///
/// Provides `feed()`, `poll()`, `close()`, and `stats()` methods.
/// The underlying `SubscriptionHandle` runs on a dedicated Rust thread.
#[pyclass]
pub struct NativeSubscription {
    handle: Mutex<Option<SubscriptionHandle>>,
}

impl NativeSubscription {
    pub fn new(handle: SubscriptionHandle) -> Self {
        Self {
            handle: Mutex::new(Some(handle)),
        }
    }
}

#[pymethods]
impl NativeSubscription {
    /// Feed text to the subscription for matching.
    fn feed(&self, py: Python<'_>, text: &str) -> PyResult<()> {
        let guard = self
            .handle
            .lock()
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("subscription lock poisoned"))?;
        let handle = guard
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("subscription is closed"))?;
        py.allow_threads(|| handle.feed(text).map_err(to_py_err))
    }

    /// Poll for the next push (non-blocking).
    /// Returns a dict or None.
    fn poll(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let guard = self
            .handle
            .lock()
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("subscription lock poisoned"))?;
        let handle = guard
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("subscription is closed"))?;
        match handle.try_recv() {
            Some(push) => Ok(Some(push_to_py(py, &push)?)),
            None => Ok(None),
        }
    }

    /// Blocking poll with timeout in seconds.
    /// Returns a dict or None if timeout expires.
    fn poll_timeout(&self, py: Python<'_>, timeout_secs: f64) -> PyResult<Option<PyObject>> {
        let guard = self
            .handle
            .lock()
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("subscription lock poisoned"))?;
        let handle = guard
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("subscription is closed"))?;
        let timeout = Duration::from_secs_f64(timeout_secs);
        let result = py.allow_threads(|| handle.recv_timeout(timeout));
        match result {
            Some(push) => Ok(Some(push_to_py(py, &push)?)),
            None => Ok(None),
        }
    }

    /// Get subscription statistics.
    fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let guard = self
            .handle
            .lock()
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("subscription lock poisoned"))?;
        let handle = guard
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("subscription is closed"))?;
        let s = handle.stats();
        let dict = PyDict::new_bound(py);
        dict.set_item("chunks_processed", s.chunks_processed)?;
        dict.set_item("chunks_bloom_rejected", s.chunks_bloom_rejected)?;
        dict.set_item("chunks_coarse_rejected", s.chunks_coarse_rejected)?;
        dict.set_item("pushes_sent", s.pushes_sent)?;
        dict.set_item("pushes_dropped", s.pushes_dropped)?;
        dict.set_item("notification_drops", s.notification_drops)?;
        Ok(dict.into())
    }

    /// Close the subscription and release resources.
    fn close(&self) -> PyResult<()> {
        let mut guard = self
            .handle
            .lock()
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("subscription lock poisoned"))?;
        if let Some(mut handle) = guard.take() {
            handle.close();
        }
        Ok(())
    }
}

fn push_to_py(py: Python<'_>, push: &SubscribePush) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("memory", memory_to_py(py, &push.memory)?)?;
    dict.set_item("confidence", push.confidence)?;
    dict.set_item("push_timestamp_us", push.push_timestamp_us)?;
    Ok(dict.into())
}
