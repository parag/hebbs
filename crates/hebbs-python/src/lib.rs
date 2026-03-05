#![allow(clippy::too_many_arguments)]
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;

mod convert;
mod engine;
mod error;
mod subscribe;

/// Native extension for the HEBBS Python SDK.
///
/// This module is imported as `hebbs._hebbs_native` and provides
/// the `NativeEngine` class that wraps `hebbs-core::Engine` for
/// embedded (no-server) mode.
#[pymodule]
fn _hebbs_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<engine::NativeEngine>()?;
    m.add_class::<subscribe::NativeSubscription>()?;
    Ok(())
}
