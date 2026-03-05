use std::os::raw::c_char;
use std::sync::Arc;

use hebbs_core::engine::Engine;
use hebbs_embed::MockEmbedder;
use hebbs_storage::RocksDbBackend;

use crate::error::{set_last_error, FFI_INVALID_HANDLE, FFI_OK};

/// Internal wrapper for safe access to the engine from raw pointers.
pub(crate) struct HandleRef;

impl HandleRef {
    /// Reconstruct a reference to the engine from a raw handle pointer.
    ///
    /// # Safety
    ///
    /// The pointer must have been returned by `hebbs_open` and not yet
    /// passed to `hebbs_close`.
    pub(crate) unsafe fn from_raw<'a>(handle: *mut libc::c_void) -> Option<&'a Engine> {
        if handle.is_null() {
            set_last_error("null handle");
            return None;
        }
        let arc_ptr = handle as *const Arc<Engine>;
        // SAFETY: caller guarantees handle is valid and not freed
        Some(unsafe { &**arc_ptr })
    }
}

/// Open a HEBBS engine with JSON configuration.
///
/// Configuration JSON structure:
/// ```json
/// {
///   "storage": { "data_dir": "./hebbs-data" },
///   "embedding": { "mock": true, "dimensions": 384 }
/// }
/// ```
///
/// Returns an opaque handle on success, null on failure.
/// The handle is thread-safe and may be used from multiple threads.
///
/// # Safety
///
/// - `config_json` must be a valid UTF-8 pointer with `config_len` bytes,
///   or null for default configuration.
/// - The returned handle must eventually be passed to `hebbs_close`.
#[no_mangle]
pub unsafe extern "C" fn hebbs_open(
    config_json: *const c_char,
    config_len: u32,
) -> *mut libc::c_void {
    let config_str = if config_json.is_null() || config_len == 0 {
        "{}"
    } else {
        match super::ptr_to_str(config_json, config_len) {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };

    let config: serde_json::Value = match serde_json::from_str(config_str) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("invalid config JSON: {}", e));
            return std::ptr::null_mut();
        }
    };

    let data_dir = config
        .get("storage")
        .and_then(|s| s.get("data_dir"))
        .and_then(|d| d.as_str())
        .unwrap_or("./hebbs-data");

    let use_mock = config
        .get("embedding")
        .and_then(|e| e.get("mock"))
        .and_then(|m| m.as_bool())
        .unwrap_or(true);

    let dimensions = config
        .get("embedding")
        .and_then(|e| e.get("dimensions"))
        .and_then(|d| d.as_u64())
        .unwrap_or(384) as usize;

    let storage: Arc<dyn hebbs_storage::StorageBackend> = match RocksDbBackend::open(data_dir) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            set_last_error(&format!("failed to open storage: {}", e));
            return std::ptr::null_mut();
        }
    };

    let embedder: Arc<dyn hebbs_embed::Embedder> = if use_mock {
        Arc::new(MockEmbedder::new(dimensions))
    } else {
        set_last_error("only mock embedder is supported via FFI currently");
        return std::ptr::null_mut();
    };

    let engine = match Engine::new(storage, embedder) {
        Ok(e) => e,
        Err(e) => {
            set_last_error(&format!("failed to create engine: {}", e));
            return std::ptr::null_mut();
        }
    };

    let arc = Arc::new(engine);
    let boxed = Box::new(arc);
    Box::into_raw(boxed) as *mut libc::c_void
}

/// Close a HEBBS engine handle and free all resources.
///
/// After this call, the handle is invalid and must not be used.
///
/// # Safety
///
/// - `handle` must have been returned by `hebbs_open`.
/// - `handle` must not have been previously passed to `hebbs_close`.
/// - No other thread may be using the handle concurrently with this call.
#[no_mangle]
pub unsafe extern "C" fn hebbs_close(handle: *mut libc::c_void) -> i32 {
    if handle.is_null() {
        set_last_error("null handle");
        return FFI_INVALID_HANDLE;
    }
    let arc_ptr = handle as *mut Arc<Engine>;
    // SAFETY: caller guarantees this pointer was returned by hebbs_open
    // and has not been freed. We reconstruct the Box and drop it.
    drop(unsafe { Box::from_raw(arc_ptr) });
    FFI_OK
}

/// Get the number of memories in the engine.
///
/// # Safety
///
/// `handle` must be valid.
#[no_mangle]
pub unsafe extern "C" fn hebbs_count(handle: *mut libc::c_void) -> i64 {
    let engine = match HandleRef::from_raw(handle) {
        Some(e) => e,
        None => return -1,
    };

    match engine.count() {
        Ok(n) => n as i64,
        Err(e) => {
            set_last_error(&format!("count failed: {}", e));
            -1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_close_lifecycle() {
        let dir = tempfile::tempdir().unwrap();
        let config = serde_json::json!({
            "storage": { "data_dir": dir.path().to_str().unwrap() },
            "embedding": { "mock": true, "dimensions": 8 }
        });
        let config_str = config.to_string();

        unsafe {
            let handle = hebbs_open(
                config_str.as_ptr() as *const c_char,
                config_str.len() as u32,
            );
            assert!(!handle.is_null(), "handle should be non-null");

            let count = hebbs_count(handle);
            assert_eq!(count, 0);

            let rc = hebbs_close(handle);
            assert_eq!(rc, FFI_OK);
        }
    }

    #[test]
    fn open_with_default_config() {
        let dir = tempfile::tempdir().unwrap();
        let config = format!(
            r#"{{"storage":{{"data_dir":"{}"}}}}"#,
            dir.path().to_str().unwrap()
        );

        unsafe {
            let handle = hebbs_open(config.as_ptr() as *const c_char, config.len() as u32);
            assert!(!handle.is_null());
            hebbs_close(handle);
        }
    }

    #[test]
    fn open_with_invalid_json() {
        let bad = "not json";
        unsafe {
            let handle = hebbs_open(bad.as_ptr() as *const c_char, bad.len() as u32);
            assert!(handle.is_null());
            let err = crate::hebbs_last_error();
            assert!(!err.is_null());
        }
    }

    #[test]
    fn close_null_returns_error() {
        unsafe {
            let rc = hebbs_close(std::ptr::null_mut());
            assert_eq!(rc, FFI_INVALID_HANDLE);
        }
    }

    #[test]
    fn from_raw_null_returns_none() {
        unsafe {
            assert!(HandleRef::from_raw(std::ptr::null_mut()).is_none());
        }
    }
}
