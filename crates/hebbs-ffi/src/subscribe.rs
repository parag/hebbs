use std::os::raw::c_char;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use hebbs_core::subscribe::{SubscribeConfig, SubscriptionHandle};

use crate::convert;
use crate::error::{set_last_error, FFI_INTERNAL, FFI_INVALID_HANDLE, FFI_OK};
use crate::handle::HandleRef;

/// Opaque subscription handle for FFI.
struct FfiSubscription {
    handle: SubscriptionHandle,
    closed: AtomicBool,
}

/// Callback function type for subscription pushes.
///
/// The `push_json` pointer is valid only for the duration of the callback.
/// `push_json_len` is the length in bytes (excluding null terminator).
/// `user_data` is the opaque pointer passed to `hebbs_subscribe`.
pub type HebbsSubscribeCallback = unsafe extern "C" fn(
    push_json: *const c_char,
    push_json_len: u32,
    user_data: *mut libc::c_void,
);

/// Start a subscription with callback-based push delivery.
///
/// Returns a subscription handle on success, null on failure.
///
/// # Safety
///
/// - `handle` must be valid.
/// - `config_json` must be valid UTF-8 or null for defaults.
/// - `callback` must be a valid function pointer safe to call from any thread.
/// - `user_data` is passed to the callback unchanged.
#[no_mangle]
pub unsafe extern "C" fn hebbs_subscribe(
    handle: *mut libc::c_void,
    config_json: *const c_char,
    config_len: u32,
    callback: HebbsSubscribeCallback,
    user_data: *mut libc::c_void,
) -> *mut libc::c_void {
    let engine = match HandleRef::from_raw(handle) {
        Some(e) => e,
        None => return std::ptr::null_mut(),
    };

    let config_str = if config_json.is_null() || config_len == 0 {
        "{}"
    } else {
        match super::ptr_to_str(config_json, config_len) {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };

    let config = match parse_subscribe_config(config_str) {
        Ok(c) => c,
        Err(msg) => {
            set_last_error(&msg);
            return std::ptr::null_mut();
        }
    };

    let sub_handle = match engine.subscribe(config) {
        Ok(h) => h,
        Err(e) => {
            set_last_error(&format!("subscribe failed: {}", e));
            return std::ptr::null_mut();
        }
    };

    let ffi_sub = Arc::new(FfiSubscription {
        handle: sub_handle,
        closed: AtomicBool::new(false),
    });

    let sub_clone = ffi_sub.clone();
    let user_data_ptr = user_data as usize;

    std::thread::Builder::new()
        .name("hebbs-ffi-subscribe".to_string())
        .spawn(move || {
            let ud = user_data_ptr as *mut libc::c_void;
            while !sub_clone.closed.load(Ordering::Relaxed) {
                if let Some(push) = sub_clone.handle.try_recv() {
                    let json = convert::memory_to_json(&push.memory);
                    let push_json = serde_json::json!({
                        "memory": serde_json::from_str::<serde_json::Value>(&json).unwrap_or_default(),
                        "confidence": push.confidence,
                        "push_timestamp_us": push.push_timestamp_us,
                    });
                    let json_str = push_json.to_string();

                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                        callback(
                            json_str.as_ptr() as *const c_char,
                            json_str.len() as u32,
                            ud,
                        );
                    }));

                    if result.is_err() {
                        sub_clone.closed.store(true, Ordering::Relaxed);
                        break;
                    }
                } else {
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
        })
        .ok();

    let boxed = Box::new(ffi_sub);
    Box::into_raw(boxed) as *mut libc::c_void
}

/// Poll for the next push (non-blocking alternative to callback).
///
/// Returns a JSON string if a push is available, null otherwise.
/// The returned string must be freed with `hebbs_free_string`.
///
/// # Safety
///
/// `sub_handle` must be a valid subscription handle.
#[no_mangle]
pub unsafe extern "C" fn hebbs_subscribe_poll(sub_handle: *mut libc::c_void) -> *mut c_char {
    if sub_handle.is_null() {
        return std::ptr::null_mut();
    }

    let sub = &*(sub_handle as *const Arc<FfiSubscription>);

    match sub.handle.try_recv() {
        Some(push) => {
            let json = convert::memory_to_json(&push.memory);
            let push_json = serde_json::json!({
                "memory": serde_json::from_str::<serde_json::Value>(&json).unwrap_or_default(),
                "confidence": push.confidence,
                "push_timestamp_us": push.push_timestamp_us,
            });
            let json_str = push_json.to_string();
            match std::ffi::CString::new(json_str) {
                Ok(cs) => cs.into_raw(),
                Err(_) => std::ptr::null_mut(),
            }
        }
        None => std::ptr::null_mut(),
    }
}

/// Feed text to an active subscription.
///
/// # Safety
///
/// `sub_handle` must be valid. `text` must be valid UTF-8.
#[no_mangle]
pub unsafe extern "C" fn hebbs_subscribe_feed(
    sub_handle: *mut libc::c_void,
    text: *const c_char,
    text_len: u32,
) -> i32 {
    if sub_handle.is_null() {
        set_last_error("null subscription handle");
        return FFI_INVALID_HANDLE;
    }

    let text_str = match super::ptr_to_str(text, text_len) {
        Ok(s) => s,
        Err(code) => return code,
    };

    let sub = &*(sub_handle as *const Arc<FfiSubscription>);

    match sub.handle.feed(text_str) {
        Ok(()) => FFI_OK,
        Err(e) => {
            set_last_error(&format!("feed failed: {}", e));
            FFI_INTERNAL
        }
    }
}

/// Close a subscription and free resources.
///
/// # Safety
///
/// `sub_handle` must be valid and not previously closed.
#[no_mangle]
pub unsafe extern "C" fn hebbs_subscribe_close(sub_handle: *mut libc::c_void) -> i32 {
    if sub_handle.is_null() {
        set_last_error("null subscription handle");
        return FFI_INVALID_HANDLE;
    }

    let sub_ptr = sub_handle as *mut Arc<FfiSubscription>;
    let sub = Box::from_raw(sub_ptr);
    sub.closed.store(true, Ordering::Relaxed);
    FFI_OK
}

fn parse_subscribe_config(json: &str) -> Result<SubscribeConfig, String> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("invalid JSON: {}", e))?;

    let mut config = SubscribeConfig::default();

    if let Some(eid) = v.get("entity_id").and_then(|e| e.as_str()) {
        config.entity_id = Some(eid.to_string());
    }
    if let Some(ct) = v.get("confidence_threshold").and_then(|c| c.as_f64()) {
        config.confidence_threshold = ct as f32;
    }

    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_subscribe_config_defaults() {
        let config = parse_subscribe_config("{}").unwrap();
        assert!(config.entity_id.is_none());
    }

    #[test]
    fn parse_subscribe_config_with_entity() {
        let config =
            parse_subscribe_config(r#"{"entity_id": "test", "confidence_threshold": 0.8}"#)
                .unwrap();
        assert_eq!(config.entity_id, Some("test".to_string()));
        assert!((config.confidence_threshold - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn parse_subscribe_config_invalid_json() {
        assert!(parse_subscribe_config("not json").is_err());
    }
}
