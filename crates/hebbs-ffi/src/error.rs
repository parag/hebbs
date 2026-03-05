use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::c_char;

pub const FFI_OK: i32 = 0;
pub const FFI_INVALID_ARG: i32 = -1;
pub const FFI_NOT_FOUND: i32 = -2;
pub const FFI_STORAGE: i32 = -3;
pub const FFI_EMBEDDING: i32 = -4;
pub const FFI_INTERNAL: i32 = -5;
pub const FFI_INVALID_HANDLE: i32 = -6;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

pub fn set_last_error(msg: &str) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = msg.to_string();
    });
}

/// Retrieve the last error message for the current thread.
///
/// Returns a pointer to a null-terminated UTF-8 string that remains valid
/// until the next FFI call on the same thread. The caller must NOT free
/// this pointer.
///
/// Returns null if no error has occurred.
///
/// # Safety
///
/// The returned pointer is only valid until the next `hebbs_*` call
/// on the same thread.
#[no_mangle]
pub unsafe extern "C" fn hebbs_last_error() -> *const c_char {
    thread_local! {
        static ERROR_BUF: RefCell<CString> = RefCell::new(CString::default());
    }

    let msg = LAST_ERROR.with(|e| e.borrow().clone());
    if msg.is_empty() {
        return std::ptr::null();
    }

    ERROR_BUF.with(|buf| {
        let cs = CString::new(msg).unwrap_or_default();
        let ptr = cs.as_ptr();
        *buf.borrow_mut() = cs;
        ptr
    })
}

/// Free a string allocated by the FFI layer.
///
/// # Safety
///
/// `ptr` must have been returned by a `hebbs_*` function (via result_out)
/// and must not have been freed already.
#[no_mangle]
pub unsafe extern "C" fn hebbs_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}

#[cfg(test)]
pub fn clear_last_error() {
    LAST_ERROR.with(|e| e.borrow_mut().clear());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_set_and_retrieve() {
        set_last_error("test error message");
        let msg = LAST_ERROR.with(|e| e.borrow().clone());
        assert_eq!(msg, "test error message");
    }

    #[test]
    fn clear_error() {
        set_last_error("some error");
        clear_last_error();
        let msg = LAST_ERROR.with(|e| e.borrow().clone());
        assert!(msg.is_empty());
    }

    #[test]
    fn error_thread_isolation() {
        set_last_error("main thread error");

        let handle = std::thread::spawn(|| {
            let msg = LAST_ERROR.with(|e| e.borrow().clone());
            assert!(msg.is_empty(), "other thread should have empty error");
            set_last_error("other thread error");
            let msg = LAST_ERROR.with(|e| e.borrow().clone());
            assert_eq!(msg, "other thread error");
        });
        handle.join().unwrap();

        let msg = LAST_ERROR.with(|e| e.borrow().clone());
        assert_eq!(msg, "main thread error");
    }

    #[test]
    fn last_error_null_when_empty() {
        clear_last_error();
        let ptr = unsafe { hebbs_last_error() };
        assert!(ptr.is_null());
    }

    #[test]
    fn last_error_non_null_after_set() {
        set_last_error("an error");
        let ptr = unsafe { hebbs_last_error() };
        assert!(!ptr.is_null());
        let c_str = unsafe { std::ffi::CStr::from_ptr(ptr) };
        assert_eq!(c_str.to_str().unwrap(), "an error");
    }

    #[test]
    fn free_null_is_safe() {
        unsafe {
            hebbs_free_string(std::ptr::null_mut());
        }
    }

    #[test]
    fn return_code_values() {
        assert_eq!(FFI_OK, 0);
        assert_eq!(FFI_INVALID_ARG, -1);
        assert_eq!(FFI_NOT_FOUND, -2);
        assert_eq!(FFI_STORAGE, -3);
        assert_eq!(FFI_EMBEDDING, -4);
        assert_eq!(FFI_INTERNAL, -5);
        assert_eq!(FFI_INVALID_HANDLE, -6);

        let codes = [
            FFI_INVALID_ARG,
            FFI_NOT_FOUND,
            FFI_STORAGE,
            FFI_EMBEDDING,
            FFI_INTERNAL,
            FFI_INVALID_HANDLE,
        ];
        let unique: std::collections::HashSet<_> = codes.into_iter().collect();
        assert_eq!(unique.len(), codes.len(), "error codes must be unique");
    }
}
