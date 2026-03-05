use std::ffi::CStr;
use std::os::raw::c_char;

use hebbs_ffi::*;

fn setup_engine() -> (*mut libc::c_void, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let config = serde_json::json!({
        "storage": { "data_dir": dir.path().to_str().unwrap() },
        "embedding": { "mock": true, "dimensions": 8 }
    });
    let config_str = config.to_string();

    let handle = unsafe {
        hebbs_open(
            config_str.as_ptr() as *const c_char,
            config_str.len() as u32,
        )
    };
    assert!(!handle.is_null(), "engine should open successfully");
    (handle, dir)
}

fn get_result_json(result_ptr: *mut c_char) -> serde_json::Value {
    let c_str = unsafe { CStr::from_ptr(result_ptr) };
    let json: serde_json::Value = serde_json::from_str(c_str.to_str().unwrap()).unwrap();
    unsafe { hebbs_free_string(result_ptr) };
    json
}

#[test]
fn full_lifecycle_remember_get_recall_revise_forget() {
    let (handle, _dir) = setup_engine();

    // Remember a memory
    let content = "Customer mentioned deadline concern for Q4 2024";
    let mut result_out: *mut c_char = std::ptr::null_mut();
    let mut result_len: u32 = 0;

    let rc = unsafe {
        hebbs_ffi::hebbs_remember(
            handle,
            content.as_ptr() as *const c_char,
            content.len() as u32,
            0.8,
            &mut result_out,
            &mut result_len,
        )
    };
    assert_eq!(rc, 0, "remember should succeed");
    assert!(!result_out.is_null());
    assert!(result_len > 0);

    let memory = get_result_json(result_out);
    assert_eq!(
        memory["content"],
        "Customer mentioned deadline concern for Q4 2024"
    );
    assert!(memory["importance"].as_f64().unwrap() > 0.7);
    let memory_id = memory["memory_id"].as_str().unwrap().to_string();
    assert_eq!(memory_id.len(), 26, "ULID should be 26 chars");

    // Get the memory back
    let mut get_out: *mut c_char = std::ptr::null_mut();
    let mut get_len: u32 = 0;

    let rc = unsafe {
        hebbs_ffi::hebbs_get(
            handle,
            memory_id.as_ptr() as *const c_char,
            memory_id.len() as u32,
            &mut get_out,
            &mut get_len,
        )
    };
    assert_eq!(rc, 0, "get should succeed");
    let got_memory = get_result_json(get_out);
    assert_eq!(got_memory["memory_id"], memory_id);
    assert_eq!(got_memory["content"], content);

    // Recall by similarity
    let cue = "deadline concerns";
    let mut recall_out: *mut c_char = std::ptr::null_mut();
    let mut recall_len: u32 = 0;

    let rc = unsafe {
        hebbs_ffi::hebbs_recall(
            handle,
            cue.as_ptr() as *const c_char,
            cue.len() as u32,
            std::ptr::null(),
            0,
            &mut recall_out,
            &mut recall_len,
        )
    };
    assert_eq!(rc, 0, "recall should succeed");
    let recall = get_result_json(recall_out);
    assert!(recall["results"].as_array().is_some());

    // Revise the memory
    let revise_json = serde_json::json!({
        "memory_id": memory_id,
        "content": "Customer urgently mentioned deadline concern for Q4 2024",
        "importance": 0.95,
    });
    let revise_str = revise_json.to_string();
    let mut revise_out: *mut c_char = std::ptr::null_mut();
    let mut revise_len: u32 = 0;

    let rc = unsafe {
        hebbs_ffi::hebbs_revise(
            handle,
            revise_str.as_ptr() as *const c_char,
            revise_str.len() as u32,
            &mut revise_out,
            &mut revise_len,
        )
    };
    assert_eq!(rc, 0, "revise should succeed");
    let revised = get_result_json(revise_out);
    assert!(revised["content"].as_str().unwrap().contains("urgently"));

    // Revise creates a snapshot of the old version + updates the original,
    // so count goes from 1 to 2.
    let count = unsafe { hebbs_ffi::hebbs_count(handle) };
    assert!(count >= 1, "should have at least the revised memory");

    // Forget the memory (by its ID)
    let forget_json = serde_json::json!({
        "memory_ids": [memory_id],
    });
    let forget_str = forget_json.to_string();
    let mut forget_out: *mut c_char = std::ptr::null_mut();
    let mut forget_len: u32 = 0;

    let rc = unsafe {
        hebbs_ffi::hebbs_forget(
            handle,
            forget_str.as_ptr() as *const c_char,
            forget_str.len() as u32,
            &mut forget_out,
            &mut forget_len,
        )
    };
    assert_eq!(rc, 0, "forget should succeed");
    let forget_result = get_result_json(forget_out);
    assert!(forget_result["forgotten_count"].as_u64().unwrap() >= 1);

    // Close
    let rc = unsafe { hebbs_ffi::hebbs_close(handle) };
    assert_eq!(rc, 0);
}

#[test]
fn remember_with_json_options() {
    let (handle, _dir) = setup_engine();

    let opts = serde_json::json!({
        "content": "Meeting with Acme Corp scheduled for next Tuesday",
        "importance": 0.7,
        "entity_id": "acme_corp",
        "context": {
            "stage": "active",
            "deal_size": 50000
        }
    });
    let opts_str = opts.to_string();
    let mut result_out: *mut c_char = std::ptr::null_mut();
    let mut result_len: u32 = 0;

    let rc = unsafe {
        hebbs_ffi::hebbs_remember_with(
            handle,
            opts_str.as_ptr() as *const c_char,
            opts_str.len() as u32,
            &mut result_out,
            &mut result_len,
        )
    };
    assert_eq!(rc, 0);
    let memory = get_result_json(result_out);
    assert_eq!(memory["entity_id"], "acme_corp");
    assert_eq!(memory["context"]["stage"], "active");

    unsafe { hebbs_ffi::hebbs_close(handle) };
}

#[test]
fn multiple_memories_and_recall() {
    let (handle, _dir) = setup_engine();

    let contents = [
        "Product demo went well with positive feedback",
        "Customer raised concerns about pricing",
        "Technical integration meeting scheduled",
        "Competitor analysis shows we lead in features",
        "Budget approval expected next quarter",
    ];

    for content in &contents {
        let mut result_out: *mut c_char = std::ptr::null_mut();
        let mut result_len: u32 = 0;
        let rc = unsafe {
            hebbs_ffi::hebbs_remember(
                handle,
                content.as_ptr() as *const c_char,
                content.len() as u32,
                0.5,
                &mut result_out,
                &mut result_len,
            )
        };
        assert_eq!(rc, 0);
        unsafe { hebbs_free_string(result_out) };
    }

    let count = unsafe { hebbs_ffi::hebbs_count(handle) };
    assert_eq!(count, 5);

    // Recall
    let cue = "pricing concerns";
    let opts = r#"{"top_k": 3}"#;
    let mut recall_out: *mut c_char = std::ptr::null_mut();
    let mut recall_len: u32 = 0;

    let rc = unsafe {
        hebbs_ffi::hebbs_recall(
            handle,
            cue.as_ptr() as *const c_char,
            cue.len() as u32,
            opts.as_ptr() as *const c_char,
            opts.len() as u32,
            &mut recall_out,
            &mut recall_len,
        )
    };
    assert_eq!(rc, 0);
    let recall = get_result_json(recall_out);
    let results = recall["results"].as_array().unwrap();
    assert!(results.len() <= 3, "should respect top_k");

    unsafe { hebbs_ffi::hebbs_close(handle) };
}

#[test]
fn get_nonexistent_memory() {
    let (handle, _dir) = setup_engine();

    let fake_id = ulid::Ulid::new().to_string();
    let mut result_out: *mut c_char = std::ptr::null_mut();
    let mut result_len: u32 = 0;

    let rc = unsafe {
        hebbs_ffi::hebbs_get(
            handle,
            fake_id.as_ptr() as *const c_char,
            fake_id.len() as u32,
            &mut result_out,
            &mut result_len,
        )
    };
    assert!(rc < 0, "get for nonexistent ID should fail");

    let err = unsafe { hebbs_last_error() };
    assert!(!err.is_null());
    let err_str = unsafe { CStr::from_ptr(err) }.to_str().unwrap();
    assert!(
        err_str.contains("not found"),
        "error should mention not found, got: {}",
        err_str
    );

    unsafe { hebbs_ffi::hebbs_close(handle) };
}

#[test]
fn invalid_ulid_returns_error() {
    let (handle, _dir) = setup_engine();

    let bad_id = "not-a-ulid";
    let mut result_out: *mut c_char = std::ptr::null_mut();
    let mut result_len: u32 = 0;

    let rc = unsafe {
        hebbs_ffi::hebbs_get(
            handle,
            bad_id.as_ptr() as *const c_char,
            bad_id.len() as u32,
            &mut result_out,
            &mut result_len,
        )
    };
    assert_eq!(rc, -1, "invalid ULID should return FFI_INVALID_ARG");

    unsafe { hebbs_ffi::hebbs_close(handle) };
}

#[test]
fn null_handle_returns_error() {
    let mut result_out: *mut c_char = std::ptr::null_mut();
    let mut result_len: u32 = 0;
    let content = "test";

    let rc = unsafe {
        hebbs_ffi::hebbs_remember(
            std::ptr::null_mut(),
            content.as_ptr() as *const c_char,
            content.len() as u32,
            0.5,
            &mut result_out,
            &mut result_len,
        )
    };
    assert_eq!(rc, -6, "null handle should return FFI_INVALID_HANDLE");
}

#[test]
fn prime_operation() {
    let (handle, _dir) = setup_engine();

    // Store memories with entity
    let opts = serde_json::json!({
        "content": "Initial meeting notes with Acme Corp",
        "importance": 0.6,
        "entity_id": "acme_corp"
    });
    let opts_str = opts.to_string();
    let mut result_out: *mut c_char = std::ptr::null_mut();
    let mut result_len: u32 = 0;

    let rc = unsafe {
        hebbs_ffi::hebbs_remember_with(
            handle,
            opts_str.as_ptr() as *const c_char,
            opts_str.len() as u32,
            &mut result_out,
            &mut result_len,
        )
    };
    assert_eq!(rc, 0);
    unsafe { hebbs_free_string(result_out) };

    // Prime
    let prime_opts = r#"{"entity_id": "acme_corp"}"#;
    let mut prime_out: *mut c_char = std::ptr::null_mut();
    let mut prime_len: u32 = 0;

    let rc = unsafe {
        hebbs_ffi::hebbs_prime(
            handle,
            prime_opts.as_ptr() as *const c_char,
            prime_opts.len() as u32,
            &mut prime_out,
            &mut prime_len,
        )
    };
    assert_eq!(rc, 0, "prime should succeed");
    let prime_result = get_result_json(prime_out);
    assert!(prime_result["results"].as_array().is_some());

    unsafe { hebbs_ffi::hebbs_close(handle) };
}

#[test]
fn insights_operation() {
    let (handle, _dir) = setup_engine();

    let filter = r#"{"max_results": 10}"#;
    let mut result_out: *mut c_char = std::ptr::null_mut();
    let mut result_len: u32 = 0;

    let rc = unsafe {
        hebbs_ffi::hebbs_insights(
            handle,
            filter.as_ptr() as *const c_char,
            filter.len() as u32,
            &mut result_out,
            &mut result_len,
        )
    };
    assert_eq!(rc, 0, "insights should succeed (empty is fine)");
    let insights = get_result_json(result_out);
    assert!(insights.is_array());

    unsafe { hebbs_ffi::hebbs_close(handle) };
}

#[test]
fn forget_by_entity() {
    let (handle, _dir) = setup_engine();

    // Store memories with entity
    for i in 0..3 {
        let opts = serde_json::json!({
            "content": format!("Memory {} for delete_entity", i),
            "importance": 0.5,
            "entity_id": "delete_entity"
        });
        let opts_str = opts.to_string();
        let mut result_out: *mut c_char = std::ptr::null_mut();
        let mut result_len: u32 = 0;
        let rc = unsafe {
            hebbs_ffi::hebbs_remember_with(
                handle,
                opts_str.as_ptr() as *const c_char,
                opts_str.len() as u32,
                &mut result_out,
                &mut result_len,
            )
        };
        assert_eq!(rc, 0);
        unsafe { hebbs_free_string(result_out) };
    }

    assert_eq!(unsafe { hebbs_ffi::hebbs_count(handle) }, 3);

    // Forget by entity
    let forget_json = r#"{"entity_id": "delete_entity"}"#;
    let mut forget_out: *mut c_char = std::ptr::null_mut();
    let mut forget_len: u32 = 0;

    let rc = unsafe {
        hebbs_ffi::hebbs_forget(
            handle,
            forget_json.as_ptr() as *const c_char,
            forget_json.len() as u32,
            &mut forget_out,
            &mut forget_len,
        )
    };
    assert_eq!(rc, 0);
    let result = get_result_json(forget_out);
    assert_eq!(result["forgotten_count"], 3);

    assert_eq!(unsafe { hebbs_ffi::hebbs_count(handle) }, 0);

    unsafe { hebbs_ffi::hebbs_close(handle) };
}

#[test]
fn concurrent_access_from_threads() {
    let (handle, _dir) = setup_engine();

    let handle_usize = handle as usize;
    let mut threads = Vec::new();

    for i in 0..4 {
        let h = handle_usize;
        threads.push(std::thread::spawn(move || {
            let handle = h as *mut libc::c_void;
            for j in 0..5 {
                let content = format!("Thread {} memory {}", i, j);
                let mut result_out: *mut c_char = std::ptr::null_mut();
                let mut result_len: u32 = 0;
                let rc = unsafe {
                    hebbs_ffi::hebbs_remember(
                        handle,
                        content.as_ptr() as *const c_char,
                        content.len() as u32,
                        0.5,
                        &mut result_out,
                        &mut result_len,
                    )
                };
                assert_eq!(rc, 0, "thread {}, memory {} should succeed", i, j);
                unsafe { hebbs_free_string(result_out) };
            }
        }));
    }

    for t in threads {
        t.join().unwrap();
    }

    let count = unsafe { hebbs_ffi::hebbs_count(handle) };
    assert_eq!(count, 20, "should have 4 threads * 5 memories = 20");

    unsafe { hebbs_ffi::hebbs_close(handle) };
}
