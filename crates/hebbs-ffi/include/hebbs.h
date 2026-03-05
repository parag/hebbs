/* HEBBS FFI - C header for the HEBBS cognitive memory engine.
 *
 * This header defines the C-ABI interface for embedding HEBBS directly
 * into applications without a separate server process.
 *
 * Thread Safety: All functions are thread-safe. The handle returned by
 * hebbs_open() may be shared across threads.
 *
 * Error Handling: Functions return 0 on success, negative on error.
 * Call hebbs_last_error() for a detailed error message.
 *
 * Memory Management: Result strings must be freed with hebbs_free_string().
 * The hebbs_last_error() pointer is valid until the next FFI call on the
 * same thread and must NOT be freed.
 */

#ifndef HEBBS_FFI_H
#define HEBBS_FFI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════
 *  Return codes
 * ═══════════════════════════════════════════════════════════════════════ */

#define HEBBS_OK              0
#define HEBBS_INVALID_ARG    -1
#define HEBBS_NOT_FOUND      -2
#define HEBBS_STORAGE        -3
#define HEBBS_EMBEDDING      -4
#define HEBBS_INTERNAL       -5
#define HEBBS_INVALID_HANDLE -6

/* ═══════════════════════════════════════════════════════════════════════
 *  Lifecycle
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * Open a HEBBS engine with JSON configuration.
 *
 * @param config_json  UTF-8 JSON string, or NULL for defaults.
 * @param config_len   Length of config_json in bytes.
 * @return Opaque handle, or NULL on failure. Call hebbs_last_error() for details.
 */
void* hebbs_open(const char* config_json, uint32_t config_len);

/**
 * Close a HEBBS engine and release all resources.
 *
 * @param handle  Handle from hebbs_open(). Must not be NULL.
 * @return HEBBS_OK on success.
 */
int32_t hebbs_close(void* handle);

/**
 * Get the number of memories in the engine.
 *
 * @param handle  Valid engine handle.
 * @return Count, or -1 on error.
 */
int64_t hebbs_count(void* handle);

/* ═══════════════════════════════════════════════════════════════════════
 *  Memory operations
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * Store a new memory.
 *
 * @param handle         Valid engine handle.
 * @param content        UTF-8 content string.
 * @param content_len    Length in bytes.
 * @param importance     Importance score [0.0, 1.0].
 * @param result_out     [out] JSON string of the created memory. Free with hebbs_free_string().
 * @param result_len_out [out] Length of the result string.
 * @return HEBBS_OK on success.
 */
int32_t hebbs_remember(void* handle,
                       const char* content,
                       uint32_t content_len,
                       float importance,
                       char** result_out,
                       uint32_t* result_len_out);

/**
 * Store a memory with full JSON options.
 *
 * Options JSON: { "content": "...", "importance": 0.8, "context": {...}, "entity_id": "...", "edges": [...] }
 */
int32_t hebbs_remember_with(void* handle,
                            const char* options_json,
                            uint32_t options_len,
                            char** result_out,
                            uint32_t* result_len_out);

/**
 * Retrieve a memory by its 26-character ULID string.
 */
int32_t hebbs_get(void* handle,
                  const char* memory_id,
                  uint32_t memory_id_len,
                  char** result_out,
                  uint32_t* result_len_out);

/**
 * Recall memories matching a cue.
 *
 * @param cue           Query text.
 * @param cue_len       Length of cue in bytes.
 * @param options_json  JSON options (strategy, top_k, etc.), or NULL for defaults.
 * @param options_len   Length of options_json.
 */
int32_t hebbs_recall(void* handle,
                     const char* cue,
                     uint32_t cue_len,
                     const char* options_json,
                     uint32_t options_len,
                     char** result_out,
                     uint32_t* result_len_out);

/**
 * Revise an existing memory. Options as JSON.
 *
 * Options JSON: { "memory_id": "ULID", "content": "new...", "importance": 0.9, ... }
 */
int32_t hebbs_revise(void* handle,
                     const char* options_json,
                     uint32_t options_len,
                     char** result_out,
                     uint32_t* result_len_out);

/**
 * Forget memories matching criteria.
 *
 * Criteria JSON: { "memory_ids": ["ULID", ...], "entity_id": "...", ... }
 */
int32_t hebbs_forget(void* handle,
                     const char* criteria_json,
                     uint32_t criteria_len,
                     char** result_out,
                     uint32_t* result_len_out);

/**
 * Pre-load memories for an entity.
 *
 * Options JSON: { "entity_id": "...", "max_memories": 50, ... }
 */
int32_t hebbs_prime(void* handle,
                    const char* options_json,
                    uint32_t options_len,
                    char** result_out,
                    uint32_t* result_len_out);

/**
 * Trigger reflection.
 *
 * Options JSON: { "entity_id": "..." } or { "since_us": 123456 }
 */
int32_t hebbs_reflect(void* handle,
                      const char* options_json,
                      uint32_t options_len,
                      char** result_out,
                      uint32_t* result_len_out);

/**
 * Query insights.
 *
 * Filter JSON: { "entity_id": "...", "min_confidence": 0.5, "max_results": 10 }
 */
int32_t hebbs_insights(void* handle,
                       const char* filter_json,
                       uint32_t filter_len,
                       char** result_out,
                       uint32_t* result_len_out);

/* ═══════════════════════════════════════════════════════════════════════
 *  Subscription
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * Callback type for subscription pushes.
 *
 * @param push_json      JSON string (valid only during callback).
 * @param push_json_len  Length in bytes.
 * @param user_data      Opaque pointer passed to hebbs_subscribe().
 */
typedef void (*hebbs_subscribe_callback)(const char* push_json,
                                         uint32_t push_json_len,
                                         void* user_data);

/**
 * Start a subscription with callback-based delivery.
 *
 * @param handle       Valid engine handle.
 * @param config_json  JSON config, or NULL for defaults.
 * @param config_len   Length of config_json.
 * @param callback     Function called on each push.
 * @param user_data    Passed to callback unchanged.
 * @return Subscription handle, or NULL on failure.
 */
void* hebbs_subscribe(void* handle,
                      const char* config_json,
                      uint32_t config_len,
                      hebbs_subscribe_callback callback,
                      void* user_data);

/**
 * Poll for the next push (non-blocking).
 *
 * @param sub_handle  Subscription handle from hebbs_subscribe().
 * @return JSON string if available (free with hebbs_free_string()), or NULL.
 */
char* hebbs_subscribe_poll(void* sub_handle);

/**
 * Feed text to a subscription.
 */
int32_t hebbs_subscribe_feed(void* sub_handle,
                             const char* text,
                             uint32_t text_len);

/**
 * Close a subscription.
 */
int32_t hebbs_subscribe_close(void* sub_handle);

/* ═══════════════════════════════════════════════════════════════════════
 *  Error handling
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * Get the last error message for the current thread.
 *
 * @return Error string (do NOT free), or NULL if no error.
 */
const char* hebbs_last_error(void);

/**
 * Free a string allocated by the FFI layer.
 *
 * @param ptr  String to free (from result_out or hebbs_subscribe_poll).
 */
void hebbs_free_string(char* ptr);

#ifdef __cplusplus
}
#endif

#endif /* HEBBS_FFI_H */
