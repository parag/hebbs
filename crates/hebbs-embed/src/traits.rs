use crate::error::Result;

/// Contract between the HEBBS engine and any embedding provider.
///
/// Implementors must be `Send + Sync` for concurrent inference from
/// multiple threads (Phase 8 gRPC handlers, Phase 6 subscribe pipeline).
///
/// ## Invariants
///
/// - All returned vectors are L2-normalized (unit length ±1e-5).
/// - `embed(text)` and `embed_batch(&[text])` produce identical vectors.
/// - `dimensions()` matches the length of all returned vectors.
///
/// ## Stability
///
/// This trait is immutable after Phase 2. New methods may be added
/// (with defaults), but existing signatures never change.
pub trait Embedder: Send + Sync {
    /// Embed a single text into a dense vector.
    ///
    /// The returned vector is L2-normalized and has length `self.dimensions()`.
    ///
    /// Complexity: O(seq_len × model_dim) — dominated by transformer attention.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed multiple texts in a single call.
    ///
    /// Amortizes ONNX Runtime per-call overhead. For N texts, batch inference
    /// is ~24× more throughput-efficient than N sequential `embed()` calls.
    ///
    /// Batch sizes > 256 are chunked internally (Principle 4: bounded resources).
    ///
    /// Complexity: O(batch_size × max_seq_len × model_dim).
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Return the dimensionality of vectors produced by this embedder.
    ///
    /// Used by Phase 3 to configure HNSW index parameters.
    fn dimensions(&self) -> usize;
}
