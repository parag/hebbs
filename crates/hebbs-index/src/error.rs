/// Errors produced by the HEBBS index layer.
///
/// These cover all three index types (temporal, vector/HNSW, graph)
/// and the unified IndexManager. Wrapped by `HebbsError::Index` in
/// `hebbs-core` for downstream consumers.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum IndexError {
    #[error("storage error: {0}")]
    Storage(#[from] hebbs_storage::StorageError),

    #[error("dimension mismatch: index expects {expected}-dim vectors, got {actual}-dim")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("invalid memory ID: {message}")]
    InvalidMemoryId { message: String },

    #[error("invalid edge type byte: 0x{value:02X}")]
    InvalidEdgeType { value: u8 },

    #[error("HNSW node serialization error: {message}")]
    Serialization { message: String },

    #[error("vector not L2-normalized: norm = {norm} (expected ~1.0)")]
    NotNormalized { norm: f32 },

    #[error("internal index error in {operation}: {message}")]
    Internal {
        operation: &'static str,
        message: String,
    },
}

pub type Result<T> = std::result::Result<T, IndexError>;
