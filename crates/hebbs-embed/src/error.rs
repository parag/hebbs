/// Errors produced by the embedding engine.
///
/// These are domain-specific to embedding operations and are wrapped
/// by `HebbsError::Embedding` in `hebbs-core`. Downstream consumers
/// match on `HebbsError`, not `EmbedError` directly.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EmbedError {
    #[error("model loading failed: {message}")]
    ModelLoad { message: String },

    #[error("tokenization failed: {message}")]
    Tokenization { message: String },

    #[error("inference failed: {message}")]
    Inference { message: String },

    #[error("model download failed: {message}")]
    Download { message: String },

    #[error("checksum verification failed for {file}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        file: String,
        expected: String,
        actual: String,
    },

    #[error("configuration error: {message}")]
    Config { message: String },
}

pub type Result<T> = std::result::Result<T, EmbedError>;
