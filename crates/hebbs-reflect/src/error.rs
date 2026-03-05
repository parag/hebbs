#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ReflectError {
    #[error("clustering failed: {message}")]
    Clustering { message: String },

    #[error("LLM provider error: {message}")]
    Llm { message: String },

    #[error("LLM response parse error: {message}")]
    ResponseParse { message: String },

    #[error("prompt construction error: {message}")]
    Prompt { message: String },

    #[error("pipeline error in stage '{stage}': {message}")]
    Pipeline { stage: String, message: String },

    #[error("embedding error: {0}")]
    Embedding(#[from] hebbs_embed::EmbedError),

    #[error("insufficient memories for reflection: have {have}, need {need}")]
    InsufficientMemories { have: usize, need: usize },

    #[error("configuration error: {message}")]
    Config { message: String },
}

pub type Result<T> = std::result::Result<T, ReflectError>;
