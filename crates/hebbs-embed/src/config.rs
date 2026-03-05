use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Pooling strategy for converting token-level outputs to sentence embeddings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolingStrategy {
    /// Average all non-padding token embeddings.
    /// BGE-small-en-v1.5 uses this.
    #[default]
    Mean,
    /// Use the \[CLS\] token embedding (index 0).
    Cls,
}

/// Model configuration metadata.
///
/// Stored alongside the ONNX model as `config.json` to configure
/// tokenization and post-processing without hardcoding model-specific logic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_name: String,
    /// Output embedding dimensions (e.g. 384 for BGE-small-en-v1.5).
    pub dimensions: usize,
    /// Maximum input sequence length in tokens. Longer inputs are truncated.
    pub max_seq_length: usize,
    /// How to pool token-level outputs into a sentence embedding.
    pub pooling_strategy: PoolingStrategy,
}

impl ModelConfig {
    /// Default configuration for BGE-small-en-v1.5.
    pub fn bge_small_en_v1_5() -> Self {
        Self {
            model_name: "bge-small-en-v1.5".to_string(),
            dimensions: 384,
            max_seq_length: 512,
            pooling_strategy: PoolingStrategy::Mean,
        }
    }
}

/// Full configuration for the embedding engine.
#[derive(Debug, Clone)]
pub struct EmbedderConfig {
    /// Directory to store model files.
    pub model_dir: PathBuf,
    /// Model configuration metadata.
    pub model_config: ModelConfig,
    /// Base URL for model downloads.
    pub download_base_url: String,
    /// Whether to auto-download missing model files.
    pub auto_download: bool,
}

impl EmbedderConfig {
    /// Create a default configuration for BGE-small-en-v1.5.
    ///
    /// Model files are stored under `{data_dir}/models/bge-small-en-v1.5/`.
    pub fn default_bge_small(data_dir: impl Into<PathBuf>) -> Self {
        let data_dir = data_dir.into();
        Self {
            model_dir: data_dir.join("models").join("bge-small-en-v1.5"),
            model_config: ModelConfig::bge_small_en_v1_5(),
            download_base_url: "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main"
                .to_string(),
            auto_download: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bge_small_defaults() {
        let cfg = ModelConfig::bge_small_en_v1_5();
        assert_eq!(cfg.dimensions, 384);
        assert_eq!(cfg.max_seq_length, 512);
        assert_eq!(cfg.pooling_strategy, PoolingStrategy::Mean);
    }

    #[test]
    fn config_json_roundtrip() {
        let cfg = ModelConfig::bge_small_en_v1_5();
        let json = serde_json::to_string_pretty(&cfg).unwrap();
        let restored: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.dimensions, restored.dimensions);
        assert_eq!(cfg.max_seq_length, restored.max_seq_length);
        assert_eq!(cfg.pooling_strategy, restored.pooling_strategy);
    }

    #[test]
    fn embedder_config_default_paths() {
        let cfg = EmbedderConfig::default_bge_small("/tmp/hebbs");
        assert!(cfg.model_dir.ends_with("models/bge-small-en-v1.5"));
        assert!(cfg.auto_download);
    }
}
