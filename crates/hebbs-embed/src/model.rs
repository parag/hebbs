use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::config::EmbedderConfig;
use crate::error::{EmbedError, Result};

/// Resolved paths for a model's files on disk.
#[derive(Debug)]
pub struct ModelPaths {
    pub model_onnx: PathBuf,
    pub tokenizer_json: PathBuf,
    pub config_json: PathBuf,
}

impl ModelPaths {
    pub fn from_dir(dir: &Path) -> Self {
        Self {
            model_onnx: dir.join("model.onnx"),
            tokenizer_json: dir.join("tokenizer.json"),
            config_json: dir.join("config.json"),
        }
    }

    /// Check whether all required model files are present on disk.
    pub fn all_exist(&self) -> bool {
        self.model_onnx.exists() && self.tokenizer_json.exists()
    }
}

/// Ensure model files are present, downloading if necessary and permitted.
///
/// Returns the resolved paths to all model files.
pub fn ensure_model_files(config: &EmbedderConfig) -> Result<ModelPaths> {
    let paths = ModelPaths::from_dir(&config.model_dir);

    if paths.all_exist() {
        return Ok(paths);
    }

    if !config.auto_download {
        return Err(EmbedError::ModelLoad {
            message: format!(
                "model files not found at {} and auto_download is disabled — \
                 pre-place model.onnx and tokenizer.json in the model directory",
                config.model_dir.display()
            ),
        });
    }

    fs::create_dir_all(&config.model_dir).map_err(|e| EmbedError::ModelLoad {
        message: format!(
            "failed to create model directory {}: {}",
            config.model_dir.display(),
            e
        ),
    })?;

    if !paths.model_onnx.exists() {
        let url = format!("{}/onnx/model.onnx", config.download_base_url);
        download_file(&url, &paths.model_onnx)?;
    }

    if !paths.tokenizer_json.exists() {
        let url = format!("{}/tokenizer.json", config.download_base_url);
        download_file(&url, &paths.tokenizer_json)?;
    }

    // Write config.json alongside model files
    if !paths.config_json.exists() {
        let config_json =
            serde_json::to_string_pretty(&config.model_config).map_err(|e| EmbedError::Config {
                message: format!("failed to serialize model config: {}", e),
            })?;
        fs::write(&paths.config_json, config_json).map_err(|e| EmbedError::ModelLoad {
            message: format!("failed to write config.json: {}", e),
        })?;
    }

    Ok(paths)
}

/// Download a file from a URL to a local path with atomic rename.
fn download_file(url: &str, dest: &Path) -> Result<()> {
    let tmp_path = dest.with_extension("download.tmp");

    let response = ureq::get(url).call().map_err(|e| EmbedError::Download {
        message: format!("HTTP request to {} failed: {}", url, e),
    })?;

    let mut file = fs::File::create(&tmp_path).map_err(|e| EmbedError::Download {
        message: format!("failed to create temp file {}: {}", tmp_path.display(), e),
    })?;

    // Stream the response body to disk in 64 KB chunks (bounded I/O)
    let mut reader = response.into_body().into_reader();
    let mut buffer = vec![0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buffer).map_err(|e| EmbedError::Download {
            message: format!("failed reading response body for {}: {}", url, e),
        })?;
        if n == 0 {
            break;
        }
        file.write_all(&buffer[..n])
            .map_err(|e| EmbedError::Download {
                message: format!("failed writing to {}: {}", tmp_path.display(), e),
            })?;
    }

    file.flush().map_err(|e| EmbedError::Download {
        message: format!("failed to flush {}: {}", tmp_path.display(), e),
    })?;
    drop(file);

    fs::rename(&tmp_path, dest).map_err(|e| EmbedError::Download {
        message: format!(
            "failed to rename {} → {}: {}",
            tmp_path.display(),
            dest.display(),
            e
        ),
    })?;

    Ok(())
}

/// Compute the SHA-256 hash of a file and return hex-encoded digest.
pub fn sha256_file(path: &Path) -> Result<String> {
    let data = fs::read(path).map_err(|e| EmbedError::ModelLoad {
        message: format!("failed to read file for checksum {}: {}", path.display(), e),
    })?;
    let hash = Sha256::digest(&data);
    Ok(hex::encode(hash))
}

/// Verify that a file's SHA-256 checksum matches the expected value.
pub fn verify_checksum(path: &Path, expected: &str) -> Result<()> {
    let actual = sha256_file(path)?;
    if actual != expected {
        return Err(EmbedError::ChecksumMismatch {
            file: path.display().to_string(),
            expected: expected.to_string(),
            actual,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_paths_from_dir() {
        let paths = ModelPaths::from_dir(Path::new("/data/models/bge"));
        assert_eq!(
            paths.model_onnx,
            PathBuf::from("/data/models/bge/model.onnx")
        );
        assert_eq!(
            paths.tokenizer_json,
            PathBuf::from("/data/models/bge/tokenizer.json")
        );
        assert_eq!(
            paths.config_json,
            PathBuf::from("/data/models/bge/config.json")
        );
    }

    #[test]
    fn all_exist_false_when_empty() {
        let dir = tempfile::tempdir().unwrap();
        let paths = ModelPaths::from_dir(dir.path());
        assert!(!paths.all_exist());
    }

    #[test]
    fn sha256_known_content() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("test.txt");
        fs::write(&file, b"hello world").unwrap();
        let hash = sha256_file(&file).unwrap();
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn verify_checksum_correct() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("test.txt");
        fs::write(&file, b"hello world").unwrap();
        verify_checksum(
            &file,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
        )
        .unwrap();
    }

    #[test]
    fn verify_checksum_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("test.txt");
        fs::write(&file, b"hello world").unwrap();
        let err = verify_checksum(&file, "0000000000000000").unwrap_err();
        assert!(matches!(err, EmbedError::ChecksumMismatch { .. }));
    }

    #[test]
    fn ensure_model_files_no_download() {
        let dir = tempfile::tempdir().unwrap();
        let config = EmbedderConfig {
            model_dir: dir.path().to_path_buf(),
            model_config: crate::config::ModelConfig::bge_small_en_v1_5(),
            download_base_url: "http://localhost:0".to_string(),
            auto_download: false,
        };
        let err = ensure_model_files(&config).unwrap_err();
        assert!(matches!(err, EmbedError::ModelLoad { .. }));
    }
}
