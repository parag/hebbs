use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::error::Result;
use crate::normalize::l2_normalize;
use crate::traits::Embedder;

/// Deterministic mock embedder for testing.
///
/// Produces stable, L2-normalized vectors based on a hash of the input text.
/// Does not require ONNX Runtime or any model files.
///
/// Used by `hebbs-core` tests to verify embedding integration without
/// the ~40 MB ONNX model dependency.
pub struct MockEmbedder {
    dims: usize,
}

impl MockEmbedder {
    /// Create a mock embedder that produces vectors of the given dimensionality.
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }

    /// Create a mock embedder with default 384 dimensions
    /// (matching BGE-small-en-v1.5).
    pub fn default_dims() -> Self {
        Self::new(384)
    }

    /// Generate a deterministic vector from text.
    ///
    /// Uses multiple hash seeds to fill the vector, then L2-normalizes.
    /// The hash function produces stable output across runs because
    /// `DefaultHasher` with the same seed sequence is deterministic
    /// within a single Rust version.
    fn hash_to_vector(&self, text: &str) -> Vec<f32> {
        let mut vec = Vec::with_capacity(self.dims);

        for i in 0..self.dims {
            let mut hasher = DefaultHasher::new();
            text.hash(&mut hasher);
            i.hash(&mut hasher);
            let hash_val = hasher.finish();
            let float_val = (hash_val as f64 / u64::MAX as f64) * 2.0 - 1.0;
            vec.push(float_val as f32);
        }

        l2_normalize(&mut vec);
        vec
    }
}

impl Embedder for MockEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.hash_to_vector(text))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.hash_to_vector(t)).collect())
    }

    fn dimensions(&self) -> usize {
        self.dims
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::is_normalized;

    #[test]
    fn deterministic_output() {
        let embedder = MockEmbedder::default_dims();
        let v1 = embedder.embed("hello world").unwrap();
        let v2 = embedder.embed("hello world").unwrap();
        assert_eq!(v1, v2);
    }

    #[test]
    fn different_texts_produce_different_vectors() {
        let embedder = MockEmbedder::default_dims();
        let v1 = embedder.embed("hello").unwrap();
        let v2 = embedder.embed("goodbye").unwrap();
        assert_ne!(v1, v2);
    }

    #[test]
    fn correct_dimensions() {
        let embedder = MockEmbedder::new(128);
        let v = embedder.embed("test").unwrap();
        assert_eq!(v.len(), 128);
        assert_eq!(embedder.dimensions(), 128);
    }

    #[test]
    fn output_is_normalized() {
        let embedder = MockEmbedder::default_dims();
        let v = embedder.embed("normalize test").unwrap();
        assert!(is_normalized(&v, 1e-5));
    }

    #[test]
    fn batch_single_equivalence() {
        let embedder = MockEmbedder::default_dims();
        let single = embedder.embed("test text").unwrap();
        let batch = embedder.embed_batch(&["test text"]).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(single, batch[0]);
    }

    #[test]
    fn batch_matches_individual() {
        let embedder = MockEmbedder::default_dims();
        let texts = ["first", "second", "third"];
        let batch = embedder.embed_batch(&texts).unwrap();
        assert_eq!(batch.len(), 3);

        for (i, text) in texts.iter().enumerate() {
            let single = embedder.embed(text).unwrap();
            assert_eq!(single, batch[i]);
        }
    }

    #[test]
    fn empty_batch_returns_empty() {
        let embedder = MockEmbedder::default_dims();
        let batch = embedder.embed_batch(&[]).unwrap();
        assert!(batch.is_empty());
    }

    #[test]
    fn all_batch_vectors_are_normalized() {
        let embedder = MockEmbedder::default_dims();
        let owned: Vec<String> = (0..100).map(|i| format!("text {}", i)).collect();
        let texts: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
        let batch = embedder.embed_batch(&texts).unwrap();
        for v in &batch {
            assert!(is_normalized(v, 1e-5));
        }
    }

    #[test]
    fn send_sync_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockEmbedder>();
    }

    #[test]
    fn custom_dimensions() {
        for dims in [1, 64, 128, 256, 384, 512, 768, 1024] {
            let embedder = MockEmbedder::new(dims);
            let v = embedder.embed("test").unwrap();
            assert_eq!(v.len(), dims);
            assert_eq!(embedder.dimensions(), dims);
            assert!(is_normalized(&v, 1e-5));
        }
    }

    #[test]
    fn unicode_text_embeds() {
        let embedder = MockEmbedder::default_dims();
        let v = embedder.embed("こんにちは世界 🌍").unwrap();
        assert_eq!(v.len(), 384);
        assert!(is_normalized(&v, 1e-5));
    }

    #[test]
    fn empty_string_embeds() {
        let embedder = MockEmbedder::default_dims();
        let v = embedder.embed("").unwrap();
        assert_eq!(v.len(), 384);
        assert!(is_normalized(&v, 1e-5));
    }

    #[test]
    fn long_text_embeds() {
        let embedder = MockEmbedder::default_dims();
        let long_text = "a".repeat(100_000);
        let v = embedder.embed(&long_text).unwrap();
        assert_eq!(v.len(), 384);
        assert!(is_normalized(&v, 1e-5));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::normalize::{cosine_similarity, is_normalized};
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        /// Every embedding is L2-normalized.
        #[test]
        fn embed_always_normalized(text in ".*") {
            let embedder = MockEmbedder::default_dims();
            let v = embedder.embed(&text).unwrap();
            prop_assert!(is_normalized(&v, 1e-5),
                "embedding not normalized for text: {:?}", text);
        }

        /// Embedding is deterministic: same text always produces same vector.
        #[test]
        fn embed_deterministic(text in ".*") {
            let embedder = MockEmbedder::default_dims();
            let v1 = embedder.embed(&text).unwrap();
            let v2 = embedder.embed(&text).unwrap();
            prop_assert_eq!(v1, v2);
        }

        /// embed() and embed_batch() produce identical results.
        #[test]
        fn embed_single_matches_batch(text in ".{1,100}") {
            let embedder = MockEmbedder::default_dims();
            let single = embedder.embed(&text).unwrap();
            let batch = embedder.embed_batch(&[&text]).unwrap();
            prop_assert_eq!(batch.len(), 1);
            prop_assert_eq!(&single, &batch[0]);
        }

        /// All vectors in a batch are normalized.
        #[test]
        fn batch_all_normalized(
            texts in prop::collection::vec(".{1,50}", 1..32),
        ) {
            let embedder = MockEmbedder::default_dims();
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let batch = embedder.embed_batch(&refs).unwrap();
            prop_assert_eq!(batch.len(), texts.len());
            for v in &batch {
                prop_assert!(is_normalized(v, 1e-5));
            }
        }

        /// Self-similarity is ~1.0 for any text.
        #[test]
        fn self_similarity_is_one(text in ".{1,200}") {
            let embedder = MockEmbedder::default_dims();
            let v = embedder.embed(&text).unwrap();
            let sim = cosine_similarity(&v, &v);
            prop_assert!((sim - 1.0).abs() < 1e-4,
                "self-similarity = {}", sim);
        }

        /// dimensions() always matches embed() output length.
        #[test]
        fn dimensions_match(dims in 1usize..1024, text in ".{1,50}") {
            let embedder = MockEmbedder::new(dims);
            let v = embedder.embed(&text).unwrap();
            prop_assert_eq!(v.len(), embedder.dimensions());
        }
    }
}
