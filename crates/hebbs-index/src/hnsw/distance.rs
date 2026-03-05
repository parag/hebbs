//! Inner product distance computation for HNSW search.
//!
//! For L2-normalized vectors (guaranteed by Phase 2), inner product
//! equals cosine similarity. We convert to a distance metric where
//! lower values indicate closer vectors: distance = 1.0 - inner_product.
//!
//! This gives a distance in \[0.0, 2.0\] for normalized vectors:
//! - 0.0: identical vectors (inner product = 1.0)
//! - 1.0: orthogonal vectors (inner product = 0.0)
//! - 2.0: opposite vectors (inner product = -1.0)

/// Compute the inner product (dot product) of two vectors.
///
/// Complexity: O(d) where d = dimensionality.
/// At 384-dim: 384 multiply-accumulate operations.
#[inline]
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "vectors must have equal dimensionality for inner product"
    );
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Convert inner product similarity to a distance metric.
/// distance = 1.0 - inner_product(a, b)
///
/// For L2-normalized vectors: range [0.0, 2.0].
/// Lower distance = more similar.
///
/// Complexity: O(d).
#[inline]
pub fn inner_product_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - inner_product(a, b)
}

/// Brute-force search: compute distances between query and all candidates,
/// return top-K closest sorted by distance ascending.
///
/// Used as a reference implementation for testing HNSW correctness.
///
/// Complexity: O(n * d + n * log k) where n = candidates, d = dims, k = top-k.
pub fn brute_force_search<'a>(
    query: &[f32],
    candidates: impl Iterator<Item = (&'a [u8; 16], &'a [f32])>,
    k: usize,
) -> Vec<([u8; 16], f32)> {
    let mut results: Vec<([u8; 16], f32)> = candidates
        .map(|(id, vec)| (*id, inner_product_distance(query, vec)))
        .collect();

    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_zero_distance() {
        let a = vec![0.6, 0.8];
        let dist = inner_product_distance(&a, &a);
        assert!(
            dist.abs() < 1e-6,
            "distance should be ~0 for identical vectors"
        );
    }

    #[test]
    fn orthogonal_vectors_distance_one() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let dist = inner_product_distance(&a, &b);
        assert!((dist - 1.0).abs() < 1e-6);
    }

    #[test]
    fn opposite_vectors_distance_two() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let dist = inner_product_distance(&a, &b);
        assert!((dist - 2.0).abs() < 1e-6);
    }

    #[test]
    fn inner_product_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let ip = inner_product(&a, &b);
        assert!((ip - 32.0).abs() < 1e-6); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn brute_force_returns_closest() {
        let query = [1.0, 0.0, 0.0];
        let id_a = [1u8; 16];
        let id_b = [2u8; 16];
        let id_c = [3u8; 16];

        let vec_a = [0.9, 0.1, 0.0]; // closest to query
        let vec_b = [0.0, 1.0, 0.0]; // orthogonal
        let vec_c = [-1.0, 0.0, 0.0]; // opposite

        let candidates = vec![
            (&id_a, vec_a.as_slice()),
            (&id_b, vec_b.as_slice()),
            (&id_c, vec_c.as_slice()),
        ];

        let results = brute_force_search(&query, candidates.into_iter(), 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, id_a);
        assert_eq!(results[1].0, id_b);
    }

    #[test]
    fn brute_force_empty_candidates() {
        let query = [1.0, 0.0];
        let results = brute_force_search(&query, std::iter::empty::<(&[u8; 16], &[f32])>(), 10);
        assert!(results.is_empty());
    }
}
