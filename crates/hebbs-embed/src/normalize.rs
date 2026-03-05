/// L2-normalize a vector in-place.
///
/// After normalization, ‖v‖ = 1.0 within floating-point tolerance.
/// This is critical: Phase 3 HNSW uses inner product distance which
/// equals cosine similarity only when vectors are L2-normalized.
///
/// Zero vectors (or vectors whose squared norm underflows to zero)
/// remain unchanged (avoids division by zero).
///
/// Complexity: O(d) where d = vector dimensionality.
#[inline]
pub fn l2_normalize(v: &mut [f32]) {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    if norm_sq > 0.0 {
        let inv_norm = 1.0 / norm_sq.sqrt();
        if inv_norm.is_finite() {
            for x in v.iter_mut() {
                *x *= inv_norm;
            }
        }
    }
}

/// Check whether a vector is L2-normalized within the given tolerance.
///
/// Returns `true` if |‖v‖ − 1.0| < tolerance.
#[inline]
pub fn is_normalized(v: &[f32], tolerance: f32) -> bool {
    if v.is_empty() {
        return false;
    }
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    (norm_sq.sqrt() - 1.0).abs() < tolerance
}

/// Compute the cosine similarity between two L2-normalized vectors.
///
/// Since both vectors are unit length, cosine similarity equals the
/// inner (dot) product: cos(θ) = a · b / (‖a‖ ‖b‖) = a · b.
///
/// Complexity: O(d).
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have equal dimensionality");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_unit_vector() {
        let mut v = vec![1.0, 0.0, 0.0];
        l2_normalize(&mut v);
        assert!(is_normalized(&v, 1e-6));
        assert!((v[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_3_4_triangle() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        assert!(is_normalized(&v, 1e-6));
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn normalize_zero_vector_stays_zero() {
        let mut v = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn normalize_already_normalized() {
        let mut v = vec![0.6, 0.8];
        l2_normalize(&mut v);
        assert!(is_normalized(&v, 1e-6));
    }

    #[test]
    fn normalize_384_dimensions() {
        let mut v = vec![1.0; 384];
        l2_normalize(&mut v);
        assert!(is_normalized(&v, 1e-5));
        let expected = 1.0 / (384.0f32).sqrt();
        for x in &v {
            assert!((x - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn normalize_very_small_values() {
        let mut v = vec![1e-20, 1e-20];
        l2_normalize(&mut v);
        assert!(is_normalized(&v, 1e-5));
    }

    #[test]
    fn normalize_negative_values() {
        let mut v = vec![-3.0, 4.0];
        l2_normalize(&mut v);
        assert!(is_normalized(&v, 1e-6));
        assert!((v[0] - (-0.6)).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn normalize_large_values() {
        // 1e18^2 = 1e36, within f32 range (~3.4e38)
        let mut v = vec![1e18, 1e18, 1e18];
        l2_normalize(&mut v);
        assert!(is_normalized(&v, 1e-5));
    }

    #[test]
    fn normalize_single_element() {
        let mut v = vec![42.0];
        l2_normalize(&mut v);
        assert!((v[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_idempotent() {
        let mut v = vec![3.0, 4.0, 5.0, 6.0];
        l2_normalize(&mut v);
        let after_first = v.clone();
        l2_normalize(&mut v);
        for (a, b) in after_first.iter().zip(v.iter()) {
            assert!((a - b).abs() < 1e-6, "normalization is not idempotent");
        }
    }

    #[test]
    fn normalize_preserves_direction() {
        let mut v = vec![3.0, -4.0, 5.0];
        let original = v.clone();
        l2_normalize(&mut v);
        // All components should have the same sign
        for (o, n) in original.iter().zip(v.iter()) {
            assert!(o.signum() == n.signum(), "normalization changed sign");
        }
    }

    #[test]
    fn cosine_identical_vectors() {
        let a = vec![0.6, 0.8];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_symmetry() {
        let mut a = vec![1.0, 2.0, 3.0];
        let mut b = vec![4.0, 5.0, 6.0];
        l2_normalize(&mut a);
        l2_normalize(&mut b);
        let sim_ab = cosine_similarity(&a, &b);
        let sim_ba = cosine_similarity(&b, &a);
        assert!((sim_ab - sim_ba).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_bounded() {
        let mut a = vec![1.0, -2.0, 3.0, -4.0];
        let mut b = vec![-5.0, 6.0, -7.0, 8.0];
        l2_normalize(&mut a);
        l2_normalize(&mut b);
        let sim = cosine_similarity(&a, &b);
        assert!((-1.0 - 1e-6..=1.0 + 1e-6).contains(&sim));
    }

    #[test]
    fn is_normalized_empty_returns_false() {
        assert!(!is_normalized(&[], 1e-6));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]

        /// L2 normalization always produces a unit vector for non-zero input.
        #[test]
        fn normalize_always_unit(v in prop::collection::vec(-1e6f32..1e6f32, 1..512usize)) {
            let has_nonzero = v.iter().any(|x| *x != 0.0);
            let mut v = v;
            l2_normalize(&mut v);
            if has_nonzero {
                prop_assert!(is_normalized(&v, 1e-4),
                    "vector not normalized, norm = {}",
                    v.iter().map(|x| x * x).sum::<f32>().sqrt());
            }
        }

        /// Normalization is idempotent: normalizing twice gives same result.
        #[test]
        fn normalize_idempotent_prop(v in prop::collection::vec(-100.0f32..100.0f32, 1..256usize)) {
            let mut v = v;
            l2_normalize(&mut v);
            let after_first = v.clone();
            l2_normalize(&mut v);
            for (a, b) in after_first.iter().zip(v.iter()) {
                prop_assert!((a - b).abs() < 1e-5,
                    "normalization not idempotent: {} vs {}", a, b);
            }
        }

        /// Cosine similarity of a vector with itself is ~1.0.
        #[test]
        fn cosine_self_similarity(v in prop::collection::vec(-100.0f32..100.0f32, 2..128usize)) {
            let has_nonzero = v.iter().any(|x| *x != 0.0);
            if !has_nonzero {
                return Ok(());
            }
            let mut v = v;
            l2_normalize(&mut v);
            let sim = cosine_similarity(&v, &v);
            prop_assert!((sim - 1.0).abs() < 1e-4,
                "self-similarity {} should be ~1.0", sim);
        }

        /// Cosine similarity is always in [-1, 1] for normalized vectors.
        #[test]
        fn cosine_bounded(
            a in prop::collection::vec(-100.0f32..100.0f32, 64..=64),
            b in prop::collection::vec(-100.0f32..100.0f32, 64..=64),
        ) {
            let a_nonzero = a.iter().any(|x| *x != 0.0);
            let b_nonzero = b.iter().any(|x| *x != 0.0);
            if !a_nonzero || !b_nonzero {
                return Ok(());
            }
            let mut a = a;
            let mut b = b;
            l2_normalize(&mut a);
            l2_normalize(&mut b);
            let sim = cosine_similarity(&a, &b);
            prop_assert!((-1.0 - 1e-4..=1.0 + 1e-4).contains(&sim),
                "cosine similarity {} out of bounds", sim);
        }

        /// Cosine similarity is symmetric: sim(a,b) == sim(b,a).
        #[test]
        fn cosine_symmetric(
            a in prop::collection::vec(-100.0f32..100.0f32, 64..=64),
            b in prop::collection::vec(-100.0f32..100.0f32, 64..=64),
        ) {
            let mut a = a;
            let mut b = b;
            l2_normalize(&mut a);
            l2_normalize(&mut b);
            let sim_ab = cosine_similarity(&a, &b);
            let sim_ba = cosine_similarity(&b, &a);
            prop_assert!((sim_ab - sim_ba).abs() < 1e-6,
                "symmetry violated: {} vs {}", sim_ab, sim_ba);
        }
    }
}
