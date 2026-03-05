use crate::error::{ReflectError, Result};
use crate::types::Cluster;
use rand::prelude::*;
use rand::rngs::StdRng;

pub struct ClusterConfig {
    pub min_cluster_size: usize,
    pub max_clusters: usize,
    pub seed: u64,
    pub max_iterations: usize,
    pub silhouette_subsample: usize,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            min_cluster_size: 3,
            max_clusters: 50,
            seed: 42,
            max_iterations: 50,
            silhouette_subsample: 500,
        }
    }
}

/// Cluster L2-normalized embedding vectors using spherical K-Means
/// with silhouette-guided k selection.
///
/// Time: O(n * k_max * d * iterations) for the sweep.
/// Memory: O(n * d + k * d).
pub fn cluster_embeddings(embeddings: &[Vec<f32>], config: &ClusterConfig) -> Result<Vec<Cluster>> {
    let n = embeddings.len();
    if n == 0 {
        return Err(ReflectError::Clustering {
            message: "no embeddings to cluster".into(),
        });
    }
    let d = embeddings[0].len();
    if d == 0 {
        return Err(ReflectError::Clustering {
            message: "zero-dimensional embeddings".into(),
        });
    }

    if n < config.min_cluster_size {
        return Err(ReflectError::InsufficientMemories {
            have: n,
            need: config.min_cluster_size,
        });
    }

    let k_max = config
        .max_clusters
        .min((n as f64).sqrt().ceil() as usize)
        .max(2);
    let k_min = 2usize;

    if n < k_min * config.min_cluster_size {
        let centroid = mean_centroid(embeddings, d);
        return Ok(vec![Cluster {
            cluster_id: 0,
            member_indices: (0..n).collect(),
            centroid,
        }]);
    }

    let mut rng = StdRng::seed_from_u64(config.seed);

    let subsample: Vec<usize> = if n > config.silhouette_subsample {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.shuffle(&mut rng);
        idx.truncate(config.silhouette_subsample);
        idx
    } else {
        (0..n).collect()
    };

    let mut best_k = k_min;
    let mut best_score = f64::NEG_INFINITY;

    for k in k_min..=k_max {
        let mut rng_k = StdRng::seed_from_u64(config.seed.wrapping_add(k as u64));
        let assignments = run_kmeans(embeddings, k, d, config.max_iterations, &mut rng_k);
        let score = silhouette_score(embeddings, &assignments, k, &subsample, d);
        if score > best_score {
            best_score = score;
            best_k = k;
        }
    }

    let mut rng_final = StdRng::seed_from_u64(config.seed.wrapping_add(best_k as u64));
    let assignments = run_kmeans(embeddings, best_k, d, config.max_iterations, &mut rng_final);

    build_clusters(embeddings, &assignments, best_k, d, config.min_cluster_size)
}

/// K-means++ initialization: choose k centroids that are well-separated.
fn kmeans_pp_init(embeddings: &[Vec<f32>], k: usize, d: usize, rng: &mut StdRng) -> Vec<Vec<f32>> {
    let n = embeddings.len();
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    let first = rng.gen_range(0..n);
    centroids.push(embeddings[first].clone());

    let mut min_dist = vec![f64::MAX; n];

    for _ in 1..k {
        let last_centroid = centroids.last().unwrap();
        for i in 0..n {
            let dist = cosine_distance(&embeddings[i], last_centroid, d);
            if dist < min_dist[i] {
                min_dist[i] = dist;
            }
        }
        let total: f64 = min_dist.iter().sum();
        if total <= 0.0 {
            let idx = rng.gen_range(0..n);
            centroids.push(embeddings[idx].clone());
            continue;
        }
        let threshold = rng.gen::<f64>() * total;
        let mut cumulative = 0.0;
        let mut chosen = 0;
        for (i, &d) in min_dist.iter().enumerate() {
            cumulative += d;
            if cumulative >= threshold {
                chosen = i;
                break;
            }
        }
        centroids.push(embeddings[chosen].clone());
    }

    centroids
}

/// Run spherical K-means: assign points to nearest centroid (by cosine distance),
/// then update centroids as normalized mean of assigned points.
fn run_kmeans(
    embeddings: &[Vec<f32>],
    k: usize,
    d: usize,
    max_iter: usize,
    rng: &mut StdRng,
) -> Vec<usize> {
    let n = embeddings.len();
    let mut centroids = kmeans_pp_init(embeddings, k, d, rng);
    let mut assignments = vec![0usize; n];

    for _ in 0..max_iter {
        let mut changed = false;
        for i in 0..n {
            let mut best_c = 0;
            let mut best_dist = f64::MAX;
            for (c, centroid) in centroids.iter().enumerate() {
                let dist = cosine_distance(&embeddings[i], centroid, d);
                if dist < best_dist {
                    best_dist = dist;
                    best_c = c;
                }
            }
            if assignments[i] != best_c {
                assignments[i] = best_c;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        let mut sums = vec![vec![0.0f64; d]; k];
        let mut counts = vec![0usize; k];
        for (i, &a) in assignments.iter().enumerate() {
            counts[a] += 1;
            for j in 0..d {
                sums[a][j] += embeddings[i][j] as f64;
            }
        }

        for c in 0..k {
            if counts[c] == 0 {
                let idx = rng.gen_range(0..n);
                centroids[c] = embeddings[idx].clone();
            } else {
                let inv = 1.0 / counts[c] as f64;
                let mut norm_sq = 0.0f64;
                for val in sums[c].iter_mut().take(d) {
                    *val *= inv;
                    norm_sq += *val * *val;
                }
                let norm = norm_sq.sqrt().max(1e-12);
                centroids[c] = sums[c].iter().map(|&v| (v / norm) as f32).collect();
            }
        }
    }

    assignments
}

/// Silhouette score on a subsample. Higher is better, range [-1, 1].
fn silhouette_score(
    embeddings: &[Vec<f32>],
    assignments: &[usize],
    k: usize,
    subsample: &[usize],
    d: usize,
) -> f64 {
    if k <= 1 || subsample.is_empty() {
        return -1.0;
    }

    let mut total = 0.0f64;
    let mut count = 0usize;

    for &i in subsample {
        let ci = assignments[i];

        let mut intra_sum = 0.0f64;
        let mut intra_count = 0usize;
        for (j, &aj) in assignments.iter().enumerate() {
            if j != i && aj == ci {
                intra_sum += cosine_distance(&embeddings[i], &embeddings[j], d);
                intra_count += 1;
            }
        }
        if intra_count == 0 {
            continue;
        }
        let a_i = intra_sum / intra_count as f64;

        let mut b_i = f64::MAX;
        for ck in 0..k {
            if ck == ci {
                continue;
            }
            let mut inter_sum = 0.0f64;
            let mut inter_count = 0usize;
            for (j, &aj) in assignments.iter().enumerate() {
                if aj == ck {
                    inter_sum += cosine_distance(&embeddings[i], &embeddings[j], d);
                    inter_count += 1;
                }
            }
            if inter_count > 0 {
                let avg = inter_sum / inter_count as f64;
                if avg < b_i {
                    b_i = avg;
                }
            }
        }

        if b_i == f64::MAX {
            continue;
        }

        let denom = a_i.max(b_i);
        if denom > 0.0 {
            total += (b_i - a_i) / denom;
            count += 1;
        }
    }

    if count == 0 {
        -1.0
    } else {
        total / count as f64
    }
}

fn build_clusters(
    embeddings: &[Vec<f32>],
    assignments: &[usize],
    k: usize,
    d: usize,
    min_size: usize,
) -> Result<Vec<Cluster>> {
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &a) in assignments.iter().enumerate() {
        groups[a].push(i);
    }

    let mut clusters = Vec::new();
    let mut cid = 0;
    for members in groups {
        if members.len() < min_size {
            continue;
        }
        let member_embeddings: Vec<&Vec<f32>> = members.iter().map(|&i| &embeddings[i]).collect();
        let refs: Vec<&[f32]> = member_embeddings.iter().map(|v| v.as_slice()).collect();
        let centroid = mean_centroid_refs(&refs, d);
        clusters.push(Cluster {
            cluster_id: cid,
            member_indices: members,
            centroid,
        });
        cid += 1;
    }

    Ok(clusters)
}

#[inline]
fn cosine_distance(a: &[f32], b: &[f32], d: usize) -> f64 {
    let mut dot = 0.0f64;
    for i in 0..d {
        dot += a[i] as f64 * b[i] as f64;
    }
    1.0 - dot
}

fn mean_centroid(embeddings: &[Vec<f32>], d: usize) -> Vec<f32> {
    let n = embeddings.len();
    let mut sum = vec![0.0f64; d];
    for emb in embeddings {
        for j in 0..d {
            sum[j] += emb[j] as f64;
        }
    }
    let inv = 1.0 / n as f64;
    let mut norm_sq = 0.0f64;
    for v in &mut sum {
        *v *= inv;
        norm_sq += *v * *v;
    }
    let norm = norm_sq.sqrt().max(1e-12);
    sum.iter().map(|&v| (v / norm) as f32).collect()
}

fn mean_centroid_refs(embeddings: &[&[f32]], d: usize) -> Vec<f32> {
    let n = embeddings.len();
    let mut sum = vec![0.0f64; d];
    for emb in embeddings {
        for j in 0..d {
            sum[j] += emb[j] as f64;
        }
    }
    let inv = 1.0 / n as f64;
    let mut norm_sq = 0.0f64;
    for v in &mut sum {
        *v *= inv;
        norm_sq += *v * *v;
    }
    let norm = norm_sq.sqrt().max(1e-12);
    sum.iter().map(|&v| (v / norm) as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cluster_data(centers: &[Vec<f32>], per_center: usize, noise: f32) -> Vec<Vec<f32>> {
        let d = centers[0].len();
        let mut rng = StdRng::seed_from_u64(123);
        let mut data = Vec::new();
        for center in centers {
            for _ in 0..per_center {
                let mut v: Vec<f32> = center
                    .iter()
                    .map(|&c| c + (rng.gen::<f32>() - 0.5) * noise)
                    .collect();
                l2_normalize(&mut v, d);
                data.push(v);
            }
        }
        data
    }

    fn l2_normalize(v: &mut [f32], d: usize) {
        let mut norm = 0.0f64;
        for val in v.iter().take(d) {
            norm += *val as f64 * *val as f64;
        }
        let norm = norm.sqrt().max(1e-12);
        for val in v.iter_mut().take(d) {
            *val = (*val as f64 / norm) as f32;
        }
    }

    fn unit_vec(d: usize, idx: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; d];
        v[idx % d] = 1.0;
        v
    }

    #[test]
    fn three_well_separated_clusters() {
        let centers = vec![unit_vec(16, 0), unit_vec(16, 5), unit_vec(16, 10)];
        let data = make_cluster_data(&centers, 15, 0.3);
        let config = ClusterConfig {
            min_cluster_size: 3,
            max_clusters: 10,
            seed: 42,
            max_iterations: 50,
            silhouette_subsample: 500,
        };
        let clusters = cluster_embeddings(&data, &config).unwrap();
        assert!(
            clusters.len() >= 2 && clusters.len() <= 5,
            "expected 2-5 clusters, got {}",
            clusters.len()
        );
        for c in &clusters {
            assert!(c.member_indices.len() >= 3);
        }
    }

    #[test]
    fn too_few_embeddings() {
        let data = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let config = ClusterConfig {
            min_cluster_size: 5,
            ..Default::default()
        };
        let err = cluster_embeddings(&data, &config).unwrap_err();
        assert!(matches!(
            err,
            crate::error::ReflectError::InsufficientMemories { .. }
        ));
    }

    #[test]
    fn single_cluster_when_not_enough_for_two() {
        let data = make_cluster_data(&[unit_vec(8, 0)], 4, 0.1);
        let config = ClusterConfig {
            min_cluster_size: 3,
            max_clusters: 10,
            seed: 42,
            max_iterations: 50,
            silhouette_subsample: 500,
        };
        let clusters = cluster_embeddings(&data, &config).unwrap();
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].member_indices.len(), 4);
    }

    #[test]
    fn deterministic_given_same_seed() {
        let centers = vec![unit_vec(16, 0), unit_vec(16, 5), unit_vec(16, 10)];
        let data = make_cluster_data(&centers, 20, 0.3);
        let config = ClusterConfig {
            min_cluster_size: 3,
            max_clusters: 10,
            seed: 99,
            max_iterations: 50,
            silhouette_subsample: 500,
        };
        let c1 = cluster_embeddings(&data, &config).unwrap();
        let c2 = cluster_embeddings(&data, &config).unwrap();
        assert_eq!(c1.len(), c2.len());
        for (a, b) in c1.iter().zip(c2.iter()) {
            assert_eq!(a.member_indices, b.member_indices);
        }
    }

    #[test]
    fn min_cluster_size_filters_small_groups() {
        let mut data = make_cluster_data(&[unit_vec(16, 0)], 20, 0.1);
        let mut outlier = unit_vec(16, 15);
        l2_normalize(&mut outlier, 16);
        data.push(outlier.clone());
        data.push(outlier);
        let config = ClusterConfig {
            min_cluster_size: 5,
            max_clusters: 10,
            seed: 42,
            max_iterations: 50,
            silhouette_subsample: 500,
        };
        let clusters = cluster_embeddings(&data, &config).unwrap();
        for c in &clusters {
            assert!(c.member_indices.len() >= 5);
        }
    }

    #[test]
    fn centroid_is_normalized() {
        let centers = vec![unit_vec(16, 0), unit_vec(16, 8)];
        let data = make_cluster_data(&centers, 10, 0.2);
        let config = ClusterConfig {
            min_cluster_size: 3,
            max_clusters: 10,
            seed: 42,
            max_iterations: 50,
            silhouette_subsample: 500,
        };
        let clusters = cluster_embeddings(&data, &config).unwrap();
        for c in &clusters {
            let norm: f64 = c.centroid.iter().map(|&x| x as f64 * x as f64).sum();
            assert!(
                (norm.sqrt() - 1.0).abs() < 0.01,
                "centroid not normalized: norm = {}",
                norm.sqrt()
            );
        }
    }

    #[test]
    fn empty_embeddings_error() {
        let data: Vec<Vec<f32>> = Vec::new();
        let config = ClusterConfig::default();
        assert!(cluster_embeddings(&data, &config).is_err());
    }

    #[test]
    fn no_member_appears_in_two_clusters() {
        let centers = vec![unit_vec(32, 0), unit_vec(32, 10), unit_vec(32, 20)];
        let data = make_cluster_data(&centers, 20, 0.3);
        let config = ClusterConfig {
            min_cluster_size: 3,
            max_clusters: 10,
            seed: 42,
            max_iterations: 50,
            silhouette_subsample: 500,
        };
        let clusters = cluster_embeddings(&data, &config).unwrap();
        let mut seen = std::collections::HashSet::new();
        for c in &clusters {
            for &idx in &c.member_indices {
                assert!(
                    seen.insert(idx),
                    "member {} appears in multiple clusters",
                    idx
                );
            }
        }
    }

    #[test]
    fn cluster_ids_are_sequential() {
        let centers = vec![unit_vec(16, 0), unit_vec(16, 5)];
        let data = make_cluster_data(&centers, 10, 0.2);
        let config = ClusterConfig {
            min_cluster_size: 3,
            max_clusters: 10,
            seed: 42,
            max_iterations: 50,
            silhouette_subsample: 500,
        };
        let clusters = cluster_embeddings(&data, &config).unwrap();
        for (i, c) in clusters.iter().enumerate() {
            assert_eq!(c.cluster_id, i);
        }
    }
}
