use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hebbs_reflect::cluster::{cluster_embeddings, ClusterConfig};
use hebbs_reflect::pipeline::ReflectPipeline;
use hebbs_reflect::types::{MemoryEntry, PipelineConfig, ReflectInput};
use hebbs_reflect::MockLlmProvider;
use rand::prelude::*;

fn make_clustered_memories(cluster_count: usize, per_cluster: usize, d: usize) -> Vec<MemoryEntry> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut memories = Vec::new();
    let mut idx = 0u16;
    for c in 0..cluster_count {
        let mut center = vec![0.0f32; d];
        center[c % d] = 1.0;
        for _ in 0..per_cluster {
            let mut emb: Vec<f32> = center
                .iter()
                .map(|&x| x + (rng.gen::<f32>() - 0.5) * 0.3)
                .collect();
            let norm: f64 = emb
                .iter()
                .map(|&x| x as f64 * x as f64)
                .sum::<f64>()
                .sqrt()
                .max(1e-12);
            for v in &mut emb {
                *v = (*v as f64 / norm) as f32;
            }
            let mut id = [0u8; 16];
            id[0] = (idx >> 8) as u8;
            id[1] = idx as u8;
            memories.push(MemoryEntry {
                id,
                content: format!("Cluster {c} memory {idx}"),
                importance: 0.6,
                entity_id: Some("bench".into()),
                embedding: emb,
                created_at: 1_000_000 * idx as u64,
            });
            idx += 1;
        }
    }
    memories
}

fn bench_clustering(c: &mut Criterion) {
    let memories_1k = make_clustered_memories(5, 200, 384);
    let embeddings_1k: Vec<Vec<f32>> = memories_1k.iter().map(|m| m.embedding.clone()).collect();

    c.bench_function("cluster_1k_384d", |b| {
        b.iter(|| {
            let config = ClusterConfig {
                min_cluster_size: 3,
                max_clusters: 20,
                seed: 42,
                max_iterations: 30,
                silhouette_subsample: 200,
            };
            black_box(cluster_embeddings(&embeddings_1k, &config).unwrap())
        })
    });
}

fn bench_full_pipeline(c: &mut Criterion) {
    let memories = make_clustered_memories(3, 20, 64);
    let mock = MockLlmProvider::new();

    c.bench_function("pipeline_60_memories_mock_llm", |b| {
        b.iter(|| {
            let input = ReflectInput {
                memories: memories.clone(),
                existing_insights: Vec::new(),
                config: PipelineConfig {
                    min_cluster_size: 3,
                    max_clusters: 10,
                    clustering_seed: 42,
                    max_iterations: 30,
                    proposal_max_tokens: 4000,
                    validation_max_tokens: 6000,
                    insight_importance_weight: 0.7,
                },
            };
            black_box(ReflectPipeline::run(input, &mock, &mock).unwrap())
        })
    });
}

criterion_group!(benches, bench_clustering, bench_full_pipeline);
criterion_main!(benches);
