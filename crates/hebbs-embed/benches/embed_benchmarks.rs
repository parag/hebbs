use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hebbs_embed::mock::MockEmbedder;
use hebbs_embed::normalize::l2_normalize;
use hebbs_embed::traits::Embedder;

fn bench_mock_embed_single(c: &mut Criterion) {
    let embedder = MockEmbedder::default_dims();

    c.bench_function("mock_embed_single", |b| {
        b.iter(|| {
            embedder
                .embed(black_box("Customer expressed urgency about Q4 deadline"))
                .unwrap()
        });
    });
}

fn bench_mock_embed_batch_16(c: &mut Criterion) {
    let embedder = MockEmbedder::default_dims();
    let owned: Vec<String> = (0..16)
        .map(|i| format!("This is test sentence number {} for batch embedding", i))
        .collect();
    let texts: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();

    c.bench_function("mock_embed_batch_16", |b| {
        b.iter(|| embedder.embed_batch(black_box(&texts)).unwrap());
    });
}

fn bench_mock_embed_batch_64(c: &mut Criterion) {
    let embedder = MockEmbedder::default_dims();
    let owned: Vec<String> = (0..64)
        .map(|i| format!("This is test sentence number {} for batch embedding", i))
        .collect();
    let texts: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();

    c.bench_function("mock_embed_batch_64", |b| {
        b.iter(|| embedder.embed_batch(black_box(&texts)).unwrap());
    });
}

fn bench_l2_normalize_384(c: &mut Criterion) {
    let mut v: Vec<f32> = (0..384).map(|i| i as f32 * 0.01).collect();

    c.bench_function("l2_normalize_384", |b| {
        b.iter(|| {
            l2_normalize(black_box(&mut v));
        });
    });
}

criterion_group!(
    benches,
    bench_mock_embed_single,
    bench_mock_embed_batch_16,
    bench_mock_embed_batch_64,
    bench_l2_normalize_384,
);
criterion_main!(benches);
