use criterion::{Criterion, criterion_group, criterion_main};
use proxima_ml::{
    Canberra, Chebyshev, Cosine, DistanceExt, Dot, Euclidean, Manhattan, SimilarityExt, SqEuclidean,
};

fn generate_vecs(dim: usize) -> (Vec<f64>, Vec<f64>) {
    let a: Vec<f64> = (0..dim).map(|i| (i as f64) * 0.001).collect();
    let b: Vec<f64> = (0..dim).map(|i| (i as f64) * 0.002 + 0.5).collect();
    (a, b)
}

fn bench_distances(c: &mut Criterion) {
    let dims = [128, 768, 1536];

    for &dim in &dims {
        let (a, b) = generate_vecs(dim);

        let mut group = c.benchmark_group(format!("distance_{}d", dim));

        group.bench_function("euclidean", |bencher| {
            bencher.iter(|| Euclidean::distance(&a, &b))
        });

        group.bench_function("sq_euclidean", |bencher| {
            bencher.iter(|| SqEuclidean::distance(&a, &b))
        });

        group.bench_function("manhattan", |bencher| {
            bencher.iter(|| Manhattan::distance(&a, &b))
        });

        group.bench_function("chebyshev", |bencher| {
            bencher.iter(|| Chebyshev::distance(&a, &b))
        });

        group.bench_function("canberra", |bencher| {
            bencher.iter(|| Canberra::distance(&a, &b))
        });

        group.bench_function("cosine", |bencher| {
            bencher.iter(|| Cosine::distance(&a, &b))
        });

        group.bench_function("dot", |bencher| bencher.iter(|| Dot::similarity(&a, &b)));

        group.finish();
    }
}

criterion_group!(benches, bench_distances);
criterion_main!(benches);
