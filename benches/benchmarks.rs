use criterion::{Criterion, criterion_group, criterion_main};
use proxima_ml::{
    Canberra, Chebyshev, Cosine, DistanceExt, Dot, Euclidean, Manhattan, SimilarityExt, SqEuclidean,
};

fn generate_vecs_f64(dim: usize) -> (Vec<f64>, Vec<f64>) {
    let a: Vec<f64> = (0..dim).map(|i| (i as f64) * 0.001).collect();
    let b: Vec<f64> = (0..dim).map(|i| (i as f64) * 0.002 + 0.5).collect();
    (a, b)
}

fn generate_vecs_f32(dim: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.002 + 0.5).collect();
    (a, b)
}

fn bench_distances_f64(c: &mut Criterion) {
    let dims = [128, 768, 1536];

    for &dim in &dims {
        let (a, b) = generate_vecs_f64(dim);
        let mut group = c.benchmark_group(format!("f64_{}d", dim));

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

fn bench_distances_f32(c: &mut Criterion) {
    let dims = [128, 768, 1536];

    for &dim in &dims {
        let (a, b) = generate_vecs_f32(dim);
        let mut group = c.benchmark_group(format!("f32_{}d", dim));

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

fn generate_points_f64(n: usize, dim: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..dim).map(|j| (i * dim + j) as f64 * 0.001).collect())
        .collect()
}

fn bench_batch_distance(c: &mut Criterion) {
    let dim = 768;
    let n = 10_000;
    let points = generate_points_f64(n, dim);
    let query: Vec<f64> = (0..dim).map(|i| i as f64 * 0.003).collect();
    let targets: Vec<&[f64]> = points.iter().map(|p| p.as_slice()).collect();

    let mut group = c.benchmark_group("batch_distance_10k_768d");
    group.bench_function("sequential", |b| {
        b.iter(|| Euclidean::batch_distance(query.as_slice(), targets.clone()))
    });
    #[cfg(feature = "parallel")]
    group.bench_function("parallel", |b| {
        b.iter(|| Euclidean::par_batch_distance(query.as_slice(), targets.clone()))
    });
    group.finish();
}

fn bench_pdist(c: &mut Criterion) {
    let dim = 128;
    let n = 1_000;
    let points = generate_points_f64(n, dim);

    let mut group = c.benchmark_group("pdist_1k_128d");
    group.bench_function("sequential", |b| {
        b.iter(|| Euclidean::pdist(&points))
    });
    #[cfg(feature = "parallel")]
    group.bench_function("parallel", |b| {
        b.iter(|| Euclidean::par_pdist(&points))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_distances_f64,
    bench_distances_f32,
    bench_batch_distance,
    bench_pdist
);
criterion_main!(benches);
