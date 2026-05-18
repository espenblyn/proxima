#![cfg(feature = "parallel")]

use approx::assert_abs_diff_eq;
use proxima_ml::{Chebyshev, DistanceExt, Euclidean, Manhattan};

fn sample_points_f64() -> Vec<Vec<f64>> {
    vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
        vec![0.0, 1.0, 0.0],
        vec![2.5, 3.5, 4.5],
    ]
}

fn sample_points_f32() -> Vec<Vec<f32>> {
    vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
        vec![0.0, 1.0, 0.0],
        vec![2.5, 3.5, 4.5],
    ]
}

#[test]
fn par_pdist_matches_pdist_euclidean_f64() {
    let points = sample_points_f64();
    let sequential = Euclidean::pdist(&points);
    let parallel = Euclidean::par_pdist(&points);
    assert_eq!(sequential.len(), parallel.len());
    for (s, p) in sequential.iter().zip(parallel.iter()) {
        assert_abs_diff_eq!(s, p, epsilon = 1e-10);
    }
}

#[test]
fn par_pdist_matches_pdist_manhattan_f64() {
    let points = sample_points_f64();
    let sequential = Manhattan::pdist(&points);
    let parallel = Manhattan::par_pdist(&points);
    assert_eq!(sequential.len(), parallel.len());
    for (s, p) in sequential.iter().zip(parallel.iter()) {
        assert_abs_diff_eq!(s, p, epsilon = 1e-10);
    }
}

#[test]
fn par_pdist_matches_pdist_chebyshev_f64() {
    let points = sample_points_f64();
    let sequential = Chebyshev::pdist(&points);
    let parallel = Chebyshev::par_pdist(&points);
    assert_eq!(sequential.len(), parallel.len());
    for (s, p) in sequential.iter().zip(parallel.iter()) {
        assert_abs_diff_eq!(s, p, epsilon = 1e-10);
    }
}

#[test]
fn par_pdist_matches_pdist_euclidean_f32() {
    let points = sample_points_f32();
    let sequential = Euclidean::pdist(&points);
    let parallel = Euclidean::par_pdist(&points);
    assert_eq!(sequential.len(), parallel.len());
    for (s, p) in sequential.iter().zip(parallel.iter()) {
        assert_abs_diff_eq!(s, p, epsilon = 1e-5);
    }
}

#[test]
fn par_batch_distance_matches_batch_distance_euclidean_f64() {
    let points = sample_points_f64();
    let query = vec![1.0, 1.0, 1.0];
    let targets: Vec<&[f64]> = points.iter().map(|p| p.as_slice()).collect();
    let sequential = Euclidean::batch_distance(query.as_slice(), targets.clone());
    let parallel = Euclidean::par_batch_distance(query.as_slice(), targets);
    assert_eq!(sequential.len(), parallel.len());
    for (s, p) in sequential.iter().zip(parallel.iter()) {
        assert_abs_diff_eq!(s, p, epsilon = 1e-10);
    }
}

#[test]
fn par_batch_distance_matches_batch_distance_manhattan_f64() {
    let points = sample_points_f64();
    let query = vec![0.0, 0.0, 0.0];
    let targets: Vec<&[f64]> = points.iter().map(|p| p.as_slice()).collect();
    let sequential = Manhattan::batch_distance(query.as_slice(), targets.clone());
    let parallel = Manhattan::par_batch_distance(query.as_slice(), targets);
    assert_eq!(sequential.len(), parallel.len());
    for (s, p) in sequential.iter().zip(parallel.iter()) {
        assert_abs_diff_eq!(s, p, epsilon = 1e-10);
    }
}

#[test]
fn par_batch_distance_matches_batch_distance_euclidean_f32() {
    let points = sample_points_f32();
    let query = vec![1.0_f32, 1.0, 1.0];
    let targets: Vec<&[f32]> = points.iter().map(|p| p.as_slice()).collect();
    let sequential = Euclidean::batch_distance(query.as_slice(), targets.clone());
    let parallel = Euclidean::par_batch_distance(query.as_slice(), targets);
    assert_eq!(sequential.len(), parallel.len());
    for (s, p) in sequential.iter().zip(parallel.iter()) {
        assert_abs_diff_eq!(s, p, epsilon = 1e-5);
    }
}
