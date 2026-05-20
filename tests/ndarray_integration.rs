#[cfg(feature = "ndarray")]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2, array};
    use proxima_ml::{Cosine, Dot, Euclidean, Manhattan, SqEuclidean};
    use proxima_ml::{DistanceExt, SimilarityExt};

    #[test]
    fn test_standard_slices() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];

        let dist = Euclidean::distance(a.as_slice(), b.as_slice());
        assert!(dist > 1.41 && dist < 1.42);
    }

    #[test]
    fn test_ndarray_contiguous() {
        let a = array![1.0, 0.0];
        let b = array![0.0, 1.0];

        let dist = Euclidean::distance(&a, &b);
        assert!(dist > 1.41 && dist < 1.42);
    }

    #[test]
    fn test_ndarray_batch_distance() {
        let query = array![1.0, 1.0];

        let targets = array![[1.0, 1.0], [0.0, 1.0], [1.0, 0.0],];

        let distances = Euclidean::batch_distance(&query, targets.outer_iter());

        assert_eq!(distances.len(), 3);
        assert_eq!(distances[0], 0.0);
        assert_eq!(distances[1], 1.0);
        assert_eq!(distances[2], 1.0);
    }

    #[test]
    fn test_similarity_mixed_types() {
        let a_slice = vec![1.0, 2.0, 3.0];
        let b_array = array![1.0, 2.0, 3.0];

        let sim = Cosine::similarity(a_slice.as_slice(), &b_array);

        assert!(sim > 0.999);
    }

    #[test]
    fn euclidean_ndarray() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        assert_abs_diff_eq!(Euclidean::distance(&a, &b), 5.0);
    }

    #[test]
    fn sq_euclidean_ndarray() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        assert_abs_diff_eq!(SqEuclidean::distance(&a, &b), 25.0);
    }

    #[test]
    fn manhattan_ndarray() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];
        assert_abs_diff_eq!(Manhattan::distance(&a, &b), 9.0);
    }

    #[test]
    fn cosine_similarity_ndarray() {
        let a = array![1.0, 0.0];
        let b = array![1.0, 0.0];
        assert_abs_diff_eq!(Cosine::similarity(&a, &b), 1.0);
    }

    #[test]
    fn cosine_distance_ndarray() {
        let a = array![1.0, 0.0];
        let b = array![0.0, 1.0];
        assert_abs_diff_eq!(Cosine::distance(&a, &b), 1.0);
    }

    #[test]
    fn dot_similarity_ndarray() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];
        assert_abs_diff_eq!(Dot::similarity(&a, &b), 32.0);
    }

    #[test]
    fn mixed_ndarray_and_slice() {
        let a = array![0.0, 0.0];
        let b: &[f64] = &[3.0, 4.0];
        assert_abs_diff_eq!(Euclidean::distance(&a, b), 5.0);
    }

    #[test]
    fn batch_distance_ndarray() {
        let query = array![0.0, 0.0];
        let t1 = array![3.0, 4.0];
        let t2 = array![1.0, 0.0];
        let targets = vec![&t1, &t2];
        let distances = Euclidean::batch_distance(&query, targets);
        assert_abs_diff_eq!(distances[0], 5.0);
        assert_abs_diff_eq!(distances[1], 1.0);
    }

    #[test]
    fn batch_similarity_ndarray() {
        let query = array![1.0, 0.0];
        let t1 = array![1.0, 0.0];
        let t2 = array![0.0, 1.0];
        let targets = vec![&t1, &t2];
        let similarities = Cosine::batch_similarity(&query, targets);
        assert_abs_diff_eq!(similarities[0], 1.0);
        assert_abs_diff_eq!(similarities[1], 0.0);
    }

    #[test]
    fn ndarray_f32() {
        let a = array![0.0_f32, 0.0];
        let b = array![3.0_f32, 4.0];
        assert_abs_diff_eq!(Euclidean::distance(&a, &b), 5.0_f32);
    }

    #[test]
    fn ndarray_slice_view() {
        let a = array![1.0, 2.0, 3.0, 4.0];
        let view = a.slice(ndarray::s![0..2]);
        let b = array![3.0, 4.0];
        assert_abs_diff_eq!(Euclidean::distance(&view, &b), 8.0_f64.sqrt());
    }

    // ---- Array2 tests ----

    #[test]
    fn pdist_2d_basic() {
        let points = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 3.0, 4.0, 1.0, 0.0]).unwrap();
        let result = Euclidean::pdist_2d(&points);
        assert_eq!(result.len(), 3);
        assert_abs_diff_eq!(result[0], 5.0);
        assert_abs_diff_eq!(result[1], 1.0);
        assert_abs_diff_eq!(result[2], (4.0_f64 + 16.0).sqrt());
    }

    #[test]
    fn pdist_2d_matches_pdist() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![0.0, 1.0, 0.0],
        ];
        let flat: Vec<f64> = data.iter().flatten().copied().collect();
        let points_2d = Array2::from_shape_vec((4, 3), flat).unwrap();

        let from_vec = Euclidean::pdist(&data);
        let from_2d = Euclidean::pdist_2d(&points_2d);

        assert_eq!(from_vec.len(), from_2d.len());
        for (a, b) in from_vec.iter().zip(from_2d.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn pdist_2d_single_row() {
        let points = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let result = Euclidean::pdist_2d(&points);
        assert!(result.is_empty());
    }

    #[test]
    fn pdist_2d_with_manhattan() {
        let points = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 3.0, 4.0, 1.0, 1.0]).unwrap();
        let result = Manhattan::pdist_2d(&points);
        assert_eq!(result.len(), 3);
        assert_abs_diff_eq!(result[0], 7.0);
        assert_abs_diff_eq!(result[1], 2.0);
        assert_abs_diff_eq!(result[2], 5.0);
    }

    #[test]
    fn pdist_2d_f32() {
        let points = Array2::from_shape_vec((2, 2), vec![0.0_f32, 0.0, 3.0, 4.0]).unwrap();
        let result = Euclidean::pdist_2d(&points);
        assert_eq!(result.len(), 1);
        assert_abs_diff_eq!(result[0], 5.0_f32);
    }

    #[test]
    fn batch_distance_2d_basic() {
        let query = Array1::from_vec(vec![0.0, 0.0]);
        let targets = Array2::from_shape_vec((3, 2), vec![3.0, 4.0, 1.0, 0.0, 0.0, 5.0]).unwrap();
        let result = Euclidean::batch_distance_2d(&query, &targets);
        assert_eq!(result.len(), 3);
        assert_abs_diff_eq!(result[0], 5.0);
        assert_abs_diff_eq!(result[1], 1.0);
        assert_abs_diff_eq!(result[2], 5.0);
    }

    #[test]
    fn batch_distance_2d_matches_batch_distance() {
        let query_vec = vec![1.0, 2.0, 3.0];
        let targets_vec = vec![
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![0.0, 1.0, 0.0],
        ];

        let query_arr = Array1::from_vec(query_vec.clone());
        let flat: Vec<f64> = targets_vec.iter().flatten().copied().collect();
        let targets_arr = Array2::from_shape_vec((3, 3), flat).unwrap();

        let from_vec = Euclidean::batch_distance(query_vec.as_slice(), &targets_vec);
        let from_2d = Euclidean::batch_distance_2d(&query_arr, &targets_arr);

        assert_eq!(from_vec.len(), from_2d.len());
        for (a, b) in from_vec.iter().zip(from_2d.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn batch_distance_2d_single_target() {
        let query = Array1::from_vec(vec![0.0, 0.0]);
        let targets = Array2::from_shape_vec((1, 2), vec![3.0, 4.0]).unwrap();
        let result = Euclidean::batch_distance_2d(&query, &targets);
        assert_eq!(result.len(), 1);
        assert_abs_diff_eq!(result[0], 5.0);
    }

    #[test]
    fn batch_distance_2d_with_cosine() {
        let query = Array1::from_vec(vec![1.0, 0.0]);
        let targets = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let result = Cosine::batch_distance_2d(&query, &targets);
        assert_eq!(result.len(), 2);
        assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 1.0, epsilon = 1e-10);
    }

    // ---- Parallel Array2 tests ----

    #[cfg(feature = "parallel")]
    #[test]
    fn par_pdist_2d_matches_pdist_2d() {
        let points = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 1.0, 0.0, 2.0, 3.0, 1.0,
            ],
        )
        .unwrap();

        let sequential = Euclidean::pdist_2d(&points);
        let parallel = Euclidean::par_pdist_2d(&points);

        assert_eq!(sequential.len(), parallel.len());
        for (a, b) in sequential.iter().zip(parallel.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn par_batch_distance_2d_matches_batch_distance_2d() {
        let query = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let targets = Array2::from_shape_vec(
            (4, 3),
            vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 1.0, 0.0, 2.0, 3.0, 1.0],
        )
        .unwrap();

        let sequential = Euclidean::batch_distance_2d(&query, &targets);
        let parallel = Euclidean::par_batch_distance_2d(&query, &targets);

        assert_eq!(sequential.len(), parallel.len());
        for (a, b) in sequential.iter().zip(parallel.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }
}
