use crate::Distance;
use crate::validate_lengths;
use num_traits::Float;
use std::iter::Sum;

pub struct Euclidean {}

pub struct SqEuclidean {}

impl<F: Float + Sum> Distance<F> for Euclidean {
    fn compute(a: &[F], b: &[F]) -> F {
        validate_lengths(a, b);
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x - *y).powi(2))
            .sum::<F>()
            .sqrt()
    }
}

impl<F: Float + Sum> Distance<F> for SqEuclidean {
    fn compute(a: &[F], b: &[F]) -> F {
        validate_lengths(a, b);
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x - *y).powi(2))
            .sum::<F>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DistanceExt;
    use approx::assert_abs_diff_eq;

    #[test]
    fn euclidean_basic() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert_abs_diff_eq!(Euclidean::distance(&a, &b), 5.0);
    }

    #[test]
    fn euclidean_three_dimensions() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_abs_diff_eq!(Euclidean::distance(&a, &b), 5.196152, epsilon = 1e-6);
    }

    #[test]
    fn euclidean_identical_vectors() {
        let a = [1.0, 2.0, 3.0];
        assert_abs_diff_eq!(Euclidean::distance(&a, &a), 0.0);
    }

    #[test]
    fn euclidean_single_element() {
        let a = [5.0];
        let b = [3.0];
        assert_abs_diff_eq!(Euclidean::distance(&a, &b), 2.0);
    }

    #[test]
    fn sq_euclidean_basic() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert_abs_diff_eq!(SqEuclidean::distance(&a, &b), 25.0);
    }

    #[test]
    fn sq_equals_euclidean_squared() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let euclidean = Euclidean::distance(&a, &b);
        let sq = SqEuclidean::distance(&a, &b);
        assert_abs_diff_eq!(sq, euclidean * euclidean, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "proxima: slice lengths must match")]
    fn euclidean_mismatched_lengths() {
        let a = [1.0, 2.0];
        let b = [1.0, 2.0, 3.0];
        Euclidean::distance(&a, &b);
    }

    #[test]
    fn euclidean_works_with_f32() {
        let a: [f32; 2] = [0.0, 0.0];
        let b: [f32; 2] = [3.0, 4.0];
        assert_abs_diff_eq!(Euclidean::distance(&a, &b), 5.0_f32);
    }

    #[test]
    fn euclidean_batch_distance() {
        let query = [0.0, 0.0];
        let targets = vec![vec![3.0, 4.0], vec![1.0, 0.0], vec![0.0, 2.0]];
        let distances = Euclidean::batch_distance(&query, &targets);
        assert_abs_diff_eq!(distances[0], 5.0);
        assert_abs_diff_eq!(distances[1], 1.0);
        assert_abs_diff_eq!(distances[2], 2.0);
    }

    #[test]
    fn euclidean_pairwise_distances() {
        let points = vec![vec![0.0, 0.0], vec![3.0, 4.0], vec![1.0, 0.0]];
        let matrix = Euclidean::pairwise_distances(&points);

        assert_abs_diff_eq!(matrix[0][0], 0.0);
        assert_abs_diff_eq!(matrix[1][1], 0.0);
        assert_abs_diff_eq!(matrix[2][2], 0.0);

        assert_abs_diff_eq!(matrix[0][1], matrix[1][0]);
        assert_abs_diff_eq!(matrix[0][2], matrix[2][0]);
        assert_abs_diff_eq!(matrix[1][2], matrix[2][1]);

        assert_abs_diff_eq!(matrix[0][1], 5.0);
        assert_abs_diff_eq!(matrix[0][2], 1.0);
        assert_abs_diff_eq!(matrix[1][2], 4.472136, epsilon = 1e-6);
    }
}
