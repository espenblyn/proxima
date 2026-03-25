use crate::Distance;
use crate::validate_lengths;
use num_traits::Float;
use std::iter::Sum;

pub struct Manhattan;

impl<F: Float + Sum> Distance<F> for Manhattan {
    fn compute(a: &[F], b: &[F]) -> F {
        validate_lengths(a, b);
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x - *y).abs())
            .sum::<F>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DistanceExt;
    use approx::assert_abs_diff_eq;

    #[test]
    fn manhattan_basic() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert_abs_diff_eq!(Manhattan::distance(&a, &b), 7.0);
    }

    #[test]
    fn manhattan_three_dimensions() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_abs_diff_eq!(Manhattan::distance(&a, &b), 9.0);
    }

    #[test]
    fn manhattan_identical_vectors() {
        let a = [1.0, 2.0, 3.0];
        assert_abs_diff_eq!(Manhattan::distance(&a, &a), 0.0);
    }

    #[test]
    fn manhattan_single_element() {
        let a = [5.0];
        let b = [3.0];
        assert_abs_diff_eq!(Manhattan::distance(&a, &b), 2.0);
    }

    #[test]
    fn manhattan_negative_values() {
        let a = [-1.0, -2.0];
        let b = [1.0, 2.0];
        assert_abs_diff_eq!(Manhattan::distance(&a, &b), 6.0);
    }

    #[test]
    #[should_panic(expected = "proxima: slice lengths must match")]
    fn manhattan_mismatched_lengths() {
        let a = [1.0, 2.0];
        let b = [1.0, 2.0, 3.0];
        Manhattan::distance(&a, &b);
    }

    #[test]
    fn manhattan_works_with_f32() {
        let a: [f32; 2] = [0.0, 0.0];
        let b: [f32; 2] = [3.0, 4.0];
        assert_abs_diff_eq!(Manhattan::distance(&a, &b), 7.0_f32);
    }

    #[test]
    fn manhattan_always_gte_euclidean() {
        use crate::Euclidean;
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let manhattan = Manhattan::distance(&a, &b);
        let euclidean = Euclidean::distance(&a, &b);
        assert!(manhattan >= euclidean);
    }

    #[test]
    fn manhattan_batch_distance() {
        let query = [0.0, 0.0];
        let targets = vec![vec![3.0, 4.0], vec![1.0, 0.0], vec![0.0, 2.0]];
        let distances = Manhattan::batch_distance(&query, &targets);
        assert_abs_diff_eq!(distances[0], 7.0);
        assert_abs_diff_eq!(distances[1], 1.0);
        assert_abs_diff_eq!(distances[2], 2.0);
    }

    #[test]
    fn manhattan_pairwise_distances() {
        let points = vec![vec![0.0, 0.0], vec![3.0, 4.0], vec![1.0, 0.0]];
        let matrix = Manhattan::pairwise_distances(&points);

        assert_abs_diff_eq!(matrix[0][0], 0.0);
        assert_abs_diff_eq!(matrix[1][1], 0.0);
        assert_abs_diff_eq!(matrix[0][1], matrix[1][0]);
        assert_abs_diff_eq!(matrix[0][1], 7.0);
        assert_abs_diff_eq!(matrix[0][2], 1.0);
        assert_abs_diff_eq!(matrix[1][2], 6.0);
    }
}
