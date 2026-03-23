use crate::Distance;
use crate::validate_lengths;
use num_traits::Float;
use std::iter::Sum;

pub struct Manhattan;

impl<F: Float + Sum> Distance<F> for Manhattan {
    fn distance(a: &[F], b: &[F]) -> F {
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
}
