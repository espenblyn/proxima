use crate::Similarity;
use crate::validate_lengths;
use num_traits::Float;
use std::iter::Sum;

pub struct Dot;

impl<F: Float + Sum> Similarity<F> for Dot {
    fn similarity(a: &[F], b: &[F]) -> F {
        validate_lengths(a, b);
        a.iter().zip(b.iter()).map(|(x, y)| (*x) * (*y)).sum::<F>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn dot_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_abs_diff_eq!(Dot::similarity(&a, &b), 32.0);
    }

    #[test]
    fn dot_perpendicular() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        assert_abs_diff_eq!(Dot::similarity(&a, &b), 0.0);
    }

    #[test]
    fn dot_identical() {
        let a = [2.0, 3.0];
        assert_abs_diff_eq!(Dot::similarity(&a, &a), 13.0);
    }

    #[test]
    fn dot_single_element() {
        let a = [3.0];
        let b = [4.0];
        assert_abs_diff_eq!(Dot::similarity(&a, &b), 12.0);
    }

    #[test]
    fn dot_negative_values() {
        let a = [1.0, -2.0];
        let b = [-3.0, 4.0];
        assert_abs_diff_eq!(Dot::similarity(&a, &b), -11.0);
    }

    #[test]
    #[should_panic(expected = "proxima: slice lengths must match")]
    fn dot_mismatched_lengths() {
        let a = [1.0, 2.0];
        let b = [1.0, 2.0, 3.0];
        Dot::similarity(&a, &b);
    }

    #[test]
    fn dot_works_with_f32() {
        let a: [f32; 3] = [1.0, 2.0, 3.0];
        let b: [f32; 3] = [4.0, 5.0, 6.0];
        assert_abs_diff_eq!(Dot::similarity(&a, &b), 32.0_f32);
    }
}
