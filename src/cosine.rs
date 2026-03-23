use crate::Distance;
use crate::Similarity;
use crate::validate_lengths;
use num_traits::Float;
use std::iter::Sum;

pub struct Cosine;

impl<F: Float + Sum> Similarity<F> for Cosine {
    fn similarity(a: &[F], b: &[F]) -> F {
        validate_lengths(a, b);

        let dot: F = a.iter().zip(b.iter()).map(|(x, y)| (*x) * (*y)).sum();
        let mag_a: F = a.iter().map(|x| (*x) * (*x)).sum::<F>().sqrt();
        let mag_b: F = b.iter().map(|x| (*x) * (*x)).sum::<F>().sqrt();

        if mag_a == F::zero() || mag_b == F::zero() {
            return F::zero();
        }

        dot / (mag_a * mag_b)
    }
}

impl<F: Float + Sum> Distance<F> for Cosine {
    fn distance(a: &[F], b: &[F]) -> F {
        F::one() - Self::similarity(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn cosine_identical_vectors() {
        let a = [1.0, 0.0];
        let b = [1.0, 0.0];
        assert_abs_diff_eq!(Cosine::similarity(&a, &b), 1.0);
        assert_abs_diff_eq!(Cosine::distance(&a, &b), 0.0);
    }

    #[test]
    fn cosine_perpendicular_vectors() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        assert_abs_diff_eq!(Cosine::similarity(&a, &b), 0.0);
        assert_abs_diff_eq!(Cosine::distance(&a, &b), 1.0);
    }

    #[test]
    fn cosine_same_direction_different_magnitude() {
        let a = [3.0, 1.0, 0.0];
        let b = [6.0, 2.0, 0.0];
        assert_abs_diff_eq!(Cosine::similarity(&a, &b), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn cosine_three_dimensions() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_abs_diff_eq!(Cosine::similarity(&a, &b), 0.974631, epsilon = 1e-6);
    }

    #[test]
    fn cosine_zero_vector() {
        let a = [0.0, 0.0];
        let b = [1.0, 2.0];
        assert_abs_diff_eq!(Cosine::similarity(&a, &b), 0.0);
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = [1.0, 0.0];
        let b = [-1.0, 0.0];
        assert_abs_diff_eq!(Cosine::similarity(&a, &b), -1.0);
        assert_abs_diff_eq!(Cosine::distance(&a, &b), 2.0);
    }

    #[test]
    #[should_panic(expected = "proxima: slice lengths must match")]
    fn cosine_mismatched_lengths() {
        let a = [1.0, 2.0];
        let b = [1.0, 2.0, 3.0];
        Cosine::similarity(&a, &b);
    }

    #[test]
    fn cosine_works_with_f32() {
        let a: [f32; 2] = [1.0, 0.0];
        let b: [f32; 2] = [1.0, 0.0];
        assert_abs_diff_eq!(Cosine::similarity(&a, &b), 1.0_f32);
    }
}
