use crate::traits::Distance;
use crate::validation::validate_lengths;
use num_traits::Float;
use std::iter::Sum;

pub struct Canberra;

impl<F: Float + Sum> Distance<F> for Canberra {
    fn compute(a: &[F], b: &[F]) -> F {
        validate_lengths(a, b);
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let denom = x.abs() + y.abs();
                if denom == F::zero() {
                    F::zero()
                } else {
                    (*x - *y).abs() / denom
                }
            })
            .sum()
    }
}
