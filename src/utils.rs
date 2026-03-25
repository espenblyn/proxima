use std::borrow::Cow;

pub trait IntoSlice<'a, F: Clone> {
    fn into_slice(self) -> Cow<'a, [F]>;
}

impl<'a, F: Clone, const N: usize> IntoSlice<'a, F> for &'a [F; N] {
    fn into_slice(self) -> Cow<'a, [F]> {
        Cow::Borrowed(self.as_slice())
    }
}

impl<'a, F: Clone> IntoSlice<'a, F> for &'a [F] {
    fn into_slice(self) -> Cow<'a, [F]> {
        Cow::Borrowed(self)
    }
}

impl<'a, F: Clone> IntoSlice<'a, F> for &'a Vec<F> {
    fn into_slice(self) -> Cow<'a, [F]> {
        Cow::Borrowed(self.as_slice())
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod ndarray_tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use crate::{Cosine, Dot, Euclidean, Manhattan, SqEuclidean};
    use crate::{DistanceExt, SimilarityExt};

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
        assert_abs_diff_eq!(Euclidean::distance(&view, &b), 2.8284271247461903);
    }
}

#[cfg(feature = "ndarray")]
use ndarray::{ArrayBase, Data, Ix1};

#[cfg(feature = "ndarray")]
impl<'a, F, S> IntoSlice<'a, F> for &'a ArrayBase<S, Ix1>
where
    F: Clone,
    S: Data<Elem = F>,
{
    fn into_slice(self) -> Cow<'a, [F]> {
        match self.as_slice() {
            Some(s) => Cow::Borrowed(s),
            None => {
                debug_assert!(
                    false,
                    "proxima-ml warning: Strided ndarray view caused a hidden heap allocation! \
                     Consider using `.as_standard_layout()` before passing to distance metric."
                );
                Cow::Owned(self.to_vec())
            }
        }
    }
}
