use crate::IntoSlice;
use num_traits::Float;
use std::iter::Sum;

pub trait Distance<F: Float + Sum> {
    fn compute(a: &[F], b: &[F]) -> F;

    fn batch_compute(query: &[F], targets: &[&[F]]) -> Vec<F> {
        targets
            .iter()
            .map(|target| Self::compute(query, target))
            .collect()
    }
}

pub trait DistanceExt<F: Float + Sum>: Distance<F> {
    fn distance<'a>(a: impl IntoSlice<'a, F>, b: impl IntoSlice<'a, F>) -> F
    where
        F: 'a,
    {
        let a_cow = a.into_slice();
        let b_cow = b.into_slice();
        Self::compute(a_cow.as_ref(), b_cow.as_ref())
    }

    fn batch_distance<'a, T>(
        query: impl IntoSlice<'a, F>,
        targets: impl IntoIterator<Item = T>,
    ) -> Vec<F>
    where
        T: IntoSlice<'a, F>,
        F: 'a,
    {
        let query_cow = query.into_slice();
        let query_ref = query_cow.as_ref();

        targets
            .into_iter()
            .map(|t| {
                let t_cow = t.into_slice();
                Self::compute(query_ref, t_cow.as_ref())
            })
            .collect()
    }

    fn pairwise_distances<'a, T>(points: &'a [T]) -> Vec<Vec<F>>
    where
        T: AsRef<[F]>,
        F: 'a,
    {
        points
            .iter()
            .map(|a| {
                points
                    .iter()
                    .map(|b| Self::compute(a.as_ref(), b.as_ref()))
                    .collect()
            })
            .collect()
    }
}

impl<T: Distance<F>, F: Float + Sum> DistanceExt<F> for T {}

pub trait Similarity<F: Float + Sum> {
    fn compute_similarity(a: &[F], b: &[F]) -> F;

    fn batch_compute_similarity(query: &[F], targets: &[&[F]]) -> Vec<F> {
        targets
            .iter()
            .map(|target| Self::compute_similarity(query, target))
            .collect()
    }
}

pub trait SimilarityExt<F: Float + Sum>: Similarity<F> {
    fn similarity<'a>(a: impl IntoSlice<'a, F>, b: impl IntoSlice<'a, F>) -> F
    where
        F: 'a,
    {
        let a_cow = a.into_slice();
        let b_cow = b.into_slice();
        Self::compute_similarity(a_cow.as_ref(), b_cow.as_ref())
    }

    fn batch_similarity<'a, T>(
        query: impl IntoSlice<'a, F>,
        targets: impl IntoIterator<Item = T>,
    ) -> Vec<F>
    where
        T: IntoSlice<'a, F>,
        F: 'a,
    {
        let query_cow = query.into_slice();
        let query_ref = query_cow.as_ref();

        targets
            .into_iter()
            .map(|t| {
                let t_cow = t.into_slice();
                Self::compute_similarity(query_ref, t_cow.as_ref())
            })
            .collect()
    }
}

impl<T: Similarity<F>, F: Float + Sum> SimilarityExt<F> for T {}
