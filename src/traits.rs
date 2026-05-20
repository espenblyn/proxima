use crate::IntoSlice;
use num_traits::Float;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
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

    fn batch_distance<'q, 't, T>(
        query: impl IntoSlice<'q, F>,
        targets: impl IntoIterator<Item = T>,
    ) -> Vec<F>
    where
        T: IntoSlice<'t, F>,
        F: 'q + 't,
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

    fn pdist<'a, T>(points: &'a [T]) -> Vec<F>
    where
        T: AsRef<[F]>,
        F: 'a,
    {
        let n = points.len();
        let mut result = Vec::with_capacity(n * (n - 1) / 2);

        for i in 0..n {
            for j in (i + 1)..n {
                result.push(Self::compute(points[i].as_ref(), points[j].as_ref()));
            }
        }

        result
    }

    #[cfg(feature = "ndarray")]
    fn pdist_2d(points: &ndarray::Array2<F>) -> Vec<F>
    where
        F: ndarray::NdFloat,
    {
        let n = points.nrows();
        let mut result = Vec::with_capacity(n * (n - 1) / 2);

        for i in 0..n {
            let row_i = points.row(i);
            let slice_i = row_i
                .as_slice()
                .expect("proxima: Array2 must be row-major (C order)");
            for j in (i + 1)..n {
                let row_j = points.row(j);
                let slice_j = row_j
                    .as_slice()
                    .expect("proxima: Array2 must be row-major (C order)");
                result.push(Self::compute(slice_i, slice_j));
            }
        }

        result
    }

    #[cfg(feature = "ndarray")]
    fn batch_distance_2d(query: &ndarray::Array1<F>, targets: &ndarray::Array2<F>) -> Vec<F>
    where
        F: ndarray::NdFloat,
    {
        let query_slice = query
            .as_slice()
            .expect("proxima: Array1 must be contiguous");

        (0..targets.nrows())
            .map(|i| {
                let row = targets.row(i);
                let slice = row
                    .as_slice()
                    .expect("proxima: Array2 must be row-major (C order)");
                Self::compute(query_slice, slice)
            })
            .collect()
    }

    #[cfg(feature = "parallel")]
    fn par_batch_distance<'q, 't, T>(
        query: impl IntoSlice<'q, F>,
        targets: impl IntoIterator<Item = T>,
    ) -> Vec<F>
    where
        T: IntoSlice<'t, F> + Send,
        F: 'q + 't + Send + Sync,
    {
        let query_cow = query.into_slice();
        let query_ref = query_cow.as_ref();

        let targets: Vec<_> = targets.into_iter().collect();
        targets
            .into_par_iter()
            .map(|t| {
                let t_cow = t.into_slice();
                Self::compute(query_ref, t_cow.as_ref())
            })
            .collect()
    }

    #[cfg(feature = "parallel")]
    fn par_pdist<'a, T>(points: &'a [T]) -> Vec<F>
    where
        T: AsRef<[F]> + Sync,
        F: 'a + Send + Sync,
    {
        let n = points.len();
        (0..n)
            .into_par_iter()
            .flat_map(|i| {
                ((i + 1)..n)
                    .map(|j| Self::compute(points[i].as_ref(), points[j].as_ref()))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[cfg(all(feature = "ndarray", feature = "parallel"))]
    fn par_pdist_2d(points: &ndarray::Array2<F>) -> Vec<F>
    where
        F: ndarray::NdFloat + Send + Sync,
    {
        let n = points.nrows();

        (0..n)
            .into_par_iter()
            .flat_map(|i| {
                let row_i = points.row(i);
                let slice_i = row_i
                    .as_slice()
                    .expect("proxima: Array2 must be row-major (C order)");
                ((i + 1)..n)
                    .map(|j| {
                        let row_j = points.row(j);
                        let slice_j = row_j
                            .as_slice()
                            .expect("proxima: Array2 must be row-major (C order)");
                        Self::compute(slice_i, slice_j)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[cfg(all(feature = "ndarray", feature = "parallel"))]
    fn par_batch_distance_2d(query: &ndarray::Array1<F>, targets: &ndarray::Array2<F>) -> Vec<F>
    where
        F: ndarray::NdFloat + Send + Sync,
    {
        let query_slice = query
            .as_slice()
            .expect("proxima: Array1 must be contiguous");

        (0..targets.nrows())
            .into_par_iter()
            .map(|i| {
                let row = targets.row(i);
                let slice = row
                    .as_slice()
                    .expect("proxima: Array2 must be row-major (C order)");
                Self::compute(query_slice, slice)
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

    fn batch_similarity<'q, 't, T>(
        query: impl IntoSlice<'q, F>,
        targets: impl IntoIterator<Item = T>,
    ) -> Vec<F>
    where
        T: IntoSlice<'t, F>,
        F: 'q + 't,
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
