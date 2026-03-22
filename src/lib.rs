use num_traits::Float;

// Validation

fn validate_lengths<F>(a: &[F], b: &[F]) {
    assert_eq!(
        a.len(),
        b.len(),
        "proxima: slice lengths must match, got {} and {}",
        a.len(),
        b.len()
    );
}

// Defining traits

pub trait Distance<F: Float> {
    fn distance(a: &[F], b: &[F]) -> F;

    fn batch_distance(a: &[F], targets: &[Vec<F>]) -> Vec<F> {
        targets.iter().map(|b| Self::distance(a, b)).collect()
    }

    fn pairwise_distances(points: &[Vec<F>]) -> Vec<Vec<F>> {
        points
            .iter()
            .map(|a| points.iter().map(|b| Self::distance(a, b)).collect())
            .collect()
    }
}

pub trait Similarity<F: Float> {
    fn similarity(a: &[F], b: &[F]) -> F;

    fn batch_similarity(a: &[F], targets: &[Vec<F>]) -> Vec<F> {
        targets.iter().map(|b| Self::similarity(a, b)).collect()
    }
}
