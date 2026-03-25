pub fn validate_lengths<F>(a: &[F], b: &[F]) {
    assert_eq!(
        a.len(),
        b.len(),
        "proxima: slice lengths must match, got {} and {}",
        a.len(),
        b.len()
    );
}
