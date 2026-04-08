use std::collections::HashSet;
use std::hash::Hash;

pub struct Jaccard;

impl Jaccard {
    pub fn similarity<T: Eq + Hash>(a: &[T], b: &[T]) -> f64 {
        let set_a: HashSet<&T> = a.iter().collect();
        let set_b: HashSet<&T> = b.iter().collect();

        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();

        if union == 0 {
            return 1.0;
        }

        intersection as f64 / union as f64
    }

    pub fn distance<T: Eq + Hash>(a: &[T], b: &[T]) -> f64 {
        1.0 - Self::similarity(a, b)
    }
}
