pub struct Hamming;

impl Hamming {
    pub fn distance<T: PartialEq>(a: &[T], b: &[T]) -> usize {
        assert_eq!(
            a.len(),
            b.len(),
            "proxima: slice lengths must match, got {} and {}",
            a.len(),
            b.len()
        );
        a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamming_integers() {
        let a = [1, 0, 1, 1];
        let b = [1, 1, 0, 1];
        assert_eq!(Hamming::distance(&a, &b), 2);
    }

    #[test]
    fn hamming_identical() {
        let a = [1, 2, 3];
        assert_eq!(Hamming::distance(&a, &a), 0);
    }

    #[test]
    fn hamming_all_different() {
        let a = [1, 2, 3];
        let b = [4, 5, 6];
        assert_eq!(Hamming::distance(&a, &b), 3);
    }

    #[test]
    fn hamming_strings() {
        let a = ["cat", "dog", "fish"];
        let b = ["cat", "bird", "fish"];
        assert_eq!(Hamming::distance(&a, &b), 1);
    }

    #[test]
    fn hamming_single_element() {
        let a = [true];
        let b = [false];
        assert_eq!(Hamming::distance(&a, &b), 1);
    }

    #[test]
    #[should_panic(expected = "proxima: slice lengths must match")]
    fn hamming_mismatched_lengths() {
        let a = [1, 2];
        let b = [1, 2, 3];
        Hamming::distance(&a, &b);
    }
}
