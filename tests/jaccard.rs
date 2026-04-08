use approx::assert_abs_diff_eq;
use proxima_ml::Jaccard;

#[test]
fn jaccard_basic() {
    let a = ["rock", "jazz", "blues", "funk"];
    let b = ["rock", "pop", "blues", "electronic"];
    assert_abs_diff_eq!(Jaccard::similarity(&a, &b), 2.0 / 6.0, epsilon = 1e-10);
    assert_abs_diff_eq!(Jaccard::distance(&a, &b), 4.0 / 6.0, epsilon = 1e-10);
}

#[test]
fn jaccard_high_overlap() {
    let a = ["rock", "jazz", "blues", "funk"];
    let b = ["rock", "jazz", "blues"];
    assert_abs_diff_eq!(Jaccard::similarity(&a, &b), 3.0 / 4.0, epsilon = 1e-10);
}

#[test]
fn jaccard_identical() {
    let a = ["rock", "jazz", "blues"];
    assert_abs_diff_eq!(Jaccard::similarity(&a, &a), 1.0);
    assert_abs_diff_eq!(Jaccard::distance(&a, &a), 0.0);
}

#[test]
fn jaccard_no_overlap() {
    let a = ["rock", "jazz"];
    let b = ["pop", "electronic"];
    assert_abs_diff_eq!(Jaccard::similarity(&a, &b), 0.0);
    assert_abs_diff_eq!(Jaccard::distance(&a, &b), 1.0);
}

#[test]
fn jaccard_empty_sets() {
    let a: [&str; 0] = [];
    let b: [&str; 0] = [];
    assert_abs_diff_eq!(Jaccard::similarity(&a, &b), 1.0);
    assert_abs_diff_eq!(Jaccard::distance(&a, &b), 0.0);
}

#[test]
fn jaccard_one_empty() {
    let a = ["rock", "jazz"];
    let b: [&str; 0] = [];
    assert_abs_diff_eq!(Jaccard::similarity(&a, &b), 0.0);
    assert_abs_diff_eq!(Jaccard::distance(&a, &b), 1.0);
}

#[test]
fn jaccard_with_integers() {
    let a = [1, 2, 3, 4, 5];
    let b = [4, 5, 6, 7, 8];
    assert_abs_diff_eq!(Jaccard::similarity(&a, &b), 2.0 / 8.0, epsilon = 1e-10);
}

#[test]
fn jaccard_duplicates_ignored() {
    let a = ["rock", "rock", "jazz"];
    let b = ["rock", "jazz", "jazz"];
    assert_abs_diff_eq!(Jaccard::similarity(&a, &b), 1.0);
}
