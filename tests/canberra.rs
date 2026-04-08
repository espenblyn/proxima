use approx::assert_abs_diff_eq;
use proxima_ml::{Canberra, DistanceExt};

#[test]
fn canberra_basic() {
    let a = [1.0, 2.0, 3.0];
    let b = [2.0, 4.0, 6.0];
    assert_abs_diff_eq!(Canberra::distance(&a, &b), 1.0, epsilon = 1e-10);
}

#[test]
fn canberra_identical() {
    let a = [1.0, 2.0, 3.0];
    assert_abs_diff_eq!(Canberra::distance(&a, &a), 0.0);
}

#[test]
fn canberra_single_element() {
    let a = [1.0];
    let b = [3.0];
    assert_abs_diff_eq!(Canberra::distance(&a, &b), 0.5, epsilon = 1e-10);
}

#[test]
fn canberra_with_zeros() {
    let a = [0.0, 1.0, 2.0];
    let b = [0.0, 3.0, 4.0];
    assert_abs_diff_eq!(Canberra::distance(&a, &b), 0.8333333, epsilon = 1e-6);
}

#[test]
fn canberra_all_zeros() {
    let a = [0.0, 0.0];
    let b = [0.0, 0.0];
    assert_abs_diff_eq!(Canberra::distance(&a, &b), 0.0);
}

#[test]
fn canberra_negative_values() {
    let a = [-1.0, 2.0];
    let b = [1.0, -2.0];

    assert_abs_diff_eq!(Canberra::distance(&a, &b), 2.0, epsilon = 1e-10);
}

#[test]
fn canberra_one_zero() {
    let a = [0.0];
    let b = [5.0];
    assert_abs_diff_eq!(Canberra::distance(&a, &b), 1.0);
}

#[test]
#[should_panic(expected = "proxima: slice lengths must match")]
fn canberra_mismatched_lengths() {
    let a = [1.0, 2.0];
    let b = [1.0, 2.0, 3.0];
    Canberra::distance(&a, &b);
}

#[test]
fn canberra_works_with_f32() {
    let a: [f32; 3] = [1.0, 2.0, 3.0];
    let b: [f32; 3] = [2.0, 4.0, 6.0];
    assert_abs_diff_eq!(Canberra::distance(&a, &b), 1.0_f32, epsilon = 1e-6);
}
