use std::fmt::Debug;
use std::iter::zip;

use crate::{Layout, TensorView, View};

/// Trait that tests whether two values are approximately equal.
///
/// Here "approximately" means "a value that is reasonable for this crate's
/// tests".
pub trait ApproxEq: Sized {
    /// Return the default absolute tolerance value.
    fn default_tolerance() -> Self;

    /// Test if `other` is approximately equal to `self` with a maximum
    /// absolute difference of `epsilon`.
    fn approx_eq_with_tolerance(&self, other: &Self, epsilon: Self) -> bool;

    /// Test if `other` is approximately equal to `self` with the maximum
    /// absolute difference specified by `Self::default_tolerance`.
    fn approx_eq(&self, other: &Self) -> bool {
        self.approx_eq_with_tolerance(other, Self::default_tolerance())
    }
}

impl ApproxEq for f32 {
    fn default_tolerance() -> f32 {
        1e-5
    }

    fn approx_eq_with_tolerance(&self, other: &f32, epsilon: f32) -> bool {
        (self - other).abs() < epsilon
    }
}

impl ApproxEq for i32 {
    fn default_tolerance() -> i32 {
        0
    }

    fn approx_eq_with_tolerance(&self, other: &i32, eps: i32) -> bool {
        (self - other).abs() < eps
    }
}

/// Return the N-dimensional index in a tensor with a given `shape` that
/// corresponds to a linear index (ie. the index if the tensor was flattened to
/// 1D).
fn index_from_linear_index(shape: &[usize], lin_index: usize) -> Vec<usize> {
    assert!(
        lin_index < shape.iter().product(),
        "Linear index {} is out of bounds for shape {:?}",
        lin_index,
        shape,
    );
    (0..shape.len())
        .map(|dim| {
            let elts_per_index: usize = shape[dim + 1..].iter().product();
            let lin_index_for_dim = lin_index % (shape[dim] * elts_per_index);
            lin_index_for_dim / elts_per_index
        })
        .collect()
}

/// Check that the shapes of two tensors are equal and that their contents
/// are approximately equal.
///
/// If there are mismatches, this returns an `Err` with a message indicating
/// the count of mismatches and details of the first N cases.
pub fn expect_equal<V: View>(x: &V, y: &V) -> Result<(), String>
where
    V::Elem: Clone + Debug + ApproxEq,
{
    expect_equal_with_tolerance(x, y, V::Elem::default_tolerance())
}

/// Check that the shapes of two tensors are equal and that their contents
/// are approximately equal.
///
/// This is like [expect_equal] but allows a custom absolute tolerance value.
pub fn expect_equal_with_tolerance<V: View>(x: &V, y: &V, epsilon: V::Elem) -> Result<(), String>
where
    V::Elem: Clone + Debug + ApproxEq,
{
    if x.shape() != y.shape() {
        return Err(format!(
            "Tensors have different shapes. {:?} vs. {:?}",
            x.shape(),
            y.shape()
        ));
    }

    let mismatches: Vec<_> = zip(x.iter(), y.iter())
        .enumerate()
        .filter_map(|(i, (xi, yi))| {
            if !xi.approx_eq_with_tolerance(yi, epsilon.clone()) {
                Some((index_from_linear_index(x.shape().as_ref(), i), xi, yi))
            } else {
                None
            }
        })
        .collect();

    if !mismatches.is_empty() {
        let max_examples = 16;
        Err(format!(
            "Tensor values differ at {} of {} indexes: {:?}{}",
            mismatches.len(),
            x.len(),
            &mismatches[..mismatches.len().min(max_examples)],
            if mismatches.len() > max_examples {
                "..."
            } else {
                ""
            }
        ))
    } else {
        Ok(())
    }
}

// Return true if `a` and `b` have the same shape and data, treating NaN
// values as equal.
pub fn eq_with_nans(a: TensorView, b: TensorView) -> bool {
    if a.shape() != b.shape() {
        false
    } else {
        zip(a.iter(), b.iter()).all(|(a, b)| (a.is_nan() && b.is_nan()) || a == b)
    }
}
