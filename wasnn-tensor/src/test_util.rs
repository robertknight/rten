use std::fmt::Debug;
use std::iter::zip;

use crate::{Layout, TensorView, View};

/// Trait that tests whether two values are approximately equal.
///
/// Here "approximately" means "a value that is reasonable for this crate's
/// tests".
pub trait ApproxEq {
    fn approx_eq(&self, other: &Self) -> bool;
}

impl ApproxEq for f32 {
    fn approx_eq(&self, other: &f32) -> bool {
        let eps = 1e-4;
        (self - other).abs() < eps
    }
}

impl ApproxEq for i32 {
    fn approx_eq(&self, other: &i32) -> bool {
        *self == *other
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
    V::Elem: ApproxEq + Debug,
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
            if !xi.approx_eq(yi) {
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
