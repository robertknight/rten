use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::iter::zip;

use crate::{AsView, Layout, TensorView};

/// Trait that tests whether two values are approximately equal.
///
/// The comparison takes into account both the absolute difference of the values
/// and the relative difference.
///
/// The behavior of this trait is designed to match PyTorch's `torch.allclose`
/// and NumPy's `np.allclose`.
pub trait ApproxEq: Sized {
    /// Return the default absolute tolerance value.
    fn default_abs_tolerance() -> Self;

    /// Return the default relative tolerance value.
    fn default_rel_tolerance() -> Self;

    /// Test whether `self` is "close" to `other` according to the formula:
    ///
    /// ```text
    /// (self - other).abs() <= atol + rtol * other.abs()
    /// ```
    fn approx_eq_with_atol_rtol(&self, other: &Self, atol: Self, rtol: Self) -> bool;

    /// Test if `other` is approximately equal to `self` with a maximum
    /// absolute difference of `epsilon`.
    fn approx_eq_with_tolerance(&self, other: &Self, epsilon: Self) -> bool {
        self.approx_eq_with_atol_rtol(other, epsilon, Self::default_rel_tolerance())
    }

    /// Test if `other` is approximately equal to `self` with the default
    /// tolerances for this type.
    fn approx_eq(&self, other: &Self) -> bool {
        self.approx_eq_with_atol_rtol(
            other,
            Self::default_abs_tolerance(),
            Self::default_rel_tolerance(),
        )
    }
}

impl ApproxEq for f32 {
    /// Default that matches `allclose` in PyTorch, NumPy.
    #[inline]
    fn default_abs_tolerance() -> f32 {
        1e-8
    }

    /// Default that matches `allclose` in PyTorch, NumPy.
    #[inline]
    fn default_rel_tolerance() -> f32 {
        1e-5
    }

    #[inline]
    fn approx_eq_with_atol_rtol(&self, other: &f32, atol: f32, rtol: f32) -> bool {
        (self - other).abs() <= atol + rtol * other.abs()
    }
}

impl ApproxEq for i32 {
    #[inline]
    fn default_abs_tolerance() -> i32 {
        0
    }

    #[inline]
    fn default_rel_tolerance() -> i32 {
        0
    }

    #[inline]
    fn approx_eq_with_atol_rtol(&self, other: &i32, atol: i32, rtol: i32) -> bool {
        (self - other).abs() <= atol + rtol * other.abs()
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

#[derive(Debug)]
pub enum ExpectEqualError {
    ShapeMismatch(String),
    ValueMismatch(String),
}

impl Display for ExpectEqualError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ExpectEqualError::ShapeMismatch(details) => write!(f, "{}", details),
            ExpectEqualError::ValueMismatch(details) => write!(f, "{}", details),
        }
    }
}

impl Error for ExpectEqualError {}

/// Check that the shapes of two tensors are equal and that their contents
/// are approximately equal.
///
/// If there are mismatches, this returns an `Err` with a message indicating
/// the count of mismatches and details of the first N cases.
pub fn expect_equal<V: AsView>(x: &V, y: &V) -> Result<(), ExpectEqualError>
where
    V::Elem: Clone + Debug + ApproxEq,
{
    expect_equal_with_tolerance(
        x,
        y,
        V::Elem::default_abs_tolerance(),
        V::Elem::default_rel_tolerance(),
    )
}

/// Check that the shapes of two tensors are equal and that their contents
/// are approximately equal.
///
/// This is like [expect_equal] but allows a custom absolute tolerance value.
pub fn expect_equal_with_tolerance<V: AsView>(
    x: &V,
    y: &V,
    atol: V::Elem,
    rtol: V::Elem,
) -> Result<(), ExpectEqualError>
where
    V::Elem: Clone + Debug + ApproxEq,
{
    if x.shape() != y.shape() {
        return Err(ExpectEqualError::ShapeMismatch(format!(
            "Tensors have different shapes. {:?} vs. {:?}",
            x.shape(),
            y.shape()
        )));
    }

    let mismatches: Vec<_> = zip(x.iter(), y.iter())
        .enumerate()
        .filter_map(|(i, (xi, yi))| {
            if !xi.approx_eq_with_atol_rtol(yi, atol.clone(), rtol.clone()) {
                Some((index_from_linear_index(x.shape().as_ref(), i), xi, yi))
            } else {
                None
            }
        })
        .collect();

    if !mismatches.is_empty() {
        let max_examples = 16;
        Err(ExpectEqualError::ValueMismatch(format!(
            "Tensor values differ at {} of {} indexes: {:?}{}",
            mismatches.len(),
            x.len(),
            &mismatches[..mismatches.len().min(max_examples)],
            if mismatches.len() > max_examples {
                "..."
            } else {
                ""
            }
        )))
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

pub struct BenchStats {
    /// Duration in seconds.
    duration: f64,
}

const SECS_TO_MS: f64 = 1000.;
const SECS_TO_US: f64 = 1_000_000.;
const SECS_TO_NS: f64 = 1_000_000_000.;

impl BenchStats {
    /// Return total duration in milliseconds.
    pub fn duration_ms(&self) -> f64 {
        self.duration * SECS_TO_MS
    }

    /// Return total duration in microseconds.
    pub fn duration_us(&self) -> f64 {
        self.duration * SECS_TO_US
    }

    /// Return total duration in nanoseconds.
    pub fn duration_ns(&self) -> f64 {
        self.duration * SECS_TO_NS
    }
}

/// A very simple benchmark helper which runs `f` for `n_iters` iterations.
pub fn bench_loop<F: FnMut()>(n_iters: usize, mut f: F) -> BenchStats {
    let start = std::time::Instant::now();

    for _ in 0..n_iters {
        f();
    }

    BenchStats {
        duration: start.elapsed().as_secs_f64(),
    }
}

#[cfg(test)]
mod tests {
    use super::ApproxEq;

    #[test]
    fn test_approx_eq_i32() {
        let vals = [-5, -1, 0, 1, 5];
        for val in vals {
            assert!(val.approx_eq(&val));
            assert!(!val.approx_eq(&(val + 1)));
        }
    }

    #[test]
    fn test_approx_eq_f32() {
        // Same values.
        let vals = [-1000., -5., -0.5, 0., 0.5, 5., 1000.];
        for val in vals {
            assert!(val.approx_eq(&val));
        }

        // Close values
        for val in vals {
            // 9e-9 and 9e-6 are slightly smaller than the default tolerances.
            let close = val + 9e-9 + val * 9e-6;
            assert_ne!(val, close);
            assert!(val.approx_eq(&close));
        }

        // Different values
        for val in vals {
            // 2e-8 and 2e-5 are larger than the default tolerances.
            let not_close = val + 2e-8 + val * 2e-5;
            assert_ne!(val, not_close);
            assert!(!val.approx_eq(&not_close));
        }
    }
}
