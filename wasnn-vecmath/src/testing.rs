use std::io::prelude::*;
use std::iter::repeat_with;
use std::time::Instant;

use crate::ulp::Ulp;

/// Iterator over all possible f32 values.
pub struct AllF32s {
    next: u32,
}

impl AllF32s {
    pub fn new() -> AllF32s {
        AllF32s { next: 0 }
    }
}

impl Iterator for AllF32s {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.next == u32::MAX {
            None
        } else {
            let next = f32::from_bits(self.next);
            self.next += 1;
            Some(next)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (u32::MAX - self.next) as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for AllF32s {}

/// Iterator that wrapper an inner iterator and logs progress messages as
/// items are pulled from it.
pub struct Progress<I: Iterator> {
    prefix: String,
    inner: I,
    remaining: usize,
    len: usize,
    report_step: usize,
}

impl<'a, I: Iterator> Progress<I> {
    /// Wrap the iterator `inner` with an iterator that prints progress messages
    /// prefixed by `prefix`.
    pub fn wrap(inner: I, prefix: &str) -> Progress<I> {
        let remaining = inner.size_hint().0;
        let report_step = (remaining / 1000).max(1);
        Progress {
            inner,
            remaining,
            len: remaining,
            report_step,
            prefix: prefix.to_string(),
        }
    }
}

impl<I: Iterator> Iterator for Progress<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.remaining = self.remaining.saturating_sub(1);
        if self.remaining % self.report_step == 0 {
            let done = self.len - self.remaining;
            let progress = done as f32 / self.len as f32;
            print!("\r{}: {:.2}%", self.prefix, progress * 100.);
            let _ = std::io::stdout().flush();
        } else if self.remaining == 0 {
            println!("");
        }
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

/// Iterator over an arithmetic range. See [arange].
pub struct ARange<T: Copy + PartialOrd + std::ops::Add<Output = T>> {
    next: T,
    end: T,
    step: T,
}

/// Return an iterator over an arithmetic range `[start, end)` in steps of `step`.
///
/// Iteration stops if the next value in the series cannot be compared against
/// the end value (ie. if `next.partial_cmp(end)` yields `None`).
pub fn arange<T: Copy + PartialOrd + std::ops::Add<Output = T>>(
    start: T,
    end: T,
    step: T,
) -> ARange<T> {
    ARange {
        next: start,
        end,
        step,
    }
}

impl<T: Copy + PartialOrd + std::ops::Add<Output = T>> Iterator for ARange<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        use std::cmp::Ordering;
        let next = self.next;
        match next.partial_cmp(&self.end) {
            Some(Ordering::Less) => {
                self.next = self.next + self.step;
                Some(next)
            }
            _ => None,
        }
    }
}

/// Take three slices of equal length and return an iterator over triples of
/// corresponding elements.
///
/// This is a helper for use with other testing methods.
pub fn triples<'a, T: Copy>(
    input: &'a [T],
    actual: &'a [T],
    expected: &'a [T],
) -> impl Iterator<Item = (T, T, T)> + 'a {
    assert!(input.len() == actual.len() && actual.len() == expected.len());
    input
        .iter()
        .zip(actual.iter().zip(expected.iter()))
        .map(|(x, (actual, expected))| (*x, *actual, *expected))
}

/// Compare results of an operation on floats against expected results.
///
/// `results` is an iterator yielding tuples of `(input, actual, expected)`
/// values.
///
/// `max_diff` specifies the maximum allowed difference between `actual` and
/// `expected`.
pub fn check_f32s_are_equal_atol<I: Iterator<Item = (f32, f32, f32)>>(results: I, max_diff: f32) {
    for (x, actual, expected) in results {
        let diff = (actual - expected).abs();
        assert!(
            diff <= max_diff,
            "diff {} exceeds expected {} at x = {}",
            diff,
            max_diff,
            x
        );
    }
}

/// Compare results of an operation on floats against expected results.
///
/// `results` is an iterator yielding tuples of `(input, actual, expected)`
/// values.
///
/// `ulp_threshold` specifies the maximum allowed difference between
/// `actual` and `expected` in ULPs.
pub fn check_f32s_are_equal_ulps<I: Iterator<Item = (f32, f32, f32)>>(
    results: I,
    ulp_threshold: f32,
) {
    let mut max_diff_ulps = 0.0f32;
    let mut max_diff_x = 0.0f32;
    let mut max_diff_actual = 0.0f32;
    let mut max_diff_expected = 0.0f32;

    for (x, actual, expected) in results {
        if actual == expected {
            // Fast path for expected common case where results are exactly
            // equal.
            continue;
        }

        assert_eq!(
            expected.is_nan(),
            actual.is_nan(),
            "NaN mismatch at {x}. Actual {x} Expected {x}"
        );
        assert_eq!(
            expected.is_infinite(),
            actual.is_infinite(),
            "Infinite mismatch at {x}. Actual {actual} Expected {expected}"
        );

        if !expected.is_infinite() && !expected.is_nan() {
            let diff = (actual - expected).abs();
            let diff_ulps = diff / expected.ulp();
            if diff_ulps > max_diff_ulps {
                max_diff_ulps = max_diff_ulps.max(diff_ulps);
                max_diff_x = x;
                max_diff_actual = actual;
                max_diff_expected = expected;
            }
        }
    }
    assert!(
        max_diff_ulps <= ulp_threshold,
        "max diff against reference is {} ULPs for x = {}, actual = {}, expected = {}, ULP = {}. Above ULP threshold {}",
        max_diff_ulps,
        max_diff_x,
        max_diff_actual,
        max_diff_expected,
        max_diff_expected.ulp(),
        ulp_threshold
    );
}

/// Test a unary function against all possible values of a 32-bit float.
///
/// `op` is a function that takes an f32 and computes the actual and
/// expected values of the function, where the expected value is computed
/// using a reference implementation.
///
/// `ulp_threshold` specifies the maximum difference between the actual
/// and expected values, in ULPs, when the expected value is not infinite
/// or NaN.
pub fn check_with_all_f32s<F: Fn(f32) -> (f32, f32)>(
    op: F,
    ulp_threshold: f32,
    progress_msg: &str,
) {
    let actual_expected = AllF32s::new().map(|x| {
        let (actual, expected) = op(x);
        (x, actual, expected)
    });
    check_f32s_are_equal_ulps(Progress::wrap(actual_expected, progress_msg), ulp_threshold);
}

/// Benchmark a vectorized implementation of a function against a reference
/// implementation.
///
/// `reference` and `vectorized` are functions which take an input slice of
/// elements as a first argument and write the results of applying the operation
/// to the second argument.
pub fn benchmark_op<RF: Fn(&[f32], &mut [f32]), VF: Fn(&[f32], &mut [f32])>(
    reference: RF,
    vectorized: VF,
) {
    let input: Vec<_> = repeat_with(|| fastrand::f32()).take(1_000_000).collect();
    let mut output = vec![0.; input.len()];
    let iters = 100;

    let reference_start = Instant::now();
    for _ in 0..iters {
        reference(&input, &mut output);
    }
    let reference_elapsed = reference_start.elapsed().as_micros();

    let vecmath_vec_start = Instant::now();
    for _ in 0..iters {
        vectorized(&input, &mut output);
    }
    let vecmath_vec_elapsed = vecmath_vec_start.elapsed().as_micros();

    let ratio = reference_elapsed as f32 / vecmath_vec_elapsed as f32;

    println!(
        "reference {} us vectorized {} us. reference / vectorized ratio {:.3}",
        reference_elapsed, vecmath_vec_elapsed, ratio
    );
}
