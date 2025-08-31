use std::mem::MaybeUninit;

use std::io::prelude::*;
use std::iter::repeat_with;
use std::time::Instant;

use crate::ulp::Ulp;
use rten_simd::SimdUnaryOp;

/// Trait for converting containers of initialized values into uninitialized
/// ones.
pub trait AsUninit {
    type Output;

    /// Convert all elements from `T` to `MaybeUninit<T>`
    fn as_uninit(self) -> Self::Output;
}

impl<'a, T: Copy> AsUninit for &'a mut [T] {
    type Output = &'a mut [MaybeUninit<T>];

    fn as_uninit(self) -> Self::Output {
        unsafe { std::mem::transmute(self) }
    }
}

/// Iterator over all possible f32 values.
#[derive(Clone)]
pub struct AllF32s {
    next: Option<u32>,
}

impl AllF32s {
    pub fn new() -> AllF32s {
        AllF32s { next: Some(0) }
    }
}

impl Iterator for AllF32s {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        let next = self.next?;
        self.next = next.checked_add(1);
        Some(f32::from_bits(next))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self
            .next
            .map(|next| (u32::MAX - next) as usize + 1)
            .unwrap_or(0);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for AllF32s {}

/// Iterator over an arithmetic range. See [`arange`].
#[derive(Copy, Clone, Debug)]
pub struct ARange<T: Copy + PartialOrd + std::ops::Add<Output = T>> {
    next: T,
    end: T,
    step: T,
}

/// Return an iterator over an arithmetic range `[start, end)` in steps of `step`.
///
/// Iteration stops if the next value in the series cannot be compared against
/// the end value (ie. if `next.partial_cmp(end)` yields `None`).
pub const fn arange<T: Copy + PartialOrd + std::ops::Add<Output = T>>(
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
        if actual.is_nan() && expected.is_nan() {
            continue;
        }
        let diff = (actual - expected).abs();
        assert!(
            diff <= max_diff,
            "diff {} exceeds expected {} at x = {}. actual = {}, expected = {}",
            diff,
            max_diff,
            x,
            actual,
            expected
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
            actual.is_nan(),
            expected.is_nan(),
            "NaN mismatch at {x}. Actual {actual} Expected {expected}",
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

/// Benchmark a vectorized implementation of a function against a reference
/// implementation.
///
/// `reference` and `vectorized` are functions which take an input slice of
/// elements as a first argument and write the results of applying the operation
/// to the second argument.
pub fn benchmark_op<RF: Fn(&[f32], &mut [f32]), VF: Fn(&[f32], &mut [MaybeUninit<f32>])>(
    reference: RF,
    vectorized: VF,
) {
    // Generate values in [-0.5, 0.5].
    //
    // This range is chosen because a) it is a common range for inputs to NN
    // functions and b) some of the vectorized functions are activations which
    // change behavior above/below zero. If we used the default `fastrand::f32`
    // range of [0, 1] this would give misleading results for reference
    // implementations as they would always take the same branch.
    let input: Vec<_> = repeat_with(|| -0.5 + fastrand::f32())
        .take(1_000_000)
        .collect();

    let mut output = vec![0.; input.len()];
    let iters = 100;

    let reference_start = Instant::now();
    for _ in 0..iters {
        reference(&input, &mut output);
    }
    let reference_elapsed = reference_start.elapsed().as_micros();

    let vecmath_vec_start = Instant::now();
    for _ in 0..iters {
        vectorized(&input, output.as_mut_slice().as_uninit());
    }
    let vecmath_vec_elapsed = vecmath_vec_start.elapsed().as_micros();

    let ratio = reference_elapsed as f32 / vecmath_vec_elapsed as f32;

    println!(
        "reference {} us vectorized {} us. reference / vectorized ratio {:.3}",
        reference_elapsed, vecmath_vec_elapsed, ratio
    );
}

pub enum Tolerance {
    /// Number of Units of Least Precision.
    Ulp(f32),
    /// Maximum absolute difference.
    Absolute(f32),
}

/// Tests a vectorized implementation of a unary function against a reference
/// implementation.
pub struct UnaryOpTester<F: Fn(f32) -> f32, S: SimdUnaryOp<f32>, R: Iterator<Item = f32> + Clone> {
    /// Reference implementation of the operation.
    pub reference: F,

    /// Vectorized implementation of the operation.
    pub simd: S,

    /// Iterator yielding values to test.
    pub range: R,

    /// Tolerance for comparisons between reference and actual results.
    pub tolerance: Tolerance,
}

impl<F: Fn(f32) -> f32, S: SimdUnaryOp<f32>, R: Iterator<Item = f32> + Clone>
    UnaryOpTester<F, S, R>
{
    /// Run an evaluation of a vectorized operation against a reference and
    /// panic if the difference exceeds the tolerance for any input.
    pub fn run(&self) {
        self.run_impl(false)
    }

    /// Variant of [`run`](Self::run) which reports progress as it runs.
    ///
    /// This is intended for use with ranges like [`AllF32s`] with large numbers
    /// of elements.
    pub fn run_with_progress(&self) {
        self.run_impl(true)
    }

    fn run_impl(&self, with_progress: bool) {
        let mut total = 0;
        let mut remaining = self.range.clone();

        // Progress reporting uses the upper bound, so it can report a lower
        // value than the actual progress. For example a range with a filter
        // applied will jump when values are skipped.
        let (_lower_bound, upper_bound) = remaining.size_hint();

        let chunk_size = 16 * 1024;
        let mut input = Vec::with_capacity(chunk_size);
        let mut actual = Vec::with_capacity(chunk_size);
        let mut expected = Vec::with_capacity(chunk_size);

        if with_progress && upper_bound.is_none() {
            println!("Testing: ...");
        }

        loop {
            if with_progress && upper_bound.is_some() {
                let progress = total as f32 / upper_bound.unwrap() as f32;
                print!("\rTesting: {:.2}%", progress * 100.);
                let _ = std::io::stdout().flush();
            }

            input.clear();
            actual.clear();
            expected.clear();

            input.extend((&mut remaining).take(chunk_size));
            if input.is_empty() {
                break;
            }

            total += input.len();

            expected.extend(input.iter().copied().map(&self.reference));
            actual.extend(&input);
            self.simd.map_mut(&mut actual);

            let results = triples(&input, &actual, &expected);
            match self.tolerance {
                Tolerance::Ulp(max_ulps) => check_f32s_are_equal_ulps(results, max_ulps),
                Tolerance::Absolute(max_error) => check_f32s_are_equal_atol(results, max_error),
            }
        }
        assert!(total > 0, "input range was empty");
    }
}
