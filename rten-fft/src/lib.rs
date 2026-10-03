//! Fast Fourier Transform for `f32` complex signals.
//!
//! Complex signals are represented in a decomposed form, with separate slices
//! for the real and imaginary parts. This representation enables efficient use
//! of SIMD internally.

use rten_simd::ops::BitOps;
use rten_simd::{Isa, SimdOp};

mod bluestein;
mod complex;
mod stockham;

use bluestein::Bluestein;
use complex::ComplexSliceMut;
use stockham::Stockham;

/// Return `exp(i * theta)` as `(re, im)`.
///
/// The angle is taken as `f64` because the FFT's accuracy depends heavily on
/// the accuracy of its twiddle factors.
fn unit(theta: f64) -> (f32, f32) {
    let (sin, cos) = theta.sin_cos();
    (cos as f32, sin as f32)
}

/// Negate the imaginary part of a signal in place.
fn conjugate(im: &mut [f32]) {
    for x in im {
        *x = -*x;
    }
}

/// How the signals a transform operates on are laid out in memory.
#[derive(Copy, Clone, PartialEq)]
enum Layout {
    /// A single signal, with consecutive elements in consecutive positions.
    Single,
    /// One signal per SIMD lane, in `[len][lanes]` layout.
    Batch,
}

/// Return the number of `f32` lanes in a SIMD vector on the current system.
fn batch_size() -> usize {
    struct Lanes;
    impl SimdOp for Lanes {
        type Output = usize;
        fn eval<I: Isa>(self, isa: I) -> usize {
            isa.f32().len()
        }
    }
    Lanes.dispatch()
}

enum Plan {
    /// Transform of length 0 or 1, which copies the input.
    Trivial,

    /// Mixed-radix Stockham decomposition.
    Stockham(Stockham),

    /// Bluestein's algorithm, for lengths with a large prime factor.
    Bluestein(Bluestein),
}

/// Fast Fourier Transform for signals of a given length and direction.
///
/// Signals are planar: the real and imaginary parts are passed as separate
/// slices. The transform is unnormalized: an inverse transform scales the
/// signal by its length.
///
/// It can transform one signal at a time with [`process`](Fft::process)
/// or a batch of [`batch_size`](Fft::batch_size) signals at once with
/// [`process_batch`](Fft::process_batch). The batched form puts one signal
/// in each SIMD lane and is several times faster per signal, so callers with
/// many signals of the same length should prefer it.
pub struct Fft {
    len: usize,
    inverse: bool,
    plan: Plan,
}

impl Fft {
    /// Plan a transform of `len` elements.
    ///
    /// This precomputes twiddle factors, so it should be hoisted out of loops
    /// that transform many signals of the same length.
    pub fn new(len: usize, inverse: bool) -> Fft {
        Fft {
            len,
            inverse,
            plan: Self::plan_kind(len),
        }
    }

    fn plan_kind(len: usize) -> Plan {
        if len <= 1 {
            return Plan::Trivial;
        }
        match Stockham::new(len) {
            Some(stockham) => Plan::Stockham(stockham),
            None => Plan::Bluestein(Bluestein::new(len)),
        }
    }

    /// Return the length of the transform.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return true if this is a transform of zero elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return the name of the algorithm this plan uses.
    #[cfg(test)]
    fn path_name(&self) -> &'static str {
        match &self.plan {
            Plan::Trivial => "trivial",
            Plan::Stockham(_) => "stockham",
            Plan::Bluestein(_) => "bluestein",
        }
    }

    /// Return the number of signals that
    /// [`process_batch`](Fft::process_batch) transforms at once.
    ///
    /// This is the SIMD vector width for the current system.
    pub fn batch_size(&self) -> usize {
        batch_size()
    }

    /// Return the number of `f32` elements in the scratch buffer that
    /// [`process`](Fft::process) requires, per signal.
    ///
    /// [`process_batch`](Fft::process_batch) requires
    /// [`batch_size`](Fft::batch_size) times as many.
    pub fn scratch_len(&self) -> usize {
        match &self.plan {
            Plan::Trivial => 0,
            Plan::Stockham(_) => 2 * self.len,
            Plan::Bluestein(bluestein) => bluestein.scratch_len(),
        }
    }

    /// Transform the signal with real parts `re` and imaginary parts `im` in
    /// place.
    ///
    /// `re` and `im` must each have [`len`](Fft::len) elements and
    /// `scratch` must have [`scratch_len`](Fft::scratch_len) elements.
    pub fn process(&self, re: &mut [f32], im: &mut [f32], scratch: &mut [f32]) {
        assert_eq!(re.len(), self.len, "incorrect FFT buffer length");
        assert_eq!(im.len(), self.len, "incorrect FFT buffer length");
        assert_eq!(
            scratch.len(),
            self.scratch_len(),
            "incorrect FFT scratch length"
        );
        self.run(Layout::Single, re, im, scratch);
    }

    /// Transform a batch of [`batch_size`](Fft::batch_size) signals in
    /// place.
    ///
    /// The batch is stored with one signal per lane: element `i` of signal
    /// `b` is at index `i * batch_size + b` of `re` and `im`. `scratch` must
    /// have `batch_size * scratch_len` elements.
    pub fn process_batch(&self, re: &mut [f32], im: &mut [f32], scratch: &mut [f32]) {
        let lanes = self.batch_size();
        assert_eq!(re.len(), self.len * lanes, "incorrect FFT buffer length");
        assert_eq!(im.len(), self.len * lanes, "incorrect FFT buffer length");
        assert_eq!(
            scratch.len(),
            self.scratch_len() * lanes,
            "incorrect FFT scratch length"
        );
        self.run(Layout::Batch, re, im, scratch);
    }

    fn run(&self, layout: Layout, re: &mut [f32], im: &mut [f32], scratch: &mut [f32]) {
        // The inverse transform is the forward transform of the conjugated
        // signal, conjugated. This avoids duplicating every butterfly for the
        // opposite rotation direction, at the cost of two passes over the
        // signal.
        if self.inverse {
            conjugate(im);
        }
        self.forward(layout, &mut ComplexSliceMut { re, im }, scratch);
        if self.inverse {
            conjugate(im);
        }
    }

    /// Apply the unnormalized forward transform to `buf`, which holds one or
    /// more signals in the given layout.
    fn forward(&self, layout: Layout, buf: &mut ComplexSliceMut, scratch: &mut [f32]) {
        match &self.plan {
            Plan::Trivial => {}
            Plan::Stockham(stockham) => {
                stockham.forward(layout, buf, &mut ComplexSliceMut::from_halves(scratch))
            }
            Plan::Bluestein(bluestein) => bluestein.forward(layout, buf, scratch),
        }
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::rng::XorShiftRng;
    use rten_testing::TestCases;

    use super::{Fft, Plan};
    use crate::complex::ComplexVec;

    /// Compute the DFT of `signal` directly from its definition, in `f64`.
    ///
    /// This is `O(n^2)` but needs no decomposition, so it serves as a reference
    /// for every transform length.
    fn naive_dft(signal: &ComplexVec, inverse: bool) -> ComplexVec {
        let n = signal.re.len();
        let sign = if inverse { 1. } else { -1. };
        ComplexVec::from_fn(n, |k| {
            let (mut re, mut im) = (0f64, 0f64);
            for t in 0..n {
                let (xr, xi) = (signal.re[t] as f64, signal.im[t] as f64);
                let theta = sign * 2. * std::f64::consts::PI * (t * k % n) as f64 / n as f64;
                let (sin, cos) = theta.sin_cos();
                re += xr * cos - xi * sin;
                im += xr * sin + xi * cos;
            }
            (re as f32, im as f32)
        })
    }

    fn random_signal(len: usize, seed: u64) -> ComplexVec {
        let mut rng = XorShiftRng::new(seed);
        ComplexVec::from_fn(len, |_| (rng.next_f32() - 0.5, rng.next_f32() - 0.5))
    }

    /// Return the maximum difference between two signals, relative to the
    /// largest magnitude in `expected`.
    fn relative_error(actual: &ComplexVec, expected: &ComplexVec) -> f32 {
        let max_abs = expected
            .re
            .iter()
            .chain(&expected.im)
            .map(|x| x.abs())
            .fold(0f32, f32::max);
        let max_err = actual
            .re
            .iter()
            .chain(&actual.im)
            .zip(expected.re.iter().chain(&expected.im))
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        if max_abs == 0. {
            max_err
        } else {
            max_err / max_abs
        }
    }

    /// Transform `signal` with `fft`, one signal at a time.
    fn run_single(fft: &Fft, signal: &ComplexVec) -> ComplexVec {
        let mut buf = signal.clone();
        let mut scratch = vec![0.; fft.scratch_len()];
        fft.process(&mut buf.re, &mut buf.im, &mut scratch);
        buf
    }

    /// Transform `signals`, which must have `fft.batch_size()` entries, as a
    /// batch and return the results per signal.
    fn run_batch(fft: &Fft, signals: &[ComplexVec]) -> Vec<ComplexVec> {
        let lanes = fft.batch_size();
        assert_eq!(signals.len(), lanes);
        let len = fft.len();

        let mut batch = ComplexVec::from_fn(len * lanes, |i| {
            let (elem, lane) = (i / lanes, i % lanes);
            (signals[lane].re[elem], signals[lane].im[elem])
        });
        let mut scratch = vec![0.; fft.scratch_len() * lanes];
        fft.process_batch(&mut batch.re, &mut batch.im, &mut scratch);

        (0..lanes)
            .map(|lane| {
                ComplexVec::from_fn(len, |i| {
                    (batch.re[i * lanes + lane], batch.im[i * lanes + lane])
                })
            })
            .collect()
    }

    /// Return transform lengths covering each butterfly, the generic radix
    /// and Bluestein's algorithm.
    fn test_lengths() -> impl Iterator<Item = usize> {
        (0..=64).chain([
            100,  // 2^2 * 5^2
            120,  // 2^3 * 3 * 5
            128,  // 2^7
            243,  // 3^5
            352,  // 2^5 * 11, generic radix stage
            400,  // 2^4 * 5^2, Whisper's frame length
            448,  // 2^6 * 7, generic radix stage
            512,  // 2^9
            625,  // 5^4
            1000, // 2^3 * 5^3
            1024, // 2^10
            67,   // Prime, Bluestein
            97,   // Prime, Bluestein
            251,  // Prime, Bluestein
            1021, // Prime, Bluestein
            134,  // 2 * 67, Bluestein
        ])
    }

    #[test]
    fn test_fft_matches_naive_dft() {
        #[derive(Debug)]
        struct Case {
            len: usize,
        }

        let cases: Vec<Case> = test_lengths().map(|len| Case { len }).collect();

        cases.test_each(|case| {
            for inverse in [false, true] {
                let signal = random_signal(case.len, 1234);
                let expected = naive_dft(&signal, inverse);
                let fft = Fft::new(case.len, inverse);
                let actual = run_single(&fft, &signal);
                let err = relative_error(&actual, &expected);
                assert!(
                    err < 1e-5,
                    "len {} inverse {} relative error {}",
                    case.len,
                    inverse,
                    err
                );
            }
        });
    }

    #[test]
    fn test_batch_matches_naive_dft() {
        #[derive(Debug)]
        struct Case {
            len: usize,
        }

        let cases: Vec<Case> = test_lengths().map(|len| Case { len }).collect();

        cases.test_each(|case| {
            for inverse in [false, true] {
                let fft = Fft::new(case.len, inverse);
                let lanes = fft.batch_size();

                // Distinct signal per lane.
                let signals: Vec<ComplexVec> = (0..lanes)
                    .map(|lane| random_signal(case.len, 5678 + lane as u64))
                    .collect();
                let results = run_batch(&fft, &signals);

                for (lane, (signal, actual)) in signals.iter().zip(&results).enumerate() {
                    let expected = naive_dft(signal, inverse);
                    let err = relative_error(actual, &expected);
                    assert!(
                        err < 1e-5,
                        "len {} inverse {} lane {} relative error {}",
                        case.len,
                        inverse,
                        lane,
                        err
                    );
                }
            }
        });
    }

    #[test]
    fn test_roundtrip() {
        #[derive(Debug)]
        struct Case {
            len: usize,
        }

        let cases = [400, 512, 1021].map(|len| Case { len });

        cases.test_each(|case| {
            let signal = random_signal(case.len, 1234);
            let forward = Fft::new(case.len, false);
            let inverse = Fft::new(case.len, true);
            let spectrum = run_single(&forward, &signal);
            let restored = run_single(&inverse, &spectrum);

            // The transform is unnormalized, so a round trip scales the signal
            // by its length.
            let scale = 1. / case.len as f32;
            for i in 0..case.len {
                assert!((restored.re[i] * scale - signal.re[i]).abs() < 1e-5);
                assert!((restored.im[i] * scale - signal.im[i]).abs() < 1e-5);
            }
        });
    }

    #[test]
    fn test_plan_selection() {
        #[derive(Debug)]
        struct Case {
            len: usize,
            bluestein: bool,
        }

        let cases = [
            // Lengths whose prime factors all have a butterfly or fit the
            // generic one.
            Case {
                len: 1024,
                bluestein: false,
            },
            Case {
                len: 400,
                bluestein: false,
            },
            Case {
                len: 448, // 2^6 * 7
                bluestein: false,
            },
            Case {
                len: 61, // Prime, within the generic radix limit
                bluestein: false,
            },
            // Lengths with a prime factor above the generic radix limit.
            Case {
                len: 67,
                bluestein: true,
            },
            Case {
                len: 134, // 2 * 67
                bluestein: true,
            },
            Case {
                len: 8191,
                bluestein: true,
            },
        ];

        cases.test_each(|case| {
            let fft = Fft::new(case.len, false);
            assert_eq!(matches!(fft.plan, Plan::Bluestein(_)), case.bluestein);
        });
    }
}

#[cfg(test)]
mod bench {
    use rten_bench::run_bench;
    use rten_tensor::rng::XorShiftRng;

    use super::Fft;
    use crate::complex::ComplexVec;

    #[ignore]
    #[test]
    fn bench_fft() {
        // Lengths chosen to cover the paths a real model is likely to take
        // (powers of two, and the 5-smooth lengths used by audio frontends)
        // plus the generic-radix and Bluestein fallbacks.
        let lengths = [256, 400, 448, 512, 1021, 1024, 2048];

        // Number of transforms per trial, so each trial does a similar amount
        // of work regardless of length.
        let batch = 1000;

        println!(
            "{:>6} {:>12} {:>12} {:>12} {:>12} {:>12}",
            "len", "path", "single us", "single GF", "batch us", "batch GF"
        );

        for len in lengths {
            let mut rng = XorShiftRng::new(1234);
            let signal = ComplexVec::from_fn(len, |_| (rng.next_f32() - 0.5, rng.next_f32() - 0.5));

            let fft = Fft::new(len, false);
            let lanes = fft.batch_size();

            let mut buf = ComplexVec::zeros(len);
            let mut scratch = vec![0.; fft.scratch_len()];
            let single = run_bench(10, None, || {
                for _ in 0..batch {
                    buf.re.copy_from_slice(&signal.re);
                    buf.im.copy_from_slice(&signal.im);
                    fft.process(&mut buf.re, &mut buf.im, &mut scratch);
                }
            });

            // Same signal in every lane.
            let batch_signal = ComplexVec::from_fn(len * lanes, |i| {
                (signal.re[i / lanes], signal.im[i / lanes])
            });
            let mut batch_buf = ComplexVec::zeros(len * lanes);
            let mut batch_scratch = vec![0.; fft.scratch_len() * lanes];
            let batched = run_bench(10, None, || {
                for _ in 0..batch / lanes {
                    batch_buf.re.copy_from_slice(&batch_signal.re);
                    batch_buf.im.copy_from_slice(&batch_signal.im);
                    fft.process_batch(&mut batch_buf.re, &mut batch_buf.im, &mut batch_scratch);
                }
            });

            // Conventional FFT operation count, used to compare implementations
            // rather than to count the operations actually performed.
            let flops = 5. * len as f32 * (len as f32).log2();
            let single_secs = single.median / 1000. / batch as f32;
            let batch_secs = batched.median / 1000. / (batch / lanes * lanes) as f32;

            println!(
                "{:>6} {:>12} {:>12.3} {:>12.1} {:>12.3} {:>12.1}",
                len,
                fft.path_name(),
                single_secs * 1e6,
                flops / single_secs / 1e9,
                batch_secs * 1e6,
                flops / batch_secs / 1e9,
            );
        }
    }
}
