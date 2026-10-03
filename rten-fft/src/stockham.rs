use std::f64::consts::PI;

use rten_simd::ops::{BitOps, FloatOps};
use rten_simd::{Isa, Simd, SimdOp};

use crate::complex::{
    Complex, ComplexSimdSlice, ComplexSimdSliceMut, ComplexSlice, ComplexSliceMut, ComplexVec,
};
use crate::{Layout, batch_size, unit};

/// Largest radix handled by the generic `O(radix^2)` butterfly.
///
/// Lengths with a prime factor above this use Bluestein's algorithm instead.
/// The generic butterfly costs `O(len * radix)` for the whole transform, so the
/// cutoff trades a modest slowdown on lengths like `7 * 2^k` against the much
/// larger constant factor of Bluestein's algorithm.
const MAX_GENERIC_RADIX: usize = 64;

/// Apply the size-`radix` DFT to `u[..radix]` in place.
///
/// `P` is the capacity of `u`. For the small radices it equals `radix` and
/// selects a hand-written butterfly. For any other `P` the DFT is evaluated
/// directly from its definition using the twiddle table `tw`, with `v` as
/// scratch space.
///
/// These are the "codelets" or "butterflies" that form the innermost step of
/// each transform stage. They are written as straight-line code with constant
/// twiddle factors so that the values stay in registers.
#[inline(always)]
fn butterfly<O: FloatOps<f32>, const P: usize>(
    ops: O,
    u: &mut [Complex<O::Simd>; P],
    v: &mut [Complex<O::Simd>; P],
    radix: usize,
    tw: ComplexSlice,
) {
    // sin(2 * pi / 3)
    const SIN_2PI_3: f32 = 0.866_025_4;
    // cos(2 * pi / 5), sin(2 * pi / 5), cos(4 * pi / 5), sin(4 * pi / 5)
    const COS_2PI_5: f32 = 0.309_017;
    const SIN_2PI_5: f32 = 0.951_056_5;
    const COS_4PI_5: f32 = -0.809_017;
    const SIN_4PI_5: f32 = 0.587_785_25;

    match P {
        2 => {
            let (x0, x1) = (u[0], u[1]);
            u[0] = x0.add(ops, x1);
            u[1] = x0.sub(ops, x1);
        }
        3 => {
            let (x0, x1, x2) = (u[0], u[1], u[2]);
            let sum = x1.add(ops, x2);
            let diff = x1.sub(ops, x2).scale(ops, ops.splat(SIN_2PI_3));
            let mid = x0.scale_add(ops, sum, ops.splat(-0.5));
            u[0] = x0.add(ops, sum);
            u[1] = mid.add_rot(ops, diff);
            u[2] = mid.sub_rot(ops, diff);
        }
        4 => {
            let (x0, x1, x2, x3) = (u[0], u[1], u[2], u[3]);
            let (t0, t1) = (x0.add(ops, x2), x0.sub(ops, x2));
            let (t2, t3) = (x1.add(ops, x3), x1.sub(ops, x3));
            u[0] = t0.add(ops, t2);
            u[1] = t1.add_rot(ops, t3);
            u[2] = t0.sub(ops, t2);
            u[3] = t1.sub_rot(ops, t3);
        }
        5 => {
            let (x0, x1, x2, x3, x4) = (u[0], u[1], u[2], u[3], u[4]);
            let (sum1, diff1) = (x1.add(ops, x4), x1.sub(ops, x4));
            let (sum2, diff2) = (x2.add(ops, x3), x2.sub(ops, x3));

            let (c1, s1) = (ops.splat(COS_2PI_5), ops.splat(SIN_2PI_5));
            let (c2, s2) = (ops.splat(COS_4PI_5), ops.splat(SIN_4PI_5));

            let even1 = x0.scale_add(ops, sum1, c1).scale_add(ops, sum2, c2);
            let odd1 = diff1.scale(ops, s1).scale_add(ops, diff2, s2);
            let even2 = x0.scale_add(ops, sum1, c2).scale_add(ops, sum2, c1);
            let odd2 = diff1.scale(ops, s2).scale_add(ops, diff2, ops.neg(s1));

            u[0] = x0.add(ops, sum1).add(ops, sum2);
            u[1] = even1.add_rot(ops, odd1);
            u[2] = even2.add_rot(ops, odd2);
            u[3] = even2.sub_rot(ops, odd2);
            u[4] = even1.sub_rot(ops, odd1);
        }
        _ => {
            let n = tw.re.len();
            // Step between twiddle table entries for the size-`radix` DFT.
            let inner_step = n / radix;
            for i in 0..radix {
                let mut sum = u[0];
                for t in 1..radix {
                    let idx = (t * i * inner_step) % n;
                    let w = Complex::splat(ops, tw.re[idx], tw.im[idx]);
                    sum = sum.add(ops, u[t].mul(ops, w));
                }
                v[i] = sum;
            }
            u[..radix].copy_from_slice(&v[..radix]);
        }
    }
}

/// One Stockham stage, evaluated as a SIMD operation.
///
/// The stage treats its input as a `[radix][l][r]` array and its output as
/// `[l][radix][r]`, so the digit reversal that a conventional Cooley-Tukey FFT
/// performs in a separate pass is folded into the stages. Output `t` of the
/// butterfly at `(j, k)` is multiplied by twiddle `t * j * r` from `tw`.
///
/// `P` is the radix for the small radices with a dedicated butterfly, or
/// `MAX_GENERIC_RADIX` for the generic butterfly, in which case `radix` gives
/// the actual radix.
///
/// Each stage is dispatched separately rather than inlining the whole
/// transform into one operation. This keeps each instantiation small enough
/// for the butterfly values to stay in registers.
struct Stage<'a, const P: usize> {
    layout: Layout,
    radix: usize,
    src: ComplexSlice<'a>,
    dst: ComplexSliceMut<'a>,
    l: usize,
    r: usize,
    tw: ComplexSlice<'a>,
    /// Per-lane twiddles for the `Single` layout when `r` is smaller than the
    /// vector width. See [`Stockham::narrow_twiddles`].
    narrow_tw: Option<ComplexSlice<'a>>,
}

impl<const P: usize> SimdOp for Stage<'_, P> {
    type Output = ();

    #[inline(always)]
    fn eval<I: Isa>(self, isa: I) {
        let ops = isa.f32();
        let Stage {
            layout,
            radix,
            src,
            dst,
            l,
            r,
            tw,
            narrow_tw,
        } = self;
        let radix = if P == MAX_GENERIC_RADIX { radix } else { P };
        let lanes = match layout {
            Layout::Single => 1,
            Layout::Batch => ops.len(),
        };
        let n = l * r * radix;
        assert_eq!(src.re.len(), n * lanes);
        assert_eq!(src.im.len(), n * lanes);
        assert_eq!(dst.re.len(), n * lanes);
        assert_eq!(dst.im.len(), n * lanes);

        match layout {
            Layout::Batch => {
                let (src, dst) = (
                    ComplexSimdSlice::new(ops, src),
                    ComplexSimdSliceMut::new(ops, dst),
                );
                stage_batch::<_, P>(ops, radix, src, dst, l, r, tw)
            }
            Layout::Single if r >= ops.len() => {
                let (src, dst) = (
                    ComplexSimdSlice::new(ops, src),
                    ComplexSimdSliceMut::new(ops, dst),
                );
                stage_wide::<_, P>(ops, radix, src, dst, l, r, tw)
            }
            Layout::Single => {
                let narrow_tw = narrow_tw.expect("missing twiddles for narrow stage");
                stage_narrow::<_, P>(ops, radix, src, dst, l, r, tw, narrow_tw)
            }
        }
    }
}

/// Apply a stage to a batch of signals.
///
/// The batch holds one signal per SIMD lane, so element `i` of every signal
/// forms one vector. This is the scalar formulation of the stage with each
/// scalar replaced by a vector: every load and store is a whole vector and
/// the twiddle factors are broadcast, so it needs no lane shuffling at any
/// stage.
#[inline(always)]
fn stage_batch<O: FloatOps<f32>, const P: usize>(
    ops: O,
    radix: usize,
    src: ComplexSimdSlice<O>,
    mut dst: ComplexSimdSliceMut<O>,
    l: usize,
    r: usize,
    tw: ComplexSlice,
) {
    let lanes = ops.len();

    let mut u = [Complex::zero(ops); P];
    let mut v = [Complex::zero(ops); P];
    let mut w = [Complex::zero(ops); P];

    for j in 0..l {
        for (t, w) in w.iter_mut().enumerate().take(radix).skip(1) {
            let idx = t * j * r;
            *w = Complex::splat(ops, tw.re[idx], tw.im[idx]);
        }

        for k in 0..r {
            for (t, u) in u.iter_mut().enumerate().take(radix) {
                *u = src.load(lanes * (k + r * (j + t * l)));
            }

            butterfly::<_, P>(ops, &mut u, &mut v, radix, tw);

            dst.store(lanes * (k + r * j * radix), u[0]);
            if j == 0 {
                // All twiddles for `j == 0` are 1.
                for t in 1..radix {
                    dst.store(lanes * (k + r * t), u[t]);
                }
            } else {
                for t in 1..radix {
                    dst.store(lanes * (k + r * (j * radix + t)), u[t].mul(ops, w[t]));
                }
            }
        }
    }
}

/// Apply a stage to a single signal, vectorizing across `k`.
///
/// The `k` loop is contiguous in both the input and output and the twiddle
/// factors do not depend on it, so once `r` reaches the vector width each
/// vector holds consecutive `k` values.
#[inline(always)]
fn stage_wide<O: FloatOps<f32>, const P: usize>(
    ops: O,
    radix: usize,
    src: ComplexSimdSlice<O>,
    mut dst: ComplexSimdSliceMut<O>,
    l: usize,
    r: usize,
    tw: ComplexSlice,
) {
    let lanes = ops.len();

    let mut u = [Complex::zero(ops); P];
    let mut v = [Complex::zero(ops); P];
    let mut w = [Complex::zero(ops); P];

    for j in 0..l {
        for (t, w) in w.iter_mut().enumerate().take(radix).skip(1) {
            let idx = t * j * r;
            *w = Complex::splat(ops, tw.re[idx], tw.im[idx]);
        }

        let mut k = 0;
        while k < r {
            // When `r` is not a multiple of the vector width, the final
            // vector overlaps the previous one. This recomputes a few
            // outputs but avoids masked loads and stores.
            let k0 = k.min(r - lanes);

            for (t, u) in u.iter_mut().enumerate().take(radix) {
                *u = src.load(k0 + r * (j + t * l));
            }

            butterfly::<_, P>(ops, &mut u, &mut v, radix, tw);

            dst.store(k0 + r * j * radix, u[0]);
            if j == 0 {
                for t in 1..radix {
                    dst.store(k0 + r * t, u[t]);
                }
            } else {
                for t in 1..radix {
                    dst.store(k0 + r * (j * radix + t), u[t].mul(ops, w[t]));
                }
            }

            k += lanes;
        }
    }
}

/// Apply a stage to a single signal when `r` is smaller than the vector
/// width.
///
/// This vectorizes across the flattened `(j, k)` index, which is contiguous
/// in the input. The twiddle factors then vary per lane and come from the
/// precomputed `narrow_tw` table, and the outputs are scattered into the
/// output one lane at a time.
#[inline(always)]
fn stage_narrow<O: FloatOps<f32>, const P: usize>(
    ops: O,
    radix: usize,
    src: ComplexSlice,
    dst: ComplexSliceMut,
    l: usize,
    r: usize,
    tw: ComplexSlice,
    narrow_tw: ComplexSlice,
) {
    let lanes = ops.len();
    let lr = l * r;

    let mut u = [Complex::zero(ops); P];
    let mut v = [Complex::zero(ops); P];

    let mut m = 0;
    while m < lr {
        let n_lanes = (lr - m).min(lanes);

        for (t, u) in u.iter_mut().enumerate().take(radix) {
            *u = Complex::load_pad(ops, src, t * lr + m);
        }

        butterfly::<_, P>(ops, &mut u, &mut v, radix, tw);

        for (t, u) in u.iter_mut().enumerate().take(radix).skip(1) {
            *u = u.mul(ops, Complex::load_pad(ops, narrow_tw, t * lr + m));
        }

        for (t, u) in u.iter().enumerate().take(radix) {
            let re = u.re.to_array();
            let im = u.im.to_array();
            let (mut j, mut k) = (m / r, m % r);
            for lane in 0..n_lanes {
                let idx = k + r * (j * radix + t);
                dst.re[idx] = re[lane];
                dst.im[idx] = im[lane];
                k += 1;
                if k == r {
                    k = 0;
                    j += 1;
                }
            }
        }

        m += lanes;
    }
}

/// Return the sequence of radices to use for a transform of length `n`, or
/// `None` if `n` has a prime factor larger than [`MAX_GENERIC_RADIX`].
///
/// Factors of 4 are preferred over pairs of 2s because the size-4 butterfly
/// needs no multiplications.
fn plan_radices(n: usize) -> Option<Vec<usize>> {
    let mut radices = Vec::new();
    let mut n = n;

    while n.is_multiple_of(4) {
        radices.push(4);
        n /= 4;
    }
    if n.is_multiple_of(2) {
        radices.push(2);
        n /= 2;
    }
    for factor in [3, 5] {
        while n.is_multiple_of(factor) {
            radices.push(factor);
            n /= factor;
        }
    }

    let mut factor = 7;
    while factor * factor <= n {
        // Every remaining factor is >= `factor`, so if that is already too
        // large there is no point continuing the search.
        if factor > MAX_GENERIC_RADIX {
            return None;
        }
        while n.is_multiple_of(factor) {
            radices.push(factor);
            n /= factor;
        }
        factor += 2;
    }
    if n > 1 {
        if n > MAX_GENERIC_RADIX {
            return None;
        }
        radices.push(n);
    }

    Some(radices)
}

/// Return a table of `exp(-2 * pi * i * k / n)` for `k` in `0..n`.
fn twiddle_table(n: usize) -> ComplexVec {
    let direct = |k: usize| unit(-2. * PI * k as f64 / n as f64);
    if !n.is_multiple_of(4) {
        return ComplexVec::from_fn(n, direct);
    }

    // Evaluating `sin_cos` dominates the cost of creating a plan, so fill the
    // first quadrant directly and derive the rest by exact rotations. Within
    // the first quadrant, `exp(-i * (pi/2 - theta))` is `(sin, -cos)` when
    // `exp(-i * theta)` is `(cos, -sin)`, which mirrors the first octant.
    let mut table = ComplexVec::zeros(n);
    let quarter = n / 4;
    let direct_end = if n.is_multiple_of(8) {
        n / 8 + 1
    } else {
        quarter
    };
    for k in 0..direct_end {
        (table.re[k], table.im[k]) = direct(k);
    }
    for k in direct_end..quarter {
        table.re[k] = -table.im[quarter - k];
        table.im[k] = -table.re[quarter - k];
    }
    for k in 0..quarter {
        let (re, im) = (table.re[k], table.im[k]);
        // Multiply by -i, -1 and i respectively.
        (table.re[k + quarter], table.im[k + quarter]) = (im, -re);
        (table.re[k + 2 * quarter], table.im[k + 2 * quarter]) = (-re, -im);
        (table.re[k + 3 * quarter], table.im[k + 3 * quarter]) = (-im, re);
    }
    table
}

/// Precomputed data for a mixed-radix Stockham transform.
pub struct Stockham {
    len: usize,
    radices: Vec<usize>,

    /// Twiddle table for the whole transform.
    twiddles: ComplexVec,

    /// Per-lane twiddle tables for the initial stages where `r` is smaller
    /// than the SIMD vector width, in stage order.
    ///
    /// Entry `t * l * r + m` of the table for a stage holds the twiddle that
    /// stage applies to output `t` of butterfly `m`.
    narrow_twiddles: Vec<ComplexVec>,
}

impl Stockham {
    /// Create a plan for `len`-element transforms.
    ///
    /// Returns `None` if `len` has a prime factor larger than
    /// [`MAX_GENERIC_RADIX`].
    pub fn new(len: usize) -> Option<Stockham> {
        let radices = plan_radices(len)?;
        let lanes = batch_size();
        let twiddles = twiddle_table(len);

        let mut narrow_twiddles = Vec::new();
        let mut l = len;
        let mut r = 1;
        for &radix in &radices {
            l /= radix;
            if r < lanes {
                let mut table = ComplexVec::zeros(len);
                for t in 0..radix {
                    for j in 0..l {
                        let (re, im) = (twiddles.re[t * j * r], twiddles.im[t * j * r]);
                        let start = t * l * r + j * r;
                        table.re[start..start + r].fill(re);
                        table.im[start..start + r].fill(im);
                    }
                }
                narrow_twiddles.push(table);
            }
            r *= radix;
        }

        Some(Stockham {
            len,
            radices,
            twiddles,
            narrow_twiddles,
        })
    }

    /// Return the length of the transform.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Apply the unnormalized forward transform to `buf`, using `scratch` as
    /// a second buffer of the same length.
    pub fn forward(
        &self,
        layout: Layout,
        buf: &mut ComplexSliceMut,
        scratch: &mut ComplexSliceMut,
    ) {
        let tw = self.twiddles.as_slice();

        // Each stage reads from one buffer and writes to the other.
        let mut src_is_buf = true;
        let mut l = self.len;
        let mut r = 1;

        for (i, &radix) in self.radices.iter().enumerate() {
            l /= radix;

            let (src, dst) = if src_is_buf {
                (buf.as_slice(), scratch.reborrow())
            } else {
                (scratch.as_slice(), buf.reborrow())
            };
            let narrow_tw = self.narrow_twiddles.get(i).map(|tw| tw.as_slice());

            macro_rules! run_stage {
                ($p:expr) => {
                    Stage::<$p> {
                        layout,
                        radix,
                        src,
                        dst,
                        l,
                        r,
                        tw,
                        narrow_tw,
                    }
                    .dispatch()
                };
            }
            match radix {
                2 => run_stage!(2),
                3 => run_stage!(3),
                4 => run_stage!(4),
                5 => run_stage!(5),
                _ => run_stage!(MAX_GENERIC_RADIX),
            }

            src_is_buf = !src_is_buf;
            r *= radix;
        }

        if !src_is_buf {
            buf.re.copy_from_slice(scratch.re);
            buf.im.copy_from_slice(scratch.im);
        }
    }
}
