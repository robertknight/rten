use std::f64::consts::PI;

use rten_simd::ops::BitOps;
use rten_simd::{Isa, Simd, SimdOp};

use crate::complex::{
    Complex, ComplexSimdSlice, ComplexSimdSliceMut, ComplexSlice, ComplexSliceMut, ComplexVec,
};
use crate::stockham::Stockham;
use crate::{Layout, unit};

/// Bluestein's algorithm, which re-expresses the transform as a convolution
/// evaluated with a longer power-of-two transform.
pub struct Bluestein {
    /// `exp(i * pi * k^2 / len)` for `k` in `0..len`.
    chirp: ComplexVec,
    /// Transform of the cyclic filter, scaled by `1 / inner.len`.
    filter: ComplexVec,
    /// Power-of-two transform used to evaluate the convolution.
    inner: Stockham,
}

impl Bluestein {
    /// Create a plan for `len`-element transforms.
    pub fn new(len: usize) -> Bluestein {
        // The convolution is linear rather than cyclic, so it can be evaluated
        // with any transform length >= `2 * len - 1`. The next power of two is
        // the fastest length the mixed-radix path offers.
        let inner_len = (2 * len - 1).next_power_of_two();

        // `k^2` is reduced modulo `2 * len` before being converted to an angle,
        // since it otherwise loses precision rapidly as `k` grows.
        let chirp = ComplexVec::from_fn(len, |k| {
            let k = k as u64;
            let sq = (k * k % (2 * len as u64)) as f64;
            unit(PI * sq / len as f64)
        });

        let mut filter = ComplexVec::zeros(inner_len);
        filter.re[..len].copy_from_slice(&chirp.re);
        filter.im[..len].copy_from_slice(&chirp.im);
        for k in 1..len {
            filter.re[inner_len - k] = chirp.re[k];
            filter.im[inner_len - k] = chirp.im[k];
        }

        // A power of two factors into radices 4 and 2, so it always has a
        // mixed-radix plan.
        let inner = Stockham::new(inner_len).expect("power of two should have a mixed-radix plan");
        let mut scratch = ComplexVec::zeros(inner_len);
        inner.forward(
            Layout::Single,
            &mut filter.as_mut_slice(),
            &mut scratch.as_mut_slice(),
        );

        // Fold the inverse transform's normalization into the filter.
        let scale = 1. / inner_len as f32;
        for f in filter.re.iter_mut().chain(filter.im.iter_mut()) {
            *f *= scale;
        }

        Bluestein {
            chirp,
            filter,
            inner,
        }
    }

    /// Return the number of `f32` elements in the scratch buffer that
    /// [`forward`](Bluestein::forward) requires, per signal.
    pub fn scratch_len(&self) -> usize {
        // The zero-padded signal and the inner transform's scratch buffer,
        // each with real and imaginary parts.
        4 * self.inner.len()
    }

    /// Apply the unnormalized forward transform to `buf`, which holds one or
    /// more signals in the given layout.
    ///
    /// `scratch` must have [`scratch_len`](Bluestein::scratch_len) elements per
    /// signal.
    pub fn forward(&self, layout: Layout, buf: &mut ComplexSliceMut, scratch: &mut [f32]) {
        let (signal, inner_scratch) = scratch.split_at_mut(scratch.len() / 2);
        let mut signal = ComplexSliceMut::from_halves(signal);
        let mut inner_scratch = ComplexSliceMut::from_halves(inner_scratch);

        // Multiply by the conjugate chirp and zero-pad to the inner length.
        let head = buf.re.len();
        signal.re[..head].copy_from_slice(buf.re);
        signal.im[..head].copy_from_slice(buf.im);
        signal.re[head..].fill(0.);
        signal.im[head..].fill(0.);
        PointwiseMul {
            layout,
            buf: ComplexSliceMut {
                re: &mut signal.re[..head],
                im: &mut signal.im[..head],
            },
            table: self.chirp.as_slice(),
            op: Pointwise::MulConj,
        }
        .dispatch();

        self.inner.forward(layout, &mut signal, &mut inner_scratch);

        // Multiply by the filter and take the inverse transform of the result,
        // as the forward transform of its conjugate. The `1 / inner_len`
        // normalization is already folded into `filter`.
        PointwiseMul {
            layout,
            buf: signal.reborrow(),
            table: self.filter.as_slice(),
            op: Pointwise::ConjMul,
        }
        .dispatch();
        self.inner.forward(layout, &mut signal, &mut inner_scratch);

        // Conjugate the result and multiply by the conjugate chirp,
        // which is the conjugate of the product with the chirp.
        buf.re.copy_from_slice(&signal.re[..head]);
        buf.im.copy_from_slice(&signal.im[..head]);
        PointwiseMul {
            layout,
            buf: buf.reborrow(),
            table: self.chirp.as_slice(),
            op: Pointwise::ConjMul,
        }
        .dispatch();
    }
}

/// Elementwise product used by the steps of Bluestein's algorithm.
#[derive(Copy, Clone)]
enum Pointwise {
    /// `x * conj(c)`
    MulConj,
    /// `conj(x * c)`
    ConjMul,
}

/// Multiply each element of `buf` by the corresponding entry of a table.
///
/// For the `Batch` layout the table entry is broadcast across the lanes.
struct PointwiseMul<'a> {
    layout: Layout,
    buf: ComplexSliceMut<'a>,
    table: ComplexSlice<'a>,
    op: Pointwise,
}

impl SimdOp for PointwiseMul<'_> {
    type Output = ();

    #[inline(always)]
    fn eval<I: Isa>(self, isa: I) {
        let ops = isa.f32();
        let lanes = ops.len();
        let n = self.table.re.len();
        assert_eq!(self.table.im.len(), n);

        let apply = |x: Complex<I::F32>, c: Complex<I::F32>| match self.op {
            Pointwise::MulConj => x.mul(ops, c.conj(ops)),
            Pointwise::ConjMul => x.mul(ops, c).conj(ops),
        };

        match self.layout {
            Layout::Batch => {
                assert_eq!(self.buf.re.len(), n * lanes);
                assert_eq!(self.buf.im.len(), n * lanes);
                let mut buf = ComplexSimdSliceMut::new(ops, self.buf);
                for i in 0..n {
                    let c = Complex::splat(ops, self.table.re[i], self.table.im[i]);
                    buf.store(i * lanes, apply(buf.load(i * lanes), c));
                }
            }
            Layout::Single => {
                assert_eq!(self.buf.re.len(), n);
                assert_eq!(self.buf.im.len(), n);

                // Whole vectors, then a padded tail.
                let full = n - n % lanes;
                let (head_re, tail_re) = self.buf.re.split_at_mut(full);
                let (head_im, tail_im) = self.buf.im.split_at_mut(full);
                if full > 0 {
                    let table = ComplexSimdSlice::new(
                        ops,
                        ComplexSlice {
                            re: &self.table.re[..full],
                            im: &self.table.im[..full],
                        },
                    );
                    let mut buf = ComplexSimdSliceMut::new(
                        ops,
                        ComplexSliceMut {
                            re: head_re,
                            im: head_im,
                        },
                    );
                    for i in (0..full).step_by(lanes) {
                        buf.store(i, apply(buf.load(i), table.load(i)));
                    }
                }
                if full < n {
                    let tail = ComplexSlice {
                        re: tail_re,
                        im: tail_im,
                    };
                    let table = ComplexSlice {
                        re: &self.table.re[full..],
                        im: &self.table.im[full..],
                    };
                    let y = apply(
                        Complex::load_pad(ops, tail, 0),
                        Complex::load_pad(ops, table, 0),
                    );
                    let (re, im) = (y.re.to_array(), y.im.to_array());
                    for i in 0..n - full {
                        tail_re[i] = re[i];
                        tail_im[i] = im[i];
                    }
                }
            }
        }
    }
}
