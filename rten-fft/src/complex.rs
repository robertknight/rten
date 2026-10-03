use rten_simd::ops::{BitOps, FloatOps, NumOps};
use rten_simd::{Simd, SimdSlice, SimdSliceMut};

/// A vector of complex values, one per lane.
#[derive(Copy, Clone)]
pub struct Complex<S> {
    pub re: S,
    pub im: S,
}

impl<S: Simd> Complex<S> {
    #[inline(always)]
    pub fn zero<O: NumOps<f32, Simd = S>>(ops: O) -> Self {
        Complex {
            re: ops.zero(),
            im: ops.zero(),
        }
    }

    #[inline(always)]
    pub fn splat<O: NumOps<f32, Simd = S>>(ops: O, re: f32, im: f32) -> Self {
        Complex {
            re: ops.splat(re),
            im: ops.splat(im),
        }
    }

    /// Load a vector from `src`, zero-padding if the signal ends within the
    /// vector.
    #[inline(always)]
    pub fn load_pad<O: NumOps<f32, Simd = S>>(ops: O, src: ComplexSlice, offset: usize) -> Self {
        Complex {
            re: ops.load_pad(&src.re[offset..]).0,
            im: ops.load_pad(&src.im[offset..]).0,
        }
    }

    #[inline(always)]
    pub fn add<O: NumOps<f32, Simd = S>>(self, ops: O, b: Self) -> Self {
        Complex {
            re: ops.add(self.re, b.re),
            im: ops.add(self.im, b.im),
        }
    }

    #[inline(always)]
    pub fn sub<O: NumOps<f32, Simd = S>>(self, ops: O, b: Self) -> Self {
        Complex {
            re: ops.sub(self.re, b.re),
            im: ops.sub(self.im, b.im),
        }
    }

    #[inline(always)]
    pub fn mul<O: FloatOps<f32, Simd = S>>(self, ops: O, b: Self) -> Self {
        Complex {
            re: ops.mul_sub_from(self.im, b.im, ops.mul(self.re, b.re)),
            im: ops.mul_add(self.re, b.im, ops.mul(self.im, b.re)),
        }
    }

    #[inline(always)]
    pub fn conj<O: FloatOps<f32, Simd = S>>(self, ops: O) -> Self {
        Complex {
            re: self.re,
            im: ops.neg(self.im),
        }
    }

    /// Multiply each element by a real scalar.
    #[inline(always)]
    pub fn scale<O: NumOps<f32, Simd = S>>(self, ops: O, s: S) -> Self {
        Complex {
            re: ops.mul(self.re, s),
            im: ops.mul(self.im, s),
        }
    }

    /// Compute `self + b * s` for a real scalar `s`.
    #[inline(always)]
    pub fn scale_add<O: NumOps<f32, Simd = S>>(self, ops: O, b: Self, s: S) -> Self {
        Complex {
            re: ops.mul_add(b.re, s, self.re),
            im: ops.mul_add(b.im, s, self.im),
        }
    }

    /// Compute `self + b * (-i)`.
    ///
    /// Multiplying by `-i` is a rotation by a quarter turn, so this needs no
    /// multiplications.
    #[inline(always)]
    pub fn add_rot<O: NumOps<f32, Simd = S>>(self, ops: O, b: Self) -> Self {
        Complex {
            re: ops.add(self.re, b.im),
            im: ops.sub(self.im, b.re),
        }
    }

    /// Compute `self - b * (-i)`.
    #[inline(always)]
    pub fn sub_rot<O: NumOps<f32, Simd = S>>(self, ops: O, b: Self) -> Self {
        Complex {
            re: ops.sub(self.re, b.im),
            im: ops.add(self.im, b.re),
        }
    }
}

/// A complex signal stored as separate real and imaginary parts.
#[derive(Copy, Clone)]
pub struct ComplexSlice<'a> {
    pub re: &'a [f32],
    pub im: &'a [f32],
}

/// Mutable complex signal stored as separate real and imaginary slices.
pub struct ComplexSliceMut<'a> {
    pub re: &'a mut [f32],
    pub im: &'a mut [f32],
}

impl<'a> ComplexSliceMut<'a> {
    /// Use the first half of `buf` for the real parts and the second half for
    /// the imaginary parts.
    pub fn from_halves(buf: &'a mut [f32]) -> Self {
        let (re, im) = buf.split_at_mut(buf.len() / 2);
        ComplexSliceMut { re, im }
    }

    pub fn as_slice(&self) -> ComplexSlice<'_> {
        ComplexSlice {
            re: self.re,
            im: self.im,
        }
    }

    pub fn reborrow(&mut self) -> ComplexSliceMut<'_> {
        ComplexSliceMut {
            re: self.re,
            im: self.im,
        }
    }
}

/// Owned complex signal stored as separate real and imaginary vectors.
#[derive(Clone)]
pub struct ComplexVec {
    pub re: Vec<f32>,
    pub im: Vec<f32>,
}

impl ComplexVec {
    pub fn zeros(len: usize) -> ComplexVec {
        ComplexVec {
            re: vec![0.; len],
            im: vec![0.; len],
        }
    }

    /// Build a buffer from a function that returns `(re, im)` for each index.
    pub fn from_fn(len: usize, mut f: impl FnMut(usize) -> (f32, f32)) -> ComplexVec {
        let mut buf = ComplexVec::zeros(len);
        for i in 0..len {
            (buf.re[i], buf.im[i]) = f(i);
        }
        buf
    }

    pub fn as_slice(&self) -> ComplexSlice<'_> {
        ComplexSlice {
            re: &self.re,
            im: &self.im,
        }
    }

    pub fn as_mut_slice(&mut self) -> ComplexSliceMut<'_> {
        ComplexSliceMut {
            re: &mut self.re,
            im: &mut self.im,
        }
    }
}

/// Complex signal wrapped for loading SIMD vectors at arbitrary offsets.
///
/// See [`SimdSlice`] for the handling of out-of-range offsets.
#[derive(Copy, Clone)]
pub struct ComplexSimdSlice<'a, O: BitOps<f32>> {
    re: SimdSlice<'a, f32, O>,
    im: SimdSlice<'a, f32, O>,
}

impl<'a, O: BitOps<f32>> ComplexSimdSlice<'a, O> {
    /// Wrap `src`, which must hold at least one vector.
    pub fn new(ops: O, src: ComplexSlice<'a>) -> Self {
        ComplexSimdSlice {
            re: SimdSlice::new(ops, src.re).expect("signal shorter than a vector"),
            im: SimdSlice::new(ops, src.im).expect("signal shorter than a vector"),
        }
    }

    #[inline(always)]
    pub fn load(&self, offset: usize) -> Complex<O::Simd> {
        Complex {
            re: self.re.load(offset),
            im: self.im.load(offset),
        }
    }
}

/// Complex signal wrapped for loading and storing SIMD vectors at arbitrary
/// offsets.
pub struct ComplexSimdSliceMut<'a, O: BitOps<f32>> {
    re: SimdSliceMut<'a, f32, O>,
    im: SimdSliceMut<'a, f32, O>,
}

impl<'a, O: BitOps<f32>> ComplexSimdSliceMut<'a, O> {
    /// Wrap `dst`, which must hold at least one vector.
    pub fn new(ops: O, dst: ComplexSliceMut<'a>) -> Self {
        ComplexSimdSliceMut {
            re: SimdSliceMut::new(ops, dst.re).expect("signal shorter than a vector"),
            im: SimdSliceMut::new(ops, dst.im).expect("signal shorter than a vector"),
        }
    }

    #[inline(always)]
    pub fn load(&self, offset: usize) -> Complex<O::Simd> {
        Complex {
            re: self.re.load(offset),
            im: self.im.load(offset),
        }
    }

    #[inline(always)]
    pub fn store(&mut self, offset: usize, x: Complex<O::Simd>) {
        self.re.store(offset, x.re);
        self.im.store(offset, x.im);
    }
}
