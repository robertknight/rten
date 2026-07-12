use std::mem::MaybeUninit;

use rten_simd::ops::{BitOps, Extend, NarrowSaturate};
use rten_simd::{Isa, SimdOp, SliceWriter, f16};

/// Convert a slice of `f16` values to `f32`.
pub struct F16ToF32<'s, 'd> {
    src: &'s [f16],
    dest: &'d mut [MaybeUninit<f32>],
}

impl<'s, 'd> F16ToF32<'s, 'd> {
    /// Create a conversion operation which reads from `src` and writes the
    /// converted values to `dest`.
    ///
    /// Panics if `src` and `dest` have different lengths.
    pub fn new(src: &'s [f16], dest: &'d mut [MaybeUninit<f32>]) -> Self {
        assert_eq!(src.len(), dest.len());
        F16ToF32 { src, dest }
    }
}

impl<'d> SimdOp for F16ToF32<'_, 'd> {
    type Output = &'d mut [f32];

    #[inline(always)]
    fn eval<I: Isa>(self, isa: I) -> Self::Output {
        let f16_ops = isa.f16();
        let f32_ops = isa.f32();
        let f16_v_len = f16_ops.len();

        let mut dest_writer = SliceWriter::new(self.dest);

        // Main loop, unrolled by two.
        let mut chunks = self.src.chunks_exact(f16_v_len * 2);
        for chunk in chunks.by_ref() {
            let xs = f16_ops.load_many::<2>(chunk);
            let (lo0, hi0) = f16_ops.extend(xs[0]);
            let (lo1, hi1) = f16_ops.extend(xs[1]);
            // Store all four f32 vectors with a single bounds check.
            dest_writer.write_vecs(f32_ops, [lo0, hi0, lo1, hi1]);
        }

        // Convert a remaining whole `f16` vector, if any.
        let mut chunks = chunks.remainder().chunks_exact(f16_v_len);
        for chunk in chunks.by_ref() {
            let x = f16_ops.load(chunk);
            let (low, high) = f16_ops.extend(x);
            dest_writer.write_vec(f32_ops, low);
            dest_writer.write_vec(f32_ops, high);
        }

        // Convert tail elements which don't fill a whole vector.
        for &x in chunks.remainder() {
            dest_writer.write_scalar(x.to_f32());
        }

        dest_writer.into_mut_slice()
    }
}

/// Convert a slice of `f32` values to `f16`.
///
/// Values are rounded to the nearest `f16`, with ties to even. Values whose
/// magnitude exceeds the `f16` range are rounded to infinity.
pub struct F32ToF16<'s, 'd> {
    src: &'s [f32],
    dest: &'d mut [MaybeUninit<f16>],
}

impl<'s, 'd> F32ToF16<'s, 'd> {
    /// Create a conversion operation which reads from `src` and writes the
    /// converted values to `dest`.
    ///
    /// Panics if `src` and `dest` have different lengths.
    pub fn new(src: &'s [f32], dest: &'d mut [MaybeUninit<f16>]) -> Self {
        assert_eq!(src.len(), dest.len());
        F32ToF16 { src, dest }
    }
}

impl<'d> SimdOp for F32ToF16<'_, 'd> {
    type Output = &'d mut [f16];

    #[inline(always)]
    fn eval<I: Isa>(self, isa: I) -> Self::Output {
        let f32_ops = isa.f32();
        let f16_ops = isa.f16();
        let f32_v_len = f32_ops.len();

        let mut src_chunks = self.src.chunks_exact(f32_v_len * 2);
        let mut dest_writer = SliceWriter::new(self.dest);

        for src_chunk in src_chunks.by_ref() {
            let xs = f32_ops.load_many::<2>(src_chunk);
            let half = f32_ops.narrow_saturate(xs[0], xs[1]);
            dest_writer.write_vec(f16_ops, half);
        }

        for &x in src_chunks.remainder() {
            dest_writer.write_scalar(f16::from_f32(x));
        }

        dest_writer.into_mut_slice()
    }
}

#[cfg(test)]
mod tests {
    use rten_simd::ops::BitOps;
    use rten_simd::{Isa, SimdOp, f16};

    use super::{F16ToF32, F32ToF16};

    /// Return the number of `f16` lanes in a SIMD vector.
    fn f16_vec_len() -> usize {
        struct F16VecLen {}
        impl SimdOp for F16VecLen {
            type Output = usize;
            fn eval<I: Isa>(self, isa: I) -> usize {
                isa.f16().len()
            }
        }
        F16VecLen {}.dispatch()
    }

    #[test]
    fn test_f16_to_f32() {
        // Length chosen to exercise all three code paths: the main loop
        // (unrolled by two, so it needs at least two whole vectors), the
        // single-vector cleanup loop, and the scalar tail.
        let len = f16_vec_len() * 3 + 1;
        let src: Vec<f16> = (0..len)
            .map(|i| f16::from_f32(i as f32 * 0.5 - 3.0))
            .collect();
        let expected: Vec<f32> = src.iter().map(|x| x.to_f32()).collect();

        let mut buf = Vec::with_capacity(src.len());
        let actual = F16ToF32::new(&src, buf.spare_capacity_mut()).dispatch();

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_f16_to_f32_empty() {
        let src: Vec<f16> = Vec::new();
        let mut buf: Vec<f32> = Vec::new();
        let actual = F16ToF32::new(&src, buf.spare_capacity_mut()).dispatch();
        assert!(actual.is_empty());
    }

    #[test]
    fn test_f32_to_f16() {
        // Length larger than the max `f16` vector width, and not an exact
        // multiple, so we exercise both the vectorized body and the scalar
        // tail.
        let len = f16_vec_len() + 1;
        let src: Vec<f32> = (0..len).map(|i| i as f32 * 0.5 - 3.0).collect();
        let expected: Vec<f16> = src.iter().map(|&x| f16::from_f32(x)).collect();

        let mut buf = Vec::with_capacity(src.len());
        let actual = F32ToF16::new(&src, buf.spare_capacity_mut()).dispatch();

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_f32_to_f16_empty() {
        let src: Vec<f32> = Vec::new();
        let mut buf: Vec<f16> = Vec::new();
        let actual = F32ToF16::new(&src, buf.spare_capacity_mut()).dispatch();
        assert!(actual.is_empty());
    }

    // Round-trip f32 -> f16 -> f32 for values exactly representable in f16.
    #[test]
    fn test_roundtrip() {
        let len = f16_vec_len() * 2 + 3;
        let src: Vec<f32> = (0..len).map(|i| i as f32 * 0.25 - 5.0).collect();

        let mut half_buf = Vec::with_capacity(len);
        let half = F32ToF16::new(&src, half_buf.spare_capacity_mut()).dispatch();
        let half: Vec<f16> = half.to_vec();

        let mut back_buf = Vec::with_capacity(len);
        let back = F16ToF32::new(&half, back_buf.spare_capacity_mut()).dispatch();

        let expected: Vec<f32> = src.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
        assert_eq!(back, expected);
    }
}
