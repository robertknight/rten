use std::mem::MaybeUninit;

use rten_simd::safe::{Isa, NarrowSaturate, SimdFloatOps, SimdOp, SimdOps, SliceWriter};

/// Quantize a slice of `f32` elements to 8-bit integers using the formula:
///
/// ```text
/// y = saturate(round(x * inv_scale) + zero_point)
/// ```
///
/// Where `round` rounds to the nearest `i32` value with ties to even and
/// `saturate` converts `i32` to the small integer type `To` with saturation.
pub struct Quantize<'s, 'd, To> {
    src: &'s [f32],
    dest: &'d mut [MaybeUninit<To>],
    inv_scale: f32,
    zero_point: To,
}

impl<'s, 'd, To> Quantize<'s, 'd, To> {
    pub fn new(
        src: &'s [f32],
        dest: &'d mut [MaybeUninit<To>],
        inv_scale: f32,
        zero_point: To,
    ) -> Self {
        assert_eq!(src.len(), dest.len());
        Quantize {
            src,
            dest,
            inv_scale,
            zero_point,
        }
    }
}

impl<'d> SimdOp for Quantize<'_, 'd, u8> {
    type Output = &'d mut [u8];

    #[inline(always)]
    fn eval<I: Isa>(self, isa: I) -> Self::Output {
        let src_ops = isa.f32();
        let i32_ops = isa.i32();

        let zp_vec = i32_ops.splat(self.zero_point as i32);
        let scale_vec = src_ops.splat(self.inv_scale);
        let f32_v_len = src_ops.len();

        // Generate one vector of u8 elements in each iteration by quantizing
        // 4 vectors of f32 elements.
        let mut src_chunks = self.src.chunks_exact(f32_v_len * 4);
        let mut dest_writer = SliceWriter::new(self.dest);

        for src_chunk in src_chunks.by_ref() {
            let src = src_ops.load_many::<4>(src_chunk);
            let quant_i32 = src.map(|x| {
                let y = src_ops.mul(x, scale_vec);
                let y = src_ops.to_int_round(y);
                i32_ops.add(y, zp_vec)
            });
            let quant_i16_low = i32_ops.narrow_saturate(quant_i32[0], quant_i32[1]);
            let quant_i16_high = i32_ops.narrow_saturate(quant_i32[2], quant_i32[3]);
            let quant_u8 = isa.i16().narrow_saturate(quant_i16_low, quant_i16_high);
            dest_writer.write_vec(isa.u8(), quant_u8);
        }

        // Quantize tail elements.
        for src in src_chunks.remainder() {
            let y = (src * self.inv_scale).round_ties_even() as i32;
            let y = (y + self.zero_point as i32).clamp(0, u8::MAX as i32);
            dest_writer.write_scalar(y as u8);
        }

        dest_writer.into_mut_slice()
    }
}

#[cfg(test)]
mod tests {
    use rten_simd::safe::{Isa, SimdOp, SimdOps};

    use super::Quantize;

    fn reference_quantize(src: &[f32], inv_scale: f32, zero_point: u8) -> Vec<u8> {
        src.iter()
            .map(|x| {
                let tmp = (x * inv_scale).round_ties_even() + zero_point as f32;
                tmp as u8 // Saturating cast
            })
            .collect()
    }

    /// Return number of u8 lanes supported in a SIMD vector.
    fn u8_vec_len() -> usize {
        struct U8VecLen {}
        impl SimdOp for U8VecLen {
            type Output = usize;
            fn eval<I: Isa>(self, isa: I) -> usize {
                isa.u8().len()
            }
        }
        U8VecLen {}.dispatch()
    }

    #[test]
    fn test_quantize() {
        let mut rng = fastrand::Rng::with_seed(1234);

        // Larger than max u8 SIMD vector length, and not an exact multiple, so
        // we have a tail.
        let len = u8_vec_len() + 1;
        let src: Vec<f32> = std::iter::from_fn(|| Some(rng.f32())).take(len).collect();
        let inv_scale = 5.2;
        let zero_point = 10;
        let expected = reference_quantize(&src, inv_scale, zero_point);

        let mut buf = Vec::with_capacity(src.len());
        let actual = &mut buf.spare_capacity_mut();
        let actual = Quantize::new(&src, actual, inv_scale, zero_point).dispatch();

        assert_eq!(actual, expected);
    }
}
