use std::mem::{transmute, MaybeUninit};

use rten_simd::dispatch::SimdOp;
use rten_simd::{Simd, SimdFloat, SimdInt};

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
    unsafe fn eval<S: SimdFloat>(self) -> Self::Output {
        let mut n = self.src.len();
        let mut src_ptr = self.src.as_ptr();
        let mut dest_ptr = self.dest.as_mut_ptr();

        let zp_vec = S::Int::splat(self.zero_point as i32);
        let scale_vec = S::splat(self.inv_scale);
        let v_len = S::len();

        while n >= v_len {
            let q = S::load(src_ptr)
                .mul(scale_vec)
                .to_int_round()
                .add(zp_vec)
                .saturating_cast_u8();
            q.store(dest_ptr as *mut u8);

            src_ptr = src_ptr.add(v_len);
            dest_ptr = dest_ptr.add(v_len);
            n -= v_len;
        }

        while n > 0 {
            let x = *src_ptr;
            let y = (x * self.inv_scale).round_ties_even() as i32;
            let y = (y + self.zero_point as i32).clamp(0, u8::MAX as i32);
            dest_ptr.write(MaybeUninit::new(y as u8));

            src_ptr = src_ptr.add(1);
            dest_ptr = dest_ptr.add(1);
            n -= 1;
        }

        transmute::<&mut [MaybeUninit<u8>], &mut [u8]>(self.dest)
    }
}

#[cfg(test)]
mod tests {
    use rten_simd::dispatch::SimdOp;

    use super::Quantize;

    fn reference_quantize(src: &[f32], inv_scale: f32, zero_point: u8) -> Vec<u8> {
        src.iter()
            .map(|x| {
                let tmp = (x * inv_scale).round_ties_even() + zero_point as f32;
                tmp as u8 // Saturating cast
            })
            .collect()
    }

    #[test]
    fn test_quantize() {
        let mut rng = fastrand::Rng::with_seed(1234);

        // Larger than max SIMD vector length, not a multiple of one, so we have
        // a tail.
        let len = 17;
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
