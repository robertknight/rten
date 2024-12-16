use rten_simd::dispatch::{dispatch, SimdOp};
use rten_simd::SimdFloat;

struct SimdShiftScale<'a> {
    data: &'a mut [f32],
    bias: Option<&'a [f32]>,
    scale: &'a [f32],
    const_scale: f32,
}

impl<'a> SimdOp for SimdShiftScale<'a> {
    type Output = &'a mut [f32];

    #[inline(always)]
    unsafe fn eval<S: SimdFloat>(self) -> Self::Output {
        let Self {
            data,
            bias,
            scale,
            const_scale,
        } = self;

        let mut out_ptr = data.as_mut_ptr();
        let mut scale_ptr = scale.as_ptr();
        let mut bias_ptr = bias.map(|b| b.as_ptr());
        let mut n = data.len();

        let zero = S::zero();
        let const_scale_vec = S::splat(const_scale);

        while n >= S::LEN {
            let scale_vec = S::load(scale_ptr).mul(const_scale_vec);
            let bias_vec = bias_ptr.map(|b| S::load(b)).unwrap_or(zero);
            let y = S::load(out_ptr).mul_add(scale_vec, bias_vec);
            y.store(out_ptr);

            out_ptr = out_ptr.add(S::LEN);
            scale_ptr = scale_ptr.add(S::LEN);
            bias_ptr = bias_ptr.map(|b| b.add(S::LEN));

            n -= S::LEN;
        }

        if n > 0 {
            let scale_vec = S::load_partial(scale_ptr, n, 0.).mul(const_scale_vec);
            let bias_vec = bias_ptr.map(|b| S::load_partial(b, n, 0.)).unwrap_or(zero);
            let y = S::load_partial(out_ptr, n, 0.).mul_add(scale_vec, bias_vec);
            y.store_partial(out_ptr, n);
        }

        data
    }
}

/// Shift and scale each element in the input.
///
/// This scales and shifts each element using `y[i] = y[i] * const_scale *
/// scale[i] + bias[i]`.
pub fn vec_shift_scale_in_place(
    data: &mut [f32],
    const_scale: f32,
    scale: &[f32],
    bias: Option<&[f32]>,
) {
    let simd_op = SimdShiftScale {
        data,
        bias,
        scale,
        const_scale,
    };
    dispatch(simd_op);
}

#[cfg(test)]
mod tests {
    use super::vec_shift_scale_in_place;

    fn reference_shift_scale(
        data: &mut [f32],
        const_scale: f32,
        scale: &[f32],
        bias: Option<&[f32]>,
    ) {
        for i in 0..data.len() {
            data[i] = data[i].mul_add(const_scale * scale[i], bias.map(|b| b[i]).unwrap_or(0.));
        }
    }

    #[test]
    fn test_vec_shift_scale() {
        let data: Vec<_> = (0..10).map(|i| i as f32 * 0.1).collect();
        let const_scale = 0.123;
        let scale: Vec<_> = (0..data.len()).map(|i| 1.0 + i as f32 * 0.1).collect();
        let bias: Vec<_> = (0..data.len()).map(|i| -0.5 + i as f32 * 0.2).collect();

        // With bias
        let mut expected = data.clone();
        reference_shift_scale(&mut expected[..], const_scale, &scale, Some(&bias));

        let mut actual = data.clone();
        vec_shift_scale_in_place(&mut actual[..], const_scale, &scale, Some(&bias));

        assert_eq!(actual, expected);

        // Without bias
        let mut expected = data.clone();
        reference_shift_scale(&mut expected[..], const_scale, &scale, None);

        let mut actual = data.clone();
        vec_shift_scale_in_place(&mut actual[..], const_scale, &scale, None);

        assert_eq!(actual, expected);
    }
}
