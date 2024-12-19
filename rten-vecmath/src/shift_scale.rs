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

        assert_eq!(scale.len(), data.len());
        if let Some(bias) = bias {
            assert_eq!(bias.len(), data.len());
        }

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
/// This updates each element in `xs` according to the formula
/// `xs[i] = xs[i] * const_scale * scale[i] + bias[i]`.
///
/// # Panics
///
/// Panics if the length of `scale` or `bias` does not match `xs`.
pub fn vec_shift_scale_in_place(
    xs: &mut [f32],
    const_scale: f32,
    scale: &[f32],
    bias: Option<&[f32]>,
) {
    let simd_op = SimdShiftScale {
        data: xs,
        bias,
        scale,
        const_scale,
    };
    dispatch(simd_op);
}

struct SimdShiftScaleBias<'a> {
    data: &'a mut [f32],
    x_bias: f32,
    scale: f32,
    bias: f32,
}

impl<'a> SimdOp for SimdShiftScaleBias<'a> {
    type Output = &'a mut [f32];

    #[inline(always)]
    unsafe fn eval<S: SimdFloat>(self) -> Self::Output {
        let Self {
            data,
            x_bias,
            scale,
            bias,
        } = self;

        let mut out_ptr = data.as_mut_ptr();
        let mut n = data.len();

        let x_bias_vec = S::splat(x_bias);
        let scale_vec = S::splat(scale);
        let bias_vec = S::splat(bias);

        while n >= S::LEN {
            let y = S::load(out_ptr)
                .sub(x_bias_vec)
                .mul_add(scale_vec, bias_vec);
            y.store(out_ptr);

            out_ptr = out_ptr.add(S::LEN);
            n -= S::LEN;
        }

        if n > 0 {
            let y = S::load_partial(out_ptr, n, 0.)
                .sub(x_bias_vec)
                .mul_add(scale_vec, bias_vec);
            y.store_partial(out_ptr, n);
        }

        data
    }
}

/// Shift and scale each element in the input.
///
/// This updates `xs` as `xs[i] = (xs[i] - x_bias) * scale + bias`.
pub fn vec_shift_scale_bias(xs: &mut [f32], x_bias: f32, scale: f32, bias: f32) {
    let op = SimdShiftScaleBias {
        data: xs,
        x_bias,
        scale,
        bias,
    };
    dispatch(op);
}

#[cfg(test)]
mod tests {
    use super::{vec_shift_scale_bias, vec_shift_scale_in_place};

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

    fn reference_shift_scale_bias(data: &mut [f32], x_bias: f32, scale: f32, bias: f32) {
        for i in 0..data.len() {
            data[i] = (data[i] - x_bias).mul_add(scale, bias);
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

    #[test]
    fn test_vec_shift_scale_bias() {
        let data: Vec<_> = (0..10).map(|i| i as f32 * 0.1).collect();
        let x_bias = 0.123;
        let scale = 0.456;
        let bias = 0.89;

        let mut expected = data.clone();
        reference_shift_scale_bias(&mut expected, x_bias, scale, bias);

        let mut actual = data.clone();
        vec_shift_scale_bias(&mut actual, x_bias, scale, bias);

        assert_eq!(actual, expected);
    }
}
