use std::mem::MaybeUninit;

use rten_simd::dispatch::{dispatch, SimdOp};
use rten_simd::span::{MutPtrLen, PtrLen};
use rten_simd::SimdFloat;

struct Normalize<'a> {
    input: PtrLen<f32>,
    output: MutPtrLen<MaybeUninit<f32>>,

    /// Bias to subtract before scaling.
    pre_scale_bias: f32,

    scale: f32,
    element_scale: Option<&'a [f32]>,

    /// Constant bias to add after scaling
    bias: f32,

    /// Per-element bias to add after scaling
    element_bias: Option<&'a [f32]>,
}

impl SimdOp for Normalize<'_> {
    type Output = ();

    #[inline(always)]
    unsafe fn eval<S: SimdFloat>(self) -> Self::Output {
        let Self {
            input,
            output,
            pre_scale_bias,
            scale,
            element_scale,
            bias,
            element_bias,
        } = self;

        assert_eq!(input.len(), output.len());
        if let Some(scale) = element_scale {
            assert_eq!(scale.len(), input.len());
        }
        if let Some(bias) = element_bias {
            assert_eq!(bias.len(), input.len());
        }

        let mut in_ptr = input.ptr();
        let mut out_ptr = output.ptr();

        let mut scale_ptr = element_scale.map(|s| s.as_ptr());
        let mut bias_ptr = element_bias.map(|b| b.as_ptr());
        let mut n = input.len();

        let one = S::one();
        let zero = S::zero();
        let const_scale_vec = S::splat(scale);
        let const_bias_vec = S::splat(bias);
        let pre_scale_bias_vec = S::splat(pre_scale_bias);

        while n >= S::LEN {
            let scale_vec = scale_ptr
                .map(|s| S::load(s))
                .unwrap_or(one)
                .mul(const_scale_vec);
            let bias_vec = bias_ptr
                .map(|b| S::load(b))
                .unwrap_or(zero)
                .add(const_bias_vec);
            let y = S::load(in_ptr)
                .sub(pre_scale_bias_vec)
                .mul_add(scale_vec, bias_vec);
            y.store(out_ptr as *mut f32);

            in_ptr = in_ptr.add(S::LEN);
            out_ptr = out_ptr.add(S::LEN);
            scale_ptr = scale_ptr.map(|s| s.add(S::LEN));
            bias_ptr = bias_ptr.map(|b| b.add(S::LEN));

            n -= S::LEN;
        }

        if n > 0 {
            let scale_vec = scale_ptr
                .map(|s| S::load_partial(s, n))
                .unwrap_or(one)
                .mul(const_scale_vec);
            let bias_vec = bias_ptr
                .map(|b| S::load_partial(b, n))
                .unwrap_or(zero)
                .add(const_bias_vec);
            let y = S::load_partial(in_ptr, n)
                .sub(pre_scale_bias_vec)
                .mul_add(scale_vec, bias_vec);
            y.store_partial(out_ptr as *mut f32, n);
        }
    }
}

/// Normalize elements in a slice.
///
/// This normalizes elements according to the formula:
///
/// ```text
/// output[i] = (input[i] - pre_scale_bias) * scale * element_scale[i] + bias + element_bias[i]
/// ```
///
/// # Panics
///
/// Panics if any of the slices have different lengths.
pub fn normalize(
    input: &[f32],
    output: &mut [MaybeUninit<f32>],
    pre_scale_bias: f32,
    scale: f32,
    element_scale: Option<&[f32]>,
    bias: f32,
    element_bias: Option<&[f32]>,
) {
    let simd_op = Normalize {
        input: input.into(),
        output: output.into(),
        pre_scale_bias,
        scale,
        element_scale,
        bias,
        element_bias,
    };
    dispatch(simd_op)
}

/// Variant of [`normalize`] which updates elements in-place.
pub fn normalize_mut(
    input: &mut [f32],
    pre_scale_bias: f32,
    scale: f32,
    element_scale: Option<&[f32]>,
    bias: f32,
    element_bias: Option<&[f32]>,
) {
    let output: MutPtrLen<f32> = input.into();
    let simd_op = Normalize {
        input: input.into(),
        output: output.as_uninit(),
        pre_scale_bias,
        scale,
        element_scale,
        bias,
        element_bias,
    };
    dispatch(simd_op)
}

#[cfg(test)]
mod tests {
    use super::normalize_mut;

    fn reference_normalize_mut(
        data: &mut [f32],
        pre_scale_bias: f32,
        scale: f32,
        element_scale: Option<&[f32]>,
        bias: f32,
        element_bias: Option<&[f32]>,
    ) {
        for i in 0..data.len() {
            let x_scale = scale * element_scale.map(|es| es[i]).unwrap_or(1.);
            let x_bias = bias + element_bias.map(|eb| eb[i]).unwrap_or(0.);
            data[i] = (data[i] - pre_scale_bias).mul_add(x_scale, x_bias)
        }
    }

    #[test]
    fn test_normalize_mut() {
        let data: Vec<_> = (0..10).map(|i| i as f32 * 0.1).collect();
        let pre_scale_bias = 0.5;
        let scale = 0.123;
        let element_scale: Vec<_> = (0..data.len()).map(|i| 1.0 + i as f32 * 0.1).collect();
        let bias = 0.3;
        let element_bias: Vec<_> = (0..data.len()).map(|i| -0.5 + i as f32 * 0.2).collect();

        // With per-element scale and bias
        let mut expected = data.clone();
        reference_normalize_mut(
            &mut expected[..],
            pre_scale_bias,
            scale,
            Some(&element_scale),
            bias,
            Some(&element_bias),
        );

        let mut actual = data.clone();
        normalize_mut(
            &mut actual[..],
            pre_scale_bias,
            scale,
            Some(&element_scale),
            bias,
            Some(&element_bias),
        );

        assert_eq!(actual, expected);

        // Without per-element scale and bias
        let mut expected = data.clone();
        reference_normalize_mut(&mut expected[..], pre_scale_bias, scale, None, bias, None);

        let mut actual = data.clone();
        normalize_mut(&mut actual[..], pre_scale_bias, scale, None, bias, None);

        assert_eq!(actual, expected);
    }
}
