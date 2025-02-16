use std::mem::MaybeUninit;

use rten_simd::dispatch::SimdOp;
use rten_simd::span::SrcDest;
use rten_simd::SimdFloat;

/// Normalize the mean and variance of elements in a slice.
///
/// This normalizes elements according to the formula:
///
/// ```text
/// output[i] = (input[i] - pre_scale_bias) * scale * element_scale[i] + bias + element_bias[i]
/// ```
///
/// # Panics
///
/// Dispatching the operation panics if any of the slices have different lengths.
pub struct Normalize<'a> {
    src_dest: SrcDest<'a, f32>,
    opts: NormalizeOptions<'a>,
}

impl<'a> Normalize<'a> {
    /// Create a normalize operation which reads `input` and writes the normalized
    /// output to `output`.
    pub fn new(
        input: &'a [f32],
        output: &'a mut [MaybeUninit<f32>],
        opts: NormalizeOptions<'a>,
    ) -> Self {
        Normalize {
            src_dest: (input, output).into(),
            opts,
        }
    }

    /// Create a normalize operation which normalizes `input` in-place.
    pub fn new_mut(input: &'a mut [f32], opts: NormalizeOptions<'a>) -> Self {
        Normalize {
            src_dest: input.into(),
            opts,
        }
    }
}

/// Configuration for the [`Normalize`] operation.
pub struct NormalizeOptions<'a> {
    /// Bias to subtract before scaling.
    pub pre_scale_bias: f32,

    pub scale: f32,
    pub element_scale: Option<&'a [f32]>,

    /// Constant bias to add after scaling
    pub bias: f32,

    /// Per-element bias to add after scaling
    pub element_bias: Option<&'a [f32]>,
}

impl Default for NormalizeOptions<'_> {
    fn default() -> Self {
        NormalizeOptions {
            pre_scale_bias: 0.,
            scale: 1.,
            element_scale: None,
            bias: 0.,
            element_bias: None,
        }
    }
}

impl<'a> SimdOp for Normalize<'a> {
    /// The normalized elements.
    type Output = &'a mut [f32];

    #[inline(always)]
    unsafe fn eval<S: SimdFloat>(self) -> Self::Output {
        let Self {
            mut src_dest,
            opts:
                NormalizeOptions {
                    pre_scale_bias,
                    scale,
                    element_scale,
                    bias,
                    element_bias,
                },
        } = self;

        if let Some(scale) = element_scale {
            assert_eq!(scale.len(), src_dest.len());
        }
        if let Some(bias) = element_bias {
            assert_eq!(bias.len(), src_dest.len());
        }

        let (mut in_ptr, mut out_ptr, mut n) = src_dest.src_dest_ptr();

        let mut scale_ptr = element_scale.map(|s| s.as_ptr());
        let mut bias_ptr = element_bias.map(|b| b.as_ptr());

        let one = S::one();
        let zero = S::zero();
        let const_scale_vec = S::splat(scale);
        let const_bias_vec = S::splat(bias);
        let pre_scale_bias_vec = S::splat(pre_scale_bias);
        let v_len = S::len();

        while n >= v_len {
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

            in_ptr = in_ptr.add(v_len);
            out_ptr = out_ptr.add(v_len);
            scale_ptr = scale_ptr.map(|s| s.add(v_len));
            bias_ptr = bias_ptr.map(|b| b.add(v_len));

            n -= v_len;
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

        // Safety: All elements of `output` were initialized above.
        src_dest.dest_assume_init()
    }
}

#[cfg(test)]
mod tests {
    use super::{Normalize, NormalizeOptions};
    use rten_simd::dispatch::SimdOp;

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
        Normalize::new_mut(
            &mut actual[..],
            NormalizeOptions {
                pre_scale_bias,
                scale,
                element_scale: Some(&element_scale),
                bias,
                element_bias: Some(&element_bias),
            },
        )
        .dispatch();
        assert_eq!(actual, expected);

        // Without per-element scale and bias
        let mut expected = data.clone();
        reference_normalize_mut(&mut expected[..], pre_scale_bias, scale, None, bias, None);

        let mut actual = data.clone();
        Normalize::new_mut(
            &mut actual[..],
            NormalizeOptions {
                pre_scale_bias,
                scale,
                element_scale: None,
                bias,
                element_bias: None,
            },
        )
        .dispatch();

        assert_eq!(actual, expected);
    }
}
