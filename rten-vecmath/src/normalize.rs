use std::mem::MaybeUninit;

use rten_simd::functional::simd_map;
use rten_simd::span::SrcDest;
use rten_simd::{Isa, NumOps, SimdIterable, SimdOp};

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
pub struct Normalize<'src, 'dst> {
    src_dest: SrcDest<'src, 'dst, f32>,
    opts: NormalizeOptions<'src>,
}

impl<'src, 'dst> Normalize<'src, 'dst> {
    /// Create a normalize operation which reads `input` and writes the normalized
    /// output to `output`.
    pub fn new(
        input: &'src [f32],
        output: &'dst mut [MaybeUninit<f32>],
        opts: NormalizeOptions<'src>,
    ) -> Self {
        Normalize {
            src_dest: (input, output).into(),
            opts,
        }
    }

    /// Create a normalize operation which normalizes `input` in-place.
    pub fn new_mut(input: &'dst mut [f32], opts: NormalizeOptions<'src>) -> Self
    where
        'dst: 'src,
    {
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

impl<'dst> SimdOp for Normalize<'_, 'dst> {
    /// The normalized elements.
    type Output = &'dst mut [f32];

    #[inline(always)]
    fn eval<I: Isa>(self, isa: I) -> Self::Output {
        let ops = isa.f32();

        let Self {
            src_dest,
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

        let mut scale_iter = element_scale.map(|s| s.simd_iter_pad(ops));
        let mut bias_iter = element_bias.map(|b| b.simd_iter_pad(ops));

        let one = ops.one();
        let zero = ops.zero();
        let const_scale_vec = ops.splat(scale);
        let const_bias_vec = ops.splat(bias);
        let pre_scale_bias_vec = ops.splat(pre_scale_bias);

        simd_map(
            ops,
            src_dest,
            #[inline(always)]
            |x| {
                let scale_vec = scale_iter.as_mut().and_then(|s| s.next()).unwrap_or(one);
                let scale_vec = ops.mul(scale_vec, const_scale_vec);

                let bias_vec = bias_iter.as_mut().and_then(|b| b.next()).unwrap_or(zero);
                let bias_vec = ops.add(bias_vec, const_bias_vec);

                let y = ops.sub(x, pre_scale_bias_vec);
                ops.mul_add(y, scale_vec, bias_vec)
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{Normalize, NormalizeOptions};
    use rten_simd::SimdOp;

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
