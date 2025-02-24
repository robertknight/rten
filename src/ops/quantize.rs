use std::mem::MaybeUninit;

use rten_simd::dispatch::SimdOp as UnsafeSimdOp;
use rten_simd::safe::SimdOp as SafeSimdOp;
use rten_tensor::prelude::*;
use rten_tensor::{AssumeInit, NdTensor, NdTensorView, Scalar, Tensor, TensorView};
use rten_vecmath as vecmath;

use crate::ops::{
    resolve_axis, DataType, Input, InputList, IntoOpResult, OpError, Operator, Output, OutputList,
};
use crate::tensor_pool::TensorPool;

/// Convert a quantized tensor element to a higher precision value.
pub trait Dequantize<To> {
    /// Dequantize self according to `(self - zero_point) * scale`.
    fn dequantize(self, scale: To, zero_point: Self) -> To;
}

impl Dequantize<f32> for u8 {
    fn dequantize(self, scale: f32, zero_point: u8) -> f32 {
        // Promote to i32 to avoid underflow.
        let x = (self as i32) - zero_point as i32;
        (x as f32) * scale
    }
}

impl Dequantize<f32> for i8 {
    fn dequantize(self, scale: f32, zero_point: i8) -> f32 {
        // Promote to i32 to avoid underflow.
        let x = (self as i32) - zero_point as i32;
        (x as f32) * scale
    }
}

pub fn dequantize_linear<T: Copy + Default + Dequantize<f32> + Scalar>(
    pool: &TensorPool,
    input: TensorView<T>,
    scale: TensorView<f32>,
    zero_point: Option<TensorView<T>>,
    axis: isize,
) -> Result<Tensor<f32>, OpError> {
    if let Some(zero_point) = zero_point.as_ref() {
        if zero_point.shape() != scale.shape() {
            return Err(OpError::InvalidValue(
                "scale and zero_point must have same shape",
            ));
        }
    }

    match scale.ndim() {
        0 => {
            let scale = scale.item().unwrap();
            let zero_point = zero_point.and_then(|z| z.item()).unwrap();

            Ok(input.map_in(pool, |x| x.dequantize(*scale, *zero_point)))
        }
        1 => {
            let axis = resolve_axis(input.ndim(), axis)?;
            let scale: NdTensorView<f32, 1> = scale.try_into().unwrap();
            let zero = NdTensor::from(T::default());
            let zero_point: NdTensorView<T, 1> = zero_point
                .map(|zp| {
                    let zp_vec: NdTensorView<T, 1> = zp.try_into().unwrap();
                    zp_vec
                })
                .unwrap_or(zero.broadcast(scale.shape()));

            let mut output = Tensor::uninit_in(pool, input.shape());
            output
                .axis_iter_mut(axis)
                .zip(input.axis_iter(axis))
                .zip(scale.iter())
                .zip(zero_point.iter())
                .for_each(|(((mut out_slice, in_slice), &scale), &zero_point)| {
                    for (y, &x) in out_slice.iter_mut().zip(in_slice.iter()) {
                        y.write(x.dequantize(scale, zero_point));
                    }
                });

            // Safety: All elements are initialized
            Ok(unsafe { output.assume_init() })
        }
        _ => Err(OpError::UnsupportedValue(
            "Blocked dequantization is not supported",
        )),
    }
}

#[derive(Debug)]
pub struct DequantizeLinear {
    pub axis: isize,
}

impl Operator for DequantizeLinear {
    fn name(&self) -> &str {
        "DequantizeLinear"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require(0)?;
        let scale = inputs.require_as(1)?;

        match input {
            Input::Int8Tensor(x) => {
                let zero_point = inputs.get_as(2)?;
                dequantize_linear(pool, x, scale, zero_point, self.axis).into_op_result()
            }
            Input::UInt8Tensor(x) => {
                let zero_point = inputs.get_as(2)?;
                dequantize_linear(pool, x, scale, zero_point, self.axis).into_op_result()
            }
            _ => Err(OpError::UnsupportedType),
        }
    }
}

/// Convert a high precision tensor element to a quantized value.
///
/// The conversion is done according to:
///
/// ```text
/// y = saturate((self / scale) + zero_point)
/// ```
///
/// For efficiency the `quantize` method takes the reciprocal of the scale.
///
/// See https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html for
/// additional details.
pub trait Quantize<To> {
    /// Quantize a single value.
    fn quantize(self, inv_scale: Self, zero_point: To) -> To;

    /// Quantize a slice of values in `src`, writing to `dest` and returning the
    /// initialized slice.
    fn quantize_slice<'a>(
        src: &[Self],
        dest: &'a mut [MaybeUninit<To>],
        inv_scale: Self,
        zero_point: To,
    ) -> &'a mut [To]
    where
        Self: Copy + Sized,
        To: Copy,
    {
        assert_eq!(src.len(), dest.len());
        for (x, y) in src.iter().zip(dest.iter_mut()) {
            y.write(x.quantize(inv_scale, zero_point));
        }
        unsafe { dest.assume_init() }
    }
}

impl Quantize<u8> for f32 {
    fn quantize(self, inv_scale: Self, zero_point: u8) -> u8 {
        let y = (self * inv_scale).round_ties_even();
        let y = y + zero_point as f32;
        y as u8 // saturating cast
    }

    fn quantize_slice<'a>(
        src: &[f32],
        dest: &'a mut [MaybeUninit<u8>],
        inv_scale: f32,
        zero_point: u8,
    ) -> &'a mut [u8] {
        vecmath::Quantize::new(src, dest, inv_scale, zero_point).dispatch()
    }
}

impl Quantize<i8> for f32 {
    fn quantize(self, inv_scale: Self, zero_point: i8) -> i8 {
        let y = (self * inv_scale).round_ties_even();
        let y = y + zero_point as f32;
        y as i8 // saturating cast
    }
}

pub fn quantize_linear<T: Copy + Default + Scalar>(
    pool: &TensorPool,
    input: TensorView<f32>,
    scale: TensorView<f32>,
    zero_point: Option<TensorView<T>>,
    axis: isize,
) -> Result<Tensor<T>, OpError>
where
    f32: Quantize<T>,
{
    if let Some(zero_point) = zero_point.as_ref() {
        if zero_point.shape() != scale.shape() {
            return Err(OpError::InvalidValue(
                "scale and zero_point must have same shape",
            ));
        }
    }

    match scale.ndim() {
        0 => {
            let inv_scale = 1. / *scale.item().unwrap();
            let zero_point = *zero_point.and_then(|z| z.item()).unwrap();

            if let Some(data) = input.data() {
                let mut buf = pool.alloc(data.len());
                let buf_data = &mut buf.spare_capacity_mut()[..data.len()];

                Quantize::quantize_slice(data, buf_data, inv_scale, zero_point);

                // Safety: `quantize_slice` initialized `data.len()` elements
                unsafe {
                    buf.set_len(data.len());
                }
                Ok(Tensor::from_data(input.shape(), buf))
            } else {
                Ok(input.map_in(pool, |x| x.quantize(inv_scale, zero_point)))
            }
        }
        1 => {
            let axis = resolve_axis(input.ndim(), axis)?;
            let scale: NdTensorView<f32, 1> = scale.try_into().unwrap();
            let zero = NdTensor::from(T::default());
            let zero_point: NdTensorView<T, 1> = zero_point
                .map(|zp| {
                    let zp_vec: NdTensorView<T, 1> = zp.try_into().unwrap();
                    zp_vec
                })
                .unwrap_or(zero.broadcast(scale.shape()));

            let mut output = Tensor::uninit_in(pool, input.shape());
            output
                .axis_iter_mut(axis)
                .zip(input.axis_iter(axis))
                .zip(scale.iter())
                .zip(zero_point.iter())
                .for_each(|(((mut out_slice, in_slice), &scale), &zero_point)| {
                    let inv_scale = 1. / scale;
                    for (y, &x) in out_slice.iter_mut().zip(in_slice.iter()) {
                        y.write(x.quantize(inv_scale, zero_point));
                    }
                });

            // Safety: All elements are initialized
            Ok(unsafe { output.assume_init() })
        }
        _ => Err(OpError::UnsupportedValue(
            "Blocked quantization is not supported",
        )),
    }
}

#[derive(Debug)]
pub struct QuantizeLinear {
    pub axis: isize,
    pub output_dtype: Option<DataType>,
}

impl Operator for QuantizeLinear {
    fn name(&self) -> &str {
        "QuantizeLinear"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require_as(0)?;
        let y_scale = inputs.require_as(1)?;
        let y_zero_point = inputs.get(2);

        match (y_zero_point, self.output_dtype) {
            (Some(Input::UInt8Tensor(y_zero_point)), Some(DataType::UInt8) | None) => {
                quantize_linear(
                    pool,
                    input.view(),
                    y_scale.view(),
                    Some(y_zero_point.view()),
                    self.axis,
                )
                .into_op_result()
            }
            (None, Some(DataType::UInt8)) => {
                quantize_linear::<u8>(pool, input.view(), y_scale.view(), None, self.axis)
                    .into_op_result()
            }
            (Some(Input::Int8Tensor(y_zero_point)), Some(DataType::Int8) | None) => {
                quantize_linear(
                    pool,
                    input.view(),
                    y_scale.view(),
                    Some(y_zero_point.view()),
                    self.axis,
                )
                .into_op_result()
            }
            (None, Some(DataType::Int8)) => {
                quantize_linear::<i8>(pool, input.view(), y_scale.view(), None, self.axis)
                    .into_op_result()
            }
            _ => Err(OpError::UnsupportedType),
        }
    }
}

pub trait SaturatingCast<To> {
    fn saturating_cast(self) -> To;
}

impl SaturatingCast<u8> for f32 {
    fn saturating_cast(self) -> u8 {
        self.clamp(0., 255.) as u8
    }
}

pub struct DynamicQuantizeOutput<T> {
    pub quantized: Tensor<T>,
    pub scale: Tensor<f32>,
    pub zero_point: Tensor<T>,
}

pub fn dynamic_quantize_linear<T: Copy + Default + Scalar>(
    pool: &TensorPool,
    input: TensorView<f32>,
) -> Result<DynamicQuantizeOutput<T>, OpError>
where
    f32: Quantize<T> + SaturatingCast<T>,
{
    // From the ONNX spec, this operator is defined in terms of other ONNX
    // operators as:
    //
    // ```
    // DynamicQuantizeLinear (x) => (y, y_scale, y_zero_point)
    // {
    //    Q_Min = Constant <value: tensor = float {0}> ()
    //    Q_Max = Constant <value: tensor = float {255}> ()
    //    X_Min = ReduceMin <keepdims: int = 0> (x)
    //    X_Min_Adjusted = Min (X_Min, Q_Min)
    //    X_Max = ReduceMax <keepdims: int = 0> (x)
    //    X_Max_Adjusted = Max (X_Max, Q_Min)
    //    X_Range = Sub (X_Max_Adjusted, X_Min_Adjusted)
    //    Scale = Div (X_Range, Q_Max)
    //    Min_Scaled = Div (X_Min_Adjusted, Scale)
    //    Initial_ZeroPoint_FP = Sub (Q_Min, Min_Scaled)
    //    Clipped_ZeroPoint_FP = Clip (Initial_ZeroPoint_FP, Q_Min, Q_Max)
    //    Rounded_ZeroPoint_FP = Round (Clipped_ZeroPoint_FP)
    //    Zeropoint = Cast <to: int = 2> (Rounded_ZeroPoint_FP)
    //    y_scale = Identity (Scale)
    //    y_zero_point = Identity (Zeropoint)
    //    y = QuantizeLinear (x, Scale, Zeropoint)
    // }
    // ```

    if input.is_empty() {
        // If the input is empty, the zero point and scale can be chosen
        // arbitrarily. We pick zero/one as natural choices.
        return Ok(DynamicQuantizeOutput {
            quantized: Tensor::zeros(input.shape()),
            zero_point: Tensor::from(T::default()),
            scale: Tensor::from(1.),
        });
    }

    let q_min = 0.;
    let q_max = 255.;

    let input = input.to_contiguous_in(pool);
    let (x_min, x_max) = vecmath::MinMax::new(input.data().unwrap()).dispatch();
    let x_min_adjusted = x_min.min(q_min);
    let x_max_adjusted = x_max.max(q_min);
    let x_range = x_max_adjusted - x_min_adjusted;
    let scale = x_range / q_max;
    let min_scaled = x_min_adjusted / scale;
    let initial_zero_point = q_min - min_scaled;
    let clipped_zero_point = initial_zero_point.clamp(q_min, q_max);
    let rounded_zero_point = clipped_zero_point.round_ties_even();
    let zero_point: T = rounded_zero_point.saturating_cast();

    let scale_tensor = Tensor::from(scale);
    let zero_point_tensor = Tensor::from(zero_point);
    let quantized = quantize_linear(
        pool,
        input.view(),
        scale_tensor.view(),
        Some(zero_point_tensor.view()),
        1, /* axis */
    )?;

    Ok(DynamicQuantizeOutput {
        quantized,
        scale: scale_tensor,
        zero_point: zero_point_tensor,
    })
}

#[derive(Debug)]
pub struct DynamicQuantizeLinear {}

impl Operator for DynamicQuantizeLinear {
    fn name(&self) -> &str {
        "DynamicQuantizeLinear"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require_as(0)?;

        let DynamicQuantizeOutput {
            quantized,
            scale,
            zero_point,
        } = dynamic_quantize_linear::<u8>(pool, input)?;

        let quantized: Output = quantized.into();
        let scale: Output = scale.into();
        let zero_point: Output = zero_point.into();

        Ok([quantized, scale, zero_point].into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::test_util::expect_equal_with_tolerance;
    use rten_tensor::Tensor;
    use rten_testing::TestCases;

    use super::{dequantize_linear, dynamic_quantize_linear, quantize_linear};
    use crate::ops::tests::new_pool;
    use crate::ops::{OpError, Output};

    // Test dequantization followed by re-quantization. In this order the
    // result should be the input. In the opposite order this would not be the
    // case, since quantization is lossy.
    #[test]
    fn test_dequantize_quantize_linear() {
        #[derive(Debug)]
        struct Case {
            axis: isize,
            input: Output,
            scale: Tensor<f32>,
            zero_point: Option<Output>,
            expected: Result<Tensor<f32>, OpError>,
        }

        let cases = [
            // Per-tensor dequantization (u8)
            Case {
                axis: 1,
                input: Tensor::from([20u8, 30, 40]).into(),
                scale: Tensor::from(0.5),
                zero_point: Some(Tensor::from(10u8).into()),
                expected: Ok(Tensor::from([5., 10., 15.])),
            },
            // Per-tensor dequantization (i8)
            Case {
                axis: 1,
                input: Tensor::from([20i8, 30, 40]).into(),
                scale: Tensor::from(0.5),
                zero_point: Some(Tensor::from(10i8).into()),
                expected: Ok(Tensor::from([5., 10., 15.])),
            },
            // Per-row dequantization for a matrix
            Case {
                axis: 0,
                input: Tensor::from([[10u8, 20], [30, 40]]).into(),
                scale: Tensor::from([0.5, 2.]),
                zero_point: Some(Tensor::from([10u8, 20]).into()),
                expected: Ok(Tensor::from([[0., 5.], [20., 40.]])),
            },
            // Mismatched scale and zero-point shape
            Case {
                axis: 0,
                input: Tensor::from([10u8]).into(),
                scale: Tensor::from([0.5, 2.]),
                zero_point: Some(Tensor::from([1u8, 2, 3]).into()),
                expected: Err(OpError::InvalidValue(
                    "scale and zero_point must have same shape",
                )),
            },
            // Blocked dequantization
            Case {
                axis: 0,
                input: Tensor::from([[10u8, 20], [30, 40]]).into(),
                scale: Tensor::from([[1., 2.], [3., 4.]]),
                zero_point: Some(Tensor::from([[1u8, 2], [3, 4]]).into()),
                expected: Err(OpError::UnsupportedValue(
                    "Blocked dequantization is not supported",
                )),
            },
            // Empty
            Case {
                axis: 0,
                input: Tensor::<u8>::zeros(&[0]).into(),
                scale: Tensor::zeros(&[0]),
                zero_point: Some(Tensor::<u8>::zeros(&[0]).into()),
                expected: Ok(Tensor::zeros(&[0])),
            },
        ];

        cases.test_each(|case| {
            let pool = new_pool();
            let Case {
                input,
                scale,
                zero_point,
                axis,
                expected,
            } = case;

            match input {
                Output::UInt8Tensor(input) => {
                    let zero_point: Option<Tensor<u8>> =
                        zero_point.clone().map(|zp| zp.try_into().unwrap());
                    let result = dequantize_linear(
                        &pool,
                        input.view(),
                        scale.view(),
                        zero_point.as_ref().map(|zp| zp.view()),
                        *axis,
                    );
                    assert_eq!(result, *expected);

                    // Re-quantize the result, and we should get back to the
                    // input.
                    if let Ok(dequant) = result {
                        let requantized = quantize_linear(
                            &pool,
                            dequant.view(),
                            scale.view(),
                            zero_point.as_ref().map(|zp| zp.view()),
                            *axis,
                        )
                        .unwrap();
                        assert_eq!(requantized, *input);
                    }
                }
                Output::Int8Tensor(input) => {
                    let zero_point: Option<Tensor<i8>> =
                        zero_point.clone().map(|zp| zp.try_into().unwrap());
                    let result = dequantize_linear(
                        &pool,
                        input.view(),
                        scale.view(),
                        zero_point.as_ref().map(|zp| zp.view()),
                        *axis,
                    );
                    assert_eq!(result, *expected);

                    // Re-quantize the result, and we should get back to the
                    // input.
                    if let Ok(dequant) = result {
                        let requantized = quantize_linear(
                            &pool,
                            dequant.view(),
                            scale.view(),
                            zero_point.as_ref().map(|zp| zp.view()),
                            *axis,
                        )
                        .unwrap();
                        assert_eq!(requantized, *input);
                    }
                }
                _ => panic!("unsupported quantized type"),
            };
        })
    }

    #[test]
    fn test_dynamic_quantize_linear() {
        #[derive(Debug)]
        struct Case {
            input: Tensor<f32>,
            max_error: f32,
        }

        let cases = [
            // Inputs centered around zero. Zero point should be ~0.
            Case {
                input: [-2., -1., 0., 1., 2.].into(),
                max_error: 0.01,
            },
            // Positive inputs.
            Case {
                input: [1., 2., 3., 4., 5.].into(),
                max_error: 0.01,
            },
            // Negative inputs.
            Case {
                input: [-1., -2., -3., -4., -5.].into(),
                max_error: 0.01,
            },
            // Small input values
            Case {
                input: Tensor::arange(-0.1, 0.1, Some(0.01)),
                max_error: 0.001,
            },
            // All values equal (positive)
            Case {
                input: Tensor::from([234.56]),
                max_error: 0.,
            },
            // All values equal (negative)
            Case {
                input: Tensor::from([-234.56]),
                max_error: 0.,
            },
            // Empty tensor
            Case {
                input: Tensor::zeros(&[0]),
                max_error: 0.,
            },
        ];

        cases.test_each(|case| {
            let Case { input, max_error } = case;

            let pool = new_pool();

            // Quantize input.
            let output = dynamic_quantize_linear::<u8>(&pool, input.view()).unwrap();
            assert_eq!(output.quantized.shape(), input.shape());
            let zero_point = *output.zero_point.item().unwrap();
            let scale = *output.scale.item().unwrap();

            // Dequantize the results and check the value is close enough to
            // the inputs.
            let dequantized = output
                .quantized
                .map(|&q| (q as i32 - zero_point as i32) as f32 * scale);
            expect_equal_with_tolerance(&dequantized, &input, *max_error, *max_error).unwrap();
        })
    }
}
