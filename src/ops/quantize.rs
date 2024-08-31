use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, Scalar, Tensor, TensorView};

use crate::ops::{
    resolve_axis, DataType, Input, InputList, IntoOpResult, OpError, Operator, OutputList,
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
/// See https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html for
/// additional details.
pub trait Quantize<To> {
    fn quantize(self, scale: Self, zero_point: To) -> To;
}

impl Quantize<u8> for f32 {
    fn quantize(self, scale: Self, zero_point: u8) -> u8 {
        let y = (self / scale).round_ties_even();
        let y = y + zero_point as f32;
        y.clamp(0., 255.) as u8
    }
}

impl Quantize<i8> for f32 {
    fn quantize(self, scale: Self, zero_point: i8) -> i8 {
        let y = (self / scale).round_ties_even();
        let y = y + zero_point as f32;
        y.clamp(-128., 127.) as i8
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
            let scale = scale.item().unwrap();
            let zero_point = zero_point.and_then(|z| z.item()).unwrap();

            Ok(input.map_in(pool, |x| x.quantize(*scale, *zero_point)))
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
                        y.write(x.quantize(scale, zero_point));
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

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::prelude::*;
    use rten_tensor::Tensor;

    use super::{dequantize_linear, quantize_linear};
    use crate::ops::tests::new_pool;
    use crate::ops::{OpError, Output};

    // Test dequantization followed by re-quantization. In this order the
    // result should be the input. In the opposite order this would not be the
    // case, since quantization is lossy.
    #[test]
    fn test_dequantize_quantize_linear() -> Result<(), Box<dyn Error>> {
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

        let pool = new_pool();
        for Case {
            input,
            scale,
            zero_point,
            axis,
            expected,
        } in cases
        {
            match input {
                Output::UInt8Tensor(input) => {
                    let zero_point: Option<Tensor<u8>> =
                        zero_point.map(|zp| zp.try_into().unwrap());
                    let result = dequantize_linear(
                        &pool,
                        input.view(),
                        scale.view(),
                        zero_point.as_ref().map(|zp| zp.view()),
                        axis,
                    );
                    assert_eq!(result, expected);

                    // Re-quantize the result, and we should get back to the
                    // input.
                    if let Ok(dequant) = result {
                        let requantized = quantize_linear(
                            &pool,
                            dequant.view(),
                            scale.view(),
                            zero_point.as_ref().map(|zp| zp.view()),
                            axis,
                        )
                        .unwrap();
                        assert_eq!(requantized, input);
                    }
                }
                Output::Int8Tensor(input) => {
                    let zero_point: Option<Tensor<i8>> =
                        zero_point.map(|zp| zp.try_into().unwrap());
                    let result = dequantize_linear(
                        &pool,
                        input.view(),
                        scale.view(),
                        zero_point.as_ref().map(|zp| zp.view()),
                        axis,
                    );
                    assert_eq!(result, expected);

                    // Re-quantize the result, and we should get back to the
                    // input.
                    if let Ok(dequant) = result {
                        let requantized = quantize_linear(
                            &pool,
                            dequant.view(),
                            scale.view(),
                            zero_point.as_ref().map(|zp| zp.view()),
                            axis,
                        )
                        .unwrap();
                        assert_eq!(requantized, input);
                    }
                }
                _ => panic!("unsupported quantized type"),
            };
        }

        Ok(())
    }
}
