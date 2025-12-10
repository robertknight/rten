use std::ops;

use rten_base::num::Identities;
use rten_shape_inference::ops as shape_ops;
use rten_tensor::errors::DimensionError;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, Tensor, TensorView};

use crate::buffer_pool::BufferPool;
use crate::infer_shapes::{InferShapes, impl_infer_shapes};
use crate::operator::{
    IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
    static_dims,
};
use crate::ops::{map_dtype, map_value_view, resolve_axis, resolve_index};
use crate::value::{DataType, Scalar, ValueType, ValueView};

pub fn constant_of_shape<T: Copy>(
    pool: &BufferPool,
    value: T,
    shape: &NdTensorView<i32, 1>,
) -> Result<Tensor<T>, OpError> {
    let shape = shape
        .iter()
        .map(|el| (*el).try_into())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| OpError::InvalidValue("Invalid shape"))?;
    Ok(Tensor::full_in(pool, &shape, value))
}

#[derive(Debug, PartialEq)]
pub struct ConstantOfShape {
    pub value: Scalar,
}

impl Operator for ConstantOfShape {
    fn name(&self) -> &str {
        "ConstantOfShape"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(1)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let pool = ctx.pool();
        let shape = ctx.inputs().require_as(0)?;

        match self.value {
            Scalar::Int(value) => constant_of_shape(pool, value, &shape).into_op_result(),
            Scalar::Float(value) => constant_of_shape(pool, value, &shape).into_op_result(),
        }
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(self)
    }

    fn output_types(&self) -> Option<OutputTypeList> {
        Some([OutputType::Fixed(ValueType::Tensor(self.value.dtype()))].into())
    }
}

impl_infer_shapes!(
    ConstantOfShape,
    op,
    shape_ops::ConstantOfShape {
        value: match op.value {
            Scalar::Int(val) => Some(val),
            Scalar::Float(_) => None,
        },
    }
);

pub fn onehot<T: Copy + Default + PartialEq>(
    pool: &BufferPool,
    indices: TensorView<i32>,
    onehot_axis: isize,
    depth: usize,
    on_value: T,
    off_value: T,
) -> Result<Tensor<T>, OpError> {
    let onehot_axis = resolve_axis(indices.ndim() + 1, onehot_axis)?;

    let mut out_shape = Vec::with_capacity(indices.ndim() + 1);
    out_shape.extend_from_slice(indices.shape());
    out_shape.insert(onehot_axis, depth);

    let mut output = if off_value == T::default() {
        // For the common case of the "off" value being zero, use `zeros_in`
        // which can use optimized methods for zeroing buffers.
        Tensor::zeros_in(pool, &out_shape)
    } else {
        Tensor::full_in(pool, &out_shape, off_value)
    };

    output
        .lanes_mut(onehot_axis)
        .zip(indices.iter())
        .for_each(|(mut lane, index)| {
            if let Some(index) = resolve_index(depth, *index as isize) {
                *lane.nth(index).unwrap() = on_value;
            };
        });

    Ok(output)
}

/// Extract OneHot off/on values from a vector of `[off_value, on_value]`.
fn extract_off_on_values<T: Copy>(values: NdTensorView<T, 1>) -> Result<(T, T), OpError> {
    if values.len() == 2 {
        Ok((values[0], values[1]))
    } else {
        Err(OpError::InvalidValue("Expected size-2 vector"))
    }
}

#[derive(Debug)]
pub struct OneHot {
    pub axis: isize,
}

impl Operator for OneHot {
    fn name(&self) -> &str {
        "OneHot"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(3)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let indices = inputs.require_as(0)?;
        let depth: TensorView<i32> = inputs.require_as(1)?;
        let depth = depth
            .item()
            .and_then(|&val| if val > 0 { Some(val as usize) } else { None })
            .ok_or(OpError::InvalidValue("`depth` must be a positive scalar"))?;
        let values = inputs.require(2)?;

        map_value_view!(values, values, [Int32Tensor, FloatTensor], {
            let values = static_dims!(values, 1)?;
            let (off_value, on_value) = extract_off_on_values(values)?;
            onehot(ctx.pool(), indices, self.axis, depth, on_value, off_value).into_op_result()
        })
    }

    fn output_types(&self) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(2)].into())
    }
}

pub fn range<T: Copy + Default + ops::Add<Output = T> + PartialOrd>(
    start: T,
    limit: T,
    delta: T,
) -> Result<Tensor<T>, OpError> {
    if delta == T::default() {
        return Err(OpError::InvalidValue("delta must be non-zero"));
    }

    // This is not very efficient as it grows the output gradually instead of
    // allocating once. This however made the initial implementation easier by
    // minimizing the traits that T needs to implement.
    let mut output = Vec::new();
    let mut val = start;
    while (delta > T::default() && val < limit) || (delta < T::default() && val > limit) {
        output.push(val);
        val = val + delta;
    }
    Ok(output.into())
}

#[derive(Debug)]
pub struct Range {}

impl Operator for Range {
    fn name(&self) -> &str {
        "Range"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(3)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let start = inputs.require(0)?;
        let limit = inputs.require(1)?;
        let delta = inputs.require(2)?;

        map_value_view!(start, start, [FloatTensor, Int32Tensor], {
            let start = start
                .item()
                .copied()
                .ok_or(OpError::InvalidValue("`start` must be a scalar"))?;
            let limit = limit.try_into()?;
            let delta = delta.try_into()?;
            range(start, limit, delta).into_op_result()
        })
    }

    fn output_types(&self) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(&shape_ops::Range)
    }
}

pub fn eye_like<T: Copy + Default + Identities>(
    pool: &BufferPool,
    shape: [usize; 2],
    k: i32,
) -> NdTensor<T, 2> {
    let mut output = NdTensor::zeros_in(pool, shape);
    if output.is_empty() {
        return output;
    }

    let one = T::one();
    for y in 0..shape[0] {
        let x = y as i32 + k;
        if x < 0 || x >= shape[1] as i32 {
            continue;
        }
        output[[y, x as usize]] = one;
    }

    output
}

#[derive(Debug)]
pub struct EyeLike {
    pub dtype: Option<DataType>,
    pub k: i32,
}

impl Operator for EyeLike {
    fn name(&self) -> &str {
        "EyeLike"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(1)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input = ctx.inputs().require(0)?;
        let ValueType::Tensor(input_dtype) = input.dtype() else {
            return Err(OpError::InvalidValue("expected input to be a tensor"));
        };
        let dtype = self.dtype.unwrap_or(input_dtype);

        map_dtype!(dtype, T, {
            let shape: [usize; 2] = input.shape().as_ref().try_into().map_err(|_| {
                OpError::from(DimensionError {
                    actual: input.ndim(),
                    expected: 2,
                })
                .with_input_index(0)
            })?;
            eye_like::<T>(ctx.pool(), shape, self.k).into_op_result()
        })
    }

    fn output_types(&self) -> Option<OutputTypeList> {
        Some(
            [if let Some(dtype) = self.dtype {
                OutputType::Fixed(ValueType::Tensor(dtype))
            } else {
                OutputType::CopyFromInput(0)
            }]
            .into(),
        )
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::{NdTensor, Tensor};
    use rten_testing::TestCases;

    use crate::operator::{OpError, OperatorExt};
    use crate::ops::{ConstantOfShape, EyeLike, OneHot, range};
    use crate::value::{DataType, Scalar, Value};

    #[test]
    fn test_constant_of_shape() {
        let op = ConstantOfShape {
            value: Scalar::Int(42),
        };
        let shape = Tensor::from([1, 5, 10]);
        let result: Tensor<i32> = op.run_simple(&shape).unwrap();

        assert_eq!(result.shape(), &[1, 5, 10]);
        assert_eq!(result.to_vec(), vec![42; result.shape().iter().product()]);

        let shape = Tensor::from([1, -1]);
        let result: Result<Tensor<i32>, OpError> = op.run_simple(&shape);
        assert_eq!(
            result.err().unwrap(),
            OpError::InvalidValue("Invalid shape")
        );
    }

    #[test]
    fn test_eye_like() {
        #[derive(Debug)]
        struct Case {
            input: Value,
            k: i32,
            dtype: Option<DataType>,
            expected: Value,
        }

        let cases = [
            // Empty
            Case {
                input: NdTensor::<i32, 2>::zeros([0, 0]).into(),
                k: 0,
                dtype: None,
                expected: NdTensor::<i32, 2>::zeros([0, 0]).into(),
            },
            // k = 0
            Case {
                input: NdTensor::from([[1., 2.], [3., 4.]]).into(),
                k: 0,
                dtype: None,
                expected: NdTensor::from([[1., 0.], [0., 1.]]).into(),
            },
            // dtype specified
            Case {
                input: NdTensor::from([[1., 2.], [3., 4.]]).into(),
                k: 0,
                dtype: Some(DataType::Int32),
                expected: NdTensor::from([[1i32, 0], [0, 1]]).into(),
            },
            // k < 0
            Case {
                input: NdTensor::<f32, 2>::zeros([5, 4]).into(),
                k: -1,
                dtype: None,
                expected: NdTensor::from([
                    [0., 0., 0., 0.],
                    [1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.],
                ])
                .into(),
            },
            // k > 0
            Case {
                input: NdTensor::<f32, 2>::zeros([5, 4]).into(),
                k: 1,
                dtype: None,
                expected: NdTensor::from([
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                ])
                .into(),
            },
        ];

        cases.test_each(|case| {
            let result: Value = EyeLike {
                k: case.k,
                dtype: case.dtype,
            }
            .run_simple(case.input.as_view())
            .unwrap();
            assert_eq!(result, case.expected);
        });
    }

    #[test]
    fn test_onehot() {
        #[derive(Debug)]
        struct Case {
            classes: Tensor<i32>,
            axis: isize,
            depth: i32,
            on_value: f32,
            off_value: f32,
            expected: Result<Tensor<f32>, OpError>,
        }

        let cases = [
            // Common case of converting class labels to a float one-hot tensor
            // with values of 0/1.
            Case {
                classes: [0, 1, 2, 3, 4].into(),
                axis: -1,
                depth: 5,
                on_value: 1.,
                off_value: 0.,
                expected: Ok([
                    [1., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [0., 0., 1., 0., 0.],
                    [0., 0., 0., 1., 0.],
                    [0., 0., 0., 0., 1.],
                ]
                .into()),
            },
            // Non-standard on/off values.
            Case {
                classes: [0, 1].into(),
                axis: -1,
                depth: 2,
                on_value: 2.,
                off_value: -3.,
                expected: Ok([[2., -3.], [-3., 2.]].into()),
            },
            // Add classes as first axis.
            Case {
                classes: [0, 1].into(),
                axis: 0,
                depth: 2,
                on_value: 1.,
                off_value: 0.,
                expected: Ok(Tensor::from([[1., 0.], [0., 1.]]).transposed().to_tensor()),
            },
            // Invalid class index for depth.
            Case {
                classes: [0, 2].into(),
                axis: -1,
                depth: 2,
                on_value: 1.,
                off_value: 0.,
                expected: Ok([
                    [1., 0.],
                    [0., 0.], // All "off" because class is out of range.
                ]
                .into()),
            },
            // Invalid axis
            Case {
                classes: [0, 1].into(),
                axis: 2,
                depth: 2,
                on_value: 1.,
                off_value: 0.,
                expected: Err(OpError::InvalidValue("Axis is invalid")),
            },
        ];

        cases.test_each(|case| {
            let op = OneHot { axis: case.axis };
            let depth = Tensor::from(case.depth);
            let values = Tensor::from([case.off_value, case.on_value]);
            let result: Result<Tensor<f32>, _> =
                op.run_simple((case.classes.view(), depth.view(), values.view()));
            assert_eq!(result, case.expected);
        })
    }

    #[test]
    fn test_range() {
        // Int range from zero
        let r = range(0, 5, 1).unwrap();
        assert_eq!(r.to_vec(), vec![0, 1, 2, 3, 4]);

        // Float range from zero
        let r = range(0., 5., 1.).unwrap();
        assert_eq!(r.to_vec(), vec![0., 1., 2., 3., 4.]);

        // Int range from negative value with step > 1
        let r = range(-5, 5, 2).unwrap();
        assert_eq!(r.to_vec(), vec![-5, -3, -1, 1, 3]);

        // Float range from negative value with step > 1
        let r = range(-5., 5., 2.).unwrap();
        assert_eq!(r.to_vec(), vec![-5., -3., -1., 1., 3.]);

        // Negative step
        let r = range(10, 4, -2).unwrap();
        assert_eq!(r.to_vec(), vec![10, 8, 6]);
    }

    #[test]
    fn test_range_invalid_inputs() {
        let r = range(0, 5, 0);
        assert_eq!(
            r.err(),
            Some(OpError::InvalidValue("delta must be non-zero"))
        );
    }
}
