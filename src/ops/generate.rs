use std::iter::zip;
use std::ops;

use wasnn_tensor::{Layout, NdTensorView, Tensor, TensorView, View};

use crate::ops::{
    resolve_axis, resolve_index, Input, InputList, IntoOpResult, OpError, Operator, Output, Scalar,
};
use crate::static_dims;

pub fn constant_of_shape<T: Clone>(value: T, shape: &NdTensorView<i32, 1>) -> Tensor<T> {
    let shape: Vec<_> = shape.iter().map(|el| *el as usize).collect();
    let len = shape.iter().product();
    Tensor::from_data(&shape, vec![value; len])
}

#[derive(Debug)]
pub struct ConstantOfShape {
    pub value: Scalar,
}

impl Operator for ConstantOfShape {
    fn name(&self) -> &str {
        "ConstantOfShape"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let shape = inputs.require_as::<i32>(0)?;
        let shape = static_dims!(shape, 1)?;

        match self.value {
            Scalar::Int(value) => constant_of_shape(value, &shape).into_op_result(),
            Scalar::Float(value) => constant_of_shape(value, &shape).into_op_result(),
        }
    }
}

pub fn onehot<T: Copy + Default + PartialEq>(
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

    let mut output = Tensor::zeros(&out_shape);
    if off_value != T::default() {
        output.apply(|_| off_value);
    }

    for (mut index, class) in zip(indices.indices(), indices.iter()) {
        if let Some(class) = resolve_index(depth, *class as isize) {
            index.insert(onehot_axis, class);
            output[&index] = on_value;
        };
    }

    Ok(output)
}

fn extract_on_off_values<T: Copy>(values: NdTensorView<T, 1>) -> Result<(T, T), OpError> {
    if values.len() == 2 {
        Ok((values[[0]], values[[1]]))
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

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let indices = inputs.require_as::<i32>(0)?;
        let depth = inputs.require_as::<i32>(1)?;
        let depth = depth
            .item()
            .and_then(|&val| if val > 0 { Some(val as usize) } else { None })
            .ok_or(OpError::InvalidValue("`depth` must be a positive scalar"))?;
        let values = inputs.require(2)?;

        match values {
            Input::IntTensor(values) => {
                let values = static_dims!(values, 1)?;
                let (on_value, off_value) = extract_on_off_values(values)?;
                onehot(indices.view(), self.axis, depth, on_value, off_value).into_op_result()
            }
            Input::FloatTensor(values) => {
                let values = static_dims!(values, 1)?;
                let (on_value, off_value) = extract_on_off_values(values)?;
                onehot(indices.view(), self.axis, depth, on_value, off_value).into_op_result()
            }
        }
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
    Ok(Tensor::from_vec(output))
}

#[derive(Debug)]
pub struct Range {}

impl Operator for Range {
    fn name(&self) -> &str {
        "Range"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let start = inputs.require(0)?;
        let limit = inputs.require(1)?;
        let delta = inputs.require(2)?;

        match start {
            Input::FloatTensor(_) => {
                let start = start.try_into()?;
                let limit = limit.try_into()?;
                let delta = delta.try_into()?;
                range::<f32>(start, limit, delta).into_op_result()
            }
            Input::IntTensor(_) => {
                let start = start.try_into()?;
                let limit = limit.try_into()?;
                let delta = delta.try_into()?;
                range::<i32>(start, limit, delta).into_op_result()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use wasnn_tensor::{tensor, Layout, Tensor, View};

    use crate::ops::{onehot, range, ConstantOfShape, OpError, Operator, Scalar};

    #[test]
    fn test_constant_of_shape() {
        let op = ConstantOfShape {
            value: Scalar::Int(42),
        };
        let shape = Tensor::from_vec(vec![1, 5, 10]);

        let result = op
            .run((&shape).into())
            .unwrap()
            .remove(0)
            .into_int()
            .unwrap();

        assert_eq!(result.shape(), &[1, 5, 10]);
        assert_eq!(result.to_vec(), vec![42; result.shape().iter().product()]);
    }

    #[test]
    fn test_onehot() {
        struct Case {
            classes: Tensor<i32>,
            axis: isize,
            depth: usize,
            on_value: f32,
            off_value: f32,
            expected: Result<Tensor<f32>, OpError>,
        }

        let cases = [
            // Common case of converting class labels to a float one-hot tensor
            // with values of 0/1.
            Case {
                classes: tensor!([0, 1, 2, 3, 4]),
                axis: -1,
                depth: 5,
                on_value: 1.,
                off_value: 0.,
                expected: Ok(tensor!((5, 5); [
                    1., 0., 0., 0., 0., // 1
                    0., 1., 0., 0., 0., // 2
                    0., 0., 1., 0., 0., // 3
                    0., 0., 0., 1., 0., // 4
                    0., 0., 0., 0., 1. // 5
                ])),
            },
            // Non-standard on/off values.
            Case {
                classes: tensor!([0, 1]),
                axis: -1,
                depth: 2,
                on_value: 2.,
                off_value: -3.,
                expected: Ok(tensor!((2, 2); [
                    2., -3., // 1
                    -3., 2. // 2
                ])),
            },
            // Add classes as first axis.
            Case {
                classes: tensor!([0, 1]),
                axis: 0,
                depth: 2,
                on_value: 1.,
                off_value: 0.,
                expected: Ok(tensor!((2, 2); [
                    1., 0., // 1
                    0., 1. // 2
                ])
                .transposed()
                .to_tensor()),
            },
            // Invalid class index for depth.
            Case {
                classes: tensor!([0, 2]),
                axis: -1,
                depth: 2,
                on_value: 1.,
                off_value: 0.,
                expected: Ok(tensor!((2, 2); [
                    1., 0., // 1
                    0., 0. // 2. All "off" because class is out of range.
                ])),
            },
            // Invalid axis
            Case {
                classes: tensor!([0, 1]),
                axis: 2,
                depth: 2,
                on_value: 1.,
                off_value: 0.,
                expected: Err(OpError::InvalidValue("Axis is invalid")),
            },
        ];

        for Case {
            classes,
            axis,
            depth,
            on_value,
            off_value,
            expected,
        } in cases
        {
            let result = onehot(classes.view(), axis, depth, on_value, off_value);
            assert_eq!(result, expected);
        }
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
