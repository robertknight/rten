use std::ops;

use rten_tensor::prelude::*;
use rten_tensor::{NdTensorView, Tensor, TensorView};

use crate::ops::{
    resolve_axis, resolve_index, static_dims, Input, InputList, IntoOpResult, OpError, Operator,
    OutputList, Scalar,
};
use crate::tensor_pool::TensorPool;

pub fn constant_of_shape<T: Copy>(
    pool: &TensorPool,
    value: T,
    shape: &NdTensorView<i32, 1>,
) -> Tensor<T> {
    let shape: Vec<_> = shape.iter().map(|el| *el as usize).collect();
    Tensor::full_in(pool, &shape, value)
}

#[derive(Debug)]
pub struct ConstantOfShape {
    pub value: Scalar,
}

impl Operator for ConstantOfShape {
    fn name(&self) -> &str {
        "ConstantOfShape"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let shape = inputs.require_as::<i32>(0)?;
        let shape = static_dims!(shape, 1)?;

        match self.value {
            Scalar::Int(value) => constant_of_shape(pool, value, &shape).into_op_result(),
            Scalar::Float(value) => constant_of_shape(pool, value, &shape).into_op_result(),
        }
    }
}

pub fn onehot<T: Copy + Default + PartialEq>(
    pool: &TensorPool,
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

    let len = out_shape.iter().product();
    let mut data = pool.alloc(len);
    data.resize(len, off_value);
    let mut output = Tensor::from_data(&out_shape, data);

    for (mut index, class) in indices.indices().zip(indices.iter()) {
        if let Some(class) = resolve_index(depth, *class as isize) {
            index.insert(onehot_axis, class);
            output[&index] = on_value;
        };
    }

    Ok(output)
}

fn extract_on_off_values<T: Copy>(values: NdTensorView<T, 1>) -> Result<(T, T), OpError> {
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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let indices = inputs.require_as::<i32>(0)?;
        let depth = inputs.require_as::<i32>(1)?;
        let depth = depth
            .item()
            .and_then(|&val| if val > 0 { Some(val as usize) } else { None })
            .ok_or(OpError::InvalidValue("`depth` must be a positive scalar"))?;
        let values = inputs.require(2)?;

        match values {
            Input::Int32Tensor(values) => {
                let values = static_dims!(values, 1)?;
                let (on_value, off_value) = extract_on_off_values(values)?;
                onehot(pool, indices, self.axis, depth, on_value, off_value).into_op_result()
            }
            Input::FloatTensor(values) => {
                let values = static_dims!(values, 1)?;
                let (on_value, off_value) = extract_on_off_values(values)?;
                onehot(pool, indices, self.axis, depth, on_value, off_value).into_op_result()
            }
            _ => Err(OpError::UnsupportedType),
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
    Ok(output.into())
}

#[derive(Debug)]
pub struct Range {}

impl Operator for Range {
    fn name(&self) -> &str {
        "Range"
    }

    fn run(&self, _pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
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
            Input::Int32Tensor(_) => {
                let start = start.try_into()?;
                let limit = limit.try_into()?;
                let delta = delta.try_into()?;
                range::<i32>(start, limit, delta).into_op_result()
            }
            _ => Err(OpError::UnsupportedType),
        }
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::Tensor;

    use crate::ops::tests::new_pool;
    use crate::ops::{onehot, range, ConstantOfShape, OpError, Operator, Scalar};

    #[test]
    fn test_constant_of_shape() {
        let pool = new_pool();
        let op = ConstantOfShape {
            value: Scalar::Int(42),
        };
        let shape = Tensor::from([1, 5, 10]);

        let result = op
            .run(&pool, (&shape).into())
            .unwrap()
            .remove(0)
            .into_tensor::<i32>()
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

        let pool = new_pool();
        for Case {
            classes,
            axis,
            depth,
            on_value,
            off_value,
            expected,
        } in cases
        {
            let result = onehot(&pool, classes.view(), axis, depth, on_value, off_value);
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
