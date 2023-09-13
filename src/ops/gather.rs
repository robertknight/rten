use std::iter::zip;

use smallvec::SmallVec;
use wasnn_tensor::{Layout, Tensor, TensorCommon, TensorView};

use crate::ops::{
    resolve_axis, resolve_index, Input, InputList, IntoOpResult, OpError, Operator, Output,
};

/// Gather elements from `input` specified by `indices`.
///
/// See <https://onnx.ai/onnx/operators/onnx__Gather.html>. Per the ONNX spec this
/// is very similar to `numpy.take`. See
/// <https://numpy.org/doc/stable/reference/generated/numpy.take.html> for
/// additional explanation.
pub fn gather<T: Copy + Default>(
    input: TensorView<T>,
    axis: usize,
    indices: TensorView<i32>,
) -> Result<Tensor<T>, OpError> {
    if axis >= input.ndim() {
        return Err(OpError::InvalidValue("`axis` is out of range"));
    }
    for index in indices.iter().copied() {
        if index < 0 || index >= input.size(axis) as i32 {
            return Err(OpError::InvalidValue("Entry in `indices` is out of range"));
        }
    }

    let out_shape = [
        &input.shape()[..axis],
        indices.shape(),
        &input.shape()[axis + 1..],
    ]
    .concat();
    let mut output = Tensor::<T>::zeros(&out_shape);
    let mut in_index = vec![0; input.ndim()];

    match output.ndim() {
        0 => {
            // If the output index is empty, this means we are indexing a
            // 1D vector with a scalar.
            in_index[axis] = indices.item().copied().unwrap_or(0) as usize;
            output[[]] = input[&in_index[..]];
        }
        _ => {
            for (out_index, out_item) in output.indices().zip(output.iter_mut()) {
                for dim in 0..out_index.len() {
                    if dim < axis {
                        in_index[dim] = out_index[dim];
                    } else if dim == axis {
                        let idx = &out_index[dim..dim + indices.ndim()];
                        in_index[dim] = indices[idx] as usize;
                    } else if dim >= axis + indices.ndim() {
                        in_index[dim + 1 - indices.ndim()] = out_index[dim];
                    }
                }
                *out_item = input[&in_index[..]];
            }
        }
    }

    Ok(output)
}

#[derive(Debug)]
pub struct Gather {
    pub axis: usize,
}

impl Operator for Gather {
    fn name(&self) -> &str {
        "Gather"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let indices = inputs.require_as::<i32>(1)?;
        match input {
            Input::IntTensor(input) => {
                gather(input.view(), self.axis, indices.view()).into_op_result()
            }
            Input::FloatTensor(input) => {
                gather(input.view(), self.axis, indices.view()).into_op_result()
            }
        }
    }
}

pub fn scatter_elements<T: Copy + Default>(
    data: TensorView<T>,
    indices: TensorView<i32>,
    updates: TensorView<T>,
    axis: isize,
) -> Result<Tensor<T>, OpError> {
    if data.ndim() != indices.ndim() || data.ndim() != updates.ndim() {
        return Err(OpError::InvalidValue(
            "`data`, `indices` and `updates` must have same rank",
        ));
    }
    let axis = resolve_axis(data.ndim(), axis)?;

    let mut output = data.to_tensor();
    for (index, update) in zip(updates.indices(), updates.iter()) {
        let target_index: SmallVec<[usize; 5]> = index
            .iter()
            .enumerate()
            .filter_map(|(dim, idx)| {
                if dim == axis {
                    resolve_index(data.size(dim), indices[&index] as isize)
                } else {
                    Some(*idx)
                }
            })
            .collect();
        if target_index.len() < data.ndim() {
            return Err(OpError::InvalidValue("Index is invalid"));
        }
        output[target_index] = *update;
    }
    Ok(output)
}

#[derive(Debug)]
pub struct ScatterElements {
    pub axis: isize,
}

impl Operator for ScatterElements {
    fn name(&self) -> &str {
        "ScatterElements"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let data = inputs.require(0)?;
        let indices = inputs.require_as::<i32>(1)?;
        let updates = inputs.require(2)?;

        match (data, updates) {
            (Input::IntTensor(data), Input::IntTensor(updates)) => {
                scatter_elements(data.view(), indices.view(), updates.view(), self.axis)
                    .into_op_result()
            }
            (Input::FloatTensor(data), Input::FloatTensor(updates)) => {
                scatter_elements(data.view(), indices.view(), updates.view(), self.axis)
                    .into_op_result()
            }
            _ => Err(OpError::IncorrectInputType),
        }
    }
}

#[cfg(test)]
mod tests {
    use wasnn_tensor::rng::XorShiftRng;
    use wasnn_tensor::test_util::expect_equal;
    use wasnn_tensor::{tensor, Layout, Tensor, TensorCommon};

    use crate::ops::{gather, scatter_elements, OpError};

    #[test]
    fn test_gather_scalar() {
        let input = Tensor::from_vec(vec![1, 20, 30]);
        for i in 0..input.len() {
            let indices = Tensor::from_scalar(i as i32);
            let result = gather(input.view(), 0, indices.view()).unwrap();
            assert_eq!(result.item(), Some(&input[[i]]))
        }
    }

    #[test]
    fn test_gather() -> Result<(), String> {
        // Test case shrunk down from a small BERT model where `gather` is used
        // to lookup up embeddings.
        let mut rng = XorShiftRng::new(1234);
        let input = Tensor::rand(&[128, 10], &mut rng);
        let indices = Tensor::from_data(&[2, 2], vec![2, 5, 8, 50]);
        let result = gather(input.view(), 0, indices.view()).unwrap();
        assert_eq!(result.shape(), &[2, 2, 10]);

        // Test case #1 from ONNX spec.
        let input = Tensor::from_data(&[3, 2], vec![1.0, 1.2, 2.3, 3.4, 4.5, 5.7]);
        let indices = Tensor::from_data(&[2, 2], vec![0, 1, 1, 2]);
        let expected = Tensor::from_data(&[2, 2, 2], vec![1.0, 1.2, 2.3, 3.4, 2.3, 3.4, 4.5, 5.7]);
        let result = gather(input.view(), 0, indices.view()).unwrap();
        expect_equal(&result, &expected)?;

        // Test case #2 from ONNX spec.
        let input = Tensor::from_data(&[3, 3], vec![1.0, 1.2, 1.9, 2.3, 3.4, 3.9, 4.5, 5.7, 5.9]);
        let indices = Tensor::from_data(&[1, 2], vec![0, 2]);
        let expected = Tensor::from_data(&[3, 1, 2], vec![1.0, 1.9, 2.3, 3.9, 4.5, 5.9]);
        let result = gather(input.view(), 1, indices.view()).unwrap();
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_gather_invalid_inputs() {
        let mut rng = XorShiftRng::new(1234);
        let input = Tensor::rand(&[128, 10], &mut rng);
        let indices = Tensor::from_data(&[2, 2], vec![2, 5, 8, 50]);
        let result = gather(input.view(), 5, indices.view());
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("`axis` is out of range"))
        );

        let indices = Tensor::from_data(&[2, 2], vec![2, 5, 8, 130]);
        let result = gather(input.view(), 0, indices.view());
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Entry in `indices` is out of range"))
        );
    }

    #[test]
    fn test_scatter_elements() {
        // Example #1 from ONNX spec
        let data = Tensor::zeros(&[3, 3]);
        let indices = tensor!((2, 3); [
            1, 0, 2, //
            0, 2, 1 //
        ]);
        let updates = tensor!((2, 3); [
            1., 1.1, 1.2, //
            2., 2.1, 2.2 //
        ]);
        let expected = tensor!((3, 3); [
            2., 1.1, 0., //
            1., 0., 2.2, //
            0., 2.1, 1.2 //
        ]);
        let result = scatter_elements(
            data.view(),
            indices.view(),
            updates.view(),
            0, /* axis */
        )
        .unwrap();
        assert_eq!(result, expected);

        // Example #2 from ONNX spec
        let data = tensor!((1, 5); [1., 2., 3., 4., 5.]);
        let indices = tensor!((1, 2); [1, 3]);
        let updates = tensor!((1, 2); [1.1, 2.1]);
        let expected = tensor!((1, 5); [
            1., 1.1, 3., 2.1, 5.
        ]);
        let result = scatter_elements(
            data.view(),
            indices.view(),
            updates.view(),
            1, /* axis */
        )
        .unwrap();
        assert_eq!(result, expected);
    }
}
