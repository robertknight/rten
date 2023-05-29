use crate::ops::{Input, InputList, IntoOpResult, OpError, Operator, Output};
use crate::tensor::{Tensor, TensorLayout, TensorView};

/// Gather elements from `input` specified by `indices`.
///
/// See https://onnx.ai/onnx/operators/onnx__Gather.html. Per the ONNX spec this
/// is very similar to `numpy.take`. See
/// https://numpy.org/doc/stable/reference/generated/numpy.take.html for
/// additional explanation.
pub fn gather<T: Copy + Default>(
    input: TensorView<T>,
    axis: usize,
    indices: TensorView<i32>,
) -> Result<Tensor<T>, OpError> {
    if axis >= input.ndim() {
        return Err(OpError::InvalidValue("`axis` is out of range"));
    }
    for index in indices.iter() {
        if index < 0 || index >= input.shape()[axis] as i32 {
            return Err(OpError::InvalidValue("Entry in `indices` is out of range"));
        }
    }

    let out_shape = [
        &input.shape()[0..axis],
        indices.shape(),
        &input.shape()[axis + 1..],
    ]
    .concat();
    let mut output = Tensor::<T>::zeros(&out_shape);
    let mut out_index_iter = output.indices();
    let mut in_index = vec![0; input.ndim()];

    while let Some(out_index) = out_index_iter.next() {
        if out_index.is_empty() {
            // If the output index is empty, this means we are indexing a
            // 1D vector with a scalar.
            in_index[axis] = indices.item().unwrap_or(0) as usize;
        } else {
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
        }
        output[out_index] = input[&in_index[..]];
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

#[cfg(test)]
mod tests {
    use crate::ops::{gather, OpError};
    use crate::rng::XorShiftRng;
    use crate::tensor::{from_data, rand, Tensor, TensorLayout};
    use crate::test_util::expect_equal;

    #[test]
    fn test_gather_scalar() {
        let input = Tensor::from_vec(vec![1, 20, 30]);
        for i in 0..input.len() {
            let indices = Tensor::from_scalar(i as i32);
            let result = gather(input.view(), 0, indices.view()).unwrap();
            assert_eq!(result.item(), Some(input[[i]]))
        }
    }

    #[test]
    fn test_gather() -> Result<(), String> {
        // Test case shrunk down from a small BERT model where `gather` is used
        // to lookup up embeddings.
        let mut rng = XorShiftRng::new(1234);
        let input = rand(&[128, 10], &mut rng);
        let indices = from_data(&[2, 2], vec![2, 5, 8, 50]);
        let result = gather(input.view(), 0, indices.view()).unwrap();
        assert_eq!(result.shape(), &[2, 2, 10]);

        // Test case #1 from ONNX spec.
        let input = from_data(&[3, 2], vec![1.0, 1.2, 2.3, 3.4, 4.5, 5.7]);
        let indices = from_data(&[2, 2], vec![0, 1, 1, 2]);
        let expected = from_data(&[2, 2, 2], vec![1.0, 1.2, 2.3, 3.4, 2.3, 3.4, 4.5, 5.7]);
        let result = gather(input.view(), 0, indices.view()).unwrap();
        expect_equal(&result, &expected)?;

        // Test case #2 from ONNX spec.
        let input = from_data(&[3, 3], vec![1.0, 1.2, 1.9, 2.3, 3.4, 3.9, 4.5, 5.7, 5.9]);
        let indices = from_data(&[1, 2], vec![0, 2]);
        let expected = from_data(&[3, 1, 2], vec![1.0, 1.9, 2.3, 3.9, 4.5, 5.9]);
        let result = gather(input.view(), 1, indices.view()).unwrap();
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_gather_invalid_inputs() {
        let mut rng = XorShiftRng::new(1234);
        let input = rand(&[128, 10], &mut rng);
        let indices = from_data(&[2, 2], vec![2, 5, 8, 50]);
        let result = gather(input.view(), 5, indices.view());
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("`axis` is out of range"))
        );

        let indices = from_data(&[2, 2], vec![2, 5, 8, 130]);
        let result = gather(input.view(), 0, indices.view());
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Entry in `indices` is out of range"))
        );
    }
}
