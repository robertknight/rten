use wasnn_tensor::{Iter, Layout, Tensor, TensorLayout, TensorView};

use crate::ops::{Input, InputList, IntoOpResult, OpError, Operator, Output};

enum ChunkSource<'a, T: Copy> {
    Slice(&'a [T]),
    Iter(Iter<'a, T>),
}

/// Reads chunks of a tensor, where each chunk consists of one iteration over
/// N innermost dimensions.
struct TensorChunks<'a, T: Copy> {
    source: ChunkSource<'a, T>,
    chunk_size: usize,
}

impl<'a, T: Copy> TensorChunks<'a, T> {
    fn new(tensor: &'a TensorView<'a, T>, from_dim: usize) -> TensorChunks<'a, T> {
        TensorChunks {
            source: if tensor.is_contiguous() {
                ChunkSource::Slice(tensor.to_data())
            } else {
                ChunkSource::Iter(tensor.iter())
            },
            chunk_size: tensor.shape()[from_dim..].iter().product(),
        }
    }

    /// Return total remaining elements.
    fn remaining_len(&self) -> usize {
        match self.source {
            ChunkSource::Slice(it) => it.len(),
            ChunkSource::Iter(ref it) => it.len(),
        }
    }

    /// Add the next chunk of elements from this tensor to `dest`.
    fn append_next_chunk(&mut self, dest: &mut Vec<T>) {
        match self.source {
            ChunkSource::Slice(ref mut it) => {
                // Take advantage of `Vec::extend`'s fast path for slices.
                let (start, end) = it.split_at(self.chunk_size);
                *it = end;
                dest.extend(start);
            }
            ChunkSource::Iter(ref mut it) => dest.extend(it.by_ref().take(self.chunk_size)),
        }
    }
}

pub fn concat<T: Copy>(inputs: &[TensorView<T>], dim: usize) -> Result<Tensor<T>, OpError> {
    let first_shape = inputs[0].shape();
    if dim >= first_shape.len() {
        return Err(OpError::InvalidValue("dim is larger than input rank"));
    }

    for other in &inputs[1..] {
        let other_shape = other.shape();
        if other_shape.len() != first_shape.len() {
            return Err(OpError::IncompatibleInputShapes(
                "Tensors must have the same number of dimensions",
            ));
        }
        for d in 0..first_shape.len() {
            if d != dim && first_shape[d] != other_shape[d] {
                return Err(OpError::IncompatibleInputShapes(
                    "Dimensions must be the same except for concat dim",
                ));
            }
        }
    }

    let mut out_shape: Vec<_> = first_shape.into();
    for other in &inputs[1..] {
        out_shape[dim] += other.size(dim);
    }
    let mut out_data = Vec::with_capacity(out_shape.iter().product());

    let mut input_iters: Vec<TensorChunks<'_, T>> = inputs
        .iter()
        .map(|tensor| TensorChunks::new(tensor, dim))
        .collect();

    while input_iters.iter().any(|it| it.remaining_len() > 0) {
        for iter in input_iters.iter_mut() {
            iter.append_next_chunk(&mut out_data);
        }
    }

    Ok(Tensor::from_data(&out_shape, out_data))
}

#[derive(Debug)]
pub struct Concat {
    pub dim: usize,
}

impl Operator for Concat {
    fn name(&self) -> &str {
        "Concat"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let first = inputs.require(0)?;
        match first {
            Input::FloatTensor(_) => {
                let mut typed_inputs: Vec<TensorView> = Vec::new();
                for input in inputs.iter() {
                    let tensor: &Tensor<f32> = input.try_into()?;
                    typed_inputs.push(tensor.view());
                }
                concat(&typed_inputs, self.dim).into_op_result()
            }
            Input::IntTensor(_) => {
                let mut typed_inputs: Vec<TensorView<i32>> = Vec::new();
                for input in inputs.iter() {
                    let tensor: &Tensor<i32> = input.try_into()?;
                    typed_inputs.push(tensor.view());
                }
                concat(&typed_inputs, self.dim).into_op_result()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use wasnn_tensor::test_util::expect_equal;
    use wasnn_tensor::{Layout, Tensor};

    use crate::ops::{concat, OpError};

    fn from_slice<T: Clone>(data: &[T]) -> Tensor<T> {
        Tensor::from_data(&[data.len()], data.to_vec())
    }

    #[test]
    fn test_concat() -> Result<(), String> {
        let a = Tensor::from_data(&[2, 2, 1], vec![0.1, 0.2, 0.3, 0.4]);
        let b = Tensor::from_data(&[2, 2, 1], vec![1.0, 2.0, 3.0, 4.0]);

        // Concatenation along the first dimension
        let expected = Tensor::from_data(&[4, 2, 1], vec![0.1, 0.2, 0.3, 0.4, 1.0, 2.0, 3.0, 4.0]);
        let result = concat(&[a.view(), b.view()], 0).unwrap();
        expect_equal(&result, &expected)?;

        // Concatenation along a non-first dimension
        let expected = Tensor::from_data(&[2, 2, 2], vec![0.1, 1.0, 0.2, 2.0, 0.3, 3.0, 0.4, 4.0]);
        let result = concat(&[a.view(), b.view()], 2).unwrap();
        expect_equal(&result, &expected)?;

        // Concatenation with one input
        let result = concat(&[a.view()], 0).unwrap();
        expect_equal(&result, &a)?;

        // Concatenation with more than two inputs
        let result = concat(&[a.view(), b.view(), a.view()], 0).unwrap();
        assert_eq!(result.shape(), &[6, 2, 1]);

        // Concatentation with some empty inputs
        let a = from_slice(&[1, 2, 3]);
        let b = from_slice(&[]);
        let c = from_slice(&[4, 5, 6]);
        let result = concat(&[a.view(), b.view(), c.view()], 0).unwrap();
        assert_eq!(result.shape(), &[6]);
        assert_eq!(result.data(), &[1, 2, 3, 4, 5, 6]);

        Ok(())
    }

    #[test]
    fn test_concat_invalid_inputs() {
        // Invalid `dim` attribute
        let input = from_slice(&[1, 2, 3]);
        let result = concat(&[input.view(), input.view()], 1);
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("dim is larger than input rank"))
        );

        // Shape mismatch
        let a = Tensor::<f32>::zeros(&[1]);
        let b = Tensor::<f32>::zeros(&[1, 2]);
        let result = concat(&[a.view(), b.view()], 0);
        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes(
                "Tensors must have the same number of dimensions"
            ))
        );

        // Shape mismatch in non-`dim` dimension
        let a = Tensor::<f32>::zeros(&[5, 10]);
        let b = Tensor::<f32>::zeros(&[5, 11]);
        let result = concat(&[a.view(), b.view()], 0);
        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes(
                "Dimensions must be the same except for concat dim"
            ))
        );
    }
}
