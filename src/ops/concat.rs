use std::mem::MaybeUninit;

use rten_tensor::prelude::*;
use rten_tensor::{NdTensorView, Tensor, TensorView};

use crate::ops::{resolve_axis, Input, InputList, IntoOpResult, OpError, Operator, Output};
use crate::static_dims;
use crate::tensor_pool::{AutoReturn, TensorPool};

pub fn concat<T: Copy>(
    pool: &TensorPool,
    inputs: &[TensorView<T>],
    axis: isize,
) -> Result<Tensor<T>, OpError> {
    let first_shape = inputs[0].shape();
    let axis = resolve_axis(first_shape.len(), axis)?;

    for other in &inputs[1..] {
        let other_shape = other.shape();
        if other_shape.len() != first_shape.len() {
            return Err(OpError::IncompatibleInputShapes(
                "Tensors must have the same number of dimensions",
            ));
        }
        for d in 0..first_shape.len() {
            if d != axis && first_shape[d] != other_shape[d] {
                return Err(OpError::IncompatibleInputShapes(
                    "Dimensions must be the same except for concat axis",
                ));
            }
        }
    }

    let mut out_shape: Vec<_> = first_shape.into();
    for other in &inputs[1..] {
        out_shape[axis] += other.size(axis);
    }

    let mut output = Tensor::uninit_in(pool, &out_shape);

    let mut n_init = 0;
    let mut remainder = output.view_mut();
    for input in inputs {
        let (left, right) = remainder.split_at_mut(axis, input.size(axis));
        left.init_from(input);
        remainder = right;
        n_init += input.len();
    }

    assert!(n_init == output.len());
    let output = unsafe { output.assume_init() };
    Ok(output)
}

#[derive(Debug)]
pub struct Concat {
    pub axis: isize,
}

impl Operator for Concat {
    fn name(&self) -> &str {
        "Concat"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let first = inputs.require(0)?;
        match first {
            Input::FloatTensor(_) => {
                let mut typed_inputs: Vec<TensorView> = Vec::new();
                for input in inputs.iter() {
                    typed_inputs.push(input.try_into()?);
                }
                concat(pool, &typed_inputs, self.axis).into_op_result()
            }
            Input::IntTensor(_) => {
                let mut typed_inputs: Vec<TensorView<i32>> = Vec::new();
                for input in inputs.iter() {
                    typed_inputs.push(input.try_into()?);
                }
                concat(pool, &typed_inputs, self.axis).into_op_result()
            }
        }
    }
}

/// Copied from `std::MaybeUninit::write_slice` in nightly std.
fn write_slice<'a, T>(dest: &'a mut [MaybeUninit<T>], src: &[T]) -> &'a mut [T]
where
    T: Copy,
{
    // SAFETY: &[T] and &[MaybeUninit<T>] have the same layout
    let uninit_src: &[MaybeUninit<T>] = unsafe { std::mem::transmute(src) };

    dest.copy_from_slice(uninit_src);

    // SAFETY: Valid elements have just been copied into `this` so it is initialized
    unsafe { std::mem::transmute(dest) }
}

/// Recursively tile (ie. repeatly copy) chunks of `input` to `output`.
///
/// `input_shape` and `repeats` are equal-length slices specifying the size
/// of each dimension and the number of times to repeat that dim. All input
/// dim sizes and repeats must be >= 1 (ie. input and output must be non-empty).
fn tile_inner<T: Copy>(
    input: &[T],
    output: &mut [MaybeUninit<T>],
    input_shape: &[usize],
    repeats: &[usize],
) {
    let mut n_init = 0;
    match (input_shape, repeats) {
        ([size], [repeats]) => {
            assert!(input.len() == *size);
            assert!(input.len() * repeats == output.len());
            for out_chunk in output.chunks_mut(input.len()) {
                n_init += write_slice(out_chunk, input).len();
            }
        }
        ([size, inner_size @ ..], [repeats, inner_repeats @ ..]) => {
            assert!(output.len() % repeats == 0);
            let out_chunk_len = output.len() / repeats;
            let inner_input_len = input.len() / size;
            let inner_output_len = out_chunk_len / size;

            // Tile the current dimension.
            for out_chunk in output.chunks_mut(out_chunk_len) {
                // Tile the inner dimensions.
                for (inner_input, inner_output) in input
                    .chunks(inner_input_len)
                    .zip(out_chunk.chunks_mut(inner_output_len))
                {
                    tile_inner(inner_input, inner_output, inner_size, inner_repeats);
                    n_init += inner_output.len();
                }
            }
        }
        ([], []) => {
            // Input is a scalar.
            n_init += write_slice(output, input).len();
        }
        _ => panic!("input_shape.len() != repeats.len()"),
    }
    assert!(n_init == output.len());
}

pub fn tile<T: Copy>(
    pool: &TensorPool,
    input: TensorView<T>,
    repeats: NdTensorView<i32, 1>,
) -> Result<Tensor<T>, OpError> {
    if repeats.size(0) != input.ndim() || repeats.iter().any(|n| *n < 0) {
        return Err(OpError::InvalidValue("invalid repeats"));
    }

    let repeats: Vec<usize> = repeats.iter().map(|r| *r as usize).collect();
    let out_shape: Vec<_> = input
        .shape()
        .iter()
        .zip(repeats.iter())
        .map(|(size, repeat)| size * repeat)
        .collect();
    let mut output = Tensor::uninit_in(pool, &out_shape);

    if !output.is_empty() {
        tile_inner(
            input
                .to_contiguous_in(pool)
                .auto_return(pool)
                .data()
                .unwrap(),
            output.data_mut().unwrap(),
            input.shape(),
            &repeats,
        );
    }

    // Safety - `tile_inner` initialized all output elements, or the tensor
    // is empty.
    let output = unsafe { output.assume_init() };

    Ok(output)
}

#[derive(Debug)]
pub struct Tile {}

impl Operator for Tile {
    fn name(&self) -> &str {
        "Tile"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let repeats = inputs.require_as::<i32>(1)?;
        let repeats = static_dims!(repeats, 1)?;

        match input {
            Input::IntTensor(input) => tile(pool, input, repeats).into_op_result(),
            Input::FloatTensor(input) => tile(pool, input, repeats).into_op_result(),
        }
    }

    fn can_run_in_place(&self) -> bool {
        // Tile can run in place if it is a noop, ie. all the repeats are 1.
        true
    }

    fn run_in_place(
        &self,
        pool: &TensorPool,
        output: Output,
        inputs: InputList,
    ) -> Result<Output, OpError> {
        let repeats = inputs.require_as::<i32>(0)?;
        let repeats = static_dims!(repeats, 1)?;

        if repeats.iter().all(|n| *n == 1) {
            return Ok(output);
        }

        match output {
            Output::IntTensor(input) => tile(pool, input.view(), repeats).map(|t| t.into()),
            Output::FloatTensor(input) => tile(pool, input.view(), repeats).map(|t| t.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::prelude::*;
    use rten_tensor::test_util::expect_equal;
    use rten_tensor::{tensor, Tensor};

    use crate::ops::tests::new_pool;
    use crate::ops::{concat, tile, OpError};

    fn from_slice<T: Clone>(data: &[T]) -> Tensor<T> {
        Tensor::from_data(&[data.len()], data.to_vec())
    }

    #[test]
    fn test_concat() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();
        let a = Tensor::from_data(&[2, 2, 1], vec![0.1, 0.2, 0.3, 0.4]);
        let b = Tensor::from_data(&[2, 2, 1], vec![1.0, 2.0, 3.0, 4.0]);

        // Concatenation along the first dimension
        let expected = Tensor::from_data(&[4, 2, 1], vec![0.1, 0.2, 0.3, 0.4, 1.0, 2.0, 3.0, 4.0]);
        let result = concat(&pool, &[a.view(), b.view()], 0).unwrap();
        expect_equal(&result, &expected)?;

        // Concatenation along a non-first dimension
        let expected = Tensor::from_data(&[2, 2, 2], vec![0.1, 1.0, 0.2, 2.0, 0.3, 3.0, 0.4, 4.0]);
        let result = concat(&pool, &[a.view(), b.view()], 2).unwrap();
        expect_equal(&result, &expected)?;

        // Concatenation with one input
        let result = concat(&pool, &[a.view()], 0).unwrap();
        expect_equal(&result, &a)?;

        // Concatenation with more than two inputs
        let result = concat(&pool, &[a.view(), b.view(), a.view()], 0).unwrap();
        assert_eq!(result.shape(), &[6, 2, 1]);

        // Concatentation with some empty inputs
        let a = from_slice(&[1, 2, 3]);
        let b = from_slice(&[]);
        let c = from_slice(&[4, 5, 6]);
        let result = concat(&pool, &[a.view(), b.view(), c.view()], 0).unwrap();
        assert_eq!(result.shape(), &[6]);
        assert_eq!(result.to_vec(), &[1, 2, 3, 4, 5, 6]);

        Ok(())
    }

    #[test]
    fn test_concat_invalid_inputs() {
        let pool = new_pool();

        // Invalid `dim` attribute
        let input = from_slice(&[1, 2, 3]);
        let result = concat(&pool, &[input.view(), input.view()], 1);
        assert_eq!(result.err(), Some(OpError::InvalidValue("Axis is invalid")));

        // Shape mismatch
        let a = Tensor::<f32>::zeros(&[1]);
        let b = Tensor::<f32>::zeros(&[1, 2]);
        let result = concat(&pool, &[a.view(), b.view()], 0);
        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes(
                "Tensors must have the same number of dimensions"
            ))
        );

        // Shape mismatch in non-`dim` dimension
        let a = Tensor::<f32>::zeros(&[5, 10]);
        let b = Tensor::<f32>::zeros(&[5, 11]);
        let result = concat(&pool, &[a.view(), b.view()], 0);
        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes(
                "Dimensions must be the same except for concat axis"
            ))
        );
    }

    #[test]
    fn test_tile() {
        let pool = new_pool();

        // Empty
        let input = Tensor::<f32>::zeros(&[3, 4, 5]);
        let repeats = tensor!([4, 0, 1]);
        let result = tile(&pool, input.view(), repeats.nd_view()).unwrap();
        assert_eq!(result.shape(), &[12, 0, 5]);
        assert!(result.is_empty());

        // Scalar
        let input = tensor!(5.);
        let repeats = tensor!([]);
        let result = tile(&pool, input.view(), repeats.nd_view()).unwrap();
        assert_eq!(result, tensor!(5.));

        // 1D tile
        let input = tensor!([1, 2, 3, 4]);
        let repeats = tensor!([3]);
        let result = tile(&pool, input.view(), repeats.nd_view()).unwrap();
        assert_eq!(result, tensor!([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]));

        // 2D tile
        let input = tensor!((1, 1); [3.]);
        let repeats = tensor!([3, 2]);
        let result = tile(&pool, input.view(), repeats.nd_view()).unwrap();
        assert_eq!(
            result,
            tensor!(
                (3, 2);
                [
                    3., 3., //
                    3., 3., //
                    3., 3. //
                ]
            )
        );

        // Noop tile
        let input = tensor!([1, 2, 3, 4]);
        let repeats = tensor!([1]);
        let result = tile(&pool, input.view(), repeats.nd_view()).unwrap();
        assert_eq!(input, result);

        // Repeat inner dim of a 2D tensor
        let input = Tensor::from([[1, 2], [3, 4]]);
        let repeats = tensor!([1, 2]);
        let result = tile(&pool, input.view(), repeats.nd_view()).unwrap();
        assert_eq!(result, Tensor::from([[1, 2, 1, 2], [3, 4, 3, 4]]));

        // Repeat outer dim of a 2D tensor
        let input = Tensor::from([[1, 2], [3, 4]]);
        let repeats = tensor!([2, 1]);
        let result = tile(&pool, input.view(), repeats.nd_view()).unwrap();
        assert_eq!(result, Tensor::from([[1, 2], [3, 4], [1, 2], [3, 4]]));

        // Repeat inner and outer dims of a 2D tensor
        let input = Tensor::from([[1, 2], [3, 4]]);
        let repeats = tensor!([2, 2]);
        let result = tile(&pool, input.view(), repeats.nd_view()).unwrap();
        assert_eq!(
            result,
            Tensor::from([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]])
        );
    }

    #[test]
    fn test_tile_invalid_repeats() {
        let pool = new_pool();

        // Repeats length does not match input ndim.
        let input = tensor!([1, 2, 3]);
        let repeats = tensor!([1, 2]);
        let result = tile(&pool, input.view(), repeats.nd_view());
        assert_eq!(result, Err(OpError::InvalidValue("invalid repeats")));

        // Negative repeats
        let input = tensor!([1, 2, 3]);
        let repeats = tensor!([-1]);
        let result = tile(&pool, input.view(), repeats.nd_view());
        assert_eq!(result, Err(OpError::InvalidValue("invalid repeats")));
    }
}
