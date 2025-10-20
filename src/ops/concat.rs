use std::mem::MaybeUninit;

use rten_tensor::prelude::*;
use rten_tensor::{AssumeInit, NdTensorView, Tensor, TensorView};

use smallvec::SmallVec;

use crate::buffer_pool::{AutoReturn, BufferPool};
use crate::operator::{InputList, IntoOpResult, OpError, OpRunContext, Operator, OutputList};
use crate::ops::{map_value, map_value_view, resolve_axis};
use crate::value::{CastError, Value, ValueView};

/// Return the shape formed by concatenating all tensors along a given axis.
fn concatenated_shape<T: Copy>(
    first_shape: &[usize],
    inputs: &[TensorView<T>],
    axis: usize,
) -> Result<SmallVec<[usize; 4]>, OpError> {
    let mut out_shape = SmallVec::from_slice(first_shape);

    for other in inputs {
        let other_shape = other.shape();
        if other_shape.len() != first_shape.len() {
            return Err(OpError::IncompatibleInputShapes(
                "Tensors must have the same number of dimensions",
            ));
        }
        for (d, (first_size, other_size)) in first_shape.iter().zip(other_shape.iter()).enumerate()
        {
            if d != axis && first_size != other_size {
                return Err(OpError::IncompatibleInputShapes(
                    "Dimensions must be the same except for concat axis",
                ));
            } else if d == axis {
                out_shape[axis] += other_size;
            }
        }
    }

    Ok(out_shape)
}

fn typed_inputs<'a, T>(
    inputs: &InputList<'a>,
    _: TensorView<T>,
) -> Result<SmallVec<[TensorView<'a, T>; 4]>, OpError>
where
    TensorView<'a, T>: TryFrom<ValueView<'a>, Error = CastError>,
{
    let mut typed_inputs: SmallVec<_> = SmallVec::with_capacity(inputs.len());
    for input in inputs.iter().flatten() {
        typed_inputs.push(input.try_into()?);
    }
    Ok(typed_inputs)
}

fn concat_impl<T: Copy>(
    pool: &BufferPool,
    out_shape: &[usize],
    axis: usize,
    first_input: &TensorView<T>,
    inputs: &[TensorView<T>],
) -> Result<Tensor<T>, OpError> {
    let mut output = Tensor::with_capacity_in(pool, out_shape, axis);
    for input in std::iter::once(first_input).chain(inputs) {
        output.append(axis, input).expect("should have capacity");
    }
    Ok(output)
}

pub fn concat<T: Copy>(
    pool: &BufferPool,
    inputs: &[TensorView<T>],
    axis: isize,
) -> Result<Tensor<T>, OpError> {
    let axis = resolve_axis(inputs[0].ndim(), axis)?;
    let out_shape = concatenated_shape(inputs[0].shape(), &inputs[1..], axis)?;
    concat_impl(pool, &out_shape, axis, &inputs[0], &inputs[1..])
}

pub fn concat_in_place<T: Copy>(
    pool: &BufferPool,
    mut output: Tensor<T>,
    inputs: &[TensorView<T>],
    axis: isize,
) -> Result<Tensor<T>, OpError> {
    let axis = resolve_axis(output.ndim(), axis)?;
    let out_shape = concatenated_shape(output.shape(), inputs, axis)?;
    if !output.has_capacity(axis, out_shape[axis]) {
        let output = output.auto_return(pool);
        return concat_impl(pool, &out_shape, axis, &output.view(), inputs);
    }

    for input in inputs {
        output.append(axis, input).expect("should have capacity");
    }

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

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let first = inputs.require(0)?;
        map_value_view!(first, first, [FloatTensor, Int32Tensor], {
            let typed_inputs = typed_inputs(inputs, first)?;
            concat(ctx.pool(), &typed_inputs, self.axis).into_op_result()
        })
    }

    fn can_run_in_place(&self) -> bool {
        // This operator can run in place in several cases:
        //
        // - There is only one input
        // - All inputs except the first are empty
        // - Concatenation is being performed along the dimension with the
        //   largest stride, and the tensor's buffer happens to have enough
        //   spare capacity.
        // - Capacity was specifically reserved (via `Tensor::with_capacity`)
        //   by higher-level code which anticipated the concatenation.
        true
    }

    fn run_in_place(&self, input: Value, ctx: &OpRunContext) -> Result<Value, OpError> {
        map_value!(input, input, [FloatTensor, Int32Tensor], {
            let typed_inputs = typed_inputs(ctx.inputs(), input.view())?;
            concat_in_place(ctx.pool(), input, &typed_inputs, self.axis).map(|t| t.into())
        })
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
    unsafe { dest.assume_init() }
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
    // Fast path for when remaining dimensions are not repeated.
    if repeats.iter().all(|n| *n == 1) {
        write_slice(output, input);
        return;
    }

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
            assert!(output.len().is_multiple_of(*repeats));
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
    pool: &BufferPool,
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

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require(0)?;
        let repeats = inputs.require_as(1)?;

        map_value_view!(input, input, [FloatTensor, Int32Tensor], {
            tile(ctx.pool(), input, repeats).into_op_result()
        })
    }

    fn can_run_in_place(&self) -> bool {
        // Tile can run in place if it is a noop, ie. all the repeats are 1.
        true
    }

    fn run_in_place(&self, input: Value, ctx: &OpRunContext) -> Result<Value, OpError> {
        let repeats: NdTensorView<i32, 1> = ctx.inputs().require_as(0)?;

        if repeats.iter().all(|n| *n == 1) {
            return Ok(input);
        }

        map_value!(input, input, [FloatTensor, Int32Tensor], {
            tile(ctx.pool(), input.view(), repeats).map(|t| t.into())
        })
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::Tensor;
    use rten_tensor::prelude::*;
    use rten_tensor::test_util::expect_equal;
    use rten_testing::TestCases;

    use crate::ops::OpError;
    use crate::ops::tests::new_pool;

    use super::{concat, concat_in_place, tile};

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
    fn test_concat_in_place() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();

        let dest = Tensor::with_capacity_in(&pool, &[3, 3], 1);

        // Concatenate within spare capacity.
        let dest =
            concat_in_place(&pool, dest, &[Tensor::from([[1], [2], [3]]).view()], 1).unwrap();
        let dest =
            concat_in_place(&pool, dest, &[Tensor::from([[4], [5], [6]]).view()], 1).unwrap();
        let dest =
            concat_in_place(&pool, dest, &[Tensor::from([[7], [8], [9]]).view()], 1).unwrap();

        assert_eq!(dest.shape(), &[3, 3]);
        assert_eq!(dest, Tensor::from([[1, 4, 7], [2, 5, 8], [3, 6, 9],]));

        // Concatenate beyond the allocated capacity. This should fall back to
        // allocating a new tensor.
        let dest =
            concat_in_place(&pool, dest, &[Tensor::from([[10], [11], [12]]).view()], 1).unwrap();
        assert_eq!(dest.shape(), &[3, 4]);
        assert_eq!(
            dest,
            Tensor::from([[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12],])
        );

        // Shape mismatch along non-concatenation axes.
        let result = concat_in_place(&pool, dest.clone(), &[Tensor::from([[1, 2, 3]]).view()], 1);
        assert!(result.is_err());

        // Dimension count mismatch.
        let result = concat_in_place(&pool, dest.clone(), &[Tensor::from(1).view()], 1);
        assert!(result.is_err());

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
        #[derive(Debug)]
        struct Case {
            input: Tensor<i32>,
            repeats: Tensor<i32>,
            expected: Tensor<i32>,
        }

        let cases = [
            // Empty
            Case {
                input: Tensor::<i32>::zeros(&[3, 4, 5]),
                repeats: Tensor::from([4, 0, 1]),
                expected: Tensor::<i32>::zeros(&[12, 0, 5]),
            },
            // Scalar
            Case {
                input: Tensor::from(5),
                repeats: Tensor::from([] as [i32; 0]),
                expected: Tensor::from(5),
            },
            // 1D tile
            Case {
                input: Tensor::from([1, 2, 3, 4]),
                repeats: Tensor::from([3]),
                expected: Tensor::from([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]),
            },
            // 2D tile
            Case {
                input: Tensor::from([[3]]),
                repeats: Tensor::from([3, 2]),
                expected: Tensor::from([[3, 3], [3, 3], [3, 3]]),
            },
            // Noop tile
            Case {
                input: Tensor::from([1, 2, 3, 4]),
                repeats: Tensor::from([1]),
                expected: Tensor::from([1, 2, 3, 4]),
            },
            // Repeat inner dim of a 2D tensor
            Case {
                input: Tensor::from([[1, 2], [3, 4]]),
                repeats: Tensor::from([1, 2]),
                expected: Tensor::from([[1, 2, 1, 2], [3, 4, 3, 4]]),
            },
            // Repeat outer dim of a 2D tensor
            Case {
                input: Tensor::from([[1, 2], [3, 4]]),
                repeats: Tensor::from([2, 1]),
                expected: Tensor::from([[1, 2], [3, 4], [1, 2], [3, 4]]),
            },
            // Repeat inner and outer dims of a 2D tensor
            Case {
                input: Tensor::from([[1, 2], [3, 4]]),
                repeats: Tensor::from([2, 2]),
                expected: Tensor::from([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]),
            },
        ];

        cases.test_each(|case| {
            let pool = new_pool();
            let Case {
                input,
                repeats,
                expected,
            } = case;

            let result = tile(&pool, input.view(), repeats.nd_view()).unwrap();
            expect_equal(&result, &expected).unwrap();
        });
    }

    #[test]
    fn test_tile_invalid_repeats() {
        #[derive(Debug)]
        struct Case {
            input: Tensor<i32>,
            repeats: Tensor<i32>,
            expected_error: OpError,
        }

        let cases = [
            // Repeats length does not match input ndim.
            Case {
                input: Tensor::from([1, 2, 3]),
                repeats: Tensor::from([1, 2]),
                expected_error: OpError::InvalidValue("invalid repeats"),
            },
            // Negative repeats
            Case {
                input: Tensor::from([1, 2, 3]),
                repeats: Tensor::from([-1]),
                expected_error: OpError::InvalidValue("invalid repeats"),
            },
        ];

        cases.test_each(|case| {
            let pool = new_pool();
            let Case {
                input,
                repeats,
                expected_error,
            } = case;

            let result = tile(&pool, input.view(), repeats.nd_view());
            assert_eq!(result.err().as_ref(), Some(expected_error));
        });
    }
}
