//! Operators which query or change the shape of a tensor, or copy/move/reorder
//! elements.
use std::any::Any;
use std::iter::zip;

use rten_tensor::prelude::*;
use rten_tensor::{is_valid_permutation, NdTensorView, Tensor, TensorView};
use smallvec::SmallVec;

use crate::ops::binary_elementwise::{broadcast_shapes, fast_broadcast_cycles_repeats};
use crate::ops::{
    resolve_axes, resolve_axis, Input, InputList, IntoOpResult, OpError, Operator, Output,
};
use crate::static_dims;
use crate::tensor_pool::TensorPool;

/// Return the tensor shape resulting from broadcasting `input_shape` with `shape`.
fn expand_output_shape(
    input_shape: &[usize],
    shape: &NdTensorView<i32, 1>,
) -> Result<Vec<usize>, OpError> {
    let shape_vec: Vec<_> = shape.iter().map(|el| *el as usize).collect();
    broadcast_shapes(input_shape, &shape_vec).ok_or(OpError::IncompatibleInputShapes(
        "Cannot broadcast input with target shape",
    ))
}

/// Broadcast `input` to `out_shape`. This assumes that `out_shape` has already
/// been verified to be a valid broadcast target.
pub(crate) fn expand_to<T: Any + Copy>(
    pool: &TensorPool,
    input: TensorView<T>,
    out_shape: &[usize],
) -> Tensor<T> {
    let out_len = out_shape.iter().product();

    match (
        input.data(),
        fast_broadcast_cycles_repeats(input.shape(), out_shape),
    ) {
        // Fast path for common case of contiguous input and broadcast that can
        // be performed using cycle + repeat.
        (Some(in_data), Some((cycles, repeats))) => {
            assert!(out_len == input.len() * cycles * repeats);

            let mut out_data: Vec<T> = pool.alloc_vec(out_len);
            let mut out_ptr = out_data.as_mut_ptr();
            for _ in 0..cycles {
                if repeats == 1 {
                    // Super-fast path for cycling only.
                    unsafe {
                        std::ptr::copy_nonoverlapping(in_data.as_ptr(), out_ptr, in_data.len());
                        out_ptr = out_ptr.add(in_data.len());
                    }
                } else {
                    for el in in_data.iter() {
                        for _ in 0..repeats {
                            unsafe {
                                *out_ptr = *el;
                                out_ptr = out_ptr.add(1);
                            }
                        }
                    }
                }
            }
            // Safety: We have initialized all output elements.
            unsafe { out_data.set_len(out_len) };

            Tensor::from_data(out_shape, out_data)
        }
        _ => input.broadcast(out_shape).to_tensor_in(pool),
    }
}

pub fn expand<T: Any + Copy>(
    pool: &TensorPool,
    input: TensorView<T>,
    shape: &NdTensorView<i32, 1>,
) -> Result<Tensor<T>, OpError> {
    let out_shape = expand_output_shape(input.shape(), shape)?;
    Ok(expand_to(pool, input, &out_shape))
}

#[derive(Debug)]
pub struct Expand {}

impl Operator for Expand {
    fn name(&self) -> &str {
        "Expand"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let shape = inputs.require_as(1)?;
        let shape = static_dims!(shape, 1)?;

        match input {
            Input::FloatTensor(input) => expand(pool, input, &shape).into_op_result(),
            Input::IntTensor(input) => expand(pool, input, &shape).into_op_result(),
        }
    }

    fn can_run_in_place(&self) -> bool {
        // Expand can run in place if it is a noop, ie. if the broadcasted
        // shape is the same as the input shape.
        true
    }

    fn run_in_place(
        &self,
        pool: &TensorPool,
        input: Output,
        inputs: InputList,
    ) -> Result<Output, OpError> {
        let shape = inputs.require_as(0)?;
        let shape = static_dims!(shape, 1)?;

        let out_shape = expand_output_shape(input.shape(), &shape)?;
        if input.shape() == out_shape {
            return Ok(input);
        }

        let output: Output = match input {
            Output::FloatTensor(input) => expand_to(pool, input.view(), &out_shape).into(),
            Output::IntTensor(input) => expand_to(pool, input.view(), &out_shape).into(),
        };
        Ok(output)
    }
}

fn flattened_shape(shape: &[usize], axis: isize) -> Result<[usize; 2], OpError> {
    let resolved_axis = resolve_axis(shape.len(), axis)?;
    let outer_size = shape.iter().take(resolved_axis).product();
    let inner_size = shape.iter().skip(resolved_axis).product();
    Ok([outer_size, inner_size])
}

pub fn flatten<T: Any + Copy>(
    pool: &TensorPool,
    input: TensorView<T>,
    axis: isize,
) -> Result<Tensor<T>, OpError> {
    let shape = flattened_shape(input.shape(), axis)?;
    let mut output = input.to_tensor_in(pool);
    output.reshape(&shape);
    Ok(output)
}

pub fn flatten_in_place<T: Any + Copy>(
    pool: &TensorPool,
    input: &mut Tensor<T>,
    axis: isize,
) -> Result<(), OpError> {
    let shape = flattened_shape(input.shape(), axis)?;
    input.reshape_in(pool, &shape);
    Ok(())
}

#[derive(Debug)]
pub struct Flatten {
    pub axis: isize,
}

impl Operator for Flatten {
    fn name(&self) -> &str {
        "Flatten"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;

        match input {
            Input::FloatTensor(input) => flatten(pool, input, self.axis).into_op_result(),
            Input::IntTensor(input) => flatten(pool, input, self.axis).into_op_result(),
        }
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(
        &self,
        pool: &TensorPool,
        input: Output,
        _: InputList,
    ) -> Result<Output, OpError> {
        match input {
            Output::IntTensor(mut output) => {
                flatten_in_place(pool, &mut output, self.axis)?;
                Ok(output.into())
            }
            Output::FloatTensor(mut output) => {
                flatten_in_place(pool, &mut output, self.axis)?;
                Ok(output.into())
            }
        }
    }
}

/// Compute the target shape for a reshape operation, given the shape of the
/// input tensor and a target `shape` which may contain a "-1" entry to indicate
/// a dimension whose size should be inferred.
///
/// Handling of zeros in the target shape depends on `allow_zero`. If false,
/// the corresponding input dimension is copied, otherwise the zero is
/// preserved in the output shape.
fn resolve_shape(
    input_shape: &[usize],
    shape: &NdTensorView<i32, 1>,
    allow_zero: bool,
) -> Result<Vec<usize>, OpError> {
    // If exactly one of the new shape's dimensions is -1, infer the size
    // from the input length and the sizes of the other dimensions.
    let mut unspecified_dim = None;
    let mut specified_dims_size = 1;
    for (dim, &size) in shape.iter().enumerate() {
        if size < -1 {
            return Err(OpError::InvalidValue("Invalid dimension size in shape"));
        } else if size == 0 && !allow_zero {
            if dim >= input_shape.len() {
                return Err(OpError::InvalidValue(
                    "Zero dim has no corresponding input dim",
                ));
            }
            specified_dims_size *= input_shape[dim];
        } else if size != -1 {
            specified_dims_size *= size as usize;
        } else if unspecified_dim.is_some() {
            return Err(OpError::InvalidValue(
                "Multiple dimensions in new shape set to -1",
            ));
        } else {
            unspecified_dim = Some(dim);
        }
    }

    let input_len = input_shape.iter().product();
    let (unspecified_dim_size, remainder) = match input_len {
        0 => (0, 0),
        _ => {
            if specified_dims_size == 0 {
                // If `specified_dims_size` is zero but `input_len` is non-zero,
                // this means that the target shape doesn't match the input length.
                //
                // Return a non-zero remainder here to cause the appropriate
                // error to be returned.
                (0, 1)
            } else {
                (
                    input_len / specified_dims_size,
                    input_len % specified_dims_size,
                )
            }
        }
    };

    if remainder != 0 {
        return Err(OpError::InvalidValue(
            "Input length must be a multiple of specified dimensions",
        ));
    }

    Ok(shape
        .iter()
        .enumerate()
        .map(|(dim, &size)| match size {
            -1 => unspecified_dim_size,
            0 if !allow_zero => input_shape[dim],
            valid => valid as usize,
        })
        .collect())
}

pub fn reshape<T: Any + Copy>(
    pool: &TensorPool,
    input: TensorView<T>,
    shape: &NdTensorView<i32, 1>,
    allow_zero: bool,
) -> Result<Tensor<T>, OpError> {
    let out_shape = resolve_shape(input.shape(), shape, allow_zero)?;
    let mut output = pool.alloc(input.shape()).init_from(&input);
    output.reshape(&out_shape);
    Ok(output)
}

pub fn reshape_in_place<T: Any + Copy>(
    pool: &TensorPool,
    input: &mut Tensor<T>,
    shape: &NdTensorView<i32, 1>,
    allow_zero: bool,
) -> Result<(), OpError> {
    let out_shape = resolve_shape(input.shape(), shape, allow_zero)?;
    input.reshape_in(pool, &out_shape);
    Ok(())
}

#[derive(Debug)]
pub struct Reshape {
    pub allow_zero: bool,
}

impl Operator for Reshape {
    fn name(&self) -> &str {
        "Reshape"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let shape = inputs.require_as(1)?;
        let shape = static_dims!(shape, 1)?;

        match input {
            Input::IntTensor(t) => reshape(pool, t, &shape, self.allow_zero).into_op_result(),
            Input::FloatTensor(t) => reshape(pool, t, &shape, self.allow_zero).into_op_result(),
        }
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(
        &self,
        pool: &TensorPool,
        input: Output,
        other: InputList,
    ) -> Result<Output, OpError> {
        let shape = other.require_as(0)?;
        let shape = static_dims!(shape, 1)?;

        match input {
            Output::IntTensor(mut output) => {
                reshape_in_place(pool, &mut output, &shape, self.allow_zero)?;
                Ok(output.into())
            }
            Output::FloatTensor(mut output) => {
                reshape_in_place(pool, &mut output, &shape, self.allow_zero)?;
                Ok(output.into())
            }
        }
    }
}

#[derive(Debug)]
pub struct Shape {}

impl Operator for Shape {
    fn name(&self) -> &str {
        "Shape"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;

        // Allocate output from pool for consistency with other operators,
        // even though the buffer is tiny, so there is no performance benefit.
        let mut data = pool.alloc_vec(input.ndim());
        data.extend(input.shape().iter().map(|&el| el as i32));

        let shape = Tensor::from_data(&[input.ndim()], data);
        shape.into_op_result()
    }
}

#[derive(Debug)]
pub struct Size {}

impl Operator for Size {
    fn name(&self) -> &str {
        "Size"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let len = input.len() as i32;

        // Allocate output from pool for consistency with other operators,
        // even though the buffer is tiny, so there is no performance benefit.
        let mut output = pool.alloc_zeroed([]);
        output[[]] = len;

        output.into_op_result()
    }
}

pub fn squeeze_in_place<T: Clone>(
    input: &mut Tensor<T>,
    axes: Option<NdTensorView<i32, 1>>,
) -> Result<(), OpError> {
    let axes = axes
        .map(|axes| resolve_axes(input.ndim(), axes.iter()))
        .transpose()?;
    if let Some(ref axes) = axes {
        for &axis in axes.iter() {
            if axis >= input.ndim() {
                return Err(OpError::InvalidValue("Axis is invalid"));
            }
            if input.size(axis) != 1 {
                return Err(OpError::InvalidValue(
                    "Can only remove dimensions of size 1",
                ));
            }
        }
    }

    let new_shape: SmallVec<[usize; 5]> = input
        .shape()
        .iter()
        .enumerate()
        .filter(|(dim, &size)| {
            if let Some(ref axes) = axes {
                !axes.contains(dim)
            } else {
                size > 1
            }
        })
        .map(|(_, &size)| size)
        .collect();
    input.reshape(&new_shape);
    Ok(())
}

pub fn squeeze<T: Any + Copy>(
    pool: &TensorPool,
    input: TensorView<T>,
    axes: Option<NdTensorView<i32, 1>>,
) -> Result<Tensor<T>, OpError> {
    let mut output = input.to_tensor_in(pool);
    squeeze_in_place(&mut output, axes)?;
    Ok(output)
}

#[derive(Debug)]
pub struct Squeeze {}

impl Operator for Squeeze {
    fn name(&self) -> &str {
        "Squeeze"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let axes = inputs.get_as(1)?;
        let axes = axes.map(|axes| static_dims!(axes, 1)).transpose()?;

        match input {
            Input::FloatTensor(t) => squeeze(pool, t, axes).into_op_result(),
            Input::IntTensor(t) => squeeze(pool, t, axes).into_op_result(),
        }
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(
        &self,
        _pool: &TensorPool,
        input: Output,
        other: InputList,
    ) -> Result<Output, OpError> {
        let axes = other.get_as(0)?;
        let axes = axes.map(|axes| static_dims!(axes, 1)).transpose()?;

        let result = match input {
            Output::FloatTensor(mut t) => {
                squeeze_in_place(&mut t, axes)?;
                t.into()
            }
            Output::IntTensor(mut t) => {
                squeeze_in_place(&mut t, axes)?;
                t.into()
            }
        };
        Ok(result)
    }
}

pub fn transpose<T: Any + Copy>(
    pool: &TensorPool,
    input: TensorView<T>,
    permutation: Option<&[usize]>,
) -> Result<Tensor<T>, OpError> {
    let mut transposed = input.view();
    match permutation {
        Some(order) => {
            if !is_valid_permutation(input.ndim(), order) {
                return Err(OpError::InvalidValue("Permutation is invalid"));
            }
            transposed.permute(order)
        }
        None => {
            transposed.transpose();
        }
    };
    let output = pool.alloc(transposed.shape());
    Ok(output.init_from(&transposed))
}

#[derive(Debug)]
pub struct Transpose {
    /// The order of the transposed dimensions. If ommitted, the dimensions
    /// are reversed.
    pub perm: Option<Vec<usize>>,
}

impl Operator for Transpose {
    fn name(&self) -> &str {
        "Transpose"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let perm_slice = self.perm.as_deref();
        match input {
            Input::FloatTensor(input) => transpose(pool, input, perm_slice).into_op_result(),
            Input::IntTensor(input) => transpose(pool, input, perm_slice).into_op_result(),
        }
    }
}

pub fn unsqueeze_in_place<T: Clone>(
    mut input: Tensor<T>,
    axes: &NdTensorView<i32, 1>,
) -> Result<Tensor<T>, OpError> {
    let mut new_shape: SmallVec<[usize; 5]> = input.shape().iter().copied().collect();
    let mut sorted_axes = resolve_axes(input.ndim() + axes.len(), axes.iter())?;
    sorted_axes.sort();

    let axes_unique =
        zip(sorted_axes.iter().skip(1), sorted_axes.iter()).all(|(prev, current)| prev != current);
    if !axes_unique {
        return Err(OpError::InvalidValue("Axes must be unique"));
    }

    for axis in sorted_axes {
        new_shape.insert(axis, 1);
    }

    input.make_contiguous();
    input.reshape(&new_shape);

    Ok(input)
}

pub fn unsqueeze<T: Any + Copy>(
    pool: &TensorPool,
    input: TensorView<T>,
    axes: &NdTensorView<i32, 1>,
) -> Result<Tensor<T>, OpError> {
    unsqueeze_in_place(input.to_tensor_in(pool), axes)
}

#[derive(Debug)]
pub struct Unsqueeze {}

impl Operator for Unsqueeze {
    fn name(&self) -> &str {
        "Unsqueeze"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let axes = inputs.require_as(1)?;
        let axes = static_dims!(axes, 1)?;

        match input {
            Input::FloatTensor(input) => unsqueeze(pool, input, &axes).into_op_result(),
            Input::IntTensor(input) => unsqueeze(pool, input, &axes).into_op_result(),
        }
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(
        &self,
        _pool: &TensorPool,
        output: Output,
        inputs: InputList,
    ) -> Result<Output, OpError> {
        let axes = inputs.require_as(0)?;
        let axes = static_dims!(axes, 1)?;

        match output {
            Output::FloatTensor(t) => unsqueeze_in_place(t, &axes).map(Output::FloatTensor),
            Output::IntTensor(t) => unsqueeze_in_place(t, &axes).map(Output::IntTensor),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_bench::run_bench;
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::expect_equal;
    use rten_tensor::{ndtensor, tensor, Tensor};

    use crate::ops::layout::{
        expand, flatten, reshape, reshape_in_place, squeeze, squeeze_in_place, transpose,
        unsqueeze, Reshape, Shape, Size,
    };
    use crate::ops::tests::new_pool;
    use crate::ops::{OpError, Operator};

    #[test]
    fn test_expand() {
        let pool = new_pool();

        // Broadcast scalar
        let input = tensor!(5.);
        let shape = ndtensor!([2, 2]);
        let expected = Tensor::from_data(&[2, 2], vec![5., 5., 5., 5.]);
        let result = expand(&pool, input.view(), &shape.view()).unwrap();
        assert_eq!(&result, &expected);

        // Broadcast that changes dim count
        let input = Tensor::from_data(&[3, 1], (0..3).collect::<Vec<_>>());
        let shape = ndtensor!([2, 3, 1]);
        let result = expand(&pool, input.view(), &shape.view()).unwrap();
        assert_eq!(result.shape(), &[2, 3, 1]);

        // Broadcast that uses dimensions from both the input shape and target
        // shape in the output shape.
        let input = Tensor::from_data(&[3, 1], (0..3).collect::<Vec<_>>());
        let shape = ndtensor!([2, 1, 6]);
        let result = expand(&pool, input.view(), &shape.view()).unwrap();
        assert_eq!(result.shape(), &[2, 3, 6]);

        // Broadcast that does not change dim count
        let input = Tensor::from_data(&[3, 1], (0..3).collect::<Vec<_>>());
        let shape = ndtensor!([3, 4]);
        let result = expand(&pool, input.view(), &shape.view()).unwrap();
        assert_eq!(result.shape(), &[3, 4]);

        // Broadcast of leading and trailing dims
        let input = tensor!((1, 2, 1); [1, 2]);
        let shape = ndtensor!([2, 2, 2]);
        let result = expand(&pool, input.view(), &shape.view()).unwrap();
        assert_eq!(result.shape(), &[2, 2, 2]);
        assert_eq!(result.to_vec(), &[1, 1, 2, 2, 1, 1, 2, 2]);

        // Broadcast of inner dim
        let input = tensor!((2, 1, 2); [1, 2, 3, 4]);
        let shape = ndtensor!([2, 2, 2]);
        let result = expand(&pool, input.view(), &shape.view()).unwrap();
        assert_eq!(result.shape(), &[2, 2, 2]);
        assert_eq!(result.to_vec(), &[1, 2, 1, 2, 3, 4, 3, 4]);
    }

    #[test]
    fn test_expand_invalid_inputs() {
        let pool = new_pool();

        // Invalid broadcast shape
        let input = tensor!([1, 2, 3]);
        let shape = ndtensor!([2, 2]);
        let result = expand(&pool, input.view(), &shape.view());
        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes(
                "Cannot broadcast input with target shape"
            ))
        );
    }

    #[test]
    fn test_flatten() {
        let pool = new_pool();

        let input = Tensor::from_data(&[1, 5, 1, 1], vec![1, 2, 3, 4, 5]);
        let result = flatten(&pool, input.view(), 1 /* axis */).unwrap();
        assert_eq!(result.shape(), &[1, 5]);

        let input = Tensor::from_data(&[2, 3, 1, 4], (1..=24).collect::<Vec<_>>());
        let result = flatten(&pool, input.view(), 2 /* axis */).unwrap();
        assert_eq!(result.shape(), &[6, 4]);

        // Case when `axis` is zero, first output dim should always be 1
        let result = flatten(&pool, input.view(), 0 /* axis */).unwrap();
        assert_eq!(result.shape(), &[1, 24]);
    }

    #[test]
    fn test_reshape_with_unspecified_dim() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();

        // Reshape with an unspecified (-1) dim and nonzero-length input
        let input = Tensor::from_data(&[2, 2], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = ndtensor!([1, -1, 2]);
        let expected = input.to_shape([1, 2, 2].as_slice());
        let result = reshape(
            &pool,
            input.view(),
            &shape.view(),
            false, /* allow_zero */
        )
        .unwrap();
        expect_equal(&result, &expected)?;

        // Reshape with an unspecified (-1) dim and zero-length input
        let zero_sized_input = Tensor::<f32>::from_data(&[4, 0, 1], vec![]);
        let shape = ndtensor!([100, -1]);
        let result = reshape(
            &pool,
            zero_sized_input.view(),
            &shape.view(),
            false, /* allow_zero */
        )
        .unwrap();
        let expected = zero_sized_input.to_shape([100, 0].as_slice());
        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_reshape_with_zero_dim() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();

        // When the target shape has a zero dim, the corresponding input dim
        // size should be copied.
        let input = Tensor::from_data(&[1, 1, 4], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = ndtensor!([-1, 0]);
        let expected = input.to_shape([4, 1].as_slice());
        let result = reshape(
            &pool,
            input.view(),
            &shape.view(),
            false, /* allow_zero */
        )
        .unwrap();
        expect_equal(&result, &expected)?;

        // Case where copied input dim is also zero.
        let input = Tensor::<f32>::from_data(&[0], vec![]);
        let shape = ndtensor!([0]);
        let expected = input.to_shape([0].as_slice());
        let result = reshape(
            &pool,
            input.view(),
            &shape.view(),
            false, /* allow_zero */
        )
        .unwrap();
        expect_equal(&result, &expected)?;

        // Case where there is no corresponding input dim.
        let input = Tensor::from_data(&[1], vec![5.]);
        let shape = ndtensor!([1, 0]);
        let result = reshape(
            &pool,
            input.view(),
            &shape.view(),
            false, /* allow_zero */
        );
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue(
                "Zero dim has no corresponding input dim"
            ))
        );

        // Case when allow_zero is true
        let input = Tensor::<f32>::from_data(&[0, 0, 10], vec![]);
        let shape = ndtensor!([10, 0, 0]);
        let result = reshape(
            &pool,
            input.view(),
            &shape.view(),
            true, /* allow_zero */
        )
        .unwrap();
        let expected = input.to_shape([10, 0, 0].as_slice());
        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_reshape_with_multiple_unspecified_dims() {
        let pool = new_pool();
        let input = Tensor::from_data(&[2, 2], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = ndtensor!([1, -1, -1]);
        assert_eq!(
            reshape(
                &pool,
                input.view(),
                &shape.view(),
                false /* allow_zero */
            )
            .err(),
            Some(OpError::InvalidValue(
                "Multiple dimensions in new shape set to -1"
            ))
        );
    }

    #[test]
    fn test_reshape_with_unsolvable_unspecified_dim() {
        let pool = new_pool();
        let expected_err = Some(OpError::InvalidValue(
            "Input length must be a multiple of specified dimensions",
        ));

        let input = Tensor::from_data(&[2, 2], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = ndtensor!([5, -1]);
        let result = reshape(
            &pool,
            input.view(),
            &shape.view(),
            false, /* allow_zero */
        );
        assert_eq!(result.err(), expected_err);

        // Case when allow_zero is true
        let input = Tensor::from_data(&[1], vec![1]);
        let shape = ndtensor!([0, -1]);
        let result = reshape(
            &pool,
            input.view(),
            &shape.view(),
            true, /* allow_zero */
        );
        assert_eq!(result.err(), expected_err);
    }

    #[test]
    fn test_reshape_in_place() {
        let pool = new_pool();
        let mut input = Tensor::from_data(&[2, 2], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = ndtensor!([4]);
        let expected = input.to_shape([4].as_slice());
        reshape_in_place(
            &pool,
            &mut input,
            &shape.view(),
            false, /* allow_zero */
        )
        .unwrap();
        assert_eq!(&input, &expected);
    }

    #[test]
    fn test_reshape_op() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();
        let input = Tensor::from_data(&[2, 2], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = Tensor::from_data(&[1], vec![4]);
        let expected = input.to_shape([4].as_slice());

        let op = Reshape { allow_zero: false };
        let result = op
            .run(&pool, (&input, &shape).into())
            .unwrap()
            .remove(0)
            .into_float()
            .unwrap();

        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_shape() {
        let pool = new_pool();
        let op = Shape {};

        // Float input
        let input = Tensor::from_data(&[1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let result = op
            .run(&pool, (&input).into())
            .unwrap()
            .remove(0)
            .into_int()
            .unwrap();
        assert_eq!(result.shape(), &[4]);
        assert_eq!(result.to_vec(), &[1, 1, 2, 2]);

        // Int input
        let input = Tensor::from_data(&[1, 1, 2, 2], vec![1, 2, 3, 4]);
        let result = op
            .run(&pool, (&input).into())
            .unwrap()
            .remove(0)
            .into_int()
            .unwrap();
        assert_eq!(result.shape(), &[4]);
        assert_eq!(result.to_vec(), &[1, 1, 2, 2]);
    }

    #[test]
    fn test_size() {
        let pool = new_pool();
        let op = Size {};
        let input = tensor!((2, 2); [1, 2, 3, 4]);
        let result = op
            .run(&pool, (&input).into())
            .unwrap()
            .remove(0)
            .into_int()
            .unwrap();
        assert_eq!(result.ndim(), 0);
        assert_eq!(result.item(), Some(&4));
    }

    #[test]
    fn test_squeeze() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();

        let mut rng = XorShiftRng::new(5678);
        let input = Tensor::rand(&[1, 5, 5, 1], &mut rng);
        let mut expected = input.clone();

        // Remove all 1-size axes.
        expected.reshape(&[5, 5]);
        let result = squeeze(&pool, input.view(), None).unwrap();
        expect_equal(&result, &expected)?;

        // Remove final 1-size axis.
        expected.reshape(&[1, 5, 5]);
        let result = squeeze(&pool, input.view(), Some(ndtensor!([3]).view())).unwrap();
        expect_equal(&result, &expected)?;

        // Remove first 1-size axis.
        expected.reshape(&[5, 5, 1]);
        let result = squeeze(&pool, input.view(), Some(ndtensor!([0]).view())).unwrap();
        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_squeeze_in_place() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(5678);
        let mut input = Tensor::rand(&[1, 1, 5, 5], &mut rng);

        let mut expected = input.clone();
        expected.reshape(&[5, 5]);

        squeeze_in_place(&mut input, None).unwrap();

        expect_equal(&input, &expected)?;

        Ok(())
    }

    #[test]
    fn test_squeeze_invalid_inputs() {
        let mut rng = XorShiftRng::new(5678);
        let input = Tensor::rand(&[1, 5, 5, 1], &mut rng);

        let pool = new_pool();
        let result = squeeze(&pool, input.view(), Some(ndtensor!([1]).view()));

        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue(
                "Can only remove dimensions of size 1"
            ))
        );
    }

    #[test]
    fn test_transpose() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();
        let mut rng = XorShiftRng::new(5678);
        let input = Tensor::rand(&[10, 20], &mut rng);

        let mut reversed = input.clone();
        reversed.permute(&[1, 0]);

        // With no explicit permutation given, the axes should be reversed.
        let result = transpose(&pool, input.view(), None).unwrap();
        expect_equal(&result, &reversed)?;

        // With a no-op permutation given, the output should be unchanged.
        let result = transpose(&pool, input.view(), Some(&[0, 1])).unwrap();
        expect_equal(&result, &input)?;

        // With a transposed permutation given, the axes should be reversed.
        let result = transpose(&pool, input.view(), Some(&[1, 0])).unwrap();
        expect_equal(&result, &reversed)?;

        Ok(())
    }

    #[test]
    fn test_transpose_invalid_inputs() {
        let pool = new_pool();
        let mut rng = XorShiftRng::new(5678);
        let input = Tensor::rand(&[10, 20], &mut rng);

        // Too many dims
        let result = transpose(&pool, input.view(), Some(&[0, 1, 1]));
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Permutation is invalid"))
        );

        // Too few dims
        let result = transpose(&pool, input.view(), Some(&[]));
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Permutation is invalid"))
        );

        // Invalid dimension index
        let result = transpose(&pool, input.view(), Some(&[2, 1]));
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Permutation is invalid"))
        );

        // Repeated dimension index
        let result = transpose(&pool, input.view(), Some(&[1, 1]));
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Permutation is invalid"))
        );
    }

    #[test]
    fn test_unsqueeze() {
        let pool = new_pool();
        let mut rng = XorShiftRng::new(5678);
        let input = Tensor::rand(&[3, 4, 5], &mut rng);

        // Unsqueeze with axes in increasing order
        let output = unsqueeze(&pool, input.view(), &ndtensor!([0, 4]).view()).unwrap();
        assert_eq!(output.shape(), &[1, 3, 4, 5, 1]);

        // Unsqueeze with axes in decreasing order
        let output = unsqueeze(&pool, input.view(), &ndtensor!([4, 0]).view()).unwrap();
        assert_eq!(output.shape(), &[1, 3, 4, 5, 1]);

        // Unsqueeze a scalar into a 1-item vec
        let scalar = tensor!(2.0);
        let output = unsqueeze(&pool, scalar.view(), &ndtensor!([0]).view()).unwrap();
        assert_eq!(output.shape(), &[1]);
        assert_eq!(output.to_vec(), &[2.0]);
    }

    #[test]
    fn test_unsqueeze_invalid_inputs() {
        let pool = new_pool();
        let mut rng = XorShiftRng::new(5678);
        let input = Tensor::rand(&[10, 20], &mut rng);

        // Invalid dimension index
        let result = unsqueeze(&pool, input.view(), &ndtensor!([3]).view());
        assert_eq!(result.err(), Some(OpError::InvalidValue("Axis is invalid")));

        // Repeated dimension index
        let result = unsqueeze(&pool, input.view(), &ndtensor!([1, 1]).view());
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Axes must be unique"))
        );
    }

    use rten_tensor::{NdTensorView, TensorView, TensorViewMut};

    fn reference_transpose_into<'a, T: Clone>(src: TensorView<T>, mut dest: TensorViewMut<T>) {
        // Merge axes to maximize iteration count of the innermost loops.
        let mut src = src.clone();
        src.merge_axes();

        while src.ndim() < 4 {
            src.insert_axis(0);
        }

        let dest_data = dest.data_mut().unwrap();

        let src: NdTensorView<T, 4> = src.nd_view();
        let mut dest_offset = 0;
        for i0 in 0..src.size(0) {
            for i1 in 0..src.size(1) {
                for i2 in 0..src.size(2) {
                    for i3 in 0..src.size(3) {
                        unsafe {
                            let elt = src.get_unchecked([i0, i1, i2, i3]).clone();
                            *dest_data.get_unchecked_mut(dest_offset) = elt;
                            dest_offset += 1;
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn bench_transpose() {
        let mut rng = XorShiftRng::new(1234);

        struct Case<'a> {
            /// Input shape
            shape: &'a [usize],

            /// Permutation order (eg. `[1, 0]` for a matrix transpose)
            perm: &'a [usize],
        }

        let cases = [
            // No-op transpose
            Case {
                shape: &[512, 512],
                perm: &[0, 1],
            },
            // Matrix transpose with power-of-2 sizes.
            //
            // In a naive transpose implementation, these are liable to
            // experience slowdown due to poor cache usage. There can also be
            // issues to a lesser extent with sizes which are a multiple of
            // (cache_line_size / element_size).
            Case {
                shape: &[128, 128],
                perm: &[1, 0],
            },
            Case {
                shape: &[256, 256],
                perm: &[1, 0],
            },
            Case {
                shape: &[512, 512],
                perm: &[1, 0],
            },
            Case {
                shape: &[1024, 1024],
                perm: &[1, 0],
            },
            // Matrix transpose with non power-of-2 sizes.
            Case {
                shape: &[127, 127],
                perm: &[1, 0],
            },
            Case {
                shape: &[255, 255],
                perm: &[1, 0],
            },
            Case {
                shape: &[513, 513],
                perm: &[1, 0],
            },
            Case {
                shape: &[1023, 1023],
                perm: &[1, 0],
            },
            // Transpose ops taken from Whisper encoder (base model) with 4
            // batches of samples.
            //
            // Note the last two dimensions are powers of 2.
            Case {
                shape: &[4, 1500, 8, 64],
                perm: &[0, 2, 1, 3],
            },
            Case {
                shape: &[4, 8, 1500, 64],
                perm: &[0, 2, 1, 3],
            },
            // Transpose ops taken from Whisper decoder (base model)
            Case {
                shape: &[1, 1500, 8, 64],
                perm: &[0, 2, 3, 1],
            },
            Case {
                shape: &[1, 288, 8, 64],
                perm: &[0, 2, 1, 3],
            },
        ];

        for Case { shape, perm } in cases {
            let tensor = Tensor::rand(shape, &mut rng);
            let mut dest = Tensor::zeros(shape);

            // Do a simple copy. This provides a lower-bound on how fast
            // transpose can operate.
            let copy_stats = run_bench(100, None, || {
                dest.copy_from(&tensor.view());
            });
            assert_eq!(dest, tensor);

            let reference_transpose_stats = run_bench(100, None, || {
                let transposed = tensor.permuted(perm);
                reference_transpose_into(transposed.view(), dest.reshaped_mut(transposed.shape()));
            });

            let transpose_stats = run_bench(100, None, || {
                let transposed = tensor.permuted(perm);
                dest.reshape(transposed.shape());
                dest.copy_from(&transposed);
            });
            assert_eq!(dest, tensor.permuted(perm));

            let transpose_overhead =
                (transpose_stats.mean - copy_stats.mean).max(0.) / copy_stats.mean;
            println!(
                "transpose shape {:?} perm {:?} copy {:.3}ms ref transpose {:.3}ms opt transpose {:.3}ms overhead {}",
                shape, perm, copy_stats.median, reference_transpose_stats.median, transpose_stats.median, transpose_overhead
            );
        }
    }
}
