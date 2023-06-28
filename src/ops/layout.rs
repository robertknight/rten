//! Operators which query or change the shape of a tensor, or copy/move/reorder
//! elements.
use std::iter::zip;

use wasnn_tensor::{is_valid_permutation, Tensor, TensorLayout};

use crate::check_dims;
use crate::ops::binary_elementwise::broadcast_shapes;
use crate::ops::{
    resolve_axes, resolve_axis, Input, InputList, IntoOpResult, OpError, Operator, Output,
};

pub fn expand<T: Copy>(input: &Tensor<T>, shape: &Tensor<i32>) -> Result<Tensor<T>, OpError> {
    check_dims!(shape, 1);

    let shape_vec: Vec<_> = shape.iter().map(|el| *el as usize).collect();
    let out_shape = broadcast_shapes(input.shape(), &shape_vec).ok_or(
        OpError::IncompatibleInputShapes("Cannot broadcast input with target shape"),
    )?;

    let out_elts: Vec<_> = input.broadcast_iter(&out_shape).copied().collect();
    Ok(Tensor::from_data(&out_shape, out_elts))
}

#[derive(Debug)]
pub struct Expand {}

impl Operator for Expand {
    fn name(&self) -> &str {
        "Expand"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let shape = inputs.require_as(1)?;

        match input {
            Input::FloatTensor(input) => expand(input, shape).into_op_result(),
            Input::IntTensor(input) => expand(input, shape).into_op_result(),
        }
    }
}

fn flattened_shape(shape: &[usize], axis: isize) -> Result<[usize; 2], OpError> {
    let resolved_axis = resolve_axis(shape.len(), axis)?;
    let outer_size = shape.iter().take(resolved_axis).product();
    let inner_size = shape.iter().skip(resolved_axis).product();
    Ok([outer_size, inner_size])
}

pub fn flatten<T: Copy>(input: &Tensor<T>, axis: isize) -> Result<Tensor<T>, OpError> {
    let shape = flattened_shape(input.shape(), axis)?;
    Ok(input.clone_with_shape(&shape))
}

pub fn flatten_in_place<T: Copy>(input: &mut Tensor<T>, axis: isize) -> Result<(), OpError> {
    let shape = flattened_shape(input.shape(), axis)?;
    input.reshape(&shape);
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

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;

        match input {
            Input::FloatTensor(input) => flatten(input, self.axis).into_op_result(),
            Input::IntTensor(input) => flatten(input, self.axis).into_op_result(),
        }
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, _: InputList) -> Result<Output, OpError> {
        match input {
            Output::IntTensor(mut output) => {
                flatten_in_place(&mut output, self.axis)?;
                Ok(output.into())
            }
            Output::FloatTensor(mut output) => {
                flatten_in_place(&mut output, self.axis)?;
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
    shape: &Tensor<i32>,
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

pub fn reshape<T: Copy>(
    input: &Tensor<T>,
    shape: &Tensor<i32>,
    allow_zero: bool,
) -> Result<Tensor<T>, OpError> {
    let out_shape = resolve_shape(input.shape(), shape, allow_zero)?;
    Ok(input.clone_with_shape(&out_shape))
}

pub fn reshape_in_place<T: Copy>(
    input: &mut Tensor<T>,
    shape: &Tensor<i32>,
    allow_zero: bool,
) -> Result<(), OpError> {
    let out_shape = resolve_shape(input.shape(), shape, allow_zero)?;
    input.reshape(&out_shape);
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

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let shape = inputs.require_as(1)?;
        match input {
            Input::IntTensor(t) => reshape(t, shape, self.allow_zero).into_op_result(),
            Input::FloatTensor(t) => reshape(t, shape, self.allow_zero).into_op_result(),
        }
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, other: InputList) -> Result<Output, OpError> {
        let shape = other.require_as(0)?;
        match input {
            Output::IntTensor(mut output) => {
                reshape_in_place(&mut output, shape, self.allow_zero)?;
                Ok(output.into())
            }
            Output::FloatTensor(mut output) => {
                reshape_in_place(&mut output, shape, self.allow_zero)?;
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

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let shape = Tensor::from_data(
            &[input.shape().len()],
            input
                .shape()
                .iter()
                .map(|&el| el as i32)
                .collect::<Vec<_>>(),
        );
        shape.into_op_result()
    }
}

pub fn squeeze_in_place<T: Copy>(
    input: &mut Tensor<T>,
    axes: Option<&Tensor<i32>>,
) -> Result<(), OpError> {
    if let Some(axes) = axes {
        check_dims!(axes, 1);
    }

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

    let new_shape: Vec<_> = input
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

pub fn squeeze<T: Copy>(
    input: &Tensor<T>,
    axes: Option<&Tensor<i32>>,
) -> Result<Tensor<T>, OpError> {
    let mut output = input.clone();
    squeeze_in_place(&mut output, axes)?;
    Ok(output)
}

#[derive(Debug)]
pub struct Squeeze {}

impl Operator for Squeeze {
    fn name(&self) -> &str {
        "Squeeze"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let axes = inputs.get_as(1)?;
        match input {
            Input::FloatTensor(t) => squeeze(t, axes).into_op_result(),
            Input::IntTensor(t) => squeeze(t, axes).into_op_result(),
        }
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, other: InputList) -> Result<Output, OpError> {
        let axes = other.get_as(0)?;
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

pub fn transpose<T: Copy>(
    input: &Tensor<T>,
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
    Ok(transposed.to_tensor())
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

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let perm_slice = self.perm.as_deref();
        match input {
            Input::FloatTensor(input) => transpose(input, perm_slice).into_op_result(),
            Input::IntTensor(input) => transpose(input, perm_slice).into_op_result(),
        }
    }
}

pub fn unsqueeze<T: Copy>(input: &Tensor<T>, axes: &Tensor<i32>) -> Result<Tensor<T>, OpError> {
    check_dims!(axes, 1);
    let mut new_shape: Vec<_> = input.shape().to_vec();
    let mut sorted_axes: Vec<_> = resolve_axes(input.ndim() + axes.len(), axes.iter())?;
    sorted_axes.sort();

    let axes_unique =
        zip(sorted_axes.iter().skip(1), sorted_axes.iter()).all(|(prev, current)| prev != current);
    if !axes_unique {
        return Err(OpError::InvalidValue("Axes must be unique"));
    }

    for axis in sorted_axes {
        new_shape.insert(axis, 1);
    }
    Ok(input.clone_with_shape(&new_shape))
}

#[derive(Debug)]
pub struct Unsqueeze {}

impl Operator for Unsqueeze {
    fn name(&self) -> &str {
        "Unsqueeze"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let axes = inputs.require_as(1)?;
        match input {
            Input::FloatTensor(input) => unsqueeze(input, axes).into_op_result(),
            Input::IntTensor(input) => unsqueeze(input, axes).into_op_result(),
        }
    }
}

#[cfg(test)]
mod tests {
    use wasnn_tensor::rng::XorShiftRng;
    use wasnn_tensor::test_util::expect_equal;
    use wasnn_tensor::{tensor, Tensor, TensorLayout};

    use crate::ops::layout::{
        expand, flatten, reshape, reshape_in_place, squeeze, squeeze_in_place, transpose,
        unsqueeze, InputList, Reshape, Shape,
    };
    use crate::ops::{OpError, Operator};

    #[test]
    fn test_expand() {
        // Broadcast scalar
        let input = tensor!(5.);
        let shape = tensor!([2, 2]);
        let expected = Tensor::from_data(&[2, 2], vec![5., 5., 5., 5.]);
        let result = expand(&input, &shape).unwrap();
        assert_eq!(&result, &expected);

        // Broadcast that changes dim count
        let input = Tensor::from_data(&[3, 1], (0..3).collect::<Vec<_>>());
        let shape = tensor!([2, 3, 1]);
        let result = expand(&input, &shape).unwrap();
        assert_eq!(result.shape(), &[2, 3, 1]);

        // Broadcast that uses dimensions from both the input shape and target
        // shape in the output shape.
        let input = Tensor::from_data(&[3, 1], (0..3).collect::<Vec<_>>());
        let shape = tensor!([2, 1, 6]);
        let result = expand(&input, &shape).unwrap();
        assert_eq!(result.shape(), &[2, 3, 6]);

        // Broadcast that does not change dim count
        let input = Tensor::from_data(&[3, 1], (0..3).collect::<Vec<_>>());
        let shape = tensor!([3, 4]);
        let result = expand(&input, &shape).unwrap();
        assert_eq!(result.shape(), &[3, 4]);
    }

    #[test]
    fn test_expand_invalid_inputs() {
        // Invalid broadcast shape
        let input = tensor!([1, 2, 3]);
        let shape = tensor!([2, 2]);
        let result = expand(&input, &shape);
        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes(
                "Cannot broadcast input with target shape"
            ))
        );

        // Non-vector shape
        let input = tensor!(5.);
        let shape = tensor!(4);
        let result = expand(&input, &shape);
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("shape must have 1 dims"))
        );
    }

    #[test]
    fn test_flatten() {
        let input = Tensor::from_data(&[1, 5, 1, 1], vec![1, 2, 3, 4, 5]);
        let result = flatten(&input, 1 /* axis */).unwrap();
        assert_eq!(result.shape(), &[1, 5]);

        let input = Tensor::from_data(&[2, 3, 1, 4], (1..=24).collect::<Vec<_>>());
        let result = flatten(&input, 2 /* axis */).unwrap();
        assert_eq!(result.shape(), &[6, 4]);

        // Case when `axis` is zero, first output dim should always be 1
        let result = flatten(&input, 0 /* axis */).unwrap();
        assert_eq!(result.shape(), &[1, 24]);
    }

    #[test]
    fn test_reshape_with_unspecified_dim() -> Result<(), String> {
        // Reshape with an unspecified (-1) dim and nonzero-length input
        let input = Tensor::from_data(&[2, 2], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = tensor!([1, -1, 2]);
        let expected = input.clone_with_shape(&[1, 2, 2]);
        let result = reshape(&input, &shape, false /* allow_zero */).unwrap();
        expect_equal(&result, &expected)?;

        // Reshape with an unspecified (-1) dim and zero-length input
        let zero_sized_input = Tensor::<f32>::from_data(&[4, 0, 1], vec![]);
        let shape = tensor!([100, -1]);
        let result = reshape(&zero_sized_input, &shape, false /* allow_zero */).unwrap();
        let expected = zero_sized_input.clone_with_shape(&[100, 0]);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_reshape_with_zero_dim() -> Result<(), String> {
        // When the target shape has a zero dim, the corresponding input dim
        // size should be copied.
        let input = Tensor::from_data(&[1, 1, 4], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = tensor!([-1, 0]);
        let expected = input.clone_with_shape(&[4, 1]);
        let result = reshape(&input, &shape, false /* allow_zero */).unwrap();
        expect_equal(&result, &expected)?;

        // Case where copied input dim is also zero.
        let input = Tensor::<f32>::from_data(&[0], vec![]);
        let shape = tensor!([0]);
        let expected = input.clone_with_shape(&[0]);
        let result = reshape(&input, &shape, false /* allow_zero */).unwrap();
        expect_equal(&result, &expected)?;

        // Case where there is no corresponding input dim.
        let input = Tensor::from_data(&[1], vec![5.]);
        let shape = tensor!([1, 0]);
        let result = reshape(&input, &shape, false /* allow_zero */);
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue(
                "Zero dim has no corresponding input dim"
            ))
        );

        // Case when allow_zero is true
        let input = Tensor::<f32>::from_data(&[0, 0, 10], vec![]);
        let shape = tensor!([10, 0, 0]);
        let result = reshape(&input, &shape, true /* allow_zero */).unwrap();
        let expected = input.clone_with_shape(&[10, 0, 0]);
        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_reshape_with_multiple_unspecified_dims() {
        let input = Tensor::from_data(&[2, 2], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = tensor!([1, -1, -1]);
        assert_eq!(
            reshape(&input, &shape, false /* allow_zero */).err(),
            Some(OpError::InvalidValue(
                "Multiple dimensions in new shape set to -1"
            ))
        );
    }

    #[test]
    fn test_reshape_with_unsolvable_unspecified_dim() {
        let expected_err = Some(OpError::InvalidValue(
            "Input length must be a multiple of specified dimensions",
        ));

        let input = Tensor::from_data(&[2, 2], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = tensor!([5, -1]);
        let result = reshape(&input, &shape, false /* allow_zero */);
        assert_eq!(result.err(), expected_err);

        // Case when allow_zero is true
        let input = Tensor::from_data(&[1], vec![1]);
        let shape = tensor!([0, -1]);
        let result = reshape(&input, &shape, true /* allow_zero */);
        assert_eq!(result.err(), expected_err);
    }

    #[test]
    fn test_reshape_in_place() {
        let mut input = Tensor::from_data(&[2, 2], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = Tensor::from_data(&[1], vec![4]);
        let expected = input.clone_with_shape(&[4]);
        reshape_in_place(&mut input, &shape, false /* allow_zero */).unwrap();
        assert_eq!(&input, &expected);
    }

    #[test]
    fn test_reshape_op() -> Result<(), String> {
        let input = Tensor::from_data(&[2, 2], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = Tensor::from_data(&[1], vec![4]);
        let expected = input.clone_with_shape(&[4]);

        let op = Reshape { allow_zero: false };
        let result = op
            .run(InputList::from(&[(&input).into(), (&shape).into()]))
            .unwrap()
            .remove(0)
            .into_float()
            .unwrap();

        expect_equal(&result, &expected)
    }

    #[test]
    fn test_shape() {
        let op = Shape {};

        // Float input
        let input = Tensor::from_data(&[1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let result = op
            .run(InputList::from(&[(&input).into()]))
            .unwrap()
            .remove(0)
            .into_int()
            .unwrap();
        assert_eq!(result.shape(), &[4]);
        assert_eq!(result.data(), &[1, 1, 2, 2]);

        // Int input
        let input = Tensor::from_data(&[1, 1, 2, 2], vec![1, 2, 3, 4]);
        let result = op
            .run(InputList::from(&[(&input).into()]))
            .unwrap()
            .remove(0)
            .into_int()
            .unwrap();
        assert_eq!(result.shape(), &[4]);
        assert_eq!(result.data(), &[1, 1, 2, 2]);
    }

    #[test]
    fn test_squeeze() -> Result<(), String> {
        let mut rng = XorShiftRng::new(5678);
        let input = Tensor::rand(&[1, 5, 5, 1], &mut rng);
        let mut expected = input.clone();

        // Remove all 1-size axes.
        expected.reshape(&[5, 5]);
        let result = squeeze(&input, None).unwrap();
        expect_equal(&result, &expected)?;

        // Remove final 1-size axis.
        expected.reshape(&[1, 5, 5]);
        let result = squeeze(&input, Some(&tensor!([3]))).unwrap();
        expect_equal(&result, &expected)?;

        // Remove first 1-size axis.
        expected.reshape(&[5, 5, 1]);
        let result = squeeze(&input, Some(&tensor!([0]))).unwrap();
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_squeeze_in_place() -> Result<(), String> {
        let mut rng = XorShiftRng::new(5678);
        let mut input = Tensor::rand(&[1, 1, 5, 5], &mut rng);

        let mut expected = input.clone();
        expected.reshape(&[5, 5]);

        squeeze_in_place(&mut input, None).unwrap();

        expect_equal(&input, &expected)
    }

    #[test]
    fn test_squeeze_invalid_inputs() {
        let mut rng = XorShiftRng::new(5678);
        let input = Tensor::rand(&[1, 5, 5, 1], &mut rng);

        let result = squeeze(&input, Some(&tensor!([1])));

        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue(
                "Can only remove dimensions of size 1"
            ))
        );
    }

    #[test]
    fn test_transpose() -> Result<(), String> {
        let mut rng = XorShiftRng::new(5678);
        let input = Tensor::rand(&[10, 20], &mut rng);

        let mut reversed = input.clone();
        reversed.permute(&[1, 0]);

        // With no explicit permutation given, the axes should be reversed.
        let result = transpose(&input, None).unwrap();
        expect_equal(&result, &reversed)?;

        // With a no-op permutation given, the output should be unchanged.
        let result = transpose(&input, Some(&[0, 1])).unwrap();
        expect_equal(&result, &input)?;

        // With a transposed permutation given, the axes should be reversed.
        let result = transpose(&input, Some(&[1, 0])).unwrap();
        expect_equal(&result, &reversed)
    }

    #[test]
    fn test_transpose_invalid_inputs() {
        let mut rng = XorShiftRng::new(5678);
        let input = Tensor::rand(&[10, 20], &mut rng);

        // Too many dims
        let result = transpose(&input, Some(&[0, 1, 1]));
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Permutation is invalid"))
        );

        // Too few dims
        let result = transpose(&input, Some(&[]));
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Permutation is invalid"))
        );

        // Invalid dimension index
        let result = transpose(&input, Some(&[2, 1]));
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Permutation is invalid"))
        );

        // Repeated dimension index
        let result = transpose(&input, Some(&[1, 1]));
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Permutation is invalid"))
        );
    }

    #[test]
    fn test_unsqueeze() {
        let mut rng = XorShiftRng::new(5678);
        let input = Tensor::rand(&[3, 4, 5], &mut rng);

        // Unsqueeze with axes in increasing order
        let output = unsqueeze(&input, &tensor!([0, 4])).unwrap();
        assert_eq!(output.shape(), &[1, 3, 4, 5, 1]);

        // Unsqueeze with axes in decreasing order
        let output = unsqueeze(&input, &tensor!([4, 0])).unwrap();
        assert_eq!(output.shape(), &[1, 3, 4, 5, 1]);

        // Unsqueeze a scalar into a 1-item vec
        let scalar = tensor!(2.0);
        let output = unsqueeze(&scalar, &tensor!([0])).unwrap();
        assert_eq!(output.shape(), &[1]);
        assert_eq!(output.data(), &[2.0]);
    }

    #[test]
    fn test_unsqueeze_invalid_inputs() {
        let mut rng = XorShiftRng::new(5678);
        let input = Tensor::rand(&[10, 20], &mut rng);

        // Invalid dimension index
        let result = unsqueeze(&input, &tensor!([3]));
        assert_eq!(result.err(), Some(OpError::InvalidValue("Axis is invalid")));

        // Repeated dimension index
        let result = unsqueeze(&input, &tensor!([1, 1]));
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Axes must be unique"))
        );
    }
}
