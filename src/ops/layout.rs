///! Operators which query or change the shape of a tensor, or copy/move/reorder
///! elements.
use crate::ops::{get_input, Input, IntoOpResult, OpError, Operator, Output};
use crate::tensor::{from_data, Tensor};

pub fn expand<T: Copy>(input: &Tensor<T>, shape: &Tensor<i32>) -> Result<Tensor<T>, OpError> {
    if shape.ndim() != 1 {
        return Err(OpError::InvalidValue("shape must be a vector"));
    }

    let out_shape: Vec<_> = shape.elements().map(|el| el as usize).collect();
    if !input.can_broadcast(&out_shape) {
        return Err(OpError::IncompatibleInputShapes(
            "Cannot broadcast to output shape",
        ));
    }

    let out_elts = input.broadcast_elements(&out_shape).collect();
    Ok(from_data(out_shape, out_elts))
}

#[derive(Debug)]
pub struct Expand {}

impl Operator for Expand {
    fn name(&self) -> &str {
        "Expand"
    }

    fn run(&self, inputs: &[Input]) -> Result<Vec<Output>, OpError> {
        let input = inputs.get(0).ok_or(OpError::MissingInputs)?;
        let shape = get_input(inputs, 1)?;

        match input {
            Input::FloatTensor(input) => expand(&input, &shape).into_op_result(),
            Input::IntTensor(input) => expand(&input, &shape).into_op_result(),
        }
    }
}

pub fn reshape<T: Copy>(input: &Tensor<T>, shape: &Tensor<i32>) -> Result<Tensor<T>, OpError> {
    // If exactly one of the new shape's dimensions is -1, infer the size
    // from the input length and the sizes of the other dimensions.
    let mut unspecified_dim = None;
    let mut specified_dims_size = 1;
    for (dim, size) in shape.elements().enumerate() {
        if size < -1 {
            return Err(OpError::InvalidValue("Invalid dimension size in shape"));
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
    let (unspecified_dim_size, remainder) = match input.len() {
        0 => (0, 0),
        _ => (
            input.len() / specified_dims_size,
            input.len() % specified_dims_size,
        ),
    };
    if remainder != 0 {
        return Err(OpError::InvalidValue(
            "Input length must be a multiple of specified dimensions",
        ));
    }

    let complete_shape: Vec<_> = shape
        .elements()
        .map(|size| match size {
            -1 => unspecified_dim_size,
            valid => valid as usize,
        })
        .collect();

    Ok(input.clone_with_shape(&complete_shape))
}

#[derive(Debug)]
pub struct Reshape {}
impl Operator for Reshape {
    fn name(&self) -> &str {
        "Reshape"
    }

    fn run(&self, inputs: &[Input]) -> Result<Vec<Output>, OpError> {
        let input = inputs.get(0).ok_or(OpError::MissingInputs)?;
        let shape = get_input(inputs, 1)?;
        match input {
            Input::IntTensor(t) => reshape(t, shape).into_op_result(),
            Input::FloatTensor(t) => reshape(t, shape).into_op_result(),
        }
    }

    fn can_run_in_place(&self) -> bool {
        // The ability to reshape in place depends on input and target types.
        // If the planned inputs were passed to this method, we could do an
        // in-place reshape if the inputs/targets were compatible.
        false
    }
}

#[derive(Debug)]
pub struct Shape {}

impl Operator for Shape {
    fn name(&self) -> &str {
        "Shape"
    }

    fn run(&self, inputs: &[Input]) -> Result<Vec<Output>, OpError> {
        let input = inputs.get(0).ok_or(OpError::MissingInputs)?;
        let shape = from_data(
            vec![input.shape().len()],
            input.shape().iter().map(|&el| el as i32).collect(),
        );
        shape.into_op_result()
    }
}

pub fn squeeze_in_place<T: Copy>(input: &mut Tensor<T>, axes: Option<&[usize]>) {
    let new_shape: Vec<_> = input
        .shape()
        .iter()
        .enumerate()
        .filter(|(dim, &size)| {
            if let Some(axes) = axes {
                let keep_axis = !axes.contains(dim);
                // TODO - Turn this into a result
                assert!(
                    keep_axis || size == 1,
                    "Can only remove dimensions of size 1"
                );
                keep_axis
            } else {
                size > 1
            }
        })
        .map(|(_, &size)| size)
        .collect();
    input.reshape(&new_shape);
}

pub fn squeeze<T: Copy>(input: &Tensor<T>, axes: Option<&[usize]>) -> Tensor<T> {
    let mut output = input.clone();
    squeeze_in_place(&mut output, axes);
    output
}

#[derive(Debug)]
pub struct Squeeze {
    pub axes: Option<Vec<usize>>,
}

impl Operator for Squeeze {
    fn name(&self) -> &str {
        "Squeeze"
    }

    fn run(&self, inputs: &[Input]) -> Result<Vec<Output>, OpError> {
        let input = inputs.get(0).ok_or(OpError::MissingInputs)?;
        let axes = self.axes.as_ref().map(|a| &a[..]);
        let result: Output = match input {
            Input::FloatTensor(t) => squeeze(t, axes).into(),
            Input::IntTensor(t) => squeeze(t, axes).into(),
        };
        result.into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, _: &[Input]) -> Result<Output, OpError> {
        let axes = self.axes.as_ref().map(|a| &a[..]);
        let result = match input {
            Output::FloatTensor(mut t) => {
                squeeze_in_place(&mut t, axes);
                t.into()
            }
            Output::IntTensor(mut t) => {
                squeeze_in_place(&mut t, axes);
                t.into()
            }
        };
        Ok(result)
    }
}

pub fn transpose<T: Copy>(input: &Tensor<T>, permutation: Option<&[usize]>) -> Tensor<T> {
    let mut transposed = input.clone();
    match permutation {
        Some(order) => transposed.permute(order),
        None => {
            let reversed: Vec<usize> = (0..transposed.shape().len()).rev().collect();
            transposed.permute(&reversed);
        }
    };
    transposed
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

    fn run(&self, inputs: &[Input]) -> Result<Vec<Output>, OpError> {
        let input = inputs.get(0).ok_or(OpError::MissingInputs)?;
        let perm_slice = self.perm.as_deref();
        let result: Output = match input {
            Input::FloatTensor(input) => transpose(input, perm_slice).into(),
            Input::IntTensor(input) => transpose(input, perm_slice).into(),
        };
        result.into_op_result()
    }
}

pub fn unsqueeze<T: Copy>(input: &Tensor<T>, axes: &[usize]) -> Tensor<T> {
    let mut new_shape: Vec<_> = input.shape().to_vec();
    let mut sorted_axes: Vec<_> = axes.iter().collect();
    sorted_axes.sort();
    for &axis in sorted_axes {
        new_shape.insert(axis, 1);
    }
    input.clone_with_shape(&new_shape)
}

#[derive(Debug)]
pub struct Unsqueeze {
    pub axes: Vec<usize>,
}

impl Operator for Unsqueeze {
    fn name(&self) -> &str {
        "Unsqueeze"
    }

    fn run(&self, inputs: &[Input]) -> Result<Vec<Output>, OpError> {
        let input = inputs.get(0).ok_or(OpError::MissingInputs)?;
        let result: Output = match input {
            Input::FloatTensor(input) => unsqueeze(input, &self.axes).into(),
            Input::IntTensor(input) => unsqueeze(input, &self.axes).into(),
        };
        result.into_op_result()
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::layout::{
        expand, reshape, squeeze, squeeze_in_place, transpose, unsqueeze, Reshape, Shape,
    };
    use crate::ops::{OpError, Operator};
    use crate::rng::XorShiftRNG;
    use crate::tensor::{from_data, from_scalar, from_vec, rand};
    use crate::test_util::expect_equal;

    #[test]
    fn test_expand() {
        // Broadcast scalar
        let input = from_scalar(5.);
        let shape = from_vec(vec![2, 2]);
        let expected = from_data(vec![2, 2], vec![5., 5., 5., 5.]);
        let result = expand(&input, &shape).unwrap();
        assert_eq!(&result, &expected);

        // Broadcast that changes dim count
        let input = from_data(vec![3, 1], (0..3).collect());
        let shape = from_vec(vec![2, 1, 6]);
        let result = expand(&input, &shape).unwrap();
        assert_eq!(result.shape(), &[2, 1, 6]);

        // Broadcast that does not change dim count
        let input = from_data(vec![3, 1], (0..3).collect());
        let shape = from_vec(vec![3, 4]);
        let result = expand(&input, &shape).unwrap();
        assert_eq!(result.shape(), &[3, 4]);
    }

    #[test]
    fn test_expand_invalid_inputs() {
        // Invalid broadcast shape
        let input = from_vec(vec![1, 2, 3]);
        let shape = from_vec(vec![2, 2]);
        let result = expand(&input, &shape);
        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes(
                "Cannot broadcast to output shape"
            ))
        );

        // Non-vector shape
        let input = from_scalar(5.);
        let shape = from_scalar(4);
        let result = expand(&input, &shape);
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("shape must be a vector"))
        );
    }

    #[test]
    fn test_reshape_with_unspecified_dim() -> Result<(), String> {
        // Reshape with an unspecified (-1) dim and nonzero-length input
        let input = from_data(vec![2, 2], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = from_vec(vec![1, -1, 2]);
        let expected = input.clone_with_shape(&[1, 2, 2]);
        let result = reshape(&input, &shape).unwrap();
        expect_equal(&result, &expected)?;

        // Reshape with an unspecified (-1) dim and zero-length input
        let zero_sized_input = from_data(vec![4, 0, 1], vec![]);
        let shape = from_vec(vec![100, -1]);
        let result = reshape(&zero_sized_input, &shape).unwrap();
        let expected = zero_sized_input.clone_with_shape(&[100, 0]);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_reshape_with_multiple_unspecified_dims() {
        let input = from_data(vec![2, 2], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = from_vec(vec![1, -1, -1]);
        assert_eq!(
            reshape(&input, &shape).err(),
            Some(OpError::InvalidValue(
                "Multiple dimensions in new shape set to -1"
            ))
        );
    }

    #[test]
    fn test_reshape_with_unsolvable_unspecified_dim() {
        let input = from_data(vec![2, 2], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = from_vec(vec![5, -1]);
        assert_eq!(
            reshape(&input, &shape).err(),
            Some(OpError::InvalidValue(
                "Input length must be a multiple of specified dimensions"
            ))
        );
    }

    #[test]
    fn test_reshape_op() -> Result<(), String> {
        let input = from_data(vec![2, 2], vec![-0.5, 0.5, 3.0, -5.5]);
        let shape = from_data(vec![1], vec![4]);
        let expected = input.clone_with_shape(&[4]);

        let op = Reshape {};
        let result = op
            .run(&[(&input).into(), (&shape).into()])
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
        let input = from_data(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let result = op
            .run(&[(&input).into()])
            .unwrap()
            .remove(0)
            .into_int()
            .unwrap();
        assert_eq!(result.shape(), &[4]);
        assert_eq!(result.data(), &[1, 1, 2, 2]);

        // Int input
        let input = from_data(vec![1, 1, 2, 2], vec![1, 2, 3, 4]);
        let result = op
            .run(&[(&input).into()])
            .unwrap()
            .remove(0)
            .into_int()
            .unwrap();
        assert_eq!(result.shape(), &[4]);
        assert_eq!(result.data(), &[1, 1, 2, 2]);
    }

    #[test]
    fn test_squeeze() -> Result<(), String> {
        let mut rng = XorShiftRNG::new(5678);
        let input = rand(&[1, 5, 5, 1], &mut rng);
        let mut expected = input.clone();

        // Remove all 1-size axes.
        expected.reshape(&[5, 5]);
        let result = squeeze(&input, None);
        expect_equal(&result, &expected)?;

        // Remove final 1-size axis.
        expected.reshape(&[1, 5, 5]);
        let result = squeeze(&input, Some(&[3]));
        expect_equal(&result, &expected)?;

        // Remove first 1-size axis.
        expected.reshape(&[5, 5, 1]);
        let result = squeeze(&input, Some(&[0]));
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_squeeze_in_place() -> Result<(), String> {
        let mut rng = XorShiftRNG::new(5678);
        let mut input = rand(&[1, 1, 5, 5], &mut rng);

        let mut expected = input.clone();
        expected.reshape(&[5, 5]);

        squeeze_in_place(&mut input, None);

        expect_equal(&input, &expected)
    }

    #[test]
    fn test_transpose() -> Result<(), String> {
        let mut rng = XorShiftRNG::new(5678);
        let input = rand(&[10, 20], &mut rng);

        let mut reversed = input.clone();
        reversed.permute(&[1, 0]);

        // With no explicit permutation given, the axes should be reversed.
        let result = transpose(&input, None);
        expect_equal(&result, &reversed)?;

        // With a no-op permutation given, the output should be unchanged.
        let result = transpose(&input, Some(&[0, 1]));
        expect_equal(&result, &input)?;

        // With a transposed permutation given, the axes should be reversed.
        let result = transpose(&input, Some(&[1, 0]));
        expect_equal(&result, &reversed)
    }

    #[test]
    fn test_unsqueeze() {
        let mut rng = XorShiftRNG::new(5678);
        let input = rand(&[3, 4, 5], &mut rng);

        // Unsqueeze with axes in increasing order
        let output = unsqueeze(&input, &[0, 4]);
        assert_eq!(output.shape(), &[1, 3, 4, 5, 1]);

        // Unsqueeze with axes in decreasing order
        let output = unsqueeze(&input, &[4, 0]);
        assert_eq!(output.shape(), &[1, 3, 4, 5, 1]);

        // Unsqueeze a scalar into a 1-item vec
        let scalar = from_scalar(2.0);
        let output = unsqueeze(&scalar, &[0]);
        assert_eq!(output.shape(), &[1]);
        assert_eq!(output.data(), &[2.0]);
    }
}
