use std::fmt::Debug;
use std::iter::zip;

use crate::tensor::{from_data, zero_tensor, Elements, SliceRange, Tensor};

mod activations;
mod binary_elementwise;
mod conv;
mod layout;
mod matmul;
mod norm;
mod pooling;

pub use activations::{
    clip, clip_in_place, leaky_relu, leaky_relu_in_place, relu, relu_in_place, sigmoid,
    sigmoid_in_place, softmax,
};
pub use activations::{Clip, LeakyRelu, Relu, Sigmoid, Softmax};
pub use binary_elementwise::{
    add, add_in_place, choose_broadcast_shape, div, div_in_place, mul, mul_in_place, sub,
    sub_in_place,
};
pub use binary_elementwise::{Add, Div, Mul, Sub};
pub use conv::{conv_2d, conv_transpose_2d};
pub use conv::{Conv2d, ConvTranspose2d};
pub use layout::{reshape, squeeze, Reshape, Shape, Squeeze, Transpose, Unsqueeze};
pub use matmul::{gemm_op, matmul, Gemm, MatMul};
pub use norm::{batch_norm, batch_norm_in_place, BatchNormalization};
pub use pooling::{average_pool_2d, global_average_pool, max_pool_2d};
pub use pooling::{AveragePool2d, GlobalAveragePool, MaxPool2d};

#[derive(Copy, Clone, Debug)]
pub enum Padding {
    /// Apply enough padding such that the output and input have the same size.
    ///
    /// If the required amount of padding along each dimension is even, it is
    /// divided equally between the start and the end. If it is odd, one more
    /// unit is added on the end than the start. This matches the ONNX spec
    /// for the "SAME_UPPER" value for the `auto_pad` attribute.
    Same,

    /// Apply a given amount of padding to the top, left, bottom and right of
    /// the input.
    Fixed([usize; 4]),
}

#[derive(Copy, Clone, Debug)]
pub enum DataType {
    Int32,
    Float,
}

/// Enum of the different types of input tensor that an operator can accept.
#[derive(Clone, Copy)]
pub enum Input<'a> {
    FloatTensor(&'a Tensor<f32>),
    IntTensor(&'a Tensor<i32>),
}

impl<'a> Input<'a> {
    pub fn shape(&self) -> &'a [usize] {
        match self {
            Input::FloatTensor(t) => t.shape(),
            Input::IntTensor(t) => t.shape(),
        }
    }

    pub fn as_float(&self) -> Option<&'a Tensor<f32>> {
        if let Input::FloatTensor(t) = self {
            Some(t)
        } else {
            None
        }
    }

    pub fn as_int(&self) -> Option<&'a Tensor<i32>> {
        if let Input::IntTensor(t) = self {
            Some(t)
        } else {
            None
        }
    }
}

impl<'a> From<&'a Tensor<f32>> for Input<'a> {
    fn from(t: &'a Tensor<f32>) -> Input {
        Input::FloatTensor(t)
    }
}

impl<'a> From<&'a Tensor<i32>> for Input<'a> {
    fn from(t: &'a Tensor<i32>) -> Input {
        Input::IntTensor(t)
    }
}

impl<'a> From<&'a Output> for Input<'a> {
    fn from(output: &'a Output) -> Input {
        match output {
            Output::FloatTensor(ref t) => Input::FloatTensor(t),
            Output::IntTensor(ref t) => Input::IntTensor(t),
        }
    }
}

/// Enum of the different types of output tensor that an operator can produce.
pub enum Output {
    FloatTensor(Tensor<f32>),
    IntTensor(Tensor<i32>),
}

impl Output {
    pub fn into_int(self) -> Option<Tensor<i32>> {
        if let Output::IntTensor(t) = self {
            Some(t)
        } else {
            None
        }
    }

    pub fn as_int_ref(&self) -> Option<&Tensor<i32>> {
        if let Output::IntTensor(t) = self {
            Some(t)
        } else {
            None
        }
    }

    pub fn into_float(self) -> Option<Tensor<f32>> {
        if let Output::FloatTensor(t) = self {
            Some(t)
        } else {
            None
        }
    }

    pub fn as_float_ref(&self) -> Option<&Tensor<f32>> {
        if let Output::FloatTensor(t) = self {
            Some(t)
        } else {
            None
        }
    }
}

impl From<Tensor<f32>> for Output {
    fn from(t: Tensor<f32>) -> Output {
        Output::FloatTensor(t)
    }
}

impl From<Tensor<i32>> for Output {
    fn from(t: Tensor<i32>) -> Output {
        Output::IntTensor(t)
    }
}

/// Trait for values that can be converted into the result type used by
/// `Operator::run`.
pub trait IntoOpResult {
    fn into_op_result(self) -> Result<Vec<Output>, OpError>;
}

impl IntoOpResult for Result<Output, OpError> {
    fn into_op_result(self) -> Result<Vec<Output>, OpError> {
        self.map(|out| [out].into())
    }
}

impl IntoOpResult for Output {
    fn into_op_result(self) -> Result<Vec<Output>, OpError> {
        Ok([self].into())
    }
}

impl<T: Copy> IntoOpResult for Tensor<T>
where
    Output: From<Tensor<T>>,
{
    fn into_op_result(self) -> Result<Vec<Output>, OpError> {
        let output: Output = self.into();
        Ok([output].into())
    }
}

impl<T: Copy> IntoOpResult for Result<Tensor<T>, OpError>
where
    Output: From<Tensor<T>>,
{
    fn into_op_result(self) -> Result<Vec<Output>, OpError> {
        self.map(|tensor| [tensor.into()].into())
    }
}

#[derive(Eq, PartialEq, Debug)]
pub enum OpError {
    /// An operator was called with input tensors of a type that is not
    /// supported by the operator.
    UnsupportedInputType,

    /// The shapes of input tensors are not compatible with each other or with
    /// the operator's attributes.
    IncompatibleInputShapes(&'static str),

    /// Two operator inputs which should have the same type, have different types.
    IncompatibleInputTypes(&'static str),

    /// The operator was called with fewer inputs than expected.
    MissingInputs,

    /// An input has a value that is incorrect.
    InvalidValue(&'static str),

    /// An input or attribute has a value that is not currently supported.
    UnsupportedValue(&'static str),
}

/// An Operator is a computation step in a graph.
pub trait Operator: Debug {
    /// Return a display name for the operator.
    fn name(&self) -> &str;

    /// Execute the operator with the given inputs.
    fn run(&self, input: &[Input]) -> Result<Vec<Output>, OpError>;

    /// Return true if this operator supports in-place execution via
    /// `run_in_place`.
    ///
    /// In-place execution writes outputs to an existing tensor rather than
    /// allocating a new tensor. This can speed up execution by reducing the
    /// number of allocations during execution of a computation graph.
    fn can_run_in_place(&self) -> bool {
        false
    }

    /// Execute this operator in-place on an existing tensor.
    ///
    /// `input` is the first input, which the implementation may modify and
    /// return as the output. `other` are the remaining inputs.
    ///
    /// The default implementation just returns the input without modifying it.
    fn run_in_place(&self, input: Output, _other: &[Input]) -> Result<Output, OpError> {
        Ok(input)
    }
}

/// Extract a required float tensor input from `inputs`, or return an error.
pub fn get_input_as_float<'a>(
    inputs: &'a [Input],
    index: usize,
) -> Result<&'a Tensor<f32>, OpError> {
    inputs
        .get(index)
        .ok_or(OpError::MissingInputs)
        .and_then(|input| input.as_float().ok_or(OpError::UnsupportedInputType))
}

/// Extract an optional float tensor input from `inputs`, or return an error.
pub fn get_optional_input_as_float<'a>(
    inputs: &'a [Input],
    index: usize,
) -> Result<Option<&'a Tensor<f32>>, OpError> {
    inputs
        .get(index)
        .map(|input| input.as_float().ok_or(OpError::UnsupportedInputType))
        .transpose()
}

/// Extract a required int tensor input from `inputs`, or return an error.
pub fn get_input_as_int<'a>(inputs: &'a [Input], index: usize) -> Result<&'a Tensor<i32>, OpError> {
    inputs
        .get(index)
        .ok_or(OpError::MissingInputs)
        .and_then(|input| input.as_int().ok_or(OpError::UnsupportedInputType))
}

/// Extract an optional int tensor input from `inputs`, or return an error.
pub fn get_optional_input_as_int<'a>(
    inputs: &'a [Input],
    index: usize,
) -> Result<Option<&'a Tensor<i32>>, OpError> {
    inputs
        .get(index)
        .map(|input| input.as_int().ok_or(OpError::UnsupportedInputType))
        .transpose()
}

#[derive(Debug)]
pub struct Cast {
    pub to: DataType,
}

impl Operator for Cast {
    fn name(&self) -> &str {
        "Cast"
    }

    fn run(&self, inputs: &[Input]) -> Result<Vec<Output>, OpError> {
        let input = inputs.get(0).ok_or(OpError::MissingInputs)?;
        let result: Output = match input {
            Input::IntTensor(t) => match self.to {
                DataType::Int32 => (*t).clone().into(),
                DataType::Float => t.map(|x| x as f32).into(),
            },
            Input::FloatTensor(t) => match self.to {
                DataType::Int32 => t.map(|x| x as i32).into(),
                DataType::Float => (*t).clone().into(),
            },
        };
        result.into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, _: &[Input]) -> Result<Output, OpError> {
        match (input, self.to) {
            (Output::IntTensor(t), DataType::Int32) => Ok(t.into()),
            (Output::FloatTensor(t), DataType::Float) => Ok(t.into()),
            (input, _) => self
                .run(&[(&input).into()])
                .map(|mut outputs| outputs.remove(0)),
        }
    }
}

#[derive(Debug)]
pub struct ConstantOfShape {
    pub value: i32,
}

impl Operator for ConstantOfShape {
    fn name(&self) -> &str {
        "ConstantOfShape"
    }

    fn run(&self, inputs: &[Input]) -> Result<Vec<Output>, OpError> {
        let input = get_input_as_int(inputs, 0)?;
        let shape: Vec<_> = input.elements().map(|el| el as usize).collect();
        let len = shape.iter().product();
        from_data(shape, vec![self.value; len]).into_op_result()
    }
}

/// Gather elements from `input` specified by `indices`.
///
/// See https://onnx.ai/onnx/operators/onnx__Gather.html. Per the ONNX spec this
/// is very similar to `numpy.take`. See
/// https://numpy.org/doc/stable/reference/generated/numpy.take.html for
/// additional explanation.
pub fn gather<T: Copy + Default>(
    input: &Tensor<T>,
    axis: usize,
    indices: &Tensor<i32>,
) -> Result<Tensor<T>, OpError> {
    if axis >= input.ndim() {
        return Err(OpError::InvalidValue("`axis` is out of range"));
    }
    for index in indices.elements() {
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
    let mut output = zero_tensor::<T>(&out_shape);
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

    fn run(&self, inputs: &[Input]) -> Result<Vec<Output>, OpError> {
        let input = inputs.get(0).ok_or(OpError::MissingInputs)?;
        let indices = get_input_as_int(inputs, 1)?;
        match input {
            Input::IntTensor(input) => gather(input, self.axis, indices).into_op_result(),
            Input::FloatTensor(input) => gather(input, self.axis, indices).into_op_result(),
        }
    }
}

#[derive(Debug)]
pub struct Identity {}

impl Operator for Identity {
    fn name(&self) -> &str {
        "Identity"
    }

    fn run(&self, inputs: &[Input]) -> Result<Vec<Output>, OpError> {
        let input = inputs.get(0).ok_or(OpError::MissingInputs)?;
        let result: Output = match input {
            Input::IntTensor(t) => (*t).clone().into(),
            Input::FloatTensor(t) => (*t).clone().into(),
        };
        result.into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, _: &[Input]) -> Result<Output, OpError> {
        Ok(input)
    }
}

pub fn concat<T: Copy>(inputs: &[&Tensor<T>], dim: usize) -> Result<Tensor<T>, OpError> {
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
        out_shape[dim] += other.shape()[dim];
    }
    let mut out_data = Vec::with_capacity(out_shape.iter().product());

    struct ConcatIter<'a, T: Copy> {
        elements: Elements<'a, T>,
        chunk_size: usize,
    }

    let mut input_iters: Vec<ConcatIter<'_, T>> = inputs
        .iter()
        .map(|tensor| ConcatIter {
            elements: tensor.elements(),
            chunk_size: tensor.shape()[dim..].iter().product(),
        })
        .collect();

    while input_iters.iter().any(|it| it.elements.len() > 0) {
        for iter in input_iters.iter_mut() {
            out_data.extend(iter.elements.by_ref().take(iter.chunk_size));
        }
    }

    Ok(from_data(out_shape, out_data))
}

#[derive(Debug)]
pub struct Concat {
    pub dim: usize,
}

impl Operator for Concat {
    fn name(&self) -> &str {
        "Concat"
    }

    /// Run `concat` operator with `[a, b]` inputs.
    fn run(&self, inputs: &[Input]) -> Result<Vec<Output>, OpError> {
        let first = inputs.get(0).ok_or(OpError::MissingInputs)?;

        match first {
            Input::FloatTensor(_) => {
                let typed_inputs: Vec<_> = inputs
                    .iter()
                    .flat_map(|in_| {
                        in_.as_float().ok_or(Err::<&Tensor<f32>, OpError>(
                            OpError::IncompatibleInputTypes("Concat inputs must have same type"),
                        ))
                    })
                    .collect();
                concat(&typed_inputs, self.dim).into_op_result()
            }
            Input::IntTensor(_) => {
                let typed_inputs: Vec<_> = inputs
                    .iter()
                    .flat_map(|in_| {
                        in_.as_int().ok_or(Err::<&Tensor<i32>, OpError>(
                            OpError::IncompatibleInputTypes("Concat inputs must have same type"),
                        ))
                    })
                    .collect();
                concat(&typed_inputs, self.dim).into_op_result()
            }
        }
    }
}

pub fn pad<T: Copy>(
    input: &Tensor<T>,
    padding: &Tensor<i32>,
    const_val: T,
) -> Result<Tensor<T>, OpError> {
    if padding.ndim() != 1 || padding.shape()[0] != input.ndim() * 2 {
        return Err(OpError::InvalidValue(
            "padding should be vector of length 2 * input dimensions",
        ));
    }
    if !padding.elements().all(|x| x >= 0) {
        return Err(OpError::InvalidValue("Pad only supports positive pads"));
    }

    let out_shape: Vec<_> = input
        .shape()
        .iter()
        .enumerate()
        .map(|(i, size)| {
            let start_pad = padding[[i]] as usize;
            let end_pad = padding[[input.ndim() + i]] as usize;
            start_pad + size + end_pad
        })
        .collect();
    let out_len = out_shape.iter().product();

    let mut output = from_data(out_shape, vec![const_val; out_len]);
    let mut in_iter = input.indices();
    let mut out_index = vec![0; output.shape().len()];

    while let Some(in_index) = in_iter.next() {
        out_index.copy_from_slice(in_index);
        for i in 0..out_index.len() {
            out_index[i] += padding[[i]] as usize;
        }
        output[&out_index[..]] = input[in_index];
    }

    Ok(output)
}

fn extract_scalar<T: Copy>(x: &Tensor<T>) -> Result<T, OpError> {
    if let Some(scalar) = x.item() {
        Ok(scalar)
    } else {
        Err(OpError::InvalidValue("Expected scalar value"))
    }
}

fn extract_scalar_int(x: Input) -> Result<i32, OpError> {
    if let Input::IntTensor(val) = x {
        extract_scalar(val)
    } else {
        Err(OpError::IncompatibleInputTypes("Expected int input"))
    }
}

fn extract_scalar_float(x: Input) -> Result<f32, OpError> {
    if let Input::FloatTensor(val) = x {
        extract_scalar(val)
    } else {
        Err(OpError::IncompatibleInputTypes("Expected int input"))
    }
}

#[derive(Debug)]
pub struct Pad {}

impl Operator for Pad {
    fn name(&self) -> &str {
        "Pad"
    }

    fn run(&self, inputs: &[Input]) -> Result<Vec<Output>, OpError> {
        let input = inputs.get(0).ok_or(OpError::MissingInputs)?;
        let pads = get_input_as_int(inputs, 1)?;
        let const_val = inputs.get(2);
        let axes = get_optional_input_as_int(inputs, 3)?;

        if axes.is_some() {
            return Err(OpError::UnsupportedValue(
                "Pad operator does not yet support `axes` input",
            ));
        }

        match input {
            Input::IntTensor(t) => {
                let const_val = const_val.map(|&v| extract_scalar_int(v)).transpose()?;
                pad(t, pads, const_val.unwrap_or(0)).into_op_result()
            }
            Input::FloatTensor(t) => {
                let const_val = const_val.map(|&v| extract_scalar_float(v)).transpose()?;
                pad(t, pads, const_val.unwrap_or(0.0)).into_op_result()
            }
        }
    }
}

fn slice_ranges(
    input_shape: &[usize],
    starts: &Tensor<i32>,
    ends: &Tensor<i32>,
    axes: Option<&Tensor<i32>>,
) -> Vec<SliceRange> {
    let mut ranges: Vec<SliceRange> = input_shape
        .iter()
        .map(|dim_size| SliceRange::new(0, *dim_size as isize, 1))
        .collect();
    for (i, (start, end)) in zip(starts.elements(), ends.elements()).enumerate() {
        let axis = if let Some(axes) = axes {
            axes[[i]] as usize
        } else {
            i
        };
        ranges[axis] = SliceRange::new(start as isize, end as isize, 1);
    }
    ranges
}

/// Return a copy of a tensor which only retains a subset of a given dimension.
pub fn slice<T: Copy>(
    input: &Tensor<T>,
    starts: &Tensor<i32>,
    ends: &Tensor<i32>,
    axes: Option<&Tensor<i32>>,
) -> Tensor<T> {
    let ranges = slice_ranges(input.shape(), starts, ends, axes);
    let sliced_data = input.slice_elements(&ranges).collect();
    let sliced_shape = ranges
        .iter()
        .enumerate()
        .map(|(dim, range)| range.steps(input.shape()[dim]))
        .collect();
    from_data(sliced_shape, sliced_data)
}

/// Clip the dimensions of the input tensor specified by `axes` to the ranges
/// given by `starts` and `ends`. If `axes` is
/// not set, dimensions
pub fn slice_in_place<T: Copy>(
    input: &mut Tensor<T>,
    starts: &Tensor<i32>,
    ends: &Tensor<i32>,
    axes: Option<&Tensor<i32>>,
) {
    let ranges = slice_ranges(input.shape(), starts, ends, axes);
    for (dim, range) in ranges.iter().copied().enumerate() {
        // TODO - Handle negative starts/ends here.
        assert!(
            range.start >= 0 && range.end >= 0,
            "in-place slicing requires positive starts/ends"
        );
        input.clip_dim(dim, range.start as usize, range.end as usize);
    }
}

#[derive(Debug)]
pub struct Slice {}

impl Operator for Slice {
    fn name(&self) -> &str {
        "Slice"
    }

    /// Run `slice` operator with `[input, starts, ends, axes]` inputs.
    fn run(&self, inputs: &[Input]) -> Result<Vec<Output>, OpError> {
        let input = inputs.get(0).ok_or(OpError::MissingInputs)?;
        let starts = get_input_as_int(inputs, 1)?;
        let ends = get_input_as_int(inputs, 2)?;
        let axes = get_optional_input_as_int(inputs, 3)?;
        let result: Output = match input {
            Input::FloatTensor(input) => slice(input, starts, ends, axes).into(),
            Input::IntTensor(input) => slice(input, starts, ends, axes).into(),
        };
        result.into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, other: &[Input]) -> Result<Output, OpError> {
        let mut output = input.into_float().unwrap();
        let starts = other[0].as_int().unwrap();
        let ends = other[1].as_int().unwrap();
        let axes = other.get(2).map(|t| t.as_int().unwrap());
        slice_in_place(&mut output, starts, ends, axes);
        Ok(output.into())
    }
}

/// Resolve an axis given as a value in `[-rank, rank-1]` to the dimension of
/// a tensor with shape `input_shape`, where `rank` is `input_shape.len()`.
///
/// Negative axis values count backwards from the last dimension.
fn resolve_axis(input_shape: &[usize], axis: isize) -> Result<usize, OpError> {
    let rank = input_shape.len() as isize;
    if axis < -rank || axis >= rank {
        return Err(OpError::InvalidValue("axis is invalid"));
    }

    if axis >= 0 {
        Ok(axis as usize)
    } else {
        Ok((rank + axis) as usize)
    }
}

pub fn split<T: Copy>(
    input: &Tensor<T>,
    axis: isize,
    split: &[usize],
) -> Result<Vec<Tensor<T>>, OpError> {
    let axis = resolve_axis(input.shape(), axis)?;
    let split_sum: usize = split.iter().sum();
    if split_sum != input.shape()[axis] {
        return Err(OpError::InvalidValue(
            "split sizes do not sum to dimension size",
        ));
    }

    let mut outputs = Vec::new();
    let mut start = 0;

    for split_size in split {
        let slice_ranges: Vec<SliceRange> = input
            .shape()
            .iter()
            .copied()
            .enumerate()
            .map(|(dim, size)| {
                if dim == axis {
                    SliceRange::new(start as isize, (start + split_size) as isize, 1)
                } else {
                    SliceRange::new(0, size as isize, 1)
                }
            })
            .collect();
        let elements = input.slice_elements(&slice_ranges).collect();
        let slice_shape = zip(input.shape().iter(), slice_ranges)
            .map(|(&dim_size, range)| range.steps(dim_size))
            .collect();
        let tensor = from_data(slice_shape, elements);
        outputs.push(tensor);

        start += split_size;
    }

    Ok(outputs)
}

#[derive(Debug)]
pub struct Split {
    pub axis: isize,
    pub split: Vec<usize>,
}

impl Operator for Split {
    fn name(&self) -> &str {
        "Split"
    }

    fn run(&self, inputs: &[Input]) -> Result<Vec<Output>, OpError> {
        let input = get_input_as_float(inputs, 0)?;
        split(input, self.axis, &self.split[..])
            .map(|tensors| tensors.into_iter().map(|t| t.into()).collect())
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::{
        concat, gather, pad, slice, slice_in_place, split, Cast, ConstantOfShape, DataType,
        Identity, Input, OpError, Operator, Pad,
    };
    use crate::rng::XorShiftRNG;
    use crate::tensor::{from_data, from_scalar, from_vec, random_tensor, zero_tensor, Tensor};
    use crate::test_util::expect_equal;

    #[test]
    fn test_cast() -> Result<(), String> {
        let int_input = from_vec(vec![1, 2, 3]);
        let float_input = from_vec(vec![1.0, 2.0, 3.0]);

        // No-op cast from int32 => int32
        let cast_to_int = Cast {
            to: DataType::Int32,
        };
        let result = cast_to_int
            .run(&[Input::IntTensor(&int_input)])
            .unwrap()
            .remove(0)
            .into_int()
            .unwrap();

        // Flooring cast from float => int32
        assert_eq!(result, int_input);
        let result = cast_to_int
            .run(&[Input::FloatTensor(&float_input)])
            .unwrap()
            .remove(0)
            .into_int()
            .unwrap();
        assert_eq!(&result, &int_input);

        // No-op cast from float => float
        let cast_to_float = Cast {
            to: DataType::Float,
        };
        let result = cast_to_float
            .run(&[Input::FloatTensor(&float_input)])
            .unwrap()
            .remove(0)
            .into_float()
            .unwrap();
        expect_equal(&result, &float_input)?;

        // Cast from int32 => float
        let result = cast_to_float
            .run(&[Input::IntTensor(&int_input)])
            .unwrap()
            .remove(0)
            .into_float()
            .unwrap();
        expect_equal(&result, &float_input)
    }

    #[test]
    fn test_cast_out_of_range() -> Result<(), String> {
        let int_input = from_vec(vec![i32::MIN, i32::MAX]);

        // Out-of-range cast from int => float. This will simply lose some
        // significant digits.
        let cast_to_float = Cast {
            to: DataType::Float,
        };
        let result = cast_to_float
            .run(&[(&int_input).into()])
            .unwrap()
            .remove(0)
            .into_float()
            .unwrap();
        expect_equal(&result, &from_vec(vec![-2147483600.0, 2147483600.0]))?;

        // Out-of-range cast from float => int.
        let float_input = from_vec(vec![f32::MIN, f32::MAX]);
        let cast_to_int = Cast {
            to: DataType::Int32,
        };
        let result = cast_to_int
            .run(&[(&float_input).into()])
            .unwrap()
            .remove(0)
            .into_int()
            .unwrap();
        assert_eq!(&result, &from_vec(vec![i32::MIN, i32::MAX]));

        Ok(())
    }

    #[test]
    fn test_constant_of_shape() {
        let op = ConstantOfShape { value: 42 };
        let shape = from_vec(vec![1, 5, 10]);

        let result = op
            .run(&[Input::IntTensor(&shape)])
            .unwrap()
            .remove(0)
            .into_int()
            .unwrap();

        assert_eq!(result.shape(), &[1, 5, 10]);
        assert_eq!(
            result.elements().collect::<Vec<_>>(),
            vec![42; result.shape().iter().product()]
        );
    }

    #[test]
    fn test_gather_scalar() {
        let input = from_vec(vec![1, 20, 30]);
        for i in 0..input.len() {
            let indices = from_scalar(i as i32);
            let result = gather(&input, 0, &indices).unwrap();
            assert_eq!(result.item(), Some(input[[i]]))
        }
    }

    #[test]
    fn test_gather() -> Result<(), String> {
        // Test case shrunk down from a small BERT model where `gather` is used
        // to lookup up embeddings.
        let mut rng = XorShiftRNG::new(1234);
        let input = random_tensor(&[128, 10], &mut rng);
        let indices = from_data(vec![2, 2], vec![2, 5, 8, 50]);
        let result = gather(&input, 0, &indices).unwrap();
        assert_eq!(result.shape(), &[2, 2, 10]);

        // Test case #1 from ONNX spec.
        let input = from_data(vec![3, 2], vec![1.0, 1.2, 2.3, 3.4, 4.5, 5.7]);
        let indices = from_data(vec![2, 2], vec![0, 1, 1, 2]);
        let expected = from_data(vec![2, 2, 2], vec![1.0, 1.2, 2.3, 3.4, 2.3, 3.4, 4.5, 5.7]);
        let result = gather(&input, 0, &indices).unwrap();
        expect_equal(&result, &expected)?;

        // Test case #2 from ONNX spec.
        let input = from_data(
            vec![3, 3],
            vec![1.0, 1.2, 1.9, 2.3, 3.4, 3.9, 4.5, 5.7, 5.9],
        );
        let indices = from_data(vec![1, 2], vec![0, 2]);
        let expected = from_data(vec![3, 1, 2], vec![1.0, 1.9, 2.3, 3.9, 4.5, 5.9]);
        let result = gather(&input, 1, &indices).unwrap();
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_gather_invalid_inputs() {
        let mut rng = XorShiftRNG::new(1234);
        let input = random_tensor(&[128, 10], &mut rng);
        let indices = from_data(vec![2, 2], vec![2, 5, 8, 50]);
        let result = gather(&input, 5, &indices);
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("`axis` is out of range"))
        );

        let indices = from_data(vec![2, 2], vec![2, 5, 8, 130]);
        let result = gather(&input, 0, &indices);
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Entry in `indices` is out of range"))
        );
    }

    #[test]
    fn test_identity() -> Result<(), String> {
        let id_op = Identity {};

        let int_input = from_vec(vec![1, 2, 3]);
        let result = id_op
            .run(&[Input::IntTensor(&int_input)])
            .unwrap()
            .remove(0)
            .into_int()
            .unwrap();
        assert_eq!(result, int_input);

        let float_input = from_vec(vec![1.0, 2.0, 3.0]);
        let result = id_op
            .run(&[Input::FloatTensor(&float_input)])
            .unwrap()
            .remove(0)
            .into_float()
            .unwrap();
        expect_equal(&result, &float_input)
    }

    #[test]
    fn test_concat() -> Result<(), String> {
        let a = from_data(vec![2, 2, 1], vec![0.1, 0.2, 0.3, 0.4]);
        let b = from_data(vec![2, 2, 1], vec![1.0, 2.0, 3.0, 4.0]);

        // Concatenation along the first dimension
        let expected = from_data(vec![4, 2, 1], vec![0.1, 0.2, 0.3, 0.4, 1.0, 2.0, 3.0, 4.0]);
        let result = concat(&[&a, &b], 0).unwrap();
        expect_equal(&result, &expected)?;

        // Concatenation along a non-first dimension
        let expected = from_data(vec![2, 2, 2], vec![0.1, 1.0, 0.2, 2.0, 0.3, 3.0, 0.4, 4.0]);
        let result = concat(&[&a, &b], 2).unwrap();
        expect_equal(&result, &expected)?;

        // Concatenation with one input
        let result = concat(&[&a], 0).unwrap();
        expect_equal(&result, &a)?;

        // Concatenation with more than two inputs
        let result = concat(&[&a, &b, &a], 0).unwrap();
        assert_eq!(result.shape(), &[6, 2, 1]);

        // Concatentation with some empty inputs
        let a = from_slice(&[1, 2, 3]);
        let b = from_slice(&[]);
        let c = from_slice(&[4, 5, 6]);
        let result = concat(&[&a, &b, &c], 0).unwrap();
        assert_eq!(result.shape(), &[6]);
        assert_eq!(result.data(), &[1, 2, 3, 4, 5, 6]);

        Ok(())
    }

    #[test]
    fn test_concat_invalid_inputs() {
        // Invalid `dim` attribute
        let input = from_slice(&[1, 2, 3]);
        let result = concat(&[&input, &input], 1);
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("dim is larger than input rank"))
        );

        // Shape mismatch
        let a = zero_tensor::<f32>(&[1]);
        let b = zero_tensor::<f32>(&[1, 2]);
        let result = concat(&[&a, &b], 0);
        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes(
                "Tensors must have the same number of dimensions"
            ))
        );

        // Shape mismatch in non-`dim` dimension
        let a = zero_tensor::<f32>(&[5, 10]);
        let b = zero_tensor::<f32>(&[5, 11]);
        let result = concat(&[&a, &b], 0);
        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes(
                "Dimensions must be the same except for concat dim"
            ))
        );
    }

    #[test]
    fn test_pad() -> Result<(), String> {
        // Same padding around each edge.
        let input = from_data(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let expected = from_data(
            vec![4, 4],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        );
        let const_pads = from_slice(&[1, 1, 1, 1]);
        let result = pad(&input, &const_pads, 0.0).unwrap();
        expect_equal(&result, &expected)?;

        // Zero padding (no-op)
        let zero_pads = from_slice(&[0, 0, 0, 0]);
        let result = pad(&input, &zero_pads, 0.0).unwrap();
        expect_equal(&result, &input)?;

        // Un-even padding
        let input = from_data(vec![1, 2, 2], vec![1, 2, 3, 4]);
        let pads = from_slice(&[0, 0, 0, 0, 1, 0]);
        let result = pad(&input, &pads, 0).unwrap();
        assert_eq!(result.shape(), &[1, 3, 2]);
        assert_eq!(result.data(), &[1, 2, 3, 4, 0, 0]);

        Ok(())
    }

    #[test]
    fn test_pad_constant_val() -> Result<(), String> {
        let input = from_data(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let expected = from_data(
            vec![4, 4],
            vec![
                9., 9., 9., 9., 9., 1., 2., 9., 9., 3., 4., 9., 9., 9., 9., 9.,
            ],
        );
        let const_pads = from_slice(&[1, 1, 1, 1]);
        let result = pad(&input, &const_pads, 9.).unwrap();
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_pad_op() -> Result<(), String> {
        let input = from_data(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let pads = from_slice(&[1, 1, 1, 1]);
        let expected = from_data(
            vec![4, 4],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        );

        let op = Pad {};
        let result = op
            .run(&[(&input).into(), (&pads).into()])
            .unwrap()
            .remove(0)
            .into_float()
            .unwrap();
        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_pad_invalid_inputs() {
        let input = from_data(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let op = Pad {};

        // Wrong padding vector length.
        let invalid_pads = from_slice(&[1]);
        let result = op.run(&[(&input).into(), (&invalid_pads).into()]);
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue(
                "padding should be vector of length 2 * input dimensions"
            ))
        );

        // Unsupported padding amounts.
        let invalid_pads = from_slice(&[1, 1, 1, -1]);
        let result = op.run(&[(&input).into(), (&invalid_pads).into()]);
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Pad only supports positive pads"))
        );

        // Wrong constant value type.
        let invalid_pads = from_slice(&[1, 1, 1, -1]);
        let const_int = from_scalar(1);
        let result = op.run(&[(&input).into(), (&invalid_pads).into(), (&const_int).into()]);
        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputTypes("Expected int input"))
        );

        // Constant value not a scalar.
        let invalid_pads = from_slice(&[1, 1, 1, -1]);
        let int_vec = from_slice(&[1.0, 2.0]);
        let result = op.run(&[(&input).into(), (&invalid_pads).into(), (&int_vec).into()]);
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Expected scalar value"))
        );
    }

    fn from_slice<T: Copy>(data: &[T]) -> Tensor<T> {
        from_data(vec![data.len()], data.into())
    }

    #[test]
    fn test_slice_in_place() {
        let mut rng = XorShiftRNG::new(5678);
        let mut input = random_tensor(&[2, 2, 5, 3], &mut rng);

        let starts = from_slice(&[2]);
        let ends = from_slice(&[4]);
        let axes = from_slice(&[2]);

        slice_in_place(&mut input, &starts, &ends, Some(&axes));

        assert_eq!(
            input.shape(),
            vec![2, 2, ends[[0]] as usize - starts[[0]] as usize, 3]
        );
    }

    #[test]
    fn test_slice_not_first_dim() {
        let mut rng = XorShiftRNG::new(5678);
        let input = random_tensor(&[2, 2, 5, 3], &mut rng);

        let starts = from_slice(&[2]);
        let ends = from_slice(&[4]);
        let axes = from_slice(&[2]);

        let sliced = slice(&input, &starts, &ends, Some(&axes));
        let shape = sliced.shape();

        assert_eq!(
            sliced.shape(),
            vec![2, 2, ends[[0]] as usize - starts[[0]] as usize, 3]
        );
        assert_eq!(sliced.len(), shape.iter().fold(1, |len, x| len * x));

        for w in 0..shape[0] {
            for x in 0..shape[1] {
                for y in 0..shape[2] {
                    for z in 0..shape[3] {
                        assert_eq!(
                            sliced[[w, x, y, z]],
                            input[[w, x, y + starts[[0]] as usize, z]]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_slice_first_dim() {
        let mut rng = XorShiftRNG::new(5678);
        let input = random_tensor(&[5, 2, 5, 3], &mut rng);

        let starts = from_slice(&[2]);
        let ends = from_slice(&[4]);
        let axes = from_slice(&[0]);

        let sliced = slice(&input, &starts, &ends, Some(&axes));
        let shape = sliced.shape();

        assert_eq!(
            shape,
            vec![ends[[0]] as usize - starts[[0]] as usize, 2, 5, 3]
        );
        assert_eq!(sliced.len(), shape.iter().fold(1, |len, x| len * x));

        for w in 0..shape[0] {
            for x in 0..shape[1] {
                for y in 0..shape[2] {
                    for z in 0..shape[3] {
                        assert_eq!(
                            sliced[[w, x, y, z]],
                            input[[w + starts[[0]] as usize, x, y, z]]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_slice_noop() {
        let mut rng = XorShiftRNG::new(5678);
        let input = random_tensor(&[5, 2, 5, 3], &mut rng);

        for dim in 0..input.shape().len() {
            let dim_size = input.shape()[dim] as i32;

            let starts = from_slice(&[0]);
            let ends = from_slice(&[dim_size]);
            let axes = from_slice(&[dim as i32]);

            let sliced = slice(&input, &starts, &ends, Some(&axes));
            assert_eq!(sliced.shape(), input.shape());
            assert_eq!(sliced.data(), input.data());
        }
    }

    #[test]
    fn test_split() {
        let input = from_data(vec![5, 2], vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        // Split with positive axis
        let results = split(&input, 1, &[1, 1]).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].data(), &[0., 2., 4., 6., 8.]);
        assert_eq!(results[1].data(), &[1., 3., 5., 7., 9.]);

        // Split with negative axis
        let results = split(&input, -1, &[1, 1]).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].data(), &[0., 2., 4., 6., 8.]);
        assert_eq!(results[1].data(), &[1., 3., 5., 7., 9.]);
    }

    #[test]
    fn test_split_invalid_inputs() {
        let input = from_data(vec![5, 2], vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        let result = split(&input, 2, &[1, 1]);
        assert_eq!(result.err(), Some(OpError::InvalidValue("axis is invalid")));

        let result = split(&input, -3, &[1, 1]);
        assert_eq!(result.err(), Some(OpError::InvalidValue("axis is invalid")));

        let result = split(&input, 1, &[1, 2]);
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue(
                "split sizes do not sum to dimension size"
            ))
        );
    }
}
