use std::error::Error;
use std::fmt;
use std::fmt::{Debug, Display};
use std::iter::zip;

use crate::tensor::{from_data, zeros, Elements, SliceRange, Tensor};

mod binary_elementwise;
mod conv;
mod convert;
mod generate;
mod layout;
mod matmul;
mod norm;
mod pad;
mod pooling;
mod reduce;
mod resize;
mod unary_elementwise;

pub use binary_elementwise::{
    add, add_in_place, choose_broadcast_shape, div, div_in_place, equal, less, mul, mul_in_place,
    pow, pow_in_place, sub, sub_in_place, where_op,
};
pub use binary_elementwise::{Add, Div, Equal, Less, Mul, Pow, Sub, Where};
pub use conv::{conv, conv_transpose};
pub use conv::{Conv, ConvTranspose};
pub use convert::Cast;
pub use generate::{constant_of_shape, range, ConstantOfShape, Range};
pub use layout::{
    expand, reshape, squeeze, squeeze_in_place, Expand, Reshape, Shape, Squeeze, Transpose,
    Unsqueeze,
};
pub use matmul::{gemm_op, matmul, Gemm, MatMul};
pub use norm::{batch_norm, batch_norm_in_place, softmax, BatchNormalization, Softmax};
pub use pad::{pad, Pad};
pub use pooling::{average_pool, global_average_pool, max_pool};
pub use pooling::{AveragePool, GlobalAveragePool, MaxPool};
pub use reduce::{reduce_mean, ReduceMean};
pub use resize::{resize, Resize, ResizeMode, ResizeTarget};
pub use unary_elementwise::{
    clip, clip_in_place, cos, cos_in_place, erf, erf_in_place, leaky_relu, leaky_relu_in_place,
    relu, relu_in_place, sigmoid, sigmoid_in_place, sin, sin_in_place, sqrt, sqrt_in_place, tanh,
    tanh_in_place,
};
pub use unary_elementwise::{Clip, Cos, Erf, LeakyRelu, Relu, Sigmoid, Sin, Sqrt, Tanh};

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
}

impl<'a> TryFrom<Input<'a>> for &'a Tensor<f32> {
    type Error = OpError;

    fn try_from(input: Input<'a>) -> Result<&'a Tensor<f32>, Self::Error> {
        match input {
            Input::FloatTensor(t) => Ok(t),
            _ => Err(OpError::IncorrectInputType),
        }
    }
}

impl<'a> TryFrom<Input<'a>> for &'a Tensor<i32> {
    type Error = OpError;

    fn try_from(input: Input<'a>) -> Result<&'a Tensor<i32>, Self::Error> {
        match input {
            Input::IntTensor(t) => Ok(t),
            _ => Err(OpError::IncorrectInputType),
        }
    }
}

impl<'a> TryFrom<Input<'a>> for f32 {
    type Error = OpError;

    fn try_from(input: Input<'a>) -> Result<f32, Self::Error> {
        let tensor: &Tensor<_> = input.try_into()?;
        tensor
            .item()
            .ok_or(OpError::InvalidValue("Expected scalar value"))
    }
}

impl<'a> TryFrom<Input<'a>> for i32 {
    type Error = OpError;

    fn try_from(input: Input<'a>) -> Result<i32, Self::Error> {
        let tensor: &Tensor<_> = input.try_into()?;
        tensor
            .item()
            .ok_or(OpError::InvalidValue("Expected scalar value"))
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

/// Possible reasons why an operator may fail on a given input.
#[derive(Eq, PartialEq, Debug)]
pub enum OpError {
    /// Input tensors have an element type that is unsupported or incompatible
    /// with other inputs.
    IncorrectInputType,

    /// Input tensor shapes are not compatible with each other or operator
    /// attributes.
    IncompatibleInputShapes(&'static str),

    /// The number of inputs was less than the required number.
    MissingInputs,

    /// An input has a value that is incorrect.
    InvalidValue(&'static str),

    /// An input or attribute has a value that is valid, but not currently supported.
    UnsupportedValue(&'static str),
}

impl Display for OpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpError::IncorrectInputType => write!(f, "incorrect or unsupported input type"),
            OpError::IncompatibleInputShapes(details) => {
                write!(f, "incompatible input shapes: {}", details)
            }
            OpError::MissingInputs => write!(f, "required inputs were missing"),
            OpError::InvalidValue(details) => {
                write!(f, "input or attribute has invalid value: {}", details)
            }
            OpError::UnsupportedValue(details) => {
                write!(f, "unsupported input or attribute value: {}", details)
            }
        }
    }
}

impl Error for OpError {}

/// An Operator performs a computation step when executing a data flow graph.
///
/// Operators take zero or more dynamic input values, plus a set of static
/// attributes and produce one or more output values.
///
/// Operators are usually named after the ONNX operator that they implement.
/// See https://onnx.ai/onnx/operators/.
pub trait Operator: Debug {
    /// Return a display name for the operator.
    fn name(&self) -> &str;

    /// Execute the operator with the given inputs.
    fn run(&self, input: InputList) -> Result<Vec<Output>, OpError>;

    /// Return true if this operator supports in-place execution via
    /// `run_in_place`.
    ///
    /// In-place execution returns results by modifying an existing tensor
    /// instead of allocating a new one. Reducing memory allocations can
    /// significantly speed up graph runs.
    fn can_run_in_place(&self) -> bool {
        false
    }

    /// Execute this operator in-place on an existing tensor.
    ///
    /// This may only be called if `can_run_in_place` returns true.
    ///
    /// `input` is the first input, which the implementation may modify and
    /// return as the output. `other` are the remaining inputs.
    fn run_in_place(&self, _input: Output, _other: InputList) -> Result<Output, OpError> {
        unimplemented!("in-place execution not supported")
    }
}

/// List of inputs for an operator evaluation.
///
/// Conceptually this is like a `&[Option<Input>]` with methods to conveniently
/// extract inputs and produce appropriate errors if inputs are missing or of
/// the wrong type.
pub struct InputList<'a> {
    inputs: Vec<Option<Input<'a>>>,
}

impl<'a> InputList<'a> {
    pub fn from<'b>(inputs: &'b [Input<'b>]) -> InputList<'b> {
        InputList {
            inputs: inputs.iter().copied().map(Some).collect(),
        }
    }

    pub fn from_optional<'b>(inputs: &'b [Option<Input<'b>>]) -> InputList<'b> {
        InputList {
            inputs: inputs.to_vec(),
        }
    }

    /// Get an optional input.
    pub fn get(&self, index: usize) -> Option<Input<'a>> {
        self.inputs.get(index).copied().flatten()
    }

    /// Get an optional input as a tensor.
    pub fn get_as<T: Copy>(&self, index: usize) -> Result<Option<&'a Tensor<T>>, OpError>
    where
        &'a Tensor<T>: TryFrom<Input<'a>, Error = OpError>,
    {
        self.get(index).map(|input| input.try_into()).transpose()
    }

    /// Get an optional input as a scalar value.
    pub fn get_as_scalar<T: Copy + 'a>(&self, index: usize) -> Result<Option<T>, OpError>
    where
        &'a Tensor<T>: TryFrom<Input<'a>, Error = OpError>,
    {
        let tensor = self.get_as::<T>(index)?;
        tensor
            .map(|t| {
                t.item()
                    .ok_or(OpError::InvalidValue("Expected scalar value"))
            })
            .transpose()
    }

    /// Get a required operator input.
    pub fn require(&self, index: usize) -> Result<Input<'a>, OpError> {
        self.get(index).ok_or(OpError::MissingInputs)
    }

    /// Get a required operator input as a tensor.
    pub fn require_as<T: Copy>(&self, index: usize) -> Result<&'a Tensor<T>, OpError>
    where
        &'a Tensor<T>: TryFrom<Input<'a>, Error = OpError>,
    {
        self.require(index).and_then(|input| input.try_into())
    }

    #[allow(dead_code)] // Not currently used, but exists for consistency with `get_as_scalar`
    pub fn require_as_scalar<T: Copy + 'a>(&self, index: usize) -> Result<T, OpError>
    where
        T: TryFrom<Input<'a>, Error = OpError>,
    {
        self.require(index).and_then(|input| input.try_into())
    }

    /// Return an iterator over provided inputs.
    ///
    /// If the InputList was constructed with `from_optional`, this will skip
    /// over any missing inputs.
    pub fn iter(&'a self) -> impl Iterator<Item = Input<'a>> + 'a {
        self.inputs.iter().filter_map(|inp| *inp)
    }
}

#[derive(Debug)]
pub enum Scalar {
    Int(i32),
    Float(f32),
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
    let mut output = zeros::<T>(&out_shape);
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

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let result: Output = match input {
            Input::IntTensor(t) => (*t).clone().into(),
            Input::FloatTensor(t) => (*t).clone().into(),
        };
        result.into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, _: InputList) -> Result<Output, OpError> {
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

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let first = inputs.require(0)?;
        match first {
            Input::FloatTensor(_) => {
                let mut typed_inputs: Vec<_> = Vec::new();
                for input in inputs.iter() {
                    let tensor: &Tensor<f32> = input.try_into()?;
                    typed_inputs.push(tensor);
                }
                concat(&typed_inputs, self.dim).into_op_result()
            }
            Input::IntTensor(_) => {
                let mut typed_inputs: Vec<_> = Vec::new();
                for input in inputs.iter() {
                    let tensor: &Tensor<i32> = input.try_into()?;
                    typed_inputs.push(tensor);
                }
                concat(&typed_inputs, self.dim).into_op_result()
            }
        }
    }
}

/// Compute the effective starts, ends and steps for each input dimension in
/// a Slice operation.
///
/// See https://onnx.ai/onnx/operators/onnx__Slice.html.
fn slice_ranges(
    input_shape: &[usize],
    starts: &Tensor<i32>,
    ends: &Tensor<i32>,
    axes: Option<&Tensor<i32>>,
    steps: Option<&Tensor<i32>>,
) -> Result<Vec<SliceRange>, OpError> {
    // FIXME: Verify that `starts`, `ends`, `axes` and `steps` are vectors with
    // compatible lengths.

    if let Some(steps) = steps {
        if steps.ndim() != 1 {
            return Err(OpError::InvalidValue("`steps` should be a vector"));
        }
        for step in steps.elements() {
            if step == 0 {
                return Err(OpError::InvalidValue("steps must be non-zero"));
            }
        }
    }

    let mut ranges: Vec<SliceRange> = input_shape
        .iter()
        .map(|dim_size| SliceRange::new(0, *dim_size as isize, 1))
        .collect();
    for (i, (start, end)) in zip(starts.elements(), ends.elements()).enumerate() {
        let axis = if let Some(axes) = axes {
            resolve_axis(input_shape.len(), axes[[i]] as isize)?
        } else {
            i
        };

        let step = steps.map(|s| s[[i]]).unwrap_or(1);
        ranges[axis] = SliceRange::new(start as isize, end as isize, step as isize);
    }
    Ok(ranges)
}

/// Return a copy of a tensor which only retains a subset of a given dimension.
pub fn slice<T: Copy>(
    input: &Tensor<T>,
    starts: &Tensor<i32>,
    ends: &Tensor<i32>,
    axes: Option<&Tensor<i32>>,
    steps: Option<&Tensor<i32>>,
) -> Result<Tensor<T>, OpError> {
    let ranges = slice_ranges(input.shape(), starts, ends, axes, steps)?;
    let sliced_data = input.slice_elements(&ranges).collect();
    let sliced_shape = ranges
        .iter()
        .enumerate()
        .map(|(dim, range)| range.steps(input.shape()[dim]))
        .collect();
    Ok(from_data(sliced_shape, sliced_data))
}

/// Clip the dimensions of the input tensor specified by `axes` to the ranges
/// given by `starts` and `ends`.
pub fn slice_in_place<T: Copy>(
    input: &mut Tensor<T>,
    starts: &Tensor<i32>,
    ends: &Tensor<i32>,
    axes: Option<&Tensor<i32>>,
) -> Result<(), OpError> {
    let ranges = slice_ranges(input.shape(), starts, ends, axes, None)?;
    for (dim, range) in ranges.iter().enumerate() {
        // TODO - Handle negative `range.start` and `range.end` here.
        assert!(
            range.start >= 0 && range.end >= 0,
            "in-place slicing requires positive starts/ends"
        );
        input.clip_dim(dim, range.start as usize, range.end as usize);
    }
    Ok(())
}

#[derive(Debug)]
pub struct Slice {}

impl Operator for Slice {
    fn name(&self) -> &str {
        "Slice"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let starts = inputs.require_as::<i32>(1)?;
        let ends = inputs.require_as::<i32>(2)?;
        let axes = inputs.get_as::<i32>(3)?;
        let steps = inputs.get_as::<i32>(4)?;

        let result: Result<Output, OpError> = match input {
            Input::FloatTensor(input) => slice(input, starts, ends, axes, steps).map(|t| t.into()),
            Input::IntTensor(input) => slice(input, starts, ends, axes, steps).map(|t| t.into()),
        };
        result.into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, other: InputList) -> Result<Output, OpError> {
        let starts = other.require_as::<i32>(0)?;
        let ends = other.require_as::<i32>(1)?;
        let axes = other.get_as::<i32>(2)?;
        let steps = other.get_as::<i32>(3)?;

        // Fall back to copying if non-default steps are given.
        if let Some(steps) = steps {
            if steps.elements().any(|step| step != 1) {
                let mut inputs: Vec<_> = vec![(&input).into()];
                inputs.extend(other.iter());
                return self
                    .run(InputList::from(&inputs))
                    .map(|mut outputs| outputs.remove(0));
            }
        }

        match input {
            Output::IntTensor(mut output) => {
                slice_in_place(&mut output, starts, ends, axes)?;
                Ok(output.into())
            }
            Output::FloatTensor(mut output) => {
                slice_in_place(&mut output, starts, ends, axes)?;
                Ok(output.into())
            }
        }
    }
}

/// Resolve an axis given as a value in `[-ndim, ndim-1]` to the zero-based
/// dimension of a tensor with `ndim` dimensions.
///
/// Negative axis values count backwards from the last dimension.
fn resolve_axis(ndim: usize, axis: isize) -> Result<usize, OpError> {
    let rank = ndim as isize;
    if axis < -rank || axis >= rank {
        return Err(OpError::InvalidValue("axis is invalid"));
    }

    if axis >= 0 {
        Ok(axis as usize)
    } else {
        Ok((rank + axis) as usize)
    }
}

/// Resolve an array of axes values in `[-ndim, ndim-1]` to zero-based dimension
/// indexes in a tensor with `ndim` dimensions.
///
/// Negative axis values count backwards from the last dimension.
pub fn resolve_axes(ndim: usize, axes: &[i32]) -> Result<Vec<usize>, OpError> {
    let mut resolved_axes = Vec::with_capacity(axes.len());
    for &axis in axes {
        let resolved = resolve_axis(ndim, axis as isize)?;
        resolved_axes.push(resolved);
    }
    Ok(resolved_axes)
}

pub fn split<T: Copy>(
    input: &Tensor<T>,
    axis: isize,
    split: &[usize],
) -> Result<Vec<Tensor<T>>, OpError> {
    let axis = resolve_axis(input.ndim(), axis)?;
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

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require_as::<f32>(0)?;
        split(input, self.axis, &self.split[..])
            .map(|tensors| tensors.into_iter().map(|t| t.into()).collect())
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::{
        concat, gather, slice, slice_in_place, split, Identity, Input, InputList, OpError, Operator,
    };
    use crate::rng::XorShiftRNG;
    use crate::tensor::{from_data, from_scalar, from_vec, rand, zeros, Tensor};
    use crate::test_util::expect_equal;

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
        let input = rand(&[128, 10], &mut rng);
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
        let input = rand(&[128, 10], &mut rng);
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
            .run(InputList::from(&[Input::IntTensor(&int_input)]))
            .unwrap()
            .remove(0)
            .into_int()
            .unwrap();
        assert_eq!(result, int_input);

        let float_input = from_vec(vec![1.0, 2.0, 3.0]);
        let result = id_op
            .run(InputList::from(&[Input::FloatTensor(&float_input)]))
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
        let a = zeros::<f32>(&[1]);
        let b = zeros::<f32>(&[1, 2]);
        let result = concat(&[&a, &b], 0);
        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes(
                "Tensors must have the same number of dimensions"
            ))
        );

        // Shape mismatch in non-`dim` dimension
        let a = zeros::<f32>(&[5, 10]);
        let b = zeros::<f32>(&[5, 11]);
        let result = concat(&[&a, &b], 0);
        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes(
                "Dimensions must be the same except for concat dim"
            ))
        );
    }

    fn from_slice<T: Copy>(data: &[T]) -> Tensor<T> {
        from_data(vec![data.len()], data.into())
    }

    #[test]
    fn test_slice_in_place() {
        let mut rng = XorShiftRNG::new(5678);
        let mut input = rand(&[2, 2, 5, 3], &mut rng);

        let starts = from_slice(&[2]);
        let ends = from_slice(&[4]);
        let axes = from_slice(&[2]);

        slice_in_place(&mut input, &starts, &ends, Some(&axes)).unwrap();

        assert_eq!(
            input.shape(),
            vec![2, 2, ends[[0]] as usize - starts[[0]] as usize, 3]
        );
    }

    #[test]
    fn test_slice_first_dim() {
        let mut rng = XorShiftRNG::new(5678);
        let input = rand(&[5, 2, 5, 3], &mut rng);

        let starts = from_slice(&[2]);
        let ends = from_slice(&[4]);
        let axes = from_slice(&[0]);

        let sliced = slice(&input, &starts, &ends, Some(&axes), None).unwrap();
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
    fn test_slice_inner_dim() {
        let mut rng = XorShiftRNG::new(5678);
        let input = rand(&[2, 2, 5, 3], &mut rng);

        let starts = from_slice(&[2]);
        let ends = from_slice(&[4]);
        let axes = from_slice(&[2]);

        let sliced = slice(&input, &starts, &ends, Some(&axes), None).unwrap();
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
    fn test_slice_noop() {
        let mut rng = XorShiftRNG::new(5678);
        let input = rand(&[5, 2, 5, 3], &mut rng);

        for dim in 0..input.shape().len() {
            let dim_size = input.shape()[dim] as i32;

            let starts = from_slice(&[0]);
            let ends = from_slice(&[dim_size]);
            let axes = from_slice(&[dim as i32]);

            let sliced = slice(&input, &starts, &ends, Some(&axes), None).unwrap();
            assert_eq!(sliced.shape(), input.shape());
            assert_eq!(sliced.data(), input.data());
        }
    }

    #[test]
    fn test_slice_negative_axes() {
        let input = from_data(vec![3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let starts = from_slice(&[0]);
        let ends = from_slice(&[2]);

        let axes = from_slice(&[-1]);
        let sliced = slice(&input, &starts, &ends, Some(&axes), None).unwrap();
        assert_eq!(sliced.elements().collect::<Vec<_>>(), &[1, 2, 4, 5, 7, 8]);

        let axes = from_slice(&[-2]);
        let sliced = slice(&input, &starts, &ends, Some(&axes), None).unwrap();
        assert_eq!(sliced.elements().collect::<Vec<_>>(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_slice_negative_starts() {
        let input = from_data(vec![3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let axes = from_slice(&[-1]);
        let ends = from_slice(&[2]);

        let starts = from_slice(&[-3]);
        let sliced = slice(&input, &starts, &ends, Some(&axes), None).unwrap();
        assert_eq!(sliced.elements().collect::<Vec<_>>(), &[1, 2, 4, 5, 7, 8]);

        let starts = from_slice(&[-2]);
        let sliced = slice(&input, &starts, &ends, Some(&axes), None).unwrap();
        assert_eq!(sliced.elements().collect::<Vec<_>>(), &[2, 5, 8]);
    }

    #[test]
    fn test_slice_negative_ends() {
        let input = from_data(vec![3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let axes = from_slice(&[-1]);
        let starts = from_slice(&[0]);

        let ends = from_slice(&[-1]);
        let sliced = slice(&input, &starts, &ends, Some(&axes), None).unwrap();
        assert_eq!(sliced.elements().collect::<Vec<_>>(), &[1, 2, 4, 5, 7, 8]);

        let ends = from_slice(&[-2]);
        let sliced = slice(&input, &starts, &ends, Some(&axes), None).unwrap();
        assert_eq!(sliced.elements().collect::<Vec<_>>(), &[1, 4, 7]);
    }

    #[test]
    fn test_slice_clamps_starts_and_ends() -> Result<(), String> {
        let mut rng = XorShiftRNG::new(5678);
        let input = rand(&[20, 20], &mut rng);

        // Simulate how a range without a start/end may be given in a model.
        //
        // The ONNX Slice spec does not support unbounded ranges (like
        // `array[start:]` in numpy) but instead recommends the use of INT_MAX /
        // -INT_MAX together with clamping to achieve the same result.
        let starts = from_slice(&[-i32::MAX, -100]);
        let ends = from_slice(&[i32::MAX, 100]);

        let sliced = slice(&input, &starts, &ends, None, None).unwrap();

        expect_equal(&sliced, &input)
    }

    #[test]
    fn test_slice_with_step() {
        let input = from_slice(&[1, 2, 3, 4, 5]);

        struct Case<'a> {
            start: i32,
            end: i32,
            step: i32,
            expected_shape: &'a [usize],
            expected_elements: &'a [i32],
        }

        let cases = [
            // Positive step > 1
            Case {
                start: 0,
                end: 5,
                step: 2,
                expected_shape: &[3],
                expected_elements: &[1, 3, 5],
            },
            // Negative step
            Case {
                start: 5,
                end: -6,
                step: -1,
                expected_shape: &[5],
                expected_elements: &[5, 4, 3, 2, 1],
            },
            // Negative step with clamped start
            Case {
                start: 100,
                end: -6,
                step: -1,
                expected_shape: &[5],
                expected_elements: &[5, 4, 3, 2, 1],
            },
            // Negative step with clamped end
            Case {
                start: 5,
                end: -100,
                step: -1,
                expected_shape: &[5],
                expected_elements: &[5, 4, 3, 2, 1],
            },
        ];

        for case in cases {
            let starts = from_slice(&[case.start]);
            let ends = from_slice(&[case.end]);
            let axes = from_slice(&[0]);
            let steps = from_slice(&[case.step]);

            let sliced = slice(&input, &starts, &ends, Some(&axes), Some(&steps)).unwrap();

            assert_eq!(sliced.shape(), case.expected_shape);
            assert_eq!(sliced.data(), case.expected_elements);
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
