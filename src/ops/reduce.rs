use std::borrow::Cow;
use std::cmp::Ordering;

use rten_tensor;
use rten_tensor::prelude::*;
use rten_tensor::{DynIndices, NdTensor, NdTensorView, SliceItem, Tensor, TensorView};
use rten_vecmath::{vec_sum, vec_sum_square};

use crate::number::{Identities, IsNaN};
use crate::ops::layout::squeeze_in_place;
use crate::ops::{
    resolve_axes, resolve_axis, Input, InputList, IntoOpResult, OpError, Operator, OutputList,
};
use crate::slice_reductions::slice_sum;
use crate::tensor_pool::{AutoReturn, TensorPool};

/// Compute the indices of the max elements along an axis, according to a
/// comparison function `compare`.
fn select_max_index<T, Cmp: Fn(&T, &T) -> std::cmp::Ordering>(
    pool: &TensorPool,
    input: TensorView<T>,
    axis: isize,
    keep_dims: bool,
    compare: Cmp,
) -> Result<Tensor<i32>, OpError> {
    let resolved_axis = resolve_axis(input.ndim(), axis)?;
    if input.size(resolved_axis) == 0 {
        return Err(OpError::InvalidValue(
            "Cannot select index from empty sequence",
        ));
    }

    let reduced_shape: Vec<usize> = input
        .shape()
        .iter()
        .enumerate()
        .map(|(dim, &size)| if resolved_axis == dim { 1 } else { size })
        .collect();
    let mut reduced_data = pool.alloc(reduced_shape.iter().product());

    fn max_position_by<'a, T: 'a>(
        iter: impl Iterator<Item = &'a T>,
        compare: impl Fn(&'a T, &'a T) -> std::cmp::Ordering,
    ) -> usize {
        let (index, _) = iter.enumerate().max_by(|a, b| compare(a.1, b.1)).unwrap(); // Ok because we checked tensor is not empty.
        index
    }

    if !input.is_empty() {
        reduced_data.extend(input.lanes(resolved_axis).map(|lane| {
            let index = if let Some(slice) = lane.as_slice() {
                // Fast path for contiguous lanes.
                max_position_by(slice.iter(), &compare)
            } else {
                max_position_by(lane, &compare)
            };
            index as i32
        }));
    }

    let mut reduced = Tensor::<i32>::from_data(&reduced_shape, reduced_data);

    if !keep_dims {
        let axes = &[resolved_axis as i32];
        let axes = NdTensorView::from(axes);
        squeeze_in_place(&mut reduced, Some(axes)).expect("Invalid axis");
    }

    Ok(reduced)
}

/// Dispatch a reduction over multiple axes.
macro_rules! dispatch_reduce_op {
    ($pool:expr, $input:expr, $reduce_op:ident, $axes:expr, $keep_dims:expr) => {
        match $input {
            Input::FloatTensor(input) => $reduce_op(
                $pool,
                input,
                $axes.as_ref().map(|axis| &axis[..]),
                $keep_dims,
            )
            .into_op_result(),
            Input::Int32Tensor(input) => $reduce_op(
                $pool,
                input,
                $axes.as_ref().map(|axis| &axis[..]),
                $keep_dims,
            )
            .into_op_result(),
            _ => Err(OpError::UnsupportedType),
        }
    };
}

/// Dispatch a reduction over a single axis.
macro_rules! dispatch_single_axis_reduce_op {
    ($pool:expr, $input:expr, $reduce_op:ident, $axis:expr, $keep_dims:expr) => {
        match $input {
            Input::FloatTensor(input) => {
                $reduce_op($pool, input, $axis, $keep_dims).into_op_result()
            }
            Input::Int32Tensor(input) => {
                $reduce_op($pool, input, $axis, $keep_dims).into_op_result()
            }
            _ => Err(OpError::UnsupportedType),
        }
    };
}

/// Return the index of the maximum value along a given axis.
///
/// NaN values are propagated by treating NaNs as greater than other values.
pub fn arg_max<T: Copy + PartialOrd + IsNaN>(
    pool: &TensorPool,
    input: TensorView<T>,
    axis: isize,
    keep_dims: bool,
) -> Result<Tensor<i32>, OpError> {
    select_max_index(pool, input, axis, keep_dims, |a, b| cmp_nan_greater(*a, *b))
}

#[derive(Debug)]
pub struct ArgMax {
    pub axis: isize,
    pub keep_dims: bool,
}

impl Operator for ArgMax {
    fn name(&self) -> &str {
        "ArgMax"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require(0)?;
        dispatch_single_axis_reduce_op!(pool, input, arg_max, self.axis, self.keep_dims)
    }
}

/// Return the index of the minimum value along a given axis.
///
/// NaN values are propagated by treating NaNs as smaller than other values.
pub fn arg_min<T: Copy + PartialOrd + IsNaN>(
    pool: &TensorPool,
    input: TensorView<T>,
    axis: isize,
    keep_dims: bool,
) -> Result<Tensor<i32>, OpError> {
    select_max_index(pool, input, axis, keep_dims, |a, b| {
        match a.partial_cmp(b) {
            Some(ordering) => ordering.reverse(),
            None => cmp_nan_greater(*a, *b),
        }
    })
}

#[derive(Debug)]
pub struct ArgMin {
    pub axis: isize,
    pub keep_dims: bool,
}

impl Operator for ArgMin {
    fn name(&self) -> &str {
        "ArgMin"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require(0)?;
        dispatch_single_axis_reduce_op!(pool, input, arg_max, self.axis, self.keep_dims)
    }
}

pub fn cum_sum<T: Copy + Default + Identities + std::ops::AddAssign>(
    pool: &TensorPool,
    input: TensorView<T>,
    axis: isize,
) -> Result<Tensor<T>, OpError> {
    let resolved_axis = resolve_axis(input.ndim(), axis)?;
    let mut output = Tensor::uninit_in(pool, input.shape());

    let mut n_init = 0;
    if !input.is_empty() {
        for (in_slice, out_slice) in input
            .lanes(resolved_axis)
            .zip(output.lanes_mut(resolved_axis))
        {
            let mut cum_sum = T::zero();
            for (x, y) in in_slice.zip(out_slice) {
                cum_sum += *x;
                y.write(cum_sum);
                n_init += 1;
            }
        }
    }

    assert!(n_init == output.len());
    let output = unsafe { output.assume_init() };

    Ok(output)
}

#[derive(Debug)]
pub struct CumSum {}

impl Operator for CumSum {
    fn name(&self) -> &str {
        "CumSum"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require(0)?;
        let axis: i32 = inputs.require_as_scalar(1)?;
        match input {
            Input::Int32Tensor(input) => cum_sum(pool, input, axis as isize).into_op_result(),
            Input::FloatTensor(input) => cum_sum(pool, input, axis as isize).into_op_result(),
            _ => Err(OpError::UnsupportedType),
        }
    }
}

/// Return the indices of nonzero elements in `input` as a `(dim, index)` tensor.
pub fn nonzero<T: Default + PartialEq>(pool: &TensorPool, input: TensorView<T>) -> Tensor<i32> {
    // Special case for scalar inputs.
    if let (Some(item), 0) = (input.item(), input.ndim()) {
        return Tensor::zeros(&[0, if *item != T::default() { 1 } else { 0 }]);
    }

    // Build up concatenated sequence of indices of non-zero entries.
    let nonzeros: Vec<i32> = input
        .indices()
        .zip(input.iter())
        .filter(|(_index, value)| **value != T::default())
        .flat_map(|(index, _value)| {
            index.into_iter().map(|dim_idx| {
                assert!(dim_idx <= i32::MAX as usize);
                dim_idx as i32
            })
        })
        .collect();

    // Transpose from `(index, dim)` to `(dim, index)`.
    Tensor::from_data(&[nonzeros.len() / input.ndim(), input.ndim()], nonzeros)
        .transposed()
        .to_tensor_in(pool)
}

#[derive(Debug)]
pub struct NonZero {}

impl Operator for NonZero {
    fn name(&self) -> &str {
        "NonZero"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require(0)?;
        match input {
            Input::Int32Tensor(input) => nonzero(pool, input).into_op_result(),
            Input::FloatTensor(input) => nonzero(pool, input).into_op_result(),
            _ => Err(OpError::UnsupportedType),
        }
    }
}

/// Trait for reducing a subset of elements from a tensor to a single value.
trait Reducer<T> {
    /// Reduce a contiguous slice of values to a single value.
    fn reduce_slice(&self, slice: &[T]) -> T;
}

fn reduce<T: Copy>(
    pool: &TensorPool,
    input: TensorView<T>,
    axes: Option<&[i32]>,
    keep_dims: bool,
    reducer: &dyn Reducer<T>,
) -> Result<Tensor<T>, OpError> {
    let mut resolved_axes = match axes {
        Some(axes) if !axes.is_empty() => resolve_axes(input.ndim(), axes.iter())?,
        _ => (0..input.ndim()).collect(),
    };
    resolved_axes.sort();

    // Allocate temporary buffer where slices of the input to be reduced are
    // packed first if non-contiguous.
    let mut tmp_buf = if !input.is_contiguous() {
        let reduced_slice_len = resolved_axes.iter().map(|&dim| input.size(dim)).product();
        pool.alloc(reduced_slice_len)
    } else {
        Vec::new()
    }
    .auto_return(pool);

    if input.ndim() == 0 {
        let item = input.item().unwrap();
        return Ok(Tensor::from_scalar(reducer.reduce_slice(&[*item])));
    }

    // nb. Some reduce operations cannot produce a meaningful result with
    // an empty tensor, but others can, if there is a suitable identity.
    if input.is_empty() {
        return Err(OpError::InvalidValue("Cannot reduce empty tensor"));
    }

    // Number of innermost dims being iterated over, or None if we're not
    // iterating over innermost dims.
    let reduced_inner_dims: Option<usize> = resolved_axes
        .iter()
        .enumerate()
        .all(|(i, &axis)| axis == input.ndim() - 1 - i)
        .then_some(resolved_axes.len());

    let reduced_shape: Vec<usize> = input
        .shape()
        .iter()
        .enumerate()
        .map(|(dim, &size)| {
            if resolved_axes.contains(&dim) {
                1
            } else {
                size
            }
        })
        .collect();
    let mut reduced_data = pool.alloc(reduced_shape.iter().product());

    match (reduced_inner_dims, input.data()) {
        (Some(ndims), Some(input_data)) => {
            // Fast path for reducing over contiguous chunks of the input.
            let slice_len = if ndims == input.ndim() {
                input.len()
            } else {
                input.stride(input.ndim() - 1 - ndims)
            };

            reduced_data.extend(
                input_data
                    .chunks(slice_len)
                    .map(|chunk| reducer.reduce_slice(chunk)),
            );
        }
        _ => {
            if resolved_axes.len() == 1 {
                // Fast path for reducing a single axis.
                let resolved_axis = resolved_axes[0];
                reduced_data.extend(input.lanes(resolved_axis).map(|lane| {
                    if let Some(lane_slice) = lane.as_slice() {
                        reducer.reduce_slice(lane_slice)
                    } else {
                        tmp_buf.clear();
                        tmp_buf.extend(lane.copied());
                        reducer.reduce_slice(&tmp_buf)
                    }
                }));
            } else {
                // Slow case when we have to step through each index
                let outer_range: Vec<_> = (0..input.ndim())
                    .map(|dim| {
                        if resolved_axes.contains(&dim) {
                            1
                        } else {
                            input.size(dim)
                        }
                    })
                    .collect();
                let mut inner_range = Vec::with_capacity(input.ndim());
                for index in DynIndices::from_shape(&outer_range) {
                    inner_range.clear();
                    inner_range.extend(index.iter().enumerate().map(|(dim, &idx)| {
                        if resolved_axes.contains(&dim) {
                            SliceItem::range(0, Some(input.size(dim) as isize), 1)
                        } else {
                            SliceItem::Index(idx as isize)
                        }
                    }));
                    let slice = input.slice(inner_range.as_slice());

                    tmp_buf.clear();
                    tmp_buf.extend(slice.iter().copied());
                    let reduced = reducer.reduce_slice(&tmp_buf);
                    reduced_data.push(reduced);
                }
            }
        }
    }

    let mut reduced = Tensor::<T>::from_data(&reduced_shape, reduced_data);

    if !keep_dims {
        let resolved_axes_i32: NdTensor<i32, 1> =
            resolved_axes.iter().map(|&axis| axis as i32).collect();
        squeeze_in_place(&mut reduced, Some(resolved_axes_i32.view())).expect("Invalid axis");
    }

    Ok(reduced)
}

pub fn reduce_mean(
    pool: &TensorPool,
    input: TensorView,
    axes: Option<&[i32]>,
    keep_dims: bool,
) -> Result<Tensor, OpError> {
    struct MeanReducer {}
    impl Reducer<f32> for MeanReducer {
        fn reduce_slice(&self, slice: &[f32]) -> f32 {
            vec_sum(slice) / slice.len() as f32
        }
    }

    reduce(pool, input, axes, keep_dims, &MeanReducer {})
}

/// Reduces axes of a tensor using an inverse Root Mean Squared (RMS)
/// operation.
///
/// This reduces axes according to the formula:
///
/// ```text
/// 1. / (mean(x^2) + epsilon).sqrt()
/// ```
pub fn reduce_inverse_rms(
    pool: &TensorPool,
    input: TensorView,
    axes: Option<&[i32]>,
    keep_dims: bool,
    epsilon: f32,
) -> Result<Tensor, OpError> {
    struct InverseRmsReducer {
        epsilon: f32,
    }

    impl Reducer<f32> for InverseRmsReducer {
        fn reduce_slice(&self, slice: &[f32]) -> f32 {
            let mean_square = vec_sum_square(slice) / slice.len() as f32;
            1. / (mean_square + self.epsilon).sqrt()
        }
    }

    reduce(pool, input, axes, keep_dims, &InverseRmsReducer { epsilon })
}

#[derive(Debug)]
pub struct ReduceMean {
    pub axes: Option<Vec<i32>>,
    pub keep_dims: bool,
}

impl Operator for ReduceMean {
    fn name(&self) -> &str {
        "ReduceMean"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require_as(0)?;
        let axes = get_axes(&inputs, &self.axes)?;
        reduce_mean(
            pool,
            input,
            axes.as_ref().map(|axis| &axis[..]),
            self.keep_dims,
        )
        .into_op_result()
    }
}

pub fn reduce_l2(
    pool: &TensorPool,
    input: TensorView,
    axes: Option<&[i32]>,
    keep_dims: bool,
) -> Result<Tensor, OpError> {
    struct L2Reducer {}
    impl Reducer<f32> for L2Reducer {
        fn reduce_slice(&self, slice: &[f32]) -> f32 {
            vec_sum_square(slice).sqrt()
        }
    }

    reduce(pool, input, axes, keep_dims, &L2Reducer {})
}

#[derive(Debug)]
pub struct ReduceL2 {
    pub axes: Option<Vec<i32>>,
    pub keep_dims: bool,
}

impl Operator for ReduceL2 {
    fn name(&self) -> &str {
        "ReduceL2"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require_as(0)?;
        let axes = get_axes(&inputs, &self.axes)?;
        reduce_l2(
            pool,
            input,
            axes.as_ref().map(|axis| &axis[..]),
            self.keep_dims,
        )
        .into_op_result()
    }
}

/// Compare `a` and `b`, treating all NaN values as greater than non-NaN values.
pub fn cmp_nan_greater<T: PartialOrd + IsNaN>(a: T, b: T) -> std::cmp::Ordering {
    match a.partial_cmp(&b) {
        Some(ordering) => ordering,
        None => {
            if a.is_nan() {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Less
            }
        }
    }
}

/// Compare `a` and `b`, treating all NaN values as less than non-NaN values.
pub fn cmp_nan_less<T: PartialOrd + IsNaN>(a: T, b: T) -> std::cmp::Ordering {
    match a.partial_cmp(&b) {
        Some(ordering) => ordering,
        None => {
            if a.is_nan() {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        }
    }
}

fn reduce_min_max<T: Copy + PartialOrd + IsNaN>(
    pool: &TensorPool,
    input: TensorView<T>,
    axes: Option<&[i32]>,
    keep_dims: bool,
    max: bool,
) -> Result<Tensor<T>, OpError> {
    struct MinMaxReducer {
        max: bool,
    }
    impl<T: Copy + PartialOrd + IsNaN> Reducer<T> for MinMaxReducer {
        fn reduce_slice(&self, slice: &[T]) -> T {
            let reduced = if self.max {
                slice.iter().copied().max_by(|a, b| cmp_nan_greater(*a, *b))
            } else {
                slice.iter().copied().min_by(|a, b| cmp_nan_less(*a, *b))
            };
            reduced.expect("attempted to get min/max of empty axis")
        }
    }
    reduce(pool, input, axes, keep_dims, &MinMaxReducer { max })
}

/// Extract axes from input 1 in `inputs` or `attr`.
///
/// Earlier versions of the ONNX `Reduce*` operators used an attribute. In later
/// versions this was promoted to an input.
fn get_axes<'a>(
    inputs: &'a InputList,
    attr: &'a Option<Vec<i32>>,
) -> Result<Option<Cow<'a, [i32]>>, OpError> {
    let axes = inputs
        .get_as::<i32>(1)?
        .map(|x| x.to_slice())
        .or(attr.as_ref().map(|a| Cow::Borrowed(a.as_slice())));
    Ok(axes)
}

pub fn reduce_min<T: Copy + PartialOrd + IsNaN>(
    pool: &TensorPool,
    input: TensorView<T>,
    axes: Option<&[i32]>,
    keep_dims: bool,
) -> Result<Tensor<T>, OpError> {
    reduce_min_max(pool, input, axes, keep_dims, false /* max */)
}

#[derive(Debug)]
pub struct ReduceMin {
    pub axes: Option<Vec<i32>>,
    pub keep_dims: bool,
}

impl Operator for ReduceMin {
    fn name(&self) -> &str {
        "ReduceMin"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require(0)?;
        let axes = get_axes(&inputs, &self.axes)?;
        dispatch_reduce_op!(pool, input, reduce_min, axes, self.keep_dims)
    }
}

pub fn reduce_max<T: Copy + PartialOrd + IsNaN>(
    pool: &TensorPool,
    input: TensorView<T>,
    axes: Option<&[i32]>,
    keep_dims: bool,
) -> Result<Tensor<T>, OpError> {
    reduce_min_max(pool, input, axes, keep_dims, true /* max */)
}

#[derive(Debug)]
pub struct ReduceMax {
    pub axes: Option<Vec<i32>>,
    pub keep_dims: bool,
}

impl Operator for ReduceMax {
    fn name(&self) -> &str {
        "ReduceMax"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require(0)?;
        let axes = get_axes(&inputs, &self.axes)?;
        dispatch_reduce_op!(pool, input, reduce_max, axes, self.keep_dims)
    }
}

pub fn reduce_prod<T: Copy + std::iter::Product>(
    pool: &TensorPool,
    input: TensorView<T>,
    axes: Option<&[i32]>,
    keep_dims: bool,
) -> Result<Tensor<T>, OpError> {
    struct ProdReducer {}
    impl<T: Copy + std::iter::Product> Reducer<T> for ProdReducer {
        fn reduce_slice(&self, slice: &[T]) -> T {
            slice.iter().copied().product()
        }
    }
    reduce(pool, input, axes, keep_dims, &ProdReducer {})
}

#[derive(Debug)]
pub struct ReduceProd {
    pub axes: Option<Vec<i32>>,
    pub keep_dims: bool,
}

impl Operator for ReduceProd {
    fn name(&self) -> &str {
        "ReduceProd"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require(0)?;
        let axes = get_axes(&inputs, &self.axes)?;
        dispatch_reduce_op!(pool, input, reduce_prod, axes, self.keep_dims)
    }
}

pub fn reduce_sum<T: Copy + Default + std::ops::Add<T, Output = T>>(
    pool: &TensorPool,
    input: TensorView<T>,
    axes: Option<&[i32]>,
    keep_dims: bool,
) -> Result<Tensor<T>, OpError> {
    struct SumReducer {}
    impl<T: Copy + Default + std::ops::Add<T, Output = T>> Reducer<T> for SumReducer {
        fn reduce_slice(&self, slice: &[T]) -> T {
            slice_sum(slice)
        }
    }
    reduce(pool, input, axes, keep_dims, &SumReducer {})
}

#[derive(Debug)]
pub struct ReduceSum {
    pub axes: Option<Vec<i32>>,
    pub keep_dims: bool,
}

impl Operator for ReduceSum {
    fn name(&self) -> &str {
        "ReduceSum"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require(0)?;
        let axes = get_axes(&inputs, &self.axes)?;
        dispatch_reduce_op!(pool, input, reduce_sum, axes, self.keep_dims)
    }
}

pub fn reduce_sum_square<T: Copy + std::ops::Mul<T, Output = T> + std::iter::Sum>(
    pool: &TensorPool,
    input: TensorView<T>,
    axes: Option<&[i32]>,
    keep_dims: bool,
) -> Result<Tensor<T>, OpError> {
    struct SumSquareReducer {}
    impl<T: Copy + std::iter::Sum + std::ops::Mul<Output = T>> Reducer<T> for SumSquareReducer {
        fn reduce_slice(&self, slice: &[T]) -> T {
            slice.iter().copied().map(|x| x * x).sum()
        }
    }
    reduce(pool, input, axes, keep_dims, &SumSquareReducer {})
}

#[derive(Debug)]
pub struct ReduceSumSquare {
    pub axes: Option<Vec<i32>>,
    pub keep_dims: bool,
}

impl Operator for ReduceSumSquare {
    fn name(&self) -> &str {
        "ReduceSumSquare"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require(0)?;
        let axes = get_axes(&inputs, &self.axes)?;
        dispatch_reduce_op!(pool, input, reduce_sum_square, axes, self.keep_dims)
    }
}

pub fn topk<T: Copy + Default + PartialOrd + IsNaN>(
    pool: &TensorPool,
    values: TensorView<T>,
    k: usize,
    axis: Option<isize>,
    largest: bool,
    sorted: bool,
) -> Result<(Tensor<T>, Tensor<i32>), OpError> {
    let axis = resolve_axis(values.ndim(), axis.unwrap_or(-1))?;
    let out_shape: Vec<usize> = values
        .shape()
        .iter()
        .enumerate()
        .map(|(dim, size)| if dim == axis { k } else { *size })
        .collect();
    let mut out_values = Tensor::zeros_in(pool, &out_shape);
    let mut indices = Tensor::zeros_in(pool, &out_shape);

    // Handle edge case early to simplify main loop.
    if k == 0 {
        return Ok((out_values, indices));
    }

    let axis_size = values.size(axis);
    if k > axis_size {
        return Err(OpError::InvalidValue("k > dimension size"));
    }

    let topk_cmp = |(a_val, a_idx): &(T, usize), (b_val, b_idx): &(T, usize)| -> Ordering {
        // NaN values are treated as greater than other values, for consistency
        // with PyTorch (`torch.topk`) and numpy (`np.partition`). See
        // https://github.com/onnx/onnx/issues/4716. This applies regardless
        // of sort order.
        match cmp_nan_greater(*a_val, *b_val) {
            // Per spec, if values are equal, the index is used as a tie
            // breaker. Smaller indices win, regardless of value sort order.
            Ordering::Equal => a_idx.cmp(b_idx),
            order => {
                if largest {
                    order.reverse()
                } else {
                    order
                }
            }
        }
    };

    // Temporary array of (value, index).
    let mut tmp: Vec<(T, usize)> = Vec::with_capacity(axis_size);

    for (values, (out_values, indices)) in values
        .lanes(axis)
        .zip(out_values.lanes_mut(axis).zip(indices.lanes_mut(axis)))
    {
        tmp.clear();
        tmp.extend(values.copied().zip(0..axis_size));
        tmp.select_nth_unstable_by(k - 1, topk_cmp);
        tmp.truncate(k);

        if sorted {
            tmp.sort_unstable_by(topk_cmp);
        }

        for ((out_val, out_idx), (val, idx)) in out_values.zip(indices).zip(tmp.iter()) {
            *out_val = *val;
            *out_idx = *idx as i32;
        }
    }

    Ok((out_values, indices))
}

#[derive(Debug)]
pub struct TopK {
    pub axis: Option<isize>,
    pub largest: bool,
    pub sorted: bool,
}

impl Operator for TopK {
    fn name(&self) -> &str {
        "TopK"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let values = inputs.require(0)?;
        let k = inputs.require_as_scalar::<i32>(1).and_then(|k| {
            if k < 0 {
                Err(OpError::InvalidValue("k must be positive"))
            } else {
                Ok(k as usize)
            }
        })?;

        match values {
            Input::FloatTensor(values) => {
                let (values, indices) =
                    topk(pool, values, k, self.axis, self.largest, self.sorted)?;
                Ok([values.into(), indices.into()].into_iter().collect())
            }
            Input::Int32Tensor(values) => {
                let (values, indices) =
                    topk(pool, values, k, self.axis, self.largest, self.sorted)?;
                Ok([values.into(), indices.into()].into_iter().collect())
            }
            _ => Err(OpError::UnsupportedType),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::prelude::*;
    use rten_tensor::test_util::{eq_with_nans, expect_equal};
    use rten_tensor::{NdTensor, Tensor};

    use crate::ops::tests::{new_pool, run_op};
    use crate::ops::{
        arg_max, arg_min, cum_sum, nonzero, reduce_l2, reduce_max, reduce_mean, reduce_min,
        reduce_prod, reduce_sum, reduce_sum_square, topk, OpError, Operator, ReduceL2, ReduceMax,
        ReduceMean, ReduceMin, ReduceProd, ReduceSum, ReduceSumSquare,
    };

    #[test]
    fn test_arg_max() {
        let pool = new_pool();

        // Reduce a simple vector.
        let probs = Tensor::from([0.1, 0.5, 0.2, 0.9, 0.01, 0.6]);
        let class = arg_max(&pool, probs.view(), 0, false /* keep_dims */).unwrap();
        assert_eq!(class.item(), Some(&3));

        // Same, but keep dims
        let class = arg_max(&pool, probs.view(), 0, true /* keep_dims */).unwrap();
        assert_eq!(class.shape(), &[1]);
        assert_eq!(class.to_vec(), &[3]);

        // Common use case of a tensor of (batch, item, prob) where
        // `item` is eg. a token index in a sequence or box ID for object
        // detection.
        let seq_probs = Tensor::from([[
            [0.1, 0.2, 0.9],
            [0.9, 0.1, 0.2],
            [0.3, 0.8, 0.4],
            [0.1, 0.01, 0.2],
        ]]);
        let seq_classes = arg_max(&pool, seq_probs.view(), 2, false /* keep_dims */).unwrap();
        assert_eq!(seq_classes.shape(), &[1, 4]);
        assert_eq!(seq_classes.to_vec(), &[2, 0, 1, 2]);

        // Same, but keep dims
        let seq_classes = arg_max(&pool, seq_probs.view(), 2, true /* keep_dims */).unwrap();
        assert_eq!(seq_classes.shape(), &[1, 4, 1]);
        assert_eq!(seq_classes.to_vec(), &[2, 0, 1, 2]);

        // Empty tensor, axis is a non-zero-sized dim
        let empty = Tensor::<i32>::from_data(&[10, 0, 5], vec![]);
        let result = arg_max(&pool, empty.view(), 0, false /* keep_dims */).unwrap();
        assert_eq!(result.shape(), &[0, 5]);
        assert_eq!(result.to_vec(), &[] as &[i32]);

        // Empty tensor, axis is a zero-sized dim
        let empty = Tensor::<i32>::from_data(&[10, 0, 5], vec![]);
        let result = arg_max(&pool, empty.view(), 1, false /* keep_dims */);
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue(
                "Cannot select index from empty sequence"
            ))
        );

        // Non-contiguous lanes
        let mat = Tensor::from([[1, 2], [4, 8], [5, 6]]);
        let col_max = arg_max(&pool, mat.view(), 0, false /* keep_dims */).unwrap();
        assert_eq!(col_max, NdTensor::from([2, 1]));
    }

    // We only have basic tests for ArgMin since most of the implementation is
    // shared with ArgMax.
    #[test]
    fn test_arg_min() {
        let pool = new_pool();
        let probs = Tensor::from([0.1, 0.5, 0.2, 0.9, 0.01, 0.6]);
        let class = arg_min(&pool, probs.view(), 0, false /* keep_dims */).unwrap();
        assert_eq!(class.item(), Some(&4));
    }

    // ONNX does not specify how ArgMin and ArgMax should handle NaNs. We opt to
    // be consistent with ReduceMin and ReduceMax by "propagating" NaNs, which
    // for these operators means returning the index of a NaN value over other
    // indices. This is consistent with numpy's `argmin` and `argmax`.
    #[test]
    fn test_arg_min_max_nan() {
        let pool = new_pool();
        let probs = Tensor::from([0.1, 0.5, f32::NAN, 0.9, 0.01, 0.6]);
        let min_idx = arg_min(&pool, probs.view(), 0, false /* keep_dims */).unwrap();
        let max_idx = arg_max(&pool, probs.view(), 0, false /* keep_dims */).unwrap();
        assert_eq!(min_idx.item(), Some(&2));
        assert_eq!(max_idx.item(), Some(&2));
    }

    #[test]
    fn test_cum_sum() {
        let pool = new_pool();
        let elements = Tensor::from_vec((0..=5).collect());
        let sums = cum_sum(&pool, elements.view(), 0).unwrap();
        assert_eq!(sums.shape(), &[6]);
        assert_eq!(sums.to_vec(), &[0, 1, 3, 6, 10, 15]);

        let elements = Tensor::from_data(&[1, 4, 4], vec![1; 16]);
        let sums = cum_sum(&pool, elements.view(), 1).unwrap();
        assert_eq!(sums.shape(), &[1, 4, 4]);
        assert_eq!(
            sums.to_vec(),
            &[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
        );

        let sums = cum_sum(&pool, elements.view(), -1).unwrap();
        assert_eq!(sums.shape(), &[1, 4, 4]);
        assert_eq!(
            sums.to_vec(),
            &[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        );

        let elements: Tensor<f32> = Tensor::from([0.; 0]);
        let sums = cum_sum(&pool, elements.view(), 0).unwrap();
        assert_eq!(sums.shape(), &[0]);
        assert_eq!(sums.to_vec(), &[] as &[f32]);
    }

    #[test]
    fn test_nonzero() {
        let pool = new_pool();
        let input = Tensor::from([[0., 1.], [1., 1.]]);
        let result = nonzero(&pool, input.view());
        assert_eq!(result.shape(), &[2, 3]);

        // (dim, index) => (index, dim)
        let result = result.transposed();

        let indices: Vec<_> = result.iter().copied().collect();
        assert_eq!(
            indices,
            [
                0, 1, // 1st
                1, 0, // 2nd
                1, 1, // 3rd
            ]
        );
    }

    #[test]
    fn test_nonzero_scalar() {
        let pool = new_pool();
        let input = Tensor::from(3.);
        let result = nonzero(&pool, input.view());
        assert_eq!(result.shape(), &[0, 1]);

        let input = Tensor::from(0.);
        let result = nonzero(&pool, input.view());
        assert_eq!(result.shape(), &[0, 0]);
    }

    #[test]
    fn test_reduce_axes_via_input() -> Result<(), Box<dyn Error>> {
        struct Case {
            op: Box<dyn Operator>,
        }

        macro_rules! op_case {
            ($op:ident) => {
                Case {
                    op: Box::new($op {
                        // Don't set `axes` attr. Axes will come from inputs
                        // instead.
                        axes: None,
                        keep_dims: true,
                    }),
                }
            };
        }

        let cases = [
            op_case!(ReduceL2),
            op_case!(ReduceMax),
            op_case!(ReduceMean),
            op_case!(ReduceMin),
            op_case!(ReduceProd),
            op_case!(ReduceSum),
            op_case!(ReduceSumSquare),
        ];

        for Case { op } in cases {
            let input = NdTensor::from([[0., 1., 2.], [3., 4., 5.]]);
            let axes = Tensor::from([0]);
            let result: NdTensor<f32, 2> = run_op(&*op, (input.view(), axes.view()))?;
            assert_eq!(result.shape(), [1, 3]);

            let axes = Tensor::from([1]);
            let result: NdTensor<f32, 2> = run_op(&*op, (input.view(), axes.view()))?;
            assert_eq!(result.shape(), [2, 1]);
        }

        Ok(())
    }

    #[test]
    fn test_reduce_l2() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();
        let input = Tensor::from_data(&[3, 2, 2], (1..=12).map(|i| i as f32).collect::<Vec<_>>());
        let expected = Tensor::from_data(
            &[3, 2],
            vec![
                2.23606798,
                5.,
                7.81024968,
                10.63014581,
                13.45362405,
                16.2788206,
            ],
        );

        let result = reduce_l2(&pool, input.view(), Some(&[2]), false /* keep_dims */).unwrap();
        expect_equal(&result, &expected)?;

        let result = reduce_l2(&pool, input.view(), Some(&[2]), true /* keep_dims */).unwrap();
        let expected = expected.to_shape([3, 2, 1].as_slice());
        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_reduce_mean() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();
        let input = Tensor::from_data(&[3, 3], vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        // Test with `keep_dims` off
        let result = reduce_mean(&pool, input.view(), Some(&[-1]), false /* keep_dims */).unwrap();
        let expected = Tensor::from([2., 5., 8.]);
        expect_equal(&result, &expected)?;

        // Test with `keep_dims` on
        let result = reduce_mean(&pool, input.view(), Some(&[-1]), true /* keep_dims */).unwrap();
        let expected = Tensor::from_data(&[3, 1], vec![2., 5., 8.]);
        expect_equal(&result, &expected)?;

        // Reduce first dim
        let result = reduce_mean(&pool, input.view(), Some(&[0]), false /* keep_dims */).unwrap();
        let expected = Tensor::from([4., 5., 6.]);
        expect_equal(&result, &expected)?;

        // Reduce all axes
        let result = reduce_mean(&pool, input.view(), None, false /* keep_dims */).unwrap();
        let expected = Tensor::from(5.);
        expect_equal(&result, &expected)?;

        // Reduce all axes (specified via empty array)
        let result = reduce_mean(&pool, input.view(), Some(&[]), false /* keep_dims */).unwrap();
        let expected = Tensor::from(5.);
        expect_equal(&result, &expected)?;

        // Test case from ONNX spec
        let input = Tensor::from_data(
            &[3, 2, 2],
            vec![5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.],
        );
        let expected = Tensor::from_data(&[3, 2], vec![12.5, 1.5, 35., 1.5, 57.5, 1.5]);
        let result = reduce_mean(&pool, input.view(), Some(&[1]), false /* keep_dims */).unwrap();
        expect_equal(&result, &expected)?;

        // Reduce a scalar value
        let result = reduce_mean(
            &pool,
            Tensor::from(5.0).view(),
            Some(&[]),
            false, /* keep_dims */
        )
        .unwrap();
        assert_eq!(result.item(), Some(&5.0));

        // Reduce a vector
        let result = reduce_mean(
            &pool,
            Tensor::from([0., 10.]).view(),
            Some(&[0]),
            false, /* keep_dims */
        )
        .unwrap();
        assert_eq!(result.to_vec(), &[5.0]);

        Ok(())
    }

    #[test]
    fn test_reduce_mean_invalid_inputs() {
        let pool = new_pool();
        let input = Tensor::from_data(&[3, 3], vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        let result = reduce_mean(&pool, input.view(), Some(&[3]), false /* keep_dims */);
        assert_eq!(result.err(), Some(OpError::InvalidValue("Axis is invalid")));

        let result = reduce_mean(&pool, input.view(), Some(&[-3]), false /* keep_dims */);
        assert_eq!(result.err(), Some(OpError::InvalidValue("Axis is invalid")));

        // Empty tensor
        let result = reduce_mean(
            &pool,
            Tensor::from([0.; 0]).view(),
            Some(&[0]),
            false, /* keep_dims */
        );
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Cannot reduce empty tensor"))
        );
    }

    fn result_item<T: Copy>(result: Result<Tensor<T>, OpError>) -> T {
        *result.unwrap().item().unwrap()
    }

    #[test]
    fn test_reduce_min_max() {
        let pool = new_pool();
        let input: Tensor<f32> = [1.5, 2.5, 3.5, 4.5, 5.5].into();
        let min = result_item(reduce_min(
            &pool,
            input.view(),
            Some(&[0]),
            false, /* keep_dims */
        ));
        let max = result_item(reduce_max(
            &pool,
            input.view(),
            Some(&[0]),
            false, /* keep_dims */
        ));
        assert_eq!(min, 1.5);
        assert_eq!(max, 5.5);
    }

    // ONNX docs do not specify expected handling of NaNs by several operators,
    // but the corresponding numpy functions (eg. `np.min`) propagate NaNs and
    // that seems like the more sensible default behavior.
    //
    // See https://github.com/onnx/onnx/issues/4716.
    #[test]
    fn test_reduce_min_max_propagates_nan() {
        let pool = new_pool();
        let input: Tensor<f32> = [1.5, 2.5, 3.5, f32::NAN, 5.5].into();
        let min = result_item(reduce_min(
            &pool,
            input.view(),
            Some(&[0]),
            false, /* keep_dims */
        ));
        let max = result_item(reduce_max(
            &pool,
            input.view(),
            Some(&[0]),
            false, /* keep_dims */
        ));
        assert!(min.is_nan());
        assert!(max.is_nan());
    }

    #[test]
    fn test_reduce_prod() {
        let pool = new_pool();

        // Int tensor
        let input: Tensor<i32> = [1, 2, 3, 4, 5].into();
        let result = result_item(reduce_prod(
            &pool,
            input.view(),
            Some(&[0]),
            false, /* keep_dims */
        ));
        assert_eq!(result, input.iter().product::<i32>());

        // Float tensor
        let input: Tensor<f32> = [1.5, 2.5, 3.5, 4.5, 5.5].into();
        let result = result_item(reduce_prod(
            &pool,
            input.view(),
            Some(&[0]),
            false, /* keep_dims */
        ));
        assert_eq!(result, input.iter().product::<f32>());
    }

    #[test]
    fn test_reduce_sum() {
        let pool = new_pool();

        // Int tensor
        let input: Tensor<i32> = [1, 2, 3, 4, 5].into();
        let result = result_item(reduce_sum(
            &pool,
            input.view(),
            Some(&[0]),
            false, /* keep_dims */
        ));
        assert_eq!(result, input.iter().sum::<i32>());

        // Float tensor
        let input: Tensor<f32> = [1.5, 2.5, 3.5, 4.5, 5.5].into();
        let result = result_item(reduce_sum(
            &pool,
            input.view(),
            Some(&[0]),
            false, /* keep_dims */
        ));
        assert_eq!(result, input.iter().sum::<f32>());
    }

    #[test]
    fn test_reduce_sum_square() {
        let pool = new_pool();

        // Int tensor
        let input: Tensor<i32> = [1, 2, 3, 4, 5].into();
        let result = result_item(reduce_sum_square(
            &pool,
            input.view(),
            Some(&[0]),
            false, /* keep_dims */
        ));
        assert_eq!(result, input.iter().map(|x| x * x).sum::<i32>());

        // Float tensor
        let input: Tensor<f32> = [1.5, 2.5, 3.5, 4.5, 5.5].into();
        let result = result_item(reduce_sum_square(
            &pool,
            input.view(),
            Some(&[0]),
            false, /* keep_dims */
        ));
        assert_eq!(result, input.iter().map(|x| x * x).sum::<f32>());
    }

    #[test]
    fn test_topk() {
        struct Case {
            input: Tensor<f32>,
            k: usize,
            axis: Option<isize>,
            largest: bool,
            expected: Result<(Tensor<f32>, Tensor<i32>), OpError>,
        }

        impl Default for Case {
            fn default() -> Self {
                Case {
                    input: Tensor::zeros(&[]),
                    expected: Ok((Tensor::zeros(&[]), Tensor::zeros(&[]))),
                    k: 0,
                    axis: None,
                    largest: true,
                }
            }
        }

        let cases = [
            // Simple case, largest=true
            Case {
                input: [0., 1., 2.].into(),
                k: 2,
                expected: Ok((Tensor::from([2., 1.]), Tensor::from([2, 1]))),
                ..Default::default()
            },
            // Simple case, largest=false
            Case {
                input: [0., 1., 2.].into(),
                k: 2,
                largest: false,
                expected: Ok((Tensor::from([0., 1.]), Tensor::from([0, 1]))),
                ..Default::default()
            },
            // Special case where k=0
            Case {
                input: [0., 1., 2.].into(),
                k: 0,
                expected: Ok((Tensor::from([0.; 0]), Tensor::from([0; 0]))),
                ..Default::default()
            },
            // Tie break by index when input values are equal.
            Case {
                input: [1., 0., 2., 3., 1.].into(),
                k: 5,
                expected: Ok(([3., 2., 1., 1., 0.].into(), [3, 2, 0, 4, 1].into())),
                ..Default::default()
            },
            // Tie break by index when input values are equal, largest=false
            Case {
                input: [1., 0., 2., 3., 1.].into(),
                k: 5,
                largest: false,
                expected: Ok(([0., 1., 1., 2., 3.].into(), [1, 0, 4, 2, 3].into())),
                ..Default::default()
            },
            // NaN values
            Case {
                input: [0., f32::NAN, 2.].into(),
                k: 2,
                expected: Ok((Tensor::from([f32::NAN, 2.]), Tensor::from([1, 2]))),
                ..Default::default()
            },
            // NaN values, with largest=false
            Case {
                input: [0., f32::NAN, 2.].into(),
                k: 3,
                expected: Ok(([0., 2., f32::NAN].into(), [0, 2, 1].into())),
                largest: false,
                ..Default::default()
            },
            // Invalid k value
            Case {
                input: [0., 1., 2.].into(),
                k: 4,
                expected: Err(OpError::InvalidValue("k > dimension size")),
                ..Default::default()
            },
            // Scalar input
            Case {
                input: Tensor::from(0.),
                k: 2,
                expected: Err(OpError::InvalidValue("Axis is invalid")),
                ..Default::default()
            },
            // 2D input, take top-K over axis 1
            Case {
                input: [[0., 1., 2.], [0., 1., 3.], [0., 1., 4.]].into(),
                k: 2,
                expected: Ok((
                    [[2., 1.], [3., 1.], [4., 1.]].into(),
                    [[2, 1], [2, 1], [2, 1]].into(),
                )),
                ..Default::default()
            },
            // 2D input, take top-K over axis 0
            Case {
                input: Tensor::from([[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]]),
                k: 2,
                axis: Some(0),
                expected: Ok((
                    [[6., 7., 8.], [3., 4., 5.]].into(),
                    [[2, 2, 2], [1, 1, 1]].into(),
                )),
                ..Default::default()
            },
        ];

        let pool = new_pool();
        for (
            i,
            Case {
                input,
                expected,
                k,
                axis,
                largest,
            },
        ) in cases.into_iter().enumerate()
        {
            // nb. We always sort here so first result order is predictable.
            let result = topk(
                &pool,
                input.view(),
                k,
                axis,
                largest,
                true, /* sorted */
            );

            match (result, expected) {
                (Ok((values, indices)), Ok((expected_values, expected_indices))) => {
                    assert!(
                        eq_with_nans(values.view(), expected_values.view()),
                        "values differ in case {}",
                        i
                    );
                    assert_eq!(indices, expected_indices, "indices differ in case {}", i);
                }
                (result, expected) => assert_eq!(result, expected),
            }
        }
    }
}
