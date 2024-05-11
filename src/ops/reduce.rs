use std::borrow::Cow;
use std::cmp::Ordering;
use std::iter::zip;

use rten_tensor;
use rten_tensor::prelude::*;
use rten_tensor::{DynIndices, NdTensor, NdTensorView, SliceItem, Tensor, TensorView};

use crate::number::Identities;
use crate::ops::layout::squeeze_in_place;
use crate::ops::{
    resolve_axes, resolve_axis, Input, InputList, IntoOpResult, OpError, Operator, Output,
};
use crate::slice_reductions::slice_sum;
use crate::tensor_pool::TensorPool;

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

    if !input.is_empty() {
        for slice in input.lanes(resolved_axis) {
            let (index, _) = slice.enumerate().max_by(|a, b| compare(a.1, b.1)).unwrap(); // Ok because we checked tensor is not empty.
            reduced_data.push(index as i32);
        }
    }

    let mut reduced = Tensor::<i32>::from_data(&reduced_shape, reduced_data);

    if !keep_dims {
        let axes = &[resolved_axis as i32];
        let axes = NdTensorView::from_data([1], axes);
        squeeze_in_place(&mut reduced, Some(axes)).expect("Invalid axis");
    }

    Ok(reduced)
}

/// Return the index of the maximum value along a given axis.
///
/// NaN values are propagated by treating NaNs as greater than other values.
pub fn arg_max<T: Copy + PartialOrd>(
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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require_as::<f32>(0)?;
        arg_max(pool, input, self.axis, self.keep_dims).into_op_result()
    }
}

/// Return the index of the minimum value along a given axis.
///
/// NaN values are propagated by treating NaNs as smaller than other values.
pub fn arg_min<T: Copy + PartialOrd>(
    pool: &TensorPool,
    input: TensorView<T>,
    axis: isize,
    keep_dims: bool,
) -> Result<Tensor<i32>, OpError> {
    select_max_index(pool, input, axis, keep_dims, |a, b| {
        match a.partial_cmp(b) {
            Some(ordering) => ordering.reverse(),
            None => cmp_nan_greater(a, b),
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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require_as::<f32>(0)?;
        arg_min(pool, input, self.axis, self.keep_dims).into_op_result()
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
        for (in_slice, out_slice) in
            zip(input.lanes(resolved_axis), output.lanes_mut(resolved_axis))
        {
            let mut cum_sum = T::zero();
            for (x, y) in zip(in_slice, out_slice) {
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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let axis: i32 = inputs.require_as_scalar(1)?;
        match input {
            Input::IntTensor(input) => cum_sum(pool, input, axis as isize).into_op_result(),
            Input::FloatTensor(input) => cum_sum(pool, input, axis as isize).into_op_result(),
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
    let nonzeros: Vec<i32> = zip(input.indices(), input.iter())
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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        match input {
            Input::IntTensor(input) => nonzero(pool, input).into_op_result(),
            Input::FloatTensor(input) => nonzero(pool, input).into_op_result(),
        }
    }
}

/// Trait for reducing a subset of elements from a tensor to a single value.
///
/// This is a trait rather than a closure to support being invoked with
/// dynamically chosen iterator types.
trait Reducer<T> {
    fn reduce<I: ExactSizeIterator<Item = T>>(&self, iter: I) -> T;

    /// Reduce a contiguous slice of values to a single value.
    fn reduce_slice(&self, slice: &[T]) -> T
    where
        T: Copy,
    {
        self.reduce(slice.iter().copied())
    }
}

fn reduce<T: Copy, R: Reducer<T>>(
    pool: &TensorPool,
    input: TensorView<T>,
    axes: Option<&[i32]>,
    keep_dims: bool,
    reducer: R,
) -> Result<Tensor<T>, OpError> {
    let mut resolved_axes = match axes {
        Some(axes) if !axes.is_empty() => resolve_axes(input.ndim(), axes.iter())?,
        _ => (0..input.ndim()).collect(),
    };
    resolved_axes.sort();

    if input.ndim() == 0 {
        return Ok(Tensor::from_scalar(reducer.reduce(input.iter().copied())));
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
                for slice in input.lanes(resolved_axis) {
                    reduced_data.push(reducer.reduce(slice.copied()));
                }
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
                    let reduced = reducer.reduce(input.slice_iter(&inner_range).copied());
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
        fn reduce<I: ExactSizeIterator<Item = f32>>(&self, iter: I) -> f32 {
            let len = iter.len() as f32;
            iter.sum::<f32>() / len
        }

        fn reduce_slice(&self, slice: &[f32]) -> f32 {
            slice_sum(slice) / slice.len() as f32
        }
    }

    reduce(pool, input, axes, keep_dims, MeanReducer {})
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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
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
        fn reduce<I: ExactSizeIterator<Item = f32>>(&self, iter: I) -> f32 {
            let sum_of_squares: f32 = iter.map(|val| val * val).sum();
            sum_of_squares.sqrt()
        }
    }

    reduce(pool, input, axes, keep_dims, L2Reducer {})
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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
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
            Input::IntTensor(input) => $reduce_op(
                $pool,
                input,
                $axes.as_ref().map(|axis| &axis[..]),
                $keep_dims,
            )
            .into_op_result(),
        }
    };
}

fn is_nan<T: PartialOrd>(a: &T) -> bool {
    a.partial_cmp(a).is_none()
}

/// Compare `a` and `b`, treating all NaN values as greater than non-NaN values.
pub fn cmp_nan_greater<T: PartialOrd>(a: T, b: T) -> std::cmp::Ordering {
    match a.partial_cmp(&b) {
        Some(ordering) => ordering,
        None => {
            if is_nan(&a) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Less
            }
        }
    }
}

/// Compare `a` and `b`, treating all NaN values as less than non-NaN values.
pub fn cmp_nan_less<T: PartialOrd>(a: T, b: T) -> std::cmp::Ordering {
    match a.partial_cmp(&b) {
        Some(ordering) => ordering,
        None => {
            if is_nan(&a) {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        }
    }
}

fn reduce_min_max<T: Copy + PartialOrd>(
    pool: &TensorPool,
    input: TensorView<T>,
    axes: Option<&[i32]>,
    keep_dims: bool,
    max: bool,
) -> Result<Tensor<T>, OpError> {
    struct MinMaxReducer {
        max: bool,
    }
    impl<T: Copy + PartialOrd> Reducer<T> for MinMaxReducer {
        fn reduce<I: ExactSizeIterator<Item = T>>(&self, iter: I) -> T {
            let reduced = if self.max {
                iter.max_by(|a, b| cmp_nan_greater(*a, *b))
            } else {
                iter.min_by(|a, b| cmp_nan_less(*a, *b))
            };
            reduced.expect("attempted to get min/max of empty axis")
        }
    }
    reduce(pool, input, axes, keep_dims, MinMaxReducer { max })
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

pub fn reduce_min<T: Copy + PartialOrd>(
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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let axes = get_axes(&inputs, &self.axes)?;
        dispatch_reduce_op!(pool, input, reduce_min, axes, self.keep_dims)
    }
}

pub fn reduce_max<T: Copy + PartialOrd>(
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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
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
    impl<T: std::iter::Product> Reducer<T> for ProdReducer {
        fn reduce<I: ExactSizeIterator<Item = T>>(&self, iter: I) -> T {
            iter.product()
        }
    }
    reduce(pool, input, axes, keep_dims, ProdReducer {})
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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let axes = get_axes(&inputs, &self.axes)?;
        dispatch_reduce_op!(pool, input, reduce_prod, axes, self.keep_dims)
    }
}

pub fn reduce_sum<T: Copy + std::iter::Sum>(
    pool: &TensorPool,
    input: TensorView<T>,
    axes: Option<&[i32]>,
    keep_dims: bool,
) -> Result<Tensor<T>, OpError> {
    struct SumReducer {}
    impl<T: std::iter::Sum> Reducer<T> for SumReducer {
        fn reduce<I: ExactSizeIterator<Item = T>>(&self, iter: I) -> T {
            iter.sum()
        }
    }
    reduce(pool, input, axes, keep_dims, SumReducer {})
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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
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
        fn reduce<I: ExactSizeIterator<Item = T>>(&self, iter: I) -> T {
            iter.map(|x| x * x).sum()
        }
    }
    reduce(pool, input, axes, keep_dims, SumSquareReducer {})
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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let axes = get_axes(&inputs, &self.axes)?;
        dispatch_reduce_op!(pool, input, reduce_sum_square, axes, self.keep_dims)
    }
}

pub fn topk<T: Copy + Default + PartialOrd>(
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

    for (values, (out_values, indices)) in zip(
        values.lanes(axis),
        zip(out_values.lanes_mut(axis), indices.lanes_mut(axis)),
    ) {
        tmp.clear();
        tmp.extend(zip(values.copied(), 0..axis_size));
        tmp.select_nth_unstable_by(k - 1, topk_cmp);
        tmp.truncate(k);

        if sorted {
            tmp.sort_unstable_by(topk_cmp);
        }

        for ((out_val, out_idx), (val, idx)) in zip(zip(out_values, indices), tmp.iter()) {
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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
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
            Input::IntTensor(values) => {
                let (values, indices) =
                    topk(pool, values, k, self.axis, self.largest, self.sorted)?;
                Ok([values.into(), indices.into()].into_iter().collect())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::prelude::*;
    use rten_tensor::test_util::{eq_with_nans, expect_equal};
    use rten_tensor::{tensor, NdTensor, Tensor};

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
        let probs = tensor!([0.1, 0.5, 0.2, 0.9, 0.01, 0.6]);
        let class = arg_max(&pool, probs.view(), 0, false /* keep_dims */).unwrap();
        assert_eq!(class.item(), Some(&3));

        // Same, but keep dims
        let class = arg_max(&pool, probs.view(), 0, true /* keep_dims */).unwrap();
        assert_eq!(class.shape(), &[1]);
        assert_eq!(class.to_vec(), &[3]);

        // Common use case of a tensor of (batch, item, prob) where
        // `item` is eg. a token index in a sequence or box ID for object
        // detection.
        let seq_probs = Tensor::from_data(
            &[1, 4, 3],
            vec![
                0.1, 0.2, 0.9, // First item
                0.9, 0.1, 0.2, // Second item
                0.3, 0.8, 0.4, // Third item
                0.1, 0.01, 0.2, // Fourth item
            ],
        );
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
    }

    // We only have basic tests for ArgMin since most of the implementation is
    // shared with ArgMax.
    #[test]
    fn test_arg_min() {
        let pool = new_pool();
        let probs = tensor!([0.1, 0.5, 0.2, 0.9, 0.01, 0.6]);
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
        let probs = tensor!([0.1, 0.5, f32::NAN, 0.9, 0.01, 0.6]);
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

        let elements: Tensor<f32> = tensor!([]);
        let sums = cum_sum(&pool, elements.view(), 0).unwrap();
        assert_eq!(sums.shape(), &[0]);
        assert_eq!(sums.to_vec(), &[] as &[f32]);
    }

    #[test]
    fn test_nonzero() {
        let pool = new_pool();
        let input = tensor!((2, 2); [0., 1., 1., 1.]);
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
        let input = tensor!(3.);
        let result = nonzero(&pool, input.view());
        assert_eq!(result.shape(), &[0, 1]);

        let input = tensor!(0.);
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
            let axes = tensor!([0]);
            let result: NdTensor<f32, 2> = run_op(&*op, (input.view(), axes.view()))?;
            assert_eq!(result.shape(), [1, 3]);

            let axes = tensor!([1]);
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
        let expected = tensor!([2., 5., 8.]);
        expect_equal(&result, &expected)?;

        // Test with `keep_dims` on
        let result = reduce_mean(&pool, input.view(), Some(&[-1]), true /* keep_dims */).unwrap();
        let expected = Tensor::from_data(&[3, 1], vec![2., 5., 8.]);
        expect_equal(&result, &expected)?;

        // Reduce first dim
        let result = reduce_mean(&pool, input.view(), Some(&[0]), false /* keep_dims */).unwrap();
        let expected = tensor!([4., 5., 6.]);
        expect_equal(&result, &expected)?;

        // Reduce all axes
        let result = reduce_mean(&pool, input.view(), None, false /* keep_dims */).unwrap();
        let expected = Tensor::from_scalar(5.);
        expect_equal(&result, &expected)?;

        // Reduce all axes (specified via empty array)
        let result = reduce_mean(&pool, input.view(), Some(&[]), false /* keep_dims */).unwrap();
        let expected = Tensor::from_scalar(5.);
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
            Tensor::from_scalar(5.0).view(),
            Some(&[]),
            false, /* keep_dims */
        )
        .unwrap();
        assert_eq!(result.item(), Some(&5.0));

        // Reduce a vector
        let result = reduce_mean(
            &pool,
            tensor!([0., 10.]).view(),
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
            tensor!([]).view(),
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
        let input: Tensor<f32> = tensor!([1.5, 2.5, 3.5, 4.5, 5.5]);
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
        let input: Tensor<f32> = tensor!([1.5, 2.5, 3.5, f32::NAN, 5.5]);
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
        let input: Tensor<i32> = tensor!([1, 2, 3, 4, 5]);
        let result = result_item(reduce_prod(
            &pool,
            input.view(),
            Some(&[0]),
            false, /* keep_dims */
        ));
        assert_eq!(result, input.iter().product::<i32>());

        // Float tensor
        let input: Tensor<f32> = tensor!([1.5, 2.5, 3.5, 4.5, 5.5]);
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
        let input: Tensor<i32> = tensor!([1, 2, 3, 4, 5]);
        let result = result_item(reduce_sum(
            &pool,
            input.view(),
            Some(&[0]),
            false, /* keep_dims */
        ));
        assert_eq!(result, input.iter().sum::<i32>());

        // Float tensor
        let input: Tensor<f32> = tensor!([1.5, 2.5, 3.5, 4.5, 5.5]);
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
        let input: Tensor<i32> = tensor!([1, 2, 3, 4, 5]);
        let result = result_item(reduce_sum_square(
            &pool,
            input.view(),
            Some(&[0]),
            false, /* keep_dims */
        ));
        assert_eq!(result, input.iter().map(|x| x * x).sum::<i32>());

        // Float tensor
        let input: Tensor<f32> = tensor!([1.5, 2.5, 3.5, 4.5, 5.5]);
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
                input: tensor!([0., 1., 2.]),
                k: 2,
                expected: Ok((tensor!([2., 1.]), tensor!([2, 1]))),
                ..Default::default()
            },
            // Simple case, largest=false
            Case {
                input: tensor!([0., 1., 2.]),
                k: 2,
                largest: false,
                expected: Ok((tensor!([0., 1.]), tensor!([0, 1]))),
                ..Default::default()
            },
            // Special case where k=0
            Case {
                input: tensor!([0., 1., 2.]),
                k: 0,
                expected: Ok((tensor!([]), tensor!([]))),
                ..Default::default()
            },
            // Tie break by index when input values are equal.
            Case {
                input: tensor!([1., 0., 2., 3., 1.]),
                k: 5,
                expected: Ok((tensor!([3., 2., 1., 1., 0.]), tensor!([3, 2, 0, 4, 1]))),
                ..Default::default()
            },
            // Tie break by index when input values are equal, largest=false
            Case {
                input: tensor!([1., 0., 2., 3., 1.]),
                k: 5,
                largest: false,
                expected: Ok((tensor!([0., 1., 1., 2., 3.]), tensor!([1, 0, 4, 2, 3]))),
                ..Default::default()
            },
            // NaN values
            Case {
                input: tensor!([0., f32::NAN, 2.]),
                k: 2,
                expected: Ok((tensor!([f32::NAN, 2.]), tensor!([1, 2]))),
                ..Default::default()
            },
            // NaN values, with largest=false
            Case {
                input: tensor!([0., f32::NAN, 2.]),
                k: 3,
                expected: Ok((tensor!([0., 2., f32::NAN]), tensor!([0, 2, 1]))),
                largest: false,
                ..Default::default()
            },
            // Invalid k value
            Case {
                input: tensor!([0., 1., 2.]),
                k: 4,
                expected: Err(OpError::InvalidValue("k > dimension size")),
                ..Default::default()
            },
            // Scalar input
            Case {
                input: tensor!(0.),
                k: 2,
                expected: Err(OpError::InvalidValue("Axis is invalid")),
                ..Default::default()
            },
            // 2D input, take top-K over axis 1
            Case {
                input: tensor!((3, 3); [
                    0., 1., 2., //
                    0., 1., 3., //
                    0., 1., 4. //
                ]),
                k: 2,
                expected: Ok((
                    tensor!((3, 2); [
                        2., 1., //
                        3., 1., //
                        4., 1. //
                    ]),
                    tensor!((3, 2); [
                        2, 1, //
                        2, 1, //
                        2, 1 //
                    ]),
                )),
                ..Default::default()
            },
            // 2D input, take top-K over axis 0
            Case {
                input: tensor!((3, 3); [
                    0., 1., 2., //
                    3., 4., 5., //
                    6., 7., 8. //
                ]),
                k: 2,
                axis: Some(0),
                expected: Ok((
                    tensor!((2, 3); [
                        6., 7., 8., //
                        3., 4., 5. //
                    ]),
                    tensor!((2, 3); [
                        2, 2, 2, //
                        1, 1, 1 //
                    ]),
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
