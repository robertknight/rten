use std::iter::zip;

use wasnn_tensor;
use wasnn_tensor::{
    DynIndices, Layout, NdTensor, Offsets, SliceRange, Tensor, TensorCommon, TensorView,
};

use crate::number::Identities;
use crate::ops::layout::squeeze_in_place;
use crate::ops::{
    resolve_axes, resolve_axis, Input, InputList, IntoOpResult, OpError, Operator, Output,
};

/// Iterator over slices of a tensor along a target dimension of size N.
///
/// Conceptually this iterator steps through every distinct slice of a tensor
/// where a target dim is varied from 0..N and other indices are held fixed.
struct DimSlices<'a, T: Copy> {
    tensor: TensorView<'a, T>,
    slice_start_offsets: Offsets,
    dim_size: usize,
    dim_stride: usize,
}

impl<'a, T: Copy> DimSlices<'a, T> {
    /// Create a DimSlices iterator which yields all possible slices over
    /// the `dim` dimension of `tensor`.
    fn new(tensor: TensorView<'a, T>, dim: usize) -> DimSlices<'a, T> {
        let slice_starts: Vec<SliceRange> = (0..tensor.ndim())
            .map(|i| {
                if i == dim {
                    (0..1).into()
                } else {
                    (0..(tensor.shape()[i] as isize)).into()
                }
            })
            .collect();
        DimSlices {
            tensor: tensor.clone(),
            slice_start_offsets: tensor.slice_offsets(&slice_starts),
            dim_size: tensor.size(dim),
            dim_stride: tensor.stride(dim),
        }
    }

    /// Yield the next slice over the target dimension.
    fn next(&mut self) -> Option<impl ExactSizeIterator<Item = T> + 'a> {
        self.slice_start_offsets.next().map(|offset| {
            self.tensor
                .data()
                .iter()
                .copied()
                .skip(offset)
                .step_by(self.dim_stride)
                .take(self.dim_size)
        })
    }
}

/// Compute the indices of the max elements along an axis, according to a
/// comparison function `compare`.
fn select_max_index<T: Copy, Cmp: Fn(T, T) -> std::cmp::Ordering>(
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
    let mut reduced_data = Vec::with_capacity(reduced_shape.iter().product());

    if !input.is_empty() {
        let mut slice_iter = DimSlices::new(input, resolved_axis);
        while let Some(slice) = slice_iter.next() {
            let (index, _) = slice.enumerate().max_by(|a, b| compare(a.1, b.1)).unwrap(); // Ok because we checked tensor is not empty.
            reduced_data.push(index as i32);
        }
    }

    let mut reduced = Tensor::<i32>::from_data(&reduced_shape, reduced_data);

    if !keep_dims {
        let axes = &[resolved_axis as i32];
        squeeze_in_place(&mut reduced, Some(axes.into())).expect("Invalid axis");
    }

    Ok(reduced)
}

/// Return the index of the maximum value along a given axis.
///
/// NaN values are propagated by treating NaNs as greater than other values.
pub fn arg_max<T: Copy + PartialOrd>(
    input: TensorView<T>,
    axis: isize,
    keep_dims: bool,
) -> Result<Tensor<i32>, OpError> {
    select_max_index(input, axis, keep_dims, cmp_nan_greater)
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

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require_as::<f32>(0)?;
        arg_max(input.view(), self.axis, self.keep_dims).into_op_result()
    }
}

/// Return the index of the minimum value along a given axis.
///
/// NaN values are propagated by treating NaNs as smaller than other values.
pub fn arg_min<T: Copy + PartialOrd>(
    input: TensorView<T>,
    axis: isize,
    keep_dims: bool,
) -> Result<Tensor<i32>, OpError> {
    select_max_index(input, axis, keep_dims, |a, b| match a.partial_cmp(&b) {
        Some(ordering) => ordering.reverse(),
        None => cmp_nan_greater(a, b),
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

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require_as::<f32>(0)?;
        arg_min(input.view(), self.axis, self.keep_dims).into_op_result()
    }
}

pub fn cum_sum<T: Copy + Identities + std::ops::AddAssign>(
    input: TensorView<T>,
    axis: isize,
) -> Result<Tensor<T>, OpError> {
    let resolved_axis = resolve_axis(input.ndim(), axis)?;
    let mut out_data = Vec::with_capacity(input.len());

    if !input.is_empty() {
        let mut slice_iter = DimSlices::new(input.clone(), resolved_axis);
        while let Some(slice) = slice_iter.next() {
            let mut cum_sum = T::zero();
            out_data.extend(slice.map(|val| {
                cum_sum += val;
                cum_sum
            }));
        }
    }

    Ok(Tensor::from_data(input.shape(), out_data))
}

#[derive(Debug)]
pub struct CumSum {}

impl Operator for CumSum {
    fn name(&self) -> &str {
        "CumSum"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let axis: i32 = inputs.require_as_scalar(1)?;
        match input {
            Input::IntTensor(input) => cum_sum(input.view(), axis as isize).into_op_result(),
            Input::FloatTensor(input) => cum_sum(input.view(), axis as isize).into_op_result(),
        }
    }
}

/// Return the indices of nonzero elements in `input` as a `(dim, index)` tensor.
pub fn nonzero<T: Default + PartialEq>(input: TensorView<T>) -> Tensor<i32> {
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
        .to_tensor()
}

#[derive(Debug)]
pub struct NonZero {}

impl Operator for NonZero {
    fn name(&self) -> &str {
        "NonZero"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        match input {
            Input::IntTensor(input) => nonzero(input.view()).into_op_result(),
            Input::FloatTensor(input) => nonzero(input.view()).into_op_result(),
        }
    }
}

/// Trait for reducing a subset of elements from a tensor to a single value.
///
/// This is a trait rather than a closure to support being invoked with
/// dynamically chosen iterator types.
trait Reducer<T> {
    fn reduce<I: ExactSizeIterator<Item = T>>(&self, iter: I) -> T;
}

fn reduce<T: Copy + Default, R: Reducer<T>>(
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
    let mut reduced_data = Vec::with_capacity(reduced_shape.iter().product());

    match (reduced_inner_dims, input.is_contiguous()) {
        (Some(ndims), true) => {
            // Fast path for reducing over contiguous chunks of the input.
            let slice_len = if ndims == input.ndim() {
                input.len()
            } else {
                input.stride(input.ndim() - 1 - ndims)
            };

            reduced_data.extend(
                input
                    .data()
                    .chunks(slice_len)
                    .map(|chunk| reducer.reduce(chunk.iter().copied())),
            );
        }
        _ => {
            if resolved_axes.len() == 1 {
                // Fast path for reducing a single axis.
                let resolved_axis = resolved_axes[0];
                let mut slice_iter = DimSlices::new(input, resolved_axis);
                while let Some(slice) = slice_iter.next() {
                    reduced_data.push(reducer.reduce(slice));
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
                            SliceRange::new(0, input.size(dim) as isize, 1)
                        } else {
                            SliceRange::new(idx as isize, idx as isize + 1, 1)
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
    }

    reduce(input, axes, keep_dims, MeanReducer {})
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

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require_as(0)?;
        reduce_mean(
            input.view(),
            self.axes.as_ref().map(|axis| &axis[..]),
            self.keep_dims,
        )
        .into_op_result()
    }
}

pub fn reduce_l2(
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

    reduce(input, axes, keep_dims, L2Reducer {})
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

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require_as(0)?;
        reduce_l2(
            input.view(),
            self.axes.as_ref().map(|axis| &axis[..]),
            self.keep_dims,
        )
        .into_op_result()
    }
}

macro_rules! dispatch_reduce_op {
    ($input:expr, $reduce_op:ident, $axes:expr, $keep_dims:expr) => {
        match $input {
            Input::FloatTensor(input) => $reduce_op(
                input.view(),
                $axes.as_ref().map(|axis| &axis[..]),
                $keep_dims,
            )
            .into_op_result(),
            Input::IntTensor(input) => $reduce_op(
                input.view(),
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
fn cmp_nan_greater<T: PartialOrd>(a: T, b: T) -> std::cmp::Ordering {
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
fn cmp_nan_less<T: PartialOrd>(a: T, b: T) -> std::cmp::Ordering {
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

fn reduce_min_max<T: Copy + Default + PartialOrd>(
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
    reduce(input, axes, keep_dims, MinMaxReducer { max })
}

pub fn reduce_min<T: Copy + Default + PartialOrd>(
    input: TensorView<T>,
    axes: Option<&[i32]>,
    keep_dims: bool,
) -> Result<Tensor<T>, OpError> {
    reduce_min_max(input, axes, keep_dims, false /* max */)
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

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        dispatch_reduce_op!(input, reduce_min, self.axes, self.keep_dims)
    }
}

pub fn reduce_max<T: Copy + Default + PartialOrd>(
    input: TensorView<T>,
    axes: Option<&[i32]>,
    keep_dims: bool,
) -> Result<Tensor<T>, OpError> {
    reduce_min_max(input, axes, keep_dims, true /* max */)
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

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        dispatch_reduce_op!(input, reduce_max, self.axes, self.keep_dims)
    }
}

pub fn reduce_prod<T: Copy + Default + std::iter::Product>(
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
    reduce(input, axes, keep_dims, ProdReducer {})
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

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        dispatch_reduce_op!(input, reduce_prod, self.axes, self.keep_dims)
    }
}

pub fn reduce_sum<T: Copy + Default + std::iter::Sum>(
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
    reduce(input, axes, keep_dims, SumReducer {})
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

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        dispatch_reduce_op!(input, reduce_sum, self.axes, self.keep_dims)
    }
}

#[cfg(test)]
mod tests {
    use wasnn_tensor::test_util::expect_equal;
    use wasnn_tensor::{tensor, Layout, Tensor, TensorCommon};

    use crate::ops::{
        arg_max, arg_min, cum_sum, nonzero, reduce_l2, reduce_max, reduce_mean, reduce_min,
        reduce_prod, reduce_sum, OpError,
    };

    #[test]
    fn test_arg_max() {
        // Reduce a simple vector.
        let probs = tensor!([0.1, 0.5, 0.2, 0.9, 0.01, 0.6]);
        let class = arg_max(probs.view(), 0, false /* keep_dims */).unwrap();
        assert_eq!(class.item(), Some(&3));

        // Same, but keep dims
        let class = arg_max(probs.view(), 0, true /* keep_dims */).unwrap();
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
        let seq_classes = arg_max(seq_probs.view(), 2, false /* keep_dims */).unwrap();
        assert_eq!(seq_classes.shape(), &[1, 4]);
        assert_eq!(seq_classes.to_vec(), &[2, 0, 1, 2]);

        // Same, but keep dims
        let seq_classes = arg_max(seq_probs.view(), 2, true /* keep_dims */).unwrap();
        assert_eq!(seq_classes.shape(), &[1, 4, 1]);
        assert_eq!(seq_classes.to_vec(), &[2, 0, 1, 2]);

        // Empty tensor, axis is a non-zero-sized dim
        let empty = Tensor::<i32>::from_data(&[10, 0, 5], vec![]);
        let result = arg_max(empty.view(), 0, false /* keep_dims */).unwrap();
        assert_eq!(result.shape(), &[0, 5]);
        assert_eq!(result.to_vec(), &[] as &[i32]);

        // Empty tensor, axis is a zero-sized dim
        let empty = Tensor::<i32>::from_data(&[10, 0, 5], vec![]);
        let result = arg_max(empty.view(), 1, false /* keep_dims */);
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
        let probs = tensor!([0.1, 0.5, 0.2, 0.9, 0.01, 0.6]);
        let class = arg_min(probs.view(), 0, false /* keep_dims */).unwrap();
        assert_eq!(class.item(), Some(&4));
    }

    // ONNX does not specify how ArgMin and ArgMax should handle NaNs. We opt to
    // be consistent with ReduceMin and ReduceMax by "propagating" NaNs, which
    // for these operators means returning the index of a NaN value over other
    // indices. This is consistent with numpy's `argmin` and `argmax`.
    #[test]
    fn test_arg_min_max_nan() {
        let probs = tensor!([0.1, 0.5, f32::NAN, 0.9, 0.01, 0.6]);
        let min_idx = arg_min(probs.view(), 0, false /* keep_dims */).unwrap();
        let max_idx = arg_max(probs.view(), 0, false /* keep_dims */).unwrap();
        assert_eq!(min_idx.item(), Some(&2));
        assert_eq!(max_idx.item(), Some(&2));
    }

    #[test]
    fn test_cum_sum() {
        let elements = Tensor::from_vec((0..=5).collect());
        let sums = cum_sum(elements.view(), 0).unwrap();
        assert_eq!(sums.shape(), &[6]);
        assert_eq!(sums.to_vec(), &[0, 1, 3, 6, 10, 15]);

        let elements = Tensor::from_data(&[2, 4], (0..4).chain(0..4).collect::<Vec<_>>());
        let sums = cum_sum(elements.view(), 1).unwrap();
        assert_eq!(sums.shape(), &[2, 4]);
        assert_eq!(sums.to_vec(), &[0, 1, 3, 6, 0, 1, 3, 6]);

        let sums = cum_sum(elements.view(), 0).unwrap();
        assert_eq!(sums.shape(), &[2, 4]);
        assert_eq!(sums.to_vec(), &[0, 0, 1, 2, 2, 4, 3, 6]);

        let elements: Tensor<f32> = tensor!([]);
        let sums = cum_sum(elements.view(), 0).unwrap();
        assert_eq!(sums.shape(), &[0]);
        assert_eq!(sums.to_vec(), &[] as &[f32]);
    }

    #[test]
    fn test_nonzero() {
        let input = tensor!((2, 2); [0., 1., 1., 1.]);
        let result = nonzero(input.view());
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
        let input = tensor!(3.);
        let result = nonzero(input.view());
        assert_eq!(result.shape(), &[0, 1]);

        let input = tensor!(0.);
        let result = nonzero(input.view());
        assert_eq!(result.shape(), &[0, 0]);
    }

    #[test]
    fn test_reduce_l2() -> Result<(), String> {
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

        let result = reduce_l2(input.view(), Some(&[2]), false /* keep_dims */).unwrap();
        expect_equal(&result, &expected)?;

        let result = reduce_l2(input.view(), Some(&[2]), true /* keep_dims */).unwrap();
        let expected = expected.clone_with_shape(&[3, 2, 1]);
        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_reduce_mean() -> Result<(), String> {
        let input = Tensor::from_data(&[3, 3], vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        // Test with `keep_dims` off
        let result = reduce_mean(input.view(), Some(&[-1]), false /* keep_dims */).unwrap();
        let expected = tensor!([2., 5., 8.]);
        expect_equal(&result, &expected)?;

        // Test with `keep_dims` on
        let result = reduce_mean(input.view(), Some(&[-1]), true /* keep_dims */).unwrap();
        let expected = Tensor::from_data(&[3, 1], vec![2., 5., 8.]);
        expect_equal(&result, &expected)?;

        // Reduce first dim
        let result = reduce_mean(input.view(), Some(&[0]), false /* keep_dims */).unwrap();
        let expected = tensor!([4., 5., 6.]);
        expect_equal(&result, &expected)?;

        // Reduce all axes
        let result = reduce_mean(input.view(), None, false /* keep_dims */).unwrap();
        let expected = Tensor::from_scalar(5.);
        expect_equal(&result, &expected)?;

        // Reduce all axes (specified via empty array)
        let result = reduce_mean(input.view(), Some(&[]), false /* keep_dims */).unwrap();
        let expected = Tensor::from_scalar(5.);
        expect_equal(&result, &expected)?;

        // Test case from ONNX spec
        let input = Tensor::from_data(
            &[3, 2, 2],
            vec![5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.],
        );
        let expected = Tensor::from_data(&[3, 2], vec![12.5, 1.5, 35., 1.5, 57.5, 1.5]);
        let result = reduce_mean(input.view(), Some(&[1]), false /* keep_dims */).unwrap();
        expect_equal(&result, &expected)?;

        // Reduce a scalar value
        let result = reduce_mean(
            Tensor::from_scalar(5.0).view(),
            Some(&[]),
            false, /* keep_dims */
        )
        .unwrap();
        assert_eq!(result.item(), Some(&5.0));

        // Reduce a vector
        let result = reduce_mean(
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
        let input = Tensor::from_data(&[3, 3], vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        let result = reduce_mean(input.view(), Some(&[3]), false /* keep_dims */);
        assert_eq!(result.err(), Some(OpError::InvalidValue("Axis is invalid")));

        let result = reduce_mean(input.view(), Some(&[-3]), false /* keep_dims */);
        assert_eq!(result.err(), Some(OpError::InvalidValue("Axis is invalid")));

        // Empty tensor
        let result = reduce_mean(tensor!([]).view(), Some(&[0]), false /* keep_dims */);
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
        let input: Tensor<f32> = tensor!([1.5, 2.5, 3.5, 4.5, 5.5]);
        let min = result_item(reduce_min(
            input.view(),
            Some(&[0]),
            false, /* keep_dims */
        ));
        let max = result_item(reduce_max(
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
        let input: Tensor<f32> = tensor!([1.5, 2.5, 3.5, f32::NAN, 5.5]);
        let min = result_item(reduce_min(
            input.view(),
            Some(&[0]),
            false, /* keep_dims */
        ));
        let max = result_item(reduce_max(
            input.view(),
            Some(&[0]),
            false, /* keep_dims */
        ));
        assert!(min.is_nan());
        assert!(max.is_nan());
    }

    #[test]
    fn test_reduce_prod() {
        // Int tensor
        let input: Tensor<i32> = tensor!([1, 2, 3, 4, 5]);
        let result = result_item(reduce_prod(
            input.view(),
            Some(&[0]),
            false, /* keep_dims */
        ));
        assert_eq!(result, input.iter().product::<i32>());

        // Float tensor
        let input: Tensor<f32> = tensor!([1.5, 2.5, 3.5, 4.5, 5.5]);
        let result = result_item(reduce_prod(
            input.view(),
            Some(&[0]),
            false, /* keep_dims */
        ));
        assert_eq!(result, input.iter().product::<f32>());
    }

    #[test]
    fn test_reduce_sum() {
        // Int tensor
        let input: Tensor<i32> = tensor!([1, 2, 3, 4, 5]);
        let result = result_item(reduce_sum(
            input.view(),
            Some(&[0]),
            false, /* keep_dims */
        ));
        assert_eq!(result, input.iter().sum::<i32>());

        // Float tensor
        let input: Tensor<f32> = tensor!([1.5, 2.5, 3.5, 4.5, 5.5]);
        let result = result_item(reduce_sum(
            input.view(),
            Some(&[0]),
            false, /* keep_dims */
        ));
        assert_eq!(result, input.iter().sum::<f32>());
    }
}
