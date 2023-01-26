use crate::number::Identities;
use crate::ops::layout::squeeze_in_place;
use crate::ops::{
    resolve_axes, resolve_axis, Input, InputList, IntoOpResult, OpError, Operator, Output,
};
use crate::tensor;
use crate::tensor::{IndexIterator, Offsets, SliceRange, Tensor, TensorLayout};

/// Iterator over slices of a tensor along a target dimension of size N.
///
/// Conceptually this iterator steps through every distinct slice of a tensor
/// where a target dim is varied from 0..N and other indices are held fixed.
struct DimSlices<'a, T: Copy> {
    tensor: &'a Tensor<T>,
    slice_start_offsets: Offsets,
    dim_size: usize,
    dim_stride: usize,
}

impl<'a, T: Copy> DimSlices<'a, T> {
    /// Create a DimSlices iterator which yields all possible slices over
    /// the `dim` dimension of `tensor`.
    fn new(tensor: &'a Tensor<T>, dim: usize) -> DimSlices<'a, T> {
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
            tensor,
            slice_start_offsets: tensor.slice_offsets(&slice_starts),
            dim_size: tensor.shape()[dim],
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
fn index_select<T: Copy, Cmp: Fn(T, T) -> bool>(
    input: &Tensor<T>,
    axis: isize,
    keep_dims: bool,
    compare: Cmp,
) -> Result<Tensor<i32>, OpError> {
    let resolved_axis = resolve_axis(input.ndim(), axis)?;
    if input.shape()[resolved_axis] == 0 {
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
            let (index, _) = slice
                .enumerate()
                .fold(None, |acc, (i, val)| match acc {
                    Some((_index, max_val)) => {
                        if compare(val, max_val) {
                            Some((i, val))
                        } else {
                            acc
                        }
                    }
                    None => Some((i, val)),
                })
                .unwrap(); // Ok because we checked tensor is not empty.
            reduced_data.push(index as i32);
        }
    }

    let mut reduced = Tensor::<i32>::from_data(&reduced_shape, reduced_data);

    if !keep_dims {
        squeeze_in_place(&mut reduced, Some(&tensor!([resolved_axis as i32])))
            .expect("Invalid axis");
    }

    Ok(reduced)
}

pub fn arg_max<T: Copy + PartialOrd>(
    input: &Tensor<T>,
    axis: isize,
    keep_dims: bool,
) -> Result<Tensor<i32>, OpError> {
    index_select(input, axis, keep_dims, |a, b| a > b)
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
        arg_max(input, self.axis, self.keep_dims).into_op_result()
    }
}

pub fn arg_min<T: Copy + PartialOrd>(
    input: &Tensor<T>,
    axis: isize,
    keep_dims: bool,
) -> Result<Tensor<i32>, OpError> {
    index_select(input, axis, keep_dims, |a, b| a < b)
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
        arg_min(input, self.axis, self.keep_dims).into_op_result()
    }
}

pub fn cum_sum<T: Copy + Identities + std::ops::AddAssign>(
    input: &Tensor<T>,
    axis: isize,
) -> Result<Tensor<T>, OpError> {
    let resolved_axis = resolve_axis(input.ndim(), axis)?;
    let mut out_data = Vec::with_capacity(input.len());

    if !input.is_empty() {
        let mut slice_iter = DimSlices::new(input, resolved_axis);
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
            Input::IntTensor(input) => cum_sum(input, axis as isize).into_op_result(),
            Input::FloatTensor(input) => cum_sum(input, axis as isize).into_op_result(),
        }
    }
}

/// Trait for reducing a subset of elements from a tensor to a single value.
///
/// This is a trait rather than a closure to support being invoked with
/// dynamically chosen iterator types.
trait Reducer<T: Copy> {
    fn reduce<I: ExactSizeIterator<Item = T>>(&self, iter: I) -> T;
}

fn reduce<T: Copy + Default, R: Reducer<T>>(
    input: &Tensor<T>,
    axes: Option<&[i32]>,
    keep_dims: bool,
    reducer: R,
) -> Result<Tensor<T>, OpError> {
    let mut resolved_axes = match axes {
        Some(axes) if !axes.is_empty() => resolve_axes(input.ndim(), axes.iter().copied())?,
        _ => (0..input.ndim()).collect(),
    };
    resolved_axes.sort();

    if input.ndim() == 0 {
        return Ok(Tensor::from_scalar(reducer.reduce(input.iter())));
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
                            0..1
                        } else {
                            0..input.shape()[dim]
                        }
                    })
                    .collect();
                let mut outer_iter = IndexIterator::from_ranges(&outer_range);
                let mut inner_range = Vec::with_capacity(input.ndim());
                while let Some(index) = outer_iter.next() {
                    inner_range.clear();
                    inner_range.extend(index.iter().enumerate().map(|(dim, &idx)| {
                        if resolved_axes.contains(&dim) {
                            SliceRange::new(0, input.shape()[dim] as isize, 1)
                        } else {
                            SliceRange::new(idx as isize, idx as isize + 1, 1)
                        }
                    }));
                    let reduced = reducer.reduce(input.slice_iter(&inner_range));
                    reduced_data.push(reduced);
                }
            }
        }
    }

    let mut reduced = Tensor::<T>::from_data(&reduced_shape, reduced_data);

    if !keep_dims {
        let resolved_axes_i32 = resolved_axes.iter().map(|&axis| axis as i32).collect();
        squeeze_in_place(&mut reduced, Some(&Tensor::from_vec(resolved_axes_i32)))
            .expect("Invalid axis");
    }

    Ok(reduced)
}

pub fn reduce_mean(
    input: &Tensor,
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
            input,
            self.axes.as_ref().map(|axis| &axis[..]),
            self.keep_dims,
        )
        .into_op_result()
    }
}

pub fn reduce_l2(input: &Tensor, axes: Option<&[i32]>, keep_dims: bool) -> Result<Tensor, OpError> {
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
            input,
            self.axes.as_ref().map(|axis| &axis[..]),
            self.keep_dims,
        )
        .into_op_result()
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::{arg_max, arg_min, cum_sum, reduce_l2, reduce_mean, OpError};
    use crate::tensor;
    use crate::tensor::{from_data, from_scalar, from_vec, Tensor, TensorLayout};
    use crate::test_util::expect_equal;

    #[test]
    fn test_arg_max() {
        // Reduce a simple vector.
        let probs = tensor!([0.1, 0.5, 0.2, 0.9, 0.01, 0.6]);
        let class = arg_max(&probs, 0, false /* keep_dims */).unwrap();
        assert_eq!(class.item(), Some(3));

        // Same, but keep dims
        let class = arg_max(&probs, 0, true /* keep_dims */).unwrap();
        assert_eq!(class.shape(), &[1]);
        assert_eq!(class.to_vec(), &[3]);

        // Common use case of a tensor of (batch, item, prob) where
        // `item` is eg. a token index in a sequence or box ID for object
        // detection.
        let seq_probs = from_data(
            &[1, 4, 3],
            vec![
                0.1, 0.2, 0.9, // First item
                0.9, 0.1, 0.2, // Second item
                0.3, 0.8, 0.4, // Third item
                0.1, 0.01, 0.2, // Fourth item
            ],
        );
        let seq_classes = arg_max(&seq_probs, 2, false /* keep_dims */).unwrap();
        assert_eq!(seq_classes.shape(), &[1, 4]);
        assert_eq!(seq_classes.to_vec(), &[2, 0, 1, 2]);

        // Same, but keep dims
        let seq_classes = arg_max(&seq_probs, 2, true /* keep_dims */).unwrap();
        assert_eq!(seq_classes.shape(), &[1, 4, 1]);
        assert_eq!(seq_classes.to_vec(), &[2, 0, 1, 2]);

        // Empty tensor, axis is a non-zero-sized dim
        let empty = from_data::<i32>(&[10, 0, 5], vec![]);
        let result = arg_max(&empty, 0, false /* keep_dims */).unwrap();
        assert_eq!(result.shape(), &[0, 5]);
        assert_eq!(result.to_vec(), &[] as &[i32]);

        // Empty tensor, axis is a zero-sized dim
        let empty = from_data::<i32>(&[10, 0, 5], vec![]);
        let result = arg_max(&empty, 1, false /* keep_dims */);
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue(
                "Cannot select index from empty sequence"
            ))
        );
    }

    // We only have base tests for ArgMin since most of the implementation is
    // shared with ArgMax.
    #[test]
    fn test_arg_min() {
        let probs = tensor!([0.1, 0.5, 0.2, 0.9, 0.01, 0.6]);
        let class = arg_min(&probs, 0, false /* keep_dims */).unwrap();
        assert_eq!(class.item(), Some(4));
    }

    #[test]
    fn test_cum_sum() {
        let elements = from_vec((0..=5).collect());
        let sums = cum_sum(&elements, 0).unwrap();
        assert_eq!(sums.shape(), &[6]);
        assert_eq!(sums.to_vec(), &[0, 1, 3, 6, 10, 15]);

        let elements = from_data(&[2, 4], (0..4).chain(0..4).collect());
        let sums = cum_sum(&elements, 1).unwrap();
        assert_eq!(sums.shape(), &[2, 4]);
        assert_eq!(sums.to_vec(), &[0, 1, 3, 6, 0, 1, 3, 6]);

        let sums = cum_sum(&elements, 0).unwrap();
        assert_eq!(sums.shape(), &[2, 4]);
        assert_eq!(sums.to_vec(), &[0, 0, 1, 2, 2, 4, 3, 6]);

        let elements: Tensor<f32> = tensor!([]);
        let sums = cum_sum(&elements, 0).unwrap();
        assert_eq!(sums.shape(), &[0]);
        assert_eq!(sums.to_vec(), &[] as &[f32]);
    }

    #[test]
    fn test_reduce_l2() -> Result<(), String> {
        let input = from_data(&[3, 2, 2], (1..=12).map(|i| i as f32).collect());
        let expected = from_data(
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

        let result = reduce_l2(&input, Some(&[2]), false /* keep_dims */).unwrap();
        expect_equal(&result, &expected)?;

        let result = reduce_l2(&input, Some(&[2]), true /* keep_dims */).unwrap();
        let expected = expected.clone_with_shape(&[3, 2, 1]);
        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_reduce_mean() -> Result<(), String> {
        let input = from_data(&[3, 3], vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        // Test with `keep_dims` off
        let result = reduce_mean(&input, Some(&[-1]), false /* keep_dims */).unwrap();
        let expected = tensor!([2., 5., 8.]);
        expect_equal(&result, &expected)?;

        // Test with `keep_dims` on
        let result = reduce_mean(&input, Some(&[-1]), true /* keep_dims */).unwrap();
        let expected = from_data(&[3, 1], vec![2., 5., 8.]);
        expect_equal(&result, &expected)?;

        // Reduce first dim
        let result = reduce_mean(&input, Some(&[0]), false /* keep_dims */).unwrap();
        let expected = tensor!([4., 5., 6.]);
        expect_equal(&result, &expected)?;

        // Reduce all axes
        let result = reduce_mean(&input, None, false /* keep_dims */).unwrap();
        let expected = from_scalar(5.);
        expect_equal(&result, &expected)?;

        // Reduce all axes (specified via empty array)
        let result = reduce_mean(&input, Some(&[]), false /* keep_dims */).unwrap();
        let expected = from_scalar(5.);
        expect_equal(&result, &expected)?;

        // Test case from ONNX spec
        let input = from_data(
            &[3, 2, 2],
            vec![5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.],
        );
        let expected = from_data(&[3, 2], vec![12.5, 1.5, 35., 1.5, 57.5, 1.5]);
        let result = reduce_mean(&input, Some(&[1]), false /* keep_dims */).unwrap();
        expect_equal(&result, &expected)?;

        // Reduce a scalar value
        let result = reduce_mean(&from_scalar(5.0), Some(&[]), false /* keep_dims */).unwrap();
        assert_eq!(result.item(), Some(5.0));

        // Reduce a vector
        let result = reduce_mean(&tensor!([0., 10.]), Some(&[0]), false /* keep_dims */).unwrap();
        assert_eq!(result.to_vec(), &[5.0]);

        Ok(())
    }

    #[test]
    fn test_reduce_mean_invalid_inputs() {
        let input = from_data(&[3, 3], vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        let result = reduce_mean(&input, Some(&[3]), false /* keep_dims */);
        assert_eq!(result.err(), Some(OpError::InvalidValue("Axis is invalid")));

        let result = reduce_mean(&input, Some(&[-3]), false /* keep_dims */);
        assert_eq!(result.err(), Some(OpError::InvalidValue("Axis is invalid")));

        // Empty tensor
        let result = reduce_mean(&tensor!([]), Some(&[0]), false /* keep_dims */);
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Cannot reduce empty tensor"))
        );
    }
}
