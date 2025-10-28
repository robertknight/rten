use rayon::prelude::*;
use std::mem::MaybeUninit;

use rten_base::num::IsNaN;
use rten_tensor::prelude::*;
use rten_tensor::{
    NdTensorView, ResizeLayout, SliceItem, StorageMut, Tensor, TensorView, TensorViewMut,
    to_slice_items,
};
use smallvec::SmallVec;

use crate::buffer_pool::{AutoReturn, BufferPool};
use crate::operator::{IntoOpResult, OpError, OpRunContext, Operator, OutputList};
use crate::ops::reduce::{cmp_nan_greater, cmp_nan_less};
use crate::ops::{map_value_view, resolve_axis, resolve_index};
use crate::value::ValueView;

const INVALID_INDEX_ERR: OpError = OpError::InvalidValue("Entry in `indices` is out of range");

/// Trait for random-access to 1D slices.
trait GetItem {
    type Item;

    fn get(&self, index: usize) -> Option<&Self::Item>;
    fn len(&self) -> usize;
}

impl<T> GetItem for &[T] {
    type Item = T;

    fn get(&self, index: usize) -> Option<&T> {
        <[T]>::get(self, index)
    }

    fn len(&self) -> usize {
        <[T]>::len(self)
    }
}

impl<T> GetItem for NdTensorView<'_, T, 1> {
    type Item = T;

    fn get(&self, index: usize) -> Option<&T> {
        self.get(index)
    }

    fn len(&self) -> usize {
        self.size(0)
    }
}

/// Gather elements from `input` specified by `indices`.
///
/// See <https://onnx.ai/onnx/operators/onnx__Gather.html>. Per the ONNX spec this
/// is very similar to `numpy.take`. See
/// <https://numpy.org/doc/stable/reference/generated/numpy.take.html> for
/// additional explanation.
pub fn gather<T: Copy + Default>(
    pool: &BufferPool,
    input: TensorView<T>,
    axis: isize,
    indices: TensorView<i32>,
) -> Result<Tensor<T>, OpError> {
    let axis = resolve_axis(input.ndim(), axis)?;

    let full_range = |ndim: usize| -> SmallVec<[SliceItem; 4]> {
        (0..ndim).map(|_| SliceItem::full_range()).collect()
    };

    // Fast path for scalar `indices`. This amounts to indexing `input` along
    // `axis`.
    if indices.ndim() == 0
        && let Some(index) = indices.item()
    {
        let output = if input.ndim() == 1 {
            // Fast path for indexing a vector with a scalar. This is common
            // in subgraphs that process tensor shapes.
            let index = resolve_index(input.len(), *index as isize).ok_or(INVALID_INDEX_ERR)?;
            Tensor::full_in(pool, &[], input[[index]])
        } else {
            let mut slice_range = full_range(input.ndim());
            slice_range[axis] = SliceItem::Index(*index as isize);
            let slice = input
                .try_slice(slice_range.as_slice())
                .map_err(|_| INVALID_INDEX_ERR)?;
            slice.to_tensor_in(pool)
        };
        return Ok(output);
    }

    let out_shape = [
        &input.shape()[..axis],
        indices.shape(),
        &input.shape()[axis + 1..],
    ]
    .concat();

    // Fast path for common case of gathering from a contiguous input along
    // axis zero. For example, when gathering from a `[token_id, embed_dim]`
    // embedding matrix.
    if axis == 0
        && let Some(in_data) = input.data()
    {
        let in_slice_len = input.shape()[axis + 1..].iter().product();
        let mut out_data = pool.alloc(out_shape.iter().product());
        for index in indices.iter() {
            let Some(index) = resolve_index(input.size(axis), *index as isize) else {
                return Err(INVALID_INDEX_ERR);
            };
            let in_chunk = &in_data[index * in_slice_len..][..in_slice_len];
            out_data.extend_from_slice(in_chunk);
        }
        return Ok(Tensor::from_data(&out_shape, out_data));
    }

    // Construct layout for gathered slice of the input. Each slice has the same
    // layout so we construct it once outside the loop and then reuse it on each
    // iteration.
    let mut in_slice_layout = input.layout().clone();
    in_slice_layout.remove_axis_of_any_size(axis);
    let in_slice_layout = in_slice_layout;

    let mut output = Tensor::uninit_in(pool, &out_shape);
    let mut out_slice_layout = output.layout().clone();
    for _ in axis..axis + indices.ndim() {
        out_slice_layout.remove_axis_of_any_size(axis);
    }
    let out_slice_layout = out_slice_layout;

    let out_step = output.shape()[axis + indices.ndim()..].iter().product();
    let in_slice_data_len = in_slice_layout.min_data_len();
    let out_slice_data_len = out_slice_layout.min_data_len();

    let mut n_init = 0;
    let mut out_storage = output.storage_mut();
    for (index, out_data_offset) in indices.iter().zip((0..).step_by(out_step)) {
        let Some(index) = resolve_index(input.size(axis), *index as isize) else {
            return Err(INVALID_INDEX_ERR);
        };

        // Compute storage offsets for this slice.
        let in_offset = index * input.stride(axis);
        let in_slice_data = input
            .storage()
            .slice(in_offset..in_offset + in_slice_data_len);
        let out_slice_data =
            out_storage.slice_mut(out_data_offset..out_data_offset + out_slice_data_len);

        // Create input and output slices using the pre-computed layout.
        let out_slice =
            TensorViewMut::from_storage_and_layout(out_slice_data, out_slice_layout.clone());
        let in_slice = TensorView::from_storage_and_layout(in_slice_data, in_slice_layout.clone());

        // Copy data from input to output
        let out_slice = out_slice.init_from(&in_slice);
        n_init += out_slice.len();
    }

    assert_eq!(n_init, output.len());
    let output = unsafe { output.assume_init() };

    Ok(output)
}

#[derive(Debug)]
pub struct Gather {
    pub axis: isize,
}

impl Operator for Gather {
    fn name(&self) -> &str {
        "Gather"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(2)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require(0)?;
        let indices = inputs.require_as(1)?;

        map_value_view!(input, x, {
            gather(ctx.pool(), x, self.axis, indices).into_op_result()
        })
    }
}

pub fn gather_elements<T: Copy + Default + Send + Sync + std::fmt::Debug>(
    pool: &BufferPool,
    input: TensorView<T>,
    indices: TensorView<i32>,
    axis: isize,
) -> Result<Tensor<T>, OpError> {
    if input.ndim() != indices.ndim() {
        return Err(OpError::IncompatibleInputShapes(
            "Input and indices must have same rank",
        ));
    }
    let axis = resolve_axis(input.ndim(), axis)?;

    // Dimensions in `indices` other than `axis` can be smaller than the
    // corresponding input dimension, but not larger.
    for d in 0..input.ndim() {
        if d != axis && indices.size(d) > input.size(d) {
            return Err(OpError::IncompatibleInputShapes(
                "`indices` size must be <= input size in non-axis dimensions",
            ));
        }
    }

    // Trim the non-axis dimensions of the input to match indices, so that
    // we iterate over matching 1D lanes.
    let slice_ranges: Vec<_> = (0..input.ndim())
        .map(|d| {
            if d == axis {
                SliceItem::full_range()
            } else {
                SliceItem::range(0, Some(indices.size(d) as isize), 1)
            }
        })
        .collect();
    let input = input.slice(slice_ranges.as_slice());

    fn gather_lane<'a, T: Copy + 'a>(
        data: impl GetItem<Item = T>,
        indices: impl Iterator<Item = &'a i32>,
        output: impl Iterator<Item = &'a mut MaybeUninit<T>>,
    ) -> Result<(), OpError> {
        let axis_size = data.len() as i32;
        for (&idx, out) in indices.zip(output) {
            let idx = if idx < 0 { idx + axis_size } else { idx };
            if let Some(el) = data.get(idx as usize) {
                out.write(*el);
            } else {
                return Err(OpError::InvalidValue("Entry in `indices` is out of range"));
            }
        }
        Ok(())
    }

    let mut output = Tensor::uninit_in(pool, indices.shape());
    if output.is_empty() {
        // Safety: Output has zero elements, so is fully "initialized".
        return Ok(unsafe { output.assume_init() });
    }

    // When gathering from a stride-1 axis in a contiguous tensor, we can get
    // the 1D lanes by just splitting the data into chunks.
    if let Some(input_data) = input.data()
        && input.stride(axis) == 1
        && let Some(indices_data) = indices.data()
        && indices.stride(axis) == 1
    {
        let idx_size = indices.size(axis);
        input_data
            .par_chunks(input.size(axis))
            .zip(indices_data.par_chunks(idx_size))
            .zip(output.data_mut().unwrap().par_chunks_mut(idx_size))
            .try_for_each(|((data_lane, index_lane), out_lane)| {
                gather_lane(data_lane, index_lane.iter(), out_lane.iter_mut())
            })?;
    } else {
        for ((data_lane, index_lane), out_lane) in input
            .lanes(axis)
            .zip(indices.lanes(axis))
            .zip(output.lanes_mut(axis))
        {
            gather_lane(data_lane.as_view(), index_lane, out_lane)?;
        }
    }

    // Safety: All elements of `output` have been initialized.
    let output = unsafe { output.assume_init() };

    Ok(output)
}

#[derive(Debug)]
pub struct GatherElements {
    pub axis: isize,
}

impl Operator for GatherElements {
    fn name(&self) -> &str {
        "GatherElements"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(2)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require(0)?;
        let indices = inputs.require_as(1)?;

        map_value_view!(input, x, {
            gather_elements(ctx.pool(), x, indices, self.axis).into_op_result()
        })
    }
}

pub fn gather_nd<T: Clone + Default>(
    pool: &BufferPool,
    input: TensorView<T>,
    indices: TensorView<i32>,
    batch_dims: usize,
) -> Result<Tensor<T>, OpError> {
    if input.ndim() < 1 || indices.ndim() < 1 {
        return Err(OpError::InvalidValue(
            "Input and indices must have >= 1 dims",
        ));
    }
    if batch_dims >= input.ndim().min(indices.ndim()) {
        return Err(OpError::InvalidValue(
            "`input` and `indices` ndim must be > `batch_dims`",
        ));
    }

    if input.shape()[..batch_dims] != indices.shape()[..batch_dims] {
        return Err(OpError::InvalidValue(
            "`input` and `indices` batch dims have different sizes",
        ));
    }

    let idx_tuple_size = indices.size(indices.ndim() - 1);
    if idx_tuple_size < 1 || idx_tuple_size > input.ndim() - batch_dims {
        return Err(OpError::InvalidValue(
            "Size of last dim of `indices` is incorrect",
        ));
    }

    let idx_len = indices.size(indices.ndim() - 1);
    let out_shape: Vec<usize> = indices.shape()[..indices.ndim() - 1]
        .iter()
        .chain(input.shape()[batch_dims + idx_len..].iter())
        .copied()
        .collect();
    let out_slice_ndim = input.ndim() - batch_dims - idx_len;
    let out_slice_len = out_shape[out_shape.len() - out_slice_ndim..]
        .iter()
        .product();
    let mut output = Tensor::<T>::uninit_in(pool, &out_shape);

    let output_non_batch_dims = output.ndim() - batch_dims;
    let input_non_batch_dims = input.ndim() - batch_dims;
    let indices_non_batch_dims = indices.ndim() - batch_dims;

    // This allows the loop below to rely on index tuples being contiguous.
    let indices = indices.to_contiguous_in(pool).auto_return(pool);

    let mut n_init = 0;
    for (mut output, (input, indices)) in output.inner_iter_dyn_mut(output_non_batch_dims).zip(
        input
            .inner_iter_dyn(input_non_batch_dims)
            .zip(indices.inner_iter_dyn(indices_non_batch_dims)),
    ) {
        // For performance, work with data slices rather than tensor views here.
        let out_slices = output.data_mut().unwrap().chunks_mut(out_slice_len);
        let idx_slices = indices.data().unwrap().chunks(idx_tuple_size);

        if let Some(input_data) = input.data() {
            // Fast path for when the gathered data is contiguous. In that case
            // the gather just amounts to copying chunks of the input to the
            // output.
            for (out_slice, idx) in out_slices.zip(idx_slices) {
                let offset = idx
                    .iter()
                    .zip(input.strides())
                    .map(|(idx, stride)| *idx as usize * stride)
                    .sum();
                let in_slice = input_data
                    .get(offset..offset + out_slice.len())
                    .ok_or(OpError::InvalidValue("Invalid index"))?;
                for (out, x) in out_slice.iter_mut().zip(in_slice) {
                    out.write(x.clone());
                }
                n_init += out_slice.len();
            }
        } else {
            for (out_slice, idx) in out_slices.zip(idx_slices) {
                let slice_items = to_slice_items(idx);
                let in_slice = input
                    .try_slice(slice_items.as_slice())
                    .map_err(|_| OpError::InvalidValue("Invalid index"))?;

                for (out, x) in out_slice.iter_mut().zip(in_slice.iter()) {
                    out.write(x.clone());
                }
                n_init += out_slice.len();
            }
        }
    }

    // Safety: All elements of `output` are initialized.
    assert!(n_init == output.len());
    Ok(unsafe { output.assume_init() })
}

#[derive(Debug)]
pub struct GatherND {
    pub batch_dims: usize,
}

impl Operator for GatherND {
    fn name(&self) -> &str {
        "GatherND"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(2)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require(0)?;
        let indices = inputs.require_as(1)?;

        map_value_view!(input, x, {
            gather_nd(ctx.pool(), x, indices, self.batch_dims).into_op_result()
        })
    }
}

// Specifies how to combine an existing element value with an update in a
// scatter operation.
#[derive(Copy, Clone, Debug)]
pub enum ScatterReduction {
    /// Add the existing value and update.
    Add,

    /// Multiply the existing value with the update.
    Mul,

    /// Take the minimum of the existing value and the update, propagating NaNs.
    Min,

    /// Take the maximum of the existing value and the update, propagating NaNs.
    Max,
}

fn scatter_reduce<
    T: Copy + PartialOrd + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + IsNaN,
>(
    current: T,
    update: T,
    reduction: Option<ScatterReduction>,
) -> T {
    match reduction {
        Some(ScatterReduction::Add) => current + update,
        Some(ScatterReduction::Mul) => current * update,

        // nb. In the operations below, we prefer to keep the current value
        // unless the update is definitely less or NaN.
        Some(ScatterReduction::Min) => match cmp_nan_less(update, current) {
            std::cmp::Ordering::Less => update,
            _ => current,
        },
        Some(ScatterReduction::Max) => match cmp_nan_greater(update, current) {
            std::cmp::Ordering::Greater => update,
            _ => current,
        },
        None => update,
    }
}

pub fn scatter_elements<
    T: Copy + Default + PartialOrd + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + IsNaN,
>(
    pool: &BufferPool,
    data: TensorView<T>,
    indices: TensorView<i32>,
    updates: TensorView<T>,
    axis: isize,
    reduction: Option<ScatterReduction>,
) -> Result<Tensor<T>, OpError> {
    if indices.ndim() != data.ndim() {
        return Err(OpError::InvalidValue(
            "`data` and `indices` must have same rank",
        ));
    }
    if indices.shape() != updates.shape() {
        return Err(OpError::InvalidValue(
            "`indices` and `updates` must have same shape",
        ));
    }
    let axis = resolve_axis(data.ndim(), axis)?;

    let axis_size = data.size(axis);
    let mut output = data.to_tensor_in(pool);

    for (output_lane, (update_lane, index_lane)) in output
        .lanes_mut(axis)
        .zip(updates.lanes(axis).zip(indices.lanes(axis)))
    {
        let mut output_lane = output_lane.into_view();

        for (idx, update) in index_lane.zip(update_lane) {
            let Some(idx) = resolve_index(axis_size, *idx as isize) else {
                return Err(OpError::InvalidValue("Index is invalid"));
            };
            let out_el = &mut output_lane[[idx]];
            *out_el = scatter_reduce(*out_el, *update, reduction);
        }
    }

    Ok(output)
}

#[derive(Debug)]
pub struct ScatterElements {
    pub axis: isize,
    pub reduction: Option<ScatterReduction>,
}

impl Operator for ScatterElements {
    fn name(&self) -> &str {
        "ScatterElements"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(3)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let data = inputs.require(0)?;
        let indices = inputs.require_as(1)?;

        map_value_view!(data, x, {
            let updates = inputs.require_as(2)?;
            scatter_elements(ctx.pool(), x, indices, updates, self.axis, self.reduction)
                .into_op_result()
        })
    }
}

pub fn scatter_nd<
    T: Copy + Default + PartialOrd + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + IsNaN,
>(
    pool: &BufferPool,
    data: TensorView<T>,
    indices: TensorView<i32>,
    updates: TensorView<T>,
    reduction: Option<ScatterReduction>,
) -> Result<Tensor<T>, OpError> {
    if data.ndim() == 0 || indices.ndim() == 0 {
        return Err(OpError::InvalidValue(
            "`data` and `indices` must have rank >= 1",
        ));
    }

    // Per spec, the `indices` tensor is treated as a set of K-tuples where
    // `k <= data.ndim()`, specifying the indices of slices to update.
    let k = indices.size(indices.ndim() - 1);

    let expected_update_dim = data.ndim() + indices.ndim() - k - 1;
    if updates.ndim() != expected_update_dim {
        return Err(OpError::InvalidValue(
            "`updates` does not have expected rank",
        ));
    }

    let mut expected_update_shape: SmallVec<[usize; 5]> = SmallVec::new();
    expected_update_shape.extend_from_slice(&indices.shape()[..indices.ndim() - 1]);
    expected_update_shape.extend_from_slice(&data.shape()[k..data.ndim()]);
    if updates.shape() != expected_update_shape.as_slice() {
        return Err(OpError::InvalidValue(
            "`updates` does not have expected shape",
        ));
    }

    // Assuming the updates and indices are likely already contiguous, we can
    // optimize iterating over slices of the innermost dimensions using slice
    // chunks.
    let updates = updates.to_contiguous_in(pool).auto_return(pool);
    let update_slice_len: usize = updates.shape()[indices.ndim() - 1..].iter().product();
    let update_slices = updates.data().unwrap().chunks(update_slice_len);

    let indices = indices.to_contiguous_in(pool).auto_return(pool);
    let index_slices = indices
        .data()
        .unwrap()
        .chunks(indices.size(indices.ndim() - 1));

    let mut output = data.to_tensor_in(pool);
    for (index, update_slice) in index_slices.zip(update_slices) {
        let mut output_slice_offset = 0;
        for (i, (size, stride)) in index
            .iter()
            .zip(output.shape().iter().zip(output.strides().iter()))
        {
            let idx = resolve_index(*size, *i as isize)
                .ok_or(OpError::InvalidValue("invalid scatter index"))?;
            output_slice_offset += idx * stride;
        }
        let out_data = output.data_mut().unwrap();
        let out_slice = &mut out_data[output_slice_offset..][..update_slice_len];

        for (out_el, update) in out_slice.iter_mut().zip(update_slice.iter()) {
            *out_el = scatter_reduce(*out_el, *update, reduction);
        }
    }
    Ok(output)
}

#[derive(Debug)]
pub struct ScatterND {
    pub reduction: Option<ScatterReduction>,
}

impl Operator for ScatterND {
    fn name(&self) -> &str {
        "ScatterND"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(3)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let data = inputs.require(0)?;
        let indices = inputs.require_as(1)?;

        map_value_view!(data, x, {
            let updates = inputs.require_as(2)?;
            scatter_nd(ctx.pool(), x, indices, updates, self.reduction).into_op_result()
        })
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::Tensor;
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::expect_equal;
    use rten_testing::TestCases;

    use crate::buffer_pool::BufferPool;
    use crate::operator::OpError;
    use crate::ops::{
        ScatterReduction, gather, gather_elements, gather_nd, scatter_elements, scatter_nd,
    };

    #[test]
    fn test_gather_scalar_index() {
        let pool = BufferPool::new();

        // 1D input
        let input = Tensor::from([1, 20, 30]);
        for i in 0..input.len() {
            let indices = Tensor::from(i as i32);
            let result = gather(&pool, input.view(), 0, indices.view()).unwrap();
            assert_eq!(result.item(), Some(&input[[i]]))
        }

        // 2D input
        let input = Tensor::from([[1, 2], [3, 4]]);
        let result = gather(&pool, input.view(), 0, Tensor::from(0).view()).unwrap();
        assert_eq!(result, Tensor::from([1, 2]));
        let result = gather(&pool, input.view(), 0, Tensor::from(1).view()).unwrap();
        assert_eq!(result, Tensor::from([3, 4]));
    }

    #[test]
    fn test_gather() -> Result<(), Box<dyn Error>> {
        let pool = BufferPool::new();

        // Test case shrunk down from a small BERT model where `gather` is used
        // to lookup embeddings.
        //
        // This exercises the fast path for axis=0 with contiguous input.
        let mut rng = XorShiftRng::new(1234);
        let input = Tensor::<f32>::rand(&[128, 10], &mut rng);
        let indices = Tensor::from_data(&[2, 2], vec![2, 5, 8, 50]);
        let result = gather(&pool, input.view(), 0, indices.view()).unwrap();
        let expected = Tensor::from_fn(&[2, 2, 10], |index| {
            let [x, y, z] = index.try_into().unwrap();
            let idx = indices[[x, y]] as usize;
            input[[idx, z]]
        });
        assert_eq!(result, expected);

        // Test case #1 from ONNX spec.
        let input = Tensor::from_data(&[3, 2], vec![1.0, 1.2, 2.3, 3.4, 4.5, 5.7]);
        let indices = Tensor::from_data(&[2, 2], vec![0, 1, 1, 2]);
        let expected = Tensor::from_data(&[2, 2, 2], vec![1.0, 1.2, 2.3, 3.4, 2.3, 3.4, 4.5, 5.7]);
        let result = gather(&pool, input.view(), 0, indices.view()).unwrap();
        expect_equal(&result, &expected)?;

        // Test case #2 from ONNX spec.
        let input = Tensor::from_data(&[3, 3], vec![1.0, 1.2, 1.9, 2.3, 3.4, 3.9, 4.5, 5.7, 5.9]);
        let indices = Tensor::from_data(&[1, 2], vec![0, 2]);
        let expected = Tensor::from_data(&[3, 1, 2], vec![1.0, 1.9, 2.3, 3.9, 4.5, 5.9]);
        let result = gather(&pool, input.view(), 1, indices.view()).unwrap();
        expect_equal(&result, &expected)?;

        // Negative index values.
        let input = Tensor::from([1, 2, 3]);
        let indices = Tensor::from([-1, -2, -3]);
        let expected = Tensor::from([3, 2, 1]);
        let result = gather(&pool, input.view(), 0, indices.view()).unwrap();
        assert_eq!(&result, &expected);

        // Empty indices
        let input = Tensor::from([1, 2, 3]);
        let indices = Tensor::from([0i32; 0]);
        let expected = Tensor::from([0i32; 0]);
        let result = gather(&pool, input.view(), 0, indices.view()).unwrap();
        assert_eq!(&result, &expected);

        Ok(())
    }

    #[test]
    fn test_gather_invalid_axis() {
        let pool = BufferPool::new();

        let mut rng = XorShiftRng::new(1234);
        let input = Tensor::<f32>::rand(&[128, 10], &mut rng);
        let indices = Tensor::from_data(&[2, 2], vec![2, 5, 8, 50]);
        let result = gather(&pool, input.view(), 5, indices.view());
        assert_eq!(result.err(), Some(OpError::InvalidValue("Axis is invalid")));
    }

    #[test]
    fn test_gather_invalid_indices() {
        #[derive(Debug)]
        struct Case {
            input: Tensor<i32>,
            indices: Tensor<i32>,
        }

        let cases = [
            // Non-scalar indices
            Case {
                input: Tensor::zeros(&[128, 10]),
                indices: Tensor::from_data(&[2, 2], vec![2, 5, 8, 130]),
            },
            // Scalar indices, with 1D and ND inputs
            Case {
                input: [1, 2, 3].into(),
                indices: Tensor::from(4),
            },
            Case {
                input: [[1, 2, 3]].into(),
                indices: Tensor::from(2),
            },
        ];

        cases.test_each(|case| {
            let pool = BufferPool::new();
            let result = gather(&pool, case.input.view(), 0, case.indices.view());
            assert_eq!(
                result.err(),
                Some(OpError::InvalidValue("Entry in `indices` is out of range"))
            );
        })
    }

    #[test]
    fn test_gather_elements() {
        #[derive(Debug)]
        struct Case {
            input: Tensor<i32>,
            indices: Tensor<i32>,
            expected: Tensor<i32>,
            axis: isize,
        }

        let cases = [
            // Example #1 from ONNX spec
            Case {
                input: [[1, 2], [3, 4]].into(),
                indices: [[0, 0], [1, 0]].into(),
                axis: 1,
                expected: [[1, 1], [4, 3]].into(),
            },
            // Example #2 from ONNX spec
            Case {
                input: [[1, 2, 3], [4, 5, 6], [7, 8, 9]].into(),
                indices: [[1, 2, 0], [2, 0, 0]].into(),
                axis: 0,
                expected: [[4, 8, 3], [7, 2, 3]].into(),
            },
            // Negative indices
            Case {
                input: [1, 2, 3].into(),
                indices: [-1, -1, -2, -2].into(),
                axis: 0,
                expected: [3, 3, 2, 2].into(),
            },
            // Input with > 4 dims.
            Case {
                input: Tensor::from([1, 2, 3, 4]).into_shape([1, 1, 1, 2, 2].as_slice()),
                indices: Tensor::from([1, 1, 0, 0]).into_shape([1, 1, 1, 2, 2].as_slice()),
                axis: 4,
                expected: Tensor::from([2, 2, 3, 3]).into_shape([1, 1, 1, 2, 2].as_slice()),
            },
            // Empty input and indices
            Case {
                input: [0; 0].into(),
                indices: [0; 0].into(),
                axis: 0,
                expected: [0; 0].into(),
            },
            // Empty indices
            Case {
                input: [1, 2, 3].into(),
                indices: [0; 0].into(),
                axis: 0,
                expected: [0; 0].into(),
            },
            // Case where `input` and `indices` have dims < axis that have different
            // strides.
            Case {
                input: [[1, 2, 3], [3, 4, 5]].into(),
                indices: [[0], [2]].into(),
                axis: 1,
                expected: [[1], [5]].into(),
            },
            // Case where `indices` has dims > axis which are smaller than the
            // corresponding dims in `input`.
            Case {
                input: [[1, 2, 3], [4, 5, 6], [7, 8, 9]].into(),
                indices: [[1], [2]].into(),
                axis: 0,
                expected: [[4], [7]].into(),
            },
        ];

        cases.test_each(|case| {
            let pool = BufferPool::new();
            let result =
                gather_elements(&pool, case.input.view(), case.indices.view(), case.axis).unwrap();
            assert_eq!(result, case.expected);
        });
    }

    #[test]
    fn test_gather_elements_invalid_inputs() {
        #[derive(Debug)]
        struct Case {
            input: Tensor<i32>,
            indices: Tensor<i32>,
            expected: OpError,
            axis: isize,
        }

        let cases = [
            Case {
                input: [[1, 2], [3, 4]].into(),
                indices: [[0, 0], [1, 0]].into(),
                axis: 2,
                expected: OpError::InvalidValue("Axis is invalid"),
            },
            Case {
                input: [[1, 2], [3, 4]].into(),
                indices: [[0, 0], [1, 3]].into(),
                axis: 1,
                expected: OpError::InvalidValue("Entry in `indices` is out of range"),
            },
            Case {
                input: [[1, 2], [3, 4]].into(),
                indices: [1, 2, 3].into(),
                axis: 1,
                expected: OpError::IncompatibleInputShapes("Input and indices must have same rank"),
            },
            Case {
                input: [[1, 2], [3, 4]].into(),
                indices: [[1, 2, 3], [4, 5, 6]].into(),
                axis: 0,
                expected: OpError::IncompatibleInputShapes(
                    "`indices` size must be <= input size in non-axis dimensions",
                ),
            },
        ];

        cases.test_each_value(|case| {
            let pool = BufferPool::new();
            let result = gather_elements(&pool, case.input.view(), case.indices.view(), case.axis);
            assert_eq!(result.err(), Some(case.expected));
        });
    }

    #[test]
    fn test_gather_nd() {
        #[derive(Debug)]
        struct Case {
            batch_dims: usize,
            data: Tensor<i32>,
            transpose: bool,
            indices: Tensor<i32>,
            expected: Result<Tensor<i32>, OpError>,
        }

        let cases = [
            // Examples from ONNX spec.
            Case {
                batch_dims: 0,
                data: [[0, 1], [2, 3]].into(),
                transpose: false,
                indices: [[0, 0], [1, 1]].into(),
                expected: Ok([0, 3].into()),
            },
            Case {
                batch_dims: 0,
                data: [[0, 1], [2, 3]].into(),
                transpose: false,
                indices: [[1], [0]].into(),
                expected: Ok([[2, 3], [0, 1]].into()),
            },
            Case {
                batch_dims: 0,
                data: [[[0, 1], [2, 3]], [[4, 5], [6, 7]]].into(),
                transpose: false,
                indices: [[0, 1], [1, 0]].into(),
                expected: Ok([[2, 3], [4, 5]].into()),
            },
            Case {
                batch_dims: 0,
                data: [[[0, 1], [2, 3]], [[4, 5], [6, 7]]].into(),
                transpose: false,
                indices: [[[0, 1]], [[1, 0]]].into(),
                expected: Ok([[[2, 3]], [[4, 5]]].into()),
            },
            Case {
                batch_dims: 1,
                data: [[[0, 1], [2, 3]], [[4, 5], [6, 7]]].into(),
                transpose: false,
                indices: [[1], [0]].into(),
                expected: Ok([[2, 3], [4, 5]].into()),
            },
            // Invalid indexes
            Case {
                batch_dims: 0,
                data: [[0, 1], [2, 3]].into(),
                transpose: false,
                indices: [[0, 0], [1, 2]].into(),
                expected: Err(OpError::InvalidValue("Invalid index")),
            },
            // Transposed input
            Case {
                batch_dims: 0,
                data: [[0, 1], [2, 3]].into(),
                transpose: true,
                indices: [[0, 1], [1, 0]].into(),
                expected: Ok([2, 1].into()),
            },
            Case {
                batch_dims: 0,
                data: [[0, 1], [2, 3]].into(),
                transpose: true,
                indices: [[0, 1], [1, 2]].into(),
                expected: Err(OpError::InvalidValue("Invalid index")),
            },
        ];

        cases.test_each(|case| {
            let pool = BufferPool::new();
            let result = gather_nd(
                &pool,
                if case.transpose {
                    case.data.transposed()
                } else {
                    case.data.view()
                },
                case.indices.view(),
                case.batch_dims,
            );
            assert_eq!(result, case.expected);
        })
    }

    #[test]
    fn test_scatter_elements() {
        #[derive(Debug)]
        struct Case {
            data: Tensor,
            indices: Tensor<i32>,
            updates: Tensor,
            axis: isize,
            expected: Result<Tensor, OpError>,
        }

        let cases = [
            // Example #1 from ONNX spec
            Case {
                data: Tensor::zeros(&[3, 3]),
                indices: Tensor::from([[1, 0, 2], [0, 2, 1]]),
                updates: Tensor::from([[1., 1.1, 1.2], [2., 2.1, 2.2]]),
                axis: 0,
                expected: Ok(Tensor::from([[2., 1.1, 0.], [1., 0., 2.2], [0., 2.1, 1.2]])),
            },
            // Example #2 from ONNX spec
            Case {
                data: Tensor::from([[1., 2., 3., 4., 5.]]),
                indices: Tensor::from([[1, 3]]),
                updates: Tensor::from([[1.1, 2.1]]),
                axis: 1,
                expected: Ok(Tensor::from([[1., 1.1, 3., 2.1, 5.]])),
            },
            // Invalid index
            Case {
                data: Tensor::from([1., 2., 3.]),
                indices: Tensor::from([4]),
                updates: Tensor::from([1.]),
                axis: 0,
                expected: Err(OpError::InvalidValue("Index is invalid")),
            },
            // Rank mismatch
            Case {
                data: Tensor::from([1., 2., 3.]),
                indices: Tensor::from([[4]]),
                updates: Tensor::from([[1.]]),
                axis: 0,
                expected: Err(OpError::InvalidValue(
                    "`data` and `indices` must have same rank",
                )),
            },
            // `indices` and `updates` shape mismatch
            Case {
                data: Tensor::from([1., 2., 3.]),
                indices: Tensor::from([4]),
                updates: Tensor::from([1., 2.]),
                axis: 0,
                expected: Err(OpError::InvalidValue(
                    "`indices` and `updates` must have same shape",
                )),
            },
        ];

        cases.test_each(|case| {
            let pool = BufferPool::new();
            let result = scatter_elements(
                &pool,
                case.data.view(),
                case.indices.view(),
                case.updates.view(),
                case.axis,
                None,
            );
            assert_eq!(result, case.expected);
        });
    }

    #[test]
    fn test_scatter_elements_reduction() {
        let pool = BufferPool::new();

        let data = Tensor::from([1, 2, 3, 4]);
        let indices = Tensor::from([1, 3]);
        let updates = Tensor::from([2, 2]);

        let scatter = |reduction: Option<ScatterReduction>| {
            scatter_elements(
                &pool,
                data.view(),
                indices.view(),
                updates.view(),
                0, /* axis */
                reduction,
            )
            .unwrap()
        };

        let result = scatter(Some(ScatterReduction::Add));
        assert_eq!(result, Tensor::from([1, 4, 3, 6]));

        let result = scatter(Some(ScatterReduction::Mul));
        assert_eq!(result, Tensor::from([1, 4, 3, 8]));

        let result = scatter(Some(ScatterReduction::Min));
        assert_eq!(result, Tensor::from([1, 2, 3, 2]));

        let result = scatter(Some(ScatterReduction::Max));
        assert_eq!(result, Tensor::from([1, 2, 3, 4]));
    }

    #[test]
    fn test_scatter_nd() {
        #[derive(Debug)]
        struct Case {
            data: Tensor<i32>,
            indices: Tensor<i32>,
            updates: Tensor<i32>,
            expected: Tensor<i32>,
        }

        let cases = [
            // Example 1 from ONNX spec.
            Case {
                data: [1, 2, 3, 4, 5, 6, 7, 8].into(),
                indices: Tensor::from_data(&[4, 1], vec![4, 3, 1, 7]),
                updates: [9, 10, 11, 12].into(),
                expected: [1, 11, 3, 10, 9, 6, 7, 12].into(),
            },
            // Example 2 from ONNX spec.
            Case {
                data: [
                    [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                    [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                    [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
                    [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
                ]
                .into(),
                indices: [[0], [2]].into(),
                updates: [
                    [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                    [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
                ]
                .into(),
                expected: [
                    [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                    [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                    [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
                    [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
                ]
                .into(),
            },
            // Test for issue when `updates` has a lower rank than `indices`.
            Case {
                data: [[1, 2], [3, 4]].into(),
                indices: [[0, 0], [0, 1]].into(),
                updates: [5, 6].into(),
                expected: [[5, 6], [3, 4]].into(),
            },
        ];

        cases.test_each(|case| {
            let pool = BufferPool::new();
            let result = scatter_nd(
                &pool,
                case.data.view(),
                case.indices.view(),
                case.updates.view(),
                None,
            )
            .unwrap();
            assert_eq!(result, case.expected);
        })
    }

    #[test]
    fn test_scatter_nd_reduce() {
        #[derive(Debug)]
        struct Case {
            data: Tensor<f32>,
            indices: Tensor<i32>,
            updates: Tensor<f32>,
            expected: Tensor<f32>,
            reduction: ScatterReduction,
        }

        let cases = [
            Case {
                data: Tensor::arange(1., 5., None),
                indices: Tensor::from_data(&[4, 1], vec![0, 1, 2, 3]),
                updates: [1., 2., 3., 4.].into(),
                expected: [2., 4., 6., 8.].into(),
                reduction: ScatterReduction::Add,
            },
            Case {
                data: Tensor::arange(1., 5., None),
                indices: Tensor::from_data(&[4, 1], vec![0, 1, 2, 3]),
                updates: [1., 2., 3., 4.].into(),
                expected: [1., 4., 9., 16.].into(),
                reduction: ScatterReduction::Mul,
            },
            Case {
                data: Tensor::arange(1., 5., None),
                indices: Tensor::from_data(&[4, 1], vec![0, 1, 2, 3]),
                updates: [1., -2., 3., -4.].into(),
                expected: [1., -2., 3., -4.].into(),
                reduction: ScatterReduction::Min,
            },
            Case {
                data: Tensor::arange(1., 5., None),
                indices: Tensor::from_data(&[4, 1], vec![0, 1, 2, 3]),
                updates: [1., -2., 3., -4.].into(),
                expected: [1., 2., 3., 4.].into(),
                reduction: ScatterReduction::Max,
            },
        ];

        cases.test_each(|case| {
            let pool = BufferPool::new();
            let result = scatter_nd(
                &pool,
                case.data.view(),
                case.indices.view(),
                case.updates.view(),
                Some(case.reduction),
            )
            .unwrap();
            assert_eq!(result, case.expected);
        })
    }

    #[test]
    fn test_scatter_nd_invalid() {
        #[derive(Debug)]
        struct Case {
            data: Tensor<f32>,
            indices: Tensor<i32>,
            updates: Tensor<f32>,
            expected: OpError,
        }

        let cases = [
            Case {
                data: (5.).into(),
                indices: [0].into(),
                updates: [0.].into(),
                expected: OpError::InvalidValue("`data` and `indices` must have rank >= 1"),
            },
            Case {
                data: Tensor::from([0.]),
                indices: Tensor::from(0),
                updates: [0.].into(),
                expected: OpError::InvalidValue("`data` and `indices` must have rank >= 1"),
            },
            Case {
                data: Tensor::arange(1., 5., None),
                indices: [[0], [1], [2], [3]].into(),
                updates: [[1., 2., 3., 4.]].into(),
                expected: OpError::InvalidValue("`updates` does not have expected rank"),
            },
            Case {
                data: Tensor::arange(1., 5., None),
                indices: [[0], [1], [2], [3]].into(),
                updates: [1., 2., 3., 4., 5.].into(),
                expected: OpError::InvalidValue("`updates` does not have expected shape"),
            },
            Case {
                data: Tensor::arange(1., 5., None),
                indices: [[0], [1], [2], [4]].into(),
                updates: [1., 2., 3., 4.].into(),
                expected: OpError::InvalidValue("invalid scatter index"),
            },
        ];

        cases.test_each(|case| {
            let pool = BufferPool::new();
            let result = scatter_nd(
                &pool,
                case.data.view(),
                case.indices.view(),
                case.updates.view(),
                None,
            );
            assert_eq!(result.as_ref(), Err(&case.expected));
        })
    }
}
