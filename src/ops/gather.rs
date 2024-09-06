use std::iter::zip;

use rten_tensor::prelude::*;
use rten_tensor::{to_slice_items, NdTensorView, SliceItem, Tensor, TensorView, TensorViewMut};
use smallvec::SmallVec;

use crate::ops::reduce::{cmp_nan_greater, cmp_nan_less};
use crate::ops::{
    resolve_axis, resolve_index, Input, InputList, IntoOpResult, OpError, Operator, OutputList,
};
use crate::tensor_pool::{AutoReturn, TensorPool};

const INVALID_INDEX_ERR: OpError = OpError::InvalidValue("Entry in `indices` is out of range");

/// Gather elements from `input` specified by `indices`.
///
/// See <https://onnx.ai/onnx/operators/onnx__Gather.html>. Per the ONNX spec this
/// is very similar to `numpy.take`. See
/// <https://numpy.org/doc/stable/reference/generated/numpy.take.html> for
/// additional explanation.
pub fn gather<T: Copy + Default>(
    pool: &TensorPool,
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
    if let (0, Some(index)) = (indices.ndim(), indices.item()) {
        let output = if input.ndim() == 1 {
            // Fast path for indexing a vector with a scalar. This is common
            // in subgraphs that process tensor shapes.
            let index = resolve_index(input.len(), *index as isize).ok_or(INVALID_INDEX_ERR)?;
            Tensor::full_in(pool, &[], input[[index]])
        } else {
            let mut slice_range = full_range(input.ndim());
            slice_range[axis] = SliceItem::Index(*index as isize);
            let slice = input
                .try_slice_dyn(slice_range.as_slice())
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
    let mut output = Tensor::zeros_in(pool, &out_shape);

    let mut in_range = full_range(input.ndim());
    let mut out_range = full_range(output.ndim());

    for (index_idx, index) in zip(indices.indices(), indices.iter()) {
        in_range[axis] = SliceItem::Index(*index as isize);
        for (i, index_val) in index_idx.into_iter().enumerate() {
            out_range[axis + i] = SliceItem::Index(index_val as isize);
        }
        let in_slice = input
            .try_slice_dyn(in_range.as_slice())
            .map_err(|_| INVALID_INDEX_ERR)?;
        let mut out_slice = output.slice_mut_dyn(out_range.as_slice());
        out_slice.copy_from(&in_slice);
    }

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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require(0)?;
        let indices = inputs.require_as::<i32>(1)?;
        match input {
            Input::Int32Tensor(input) => gather(pool, input, self.axis, indices).into_op_result(),
            Input::FloatTensor(input) => gather(pool, input, self.axis, indices).into_op_result(),
            _ => Err(OpError::UnsupportedType),
        }
    }
}

/// Optimized implementation of `gather_elements` for tensor with static rank.
/// Index iteration is much faster in this case.
fn gather_elements_4d<T: Copy + Default>(
    pool: &TensorPool,
    mut output: TensorViewMut<T>,
    input: NdTensorView<T, 4>,
    indices: NdTensorView<i32, 4>,
    axis: usize,
) -> Result<(), OpError> {
    assert!(axis < input.ndim());
    assert!(output.shape() == indices.shape());

    // This allows for faster iteration, and the tensor is likely already contiguous.
    let indices = indices.to_contiguous_in(pool).auto_return(pool);
    let indices = indices.view();

    // nb. We iterate over the underlying data slices for efficiency.
    let mut out_index_iter = output
        .data_mut()
        .unwrap()
        .iter_mut()
        .zip(indices.data().unwrap().iter());
    let mut indices_valid = true;

    let indices_shape = indices.shape();
    let axis_size = input.size(axis) as isize;

    // Use nested loops to iterate over indices in `indices` as this is faster
    // than `indices.indices()`.
    for i0 in 0..indices_shape[0] {
        for i1 in 0..indices_shape[1] {
            for i2 in 0..indices_shape[2] {
                for i3 in 0..indices_shape[3] {
                    let (out_el, index) = out_index_iter.next().unwrap();

                    // nb. If axis_val is < -axis_size, it will wrap around to a value
                    // that is still out of range.
                    let index = *index as isize;

                    let mut in_index = [i0, i1, i2, i3];
                    in_index[axis] = if index < 0 {
                        (index + axis_size) as usize
                    } else {
                        index as usize
                    };

                    let maybe_el = input.get(in_index).copied();
                    *out_el = maybe_el.unwrap_or_default();
                    indices_valid &= maybe_el.is_some();
                }
            }
        }
    }

    if !indices_valid {
        return Err(OpError::InvalidValue("Entry in `indices` is out of range"));
    }

    Ok(())
}

/// Expand a tensor to 4 dims by inserting `n` axes at the front.
fn unsqueeze_n<T>(mut view: TensorView<T>, n: usize) -> TensorView<T> {
    for _ in 0..n {
        view.insert_axis(0);
    }
    view
}

pub fn gather_elements<T: Copy + Default>(
    pool: &TensorPool,
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
    let mut output = Tensor::zeros_in(pool, indices.shape());

    // For the common case of tensors with <= 4 dims, expand input to 4 dims
    // and then use a fast path for static-rank tensors.
    const FAST_PATH_NDIM: usize = 4;
    if indices.ndim() <= FAST_PATH_NDIM {
        let pad = FAST_PATH_NDIM - input.ndim();
        let mut output = output.view_mut();
        for _ in 0..pad {
            output.insert_axis(0);
        }
        gather_elements_4d(
            pool,
            output.view_mut(),
            unsqueeze_n(input, pad).nd_view(),
            unsqueeze_n(indices, pad).nd_view(),
            axis + pad,
        )?;
    } else {
        let axis_size = input.size(axis) as isize;
        let mut indices_valid = true;
        for ((mut in_index, out_el), index) in
            output.indices().zip(output.iter_mut()).zip(indices.iter())
        {
            // nb. If axis_val is < -axis_size, it will wrap around to a value
            // that is still out of range.
            let index = *index as isize;
            let axis_val = if index < 0 { index + axis_size } else { index };
            in_index[axis] = axis_val as usize;

            let maybe_el = input.get(in_index).copied();
            *out_el = maybe_el.unwrap_or_default();
            indices_valid &= maybe_el.is_some();
        }
        if !indices_valid {
            return Err(OpError::InvalidValue("Entry in `indices` is out of range"));
        }
    }

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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require(0)?;
        let indices = inputs.require_as::<i32>(1)?;
        match input {
            Input::Int32Tensor(input) => {
                gather_elements(pool, input, indices, self.axis).into_op_result()
            }
            Input::FloatTensor(input) => {
                gather_elements(pool, input, indices, self.axis).into_op_result()
            }
            _ => Err(OpError::UnsupportedType),
        }
    }
}

pub fn gather_nd<T: Clone + Default>(
    pool: &TensorPool,
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
    let mut output = Tensor::<T>::zeros_in(pool, &out_shape);

    let output_non_batch_dims = output.ndim() - batch_dims;
    let input_non_batch_dims = input.ndim() - batch_dims;
    let indices_non_batch_dims = indices.ndim() - batch_dims;

    // This allows the loop below to rely on index tuples being contiguous.
    let indices = indices.to_contiguous_in(pool).auto_return(pool);

    for (mut output, (input, indices)) in output.inner_iter_dyn_mut(output_non_batch_dims).zip(
        input
            .inner_iter_dyn(input_non_batch_dims)
            .zip(indices.inner_iter_dyn(indices_non_batch_dims)),
    ) {
        // For performance, work with data slices rather than tensor views here.
        let out_slices = output.data_mut().unwrap().chunks_mut(out_slice_len);
        let idx_slices = indices.data().unwrap().chunks(idx_tuple_size);

        for (out_slice, idx) in out_slices.zip(idx_slices) {
            let slice_items = to_slice_items(idx);
            let in_slice = input
                .try_slice_dyn(slice_items.as_slice())
                .map_err(|_| OpError::InvalidValue("Invalid index"))?;

            for (out, x) in out_slice.iter_mut().zip(in_slice.iter()) {
                *out = x.clone();
            }
        }
    }

    Ok(output)
}

#[derive(Debug)]
pub struct GatherND {
    pub batch_dims: usize,
}

impl Operator for GatherND {
    fn name(&self) -> &str {
        "GatherND"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require(0)?;
        let indices = inputs.require_as::<i32>(1)?;
        match input {
            Input::Int32Tensor(input) => {
                gather_nd(pool, input, indices, self.batch_dims).into_op_result()
            }
            Input::FloatTensor(input) => {
                gather_nd(pool, input, indices, self.batch_dims).into_op_result()
            }
            _ => Err(OpError::UnsupportedType),
        }
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

fn scatter_reduce<T: Copy + PartialOrd + std::ops::Add<Output = T> + std::ops::Mul<Output = T>>(
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
    T: Copy + Default + PartialOrd + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
>(
    pool: &TensorPool,
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

    let mut output = data.to_tensor_in(pool);
    for (index, update) in zip(updates.indices(), updates.iter()) {
        let target_index: SmallVec<[usize; 5]> = index
            .iter()
            .enumerate()
            .filter_map(|(dim, idx)| {
                if dim == axis {
                    resolve_index(data.size(dim), indices[&index] as isize)
                } else {
                    Some(*idx)
                }
            })
            .collect();
        if target_index.len() < data.ndim() {
            return Err(OpError::InvalidValue("Index is invalid"));
        }

        let out_el = &mut output[target_index];
        *out_el = scatter_reduce(*out_el, *update, reduction);
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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let data = inputs.require(0)?;
        let indices = inputs.require_as::<i32>(1)?;
        let updates = inputs.require(2)?;

        match (data, updates) {
            (Input::Int32Tensor(data), Input::Int32Tensor(updates)) => {
                scatter_elements(pool, data, indices, updates, self.axis, self.reduction)
                    .into_op_result()
            }
            (Input::FloatTensor(data), Input::FloatTensor(updates)) => {
                scatter_elements(pool, data, indices, updates, self.axis, self.reduction)
                    .into_op_result()
            }
            _ => Err(OpError::UnsupportedType),
        }
    }
}

pub fn scatter_nd<
    T: Copy + Default + PartialOrd + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
>(
    pool: &TensorPool,
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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let data = inputs.require(0)?;
        let indices = inputs.require_as::<i32>(1)?;
        let updates = inputs.require(2)?;

        match (data, updates) {
            (Input::Int32Tensor(data), Input::Int32Tensor(updates)) => {
                scatter_nd(pool, data, indices, updates, self.reduction).into_op_result()
            }
            (Input::FloatTensor(data), Input::FloatTensor(updates)) => {
                scatter_nd(pool, data, indices, updates, self.reduction).into_op_result()
            }
            _ => Err(OpError::UnsupportedType),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::expect_equal;
    use rten_tensor::Tensor;

    use crate::ops::tests::new_pool;
    use crate::ops::{
        gather, gather_elements, gather_nd, scatter_elements, scatter_nd, OpError, ScatterReduction,
    };

    #[test]
    fn test_gather_scalar_index() {
        let pool = new_pool();

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
        let pool = new_pool();

        // Test case shrunk down from a small BERT model where `gather` is used
        // to lookup embeddings.
        let mut rng = XorShiftRng::new(1234);
        let input = Tensor::rand(&[128, 10], &mut rng);
        let indices = Tensor::from_data(&[2, 2], vec![2, 5, 8, 50]);
        let result = gather(&pool, input.view(), 0, indices.view()).unwrap();
        assert_eq!(result.shape(), &[2, 2, 10]);

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

        Ok(())
    }

    #[test]
    fn test_gather_invalid_axis() {
        let pool = new_pool();

        let mut rng = XorShiftRng::new(1234);
        let input = Tensor::rand(&[128, 10], &mut rng);
        let indices = Tensor::from_data(&[2, 2], vec![2, 5, 8, 50]);
        let result = gather(&pool, input.view(), 5, indices.view());
        assert_eq!(result.err(), Some(OpError::InvalidValue("Axis is invalid")));
    }

    #[test]
    fn test_gather_invalid_indices() {
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

        for Case { input, indices } in cases {
            let pool = new_pool();
            let result = gather(&pool, input.view(), 0, indices.view());
            assert_eq!(
                result.err(),
                Some(OpError::InvalidValue("Entry in `indices` is out of range"))
            );
        }
    }

    #[test]
    fn test_gather_elements() {
        let pool = new_pool();

        // Example #1 from ONNX spec
        let input = Tensor::from([[1, 2], [3, 4]]);
        let indices = Tensor::from([[0, 0], [1, 0]]);
        let axis = 1;
        let expected = Tensor::from([[1, 1], [4, 3]]);
        let result = gather_elements(&pool, input.view(), indices.view(), axis).unwrap();
        assert_eq!(result, expected);

        // Example #2 from ONNX spec
        let input = Tensor::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let indices = Tensor::from([[1, 2, 0], [2, 0, 0]]);
        let axis = 0;
        let expected = Tensor::from([[4, 8, 3], [7, 2, 3]]);
        let result = gather_elements(&pool, input.view(), indices.view(), axis).unwrap();
        assert_eq!(result, expected);

        // Negative indices
        let input = Tensor::from([1, 2, 3]);
        let indices = Tensor::from([-1, -1, -2, -2]);
        let axis = 0;
        let expected = Tensor::from([3, 3, 2, 2]);
        let result = gather_elements(&pool, input.view(), indices.view(), axis).unwrap();
        assert_eq!(result, expected);

        // Input with > 4 dims.
        let input = Tensor::from([1, 2, 3, 4]).into_shape([1, 1, 1, 2, 2].as_slice());
        let indices = Tensor::from([1, 1, 0, 0]).into_shape([1, 1, 1, 2, 2].as_slice());
        let axis = 4;
        let expected = Tensor::from([2, 2, 3, 3]).into_shape([1, 1, 1, 2, 2].as_slice());
        let result = gather_elements(&pool, input.view(), indices.view(), axis).unwrap();
        assert_eq!(result, expected);

        // Empty input and indices
        let input = Tensor::from([0; 0]);
        let indices = Tensor::from([0; 0]);
        let axis = 0;
        let expected = Tensor::from([0; 0]);
        let result = gather_elements(&pool, input.view(), indices.view(), axis).unwrap();
        assert_eq!(result, expected);

        // Empty indices
        let input: Tensor<i32> = Tensor::from([1, 2, 3]);
        let indices = Tensor::from([0; 0]);
        let axis = 0;
        let expected = Tensor::from([0; 0]);
        let result = gather_elements(&pool, input.view(), indices.view(), axis).unwrap();
        assert_eq!(result, expected);

        // Case where `input` and `indices` have dims < axis that have different
        // strides.
        let input = Tensor::from([[1, 2], [3, 4]]);
        let indices = Tensor::from([[0], [0]]);
        let axis = 1;
        let expected = Tensor::from([[1], [3]]);
        let result = gather_elements(&pool, input.view(), indices.view(), axis).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_gather_elements_invalid_inputs() {
        let pool = new_pool();

        let input = Tensor::from([[1, 2], [3, 4]]);
        let indices = Tensor::from([[0, 0], [1, 0]]);
        let result = gather_elements(&pool, input.view(), indices.view(), 2 /* axis */);
        assert_eq!(result.err(), Some(OpError::InvalidValue("Axis is invalid")));

        let indices = Tensor::from([[0, 0], [1, 3]]);
        let result = gather_elements(&pool, input.view(), indices.view(), 1 /* axis */);
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Entry in `indices` is out of range"))
        );

        let indices = Tensor::from([1, 2, 3]);
        let result = gather_elements(&pool, input.view(), indices.view(), 1 /* axis */);
        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes(
                "Input and indices must have same rank"
            ))
        );
    }

    #[test]
    fn test_gather_nd() {
        struct Case {
            batch_dims: usize,
            data: Tensor<i32>,
            indices: Tensor<i32>,
            expected: Result<Tensor<i32>, OpError>,
        }

        let cases = [
            // Examples from ONNX spec.
            Case {
                batch_dims: 0,
                data: [[0, 1], [2, 3]].into(),
                indices: [[0, 0], [1, 1]].into(),
                expected: Ok([0, 3].into()),
            },
            Case {
                batch_dims: 0,
                data: [[0, 1], [2, 3]].into(),
                indices: [[1], [0]].into(),
                expected: Ok([[2, 3], [0, 1]].into()),
            },
            Case {
                batch_dims: 0,
                data: [[[0, 1], [2, 3]], [[4, 5], [6, 7]]].into(),
                indices: [[0, 1], [1, 0]].into(),
                expected: Ok([[2, 3], [4, 5]].into()),
            },
            Case {
                batch_dims: 0,
                data: [[[0, 1], [2, 3]], [[4, 5], [6, 7]]].into(),
                indices: [[[0, 1]], [[1, 0]]].into(),
                expected: Ok([[[2, 3]], [[4, 5]]].into()),
            },
            Case {
                batch_dims: 1,
                data: [[[0, 1], [2, 3]], [[4, 5], [6, 7]]].into(),
                indices: [[1], [0]].into(),
                expected: Ok([[2, 3], [4, 5]].into()),
            },
        ];

        let pool = new_pool();
        for Case {
            batch_dims,
            data,
            indices,
            expected,
        } in cases
        {
            let result = gather_nd(&pool, data.view(), indices.view(), batch_dims);
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_scatter_elements() {
        let pool = new_pool();

        // Example #1 from ONNX spec
        let data = Tensor::zeros(&[3, 3]);
        let indices = Tensor::from([[1, 0, 2], [0, 2, 1]]);
        let updates = Tensor::from([[1., 1.1, 1.2], [2., 2.1, 2.2]]);
        let expected = Tensor::from([[2., 1.1, 0.], [1., 0., 2.2], [0., 2.1, 1.2]]);
        let result = scatter_elements(
            &pool,
            data.view(),
            indices.view(),
            updates.view(),
            0, /* axis */
            None,
        )
        .unwrap();
        assert_eq!(result, expected);

        // Example #2 from ONNX spec
        let data = Tensor::from([[1., 2., 3., 4., 5.]]);
        let indices = Tensor::from([[1, 3]]);
        let updates = Tensor::from([[1.1, 2.1]]);
        let expected = Tensor::from([[1., 1.1, 3., 2.1, 5.]]);
        let result = scatter_elements(
            &pool,
            data.view(),
            indices.view(),
            updates.view(),
            1, /* axis */
            None,
        )
        .unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_scatter_elements_reduction() {
        let pool = new_pool();

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
        let pool = new_pool();

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

        for Case {
            data,
            indices,
            updates,
            expected,
        } in cases
        {
            let result =
                scatter_nd(&pool, data.view(), indices.view(), updates.view(), None).unwrap();
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_scatter_nd_reduce() {
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

        let pool = new_pool();
        for Case {
            data,
            indices,
            updates,
            expected,
            reduction,
        } in cases
        {
            let result = scatter_nd(
                &pool,
                data.view(),
                indices.view(),
                updates.view(),
                Some(reduction),
            )
            .unwrap();
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_scatter_nd_invalid() {
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

        let pool = new_pool();
        for Case {
            data,
            indices,
            updates,
            expected,
        } in cases
        {
            let result = scatter_nd(&pool, data.view(), indices.view(), updates.view(), None);
            assert_eq!(result, Err(expected));
        }
    }
}
