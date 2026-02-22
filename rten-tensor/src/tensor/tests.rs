use std::borrow::Cow;
use std::cell::RefCell;
use std::sync::Arc;

use super::{AsView, NdTensor, NdTensorView, NdTensorViewMut, Tensor};
use crate::errors::{ExpandError, FromDataError};
use crate::layout::{DynLayout, MatrixLayout, MutLayout};
use crate::prelude::*;
use crate::rng::XorShiftRng;
use crate::storage::{Alloc, IntoStorage};
use crate::{ArcTensor, NdLayout, SliceItem, SliceRange, Storage};

struct FakeAlloc {
    count: RefCell<usize>,
}

impl FakeAlloc {
    fn new() -> FakeAlloc {
        FakeAlloc {
            count: RefCell::new(0),
        }
    }

    fn count(&self) -> usize {
        *self.count.borrow()
    }
}

impl Alloc for FakeAlloc {
    fn alloc<T>(&self, capacity: usize) -> Vec<T> {
        *self.count.borrow_mut() += 1;
        Vec::with_capacity(capacity)
    }
}

#[test]
fn test_append() {
    let mut tensor = NdTensor::<i32, 2>::with_capacity([3, 3], 1);
    assert_eq!(tensor.shape(), [3, 0]);

    assert_eq!(
        tensor.append(1, &NdTensor::from([[1, 2, 3]])),
        Err(ExpandError::ShapeMismatch)
    );

    tensor
        .append(1, &NdTensor::from([[1, 2], [3, 4], [5, 6]]))
        .unwrap();
    assert_eq!(tensor.shape(), [3, 2]);

    tensor.append(1, &NdTensor::from([[7], [8], [9]])).unwrap();
    assert_eq!(tensor.shape(), [3, 3]);

    assert_eq!(tensor, NdTensor::from([[1, 2, 7], [3, 4, 8], [5, 6, 9],]));

    assert_eq!(
        tensor.append(1, &NdTensor::from([[10], [11], [12]])),
        Err(ExpandError::InsufficientCapacity)
    );

    // Append to an empty tensor
    let mut empty = NdTensor::<i32, 2>::zeros([0, 3]);
    empty.append(1, &NdTensor::<i32, 2>::zeros([0, 2])).unwrap();
    assert_eq!(empty.shape(), [0, 5]);
}

#[test]
fn test_concat() {
    // Concat along dim 0 (rows)
    let a = NdTensor::from([[1, 2], [3, 4]]);
    let b = NdTensor::from([[5, 6]]);
    let result = NdTensor::concat(0, &[a, b]).unwrap();
    assert_eq!(result, NdTensor::from([[1, 2], [3, 4], [5, 6]]));

    // Concat along dim 1 (columns)
    let a = NdTensor::from([[1, 2], [3, 4]]);
    let b = NdTensor::from([[5], [6]]);
    let result = NdTensor::concat(1, &[a, b]).unwrap();
    assert_eq!(result, NdTensor::from([[1, 2, 5], [3, 4, 6]]));

    // Concat three tensors
    let a = NdTensor::from([[1]]);
    let b = NdTensor::from([[2]]);
    let c = NdTensor::from([[3]]);
    let result = NdTensor::concat(1, &[a, b, c]).unwrap();
    assert_eq!(result, NdTensor::from([[1, 2, 3]]));

    // Single tensor
    let a = NdTensor::from([[1, 2], [3, 4]]);
    let result = NdTensor::concat(0, &[a]).unwrap();
    assert_eq!(result, NdTensor::from([[1, 2], [3, 4]]));

    // Empty slice
    let result = NdTensor::<i32, 2>::concat(0, &[] as &[NdTensor<i32, 2>]);
    assert_eq!(result, Err(ExpandError::ShapeMismatch));

    // Shape mismatch on non-concat dimension
    let a = NdTensor::from([[1, 2]]);
    let b = NdTensor::from([[3, 4, 5]]);
    let result = NdTensor::concat(0, &[a, b]);
    assert_eq!(result, Err(ExpandError::ShapeMismatch));

    // Dynamic-rank concat
    let a = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
    let b = Tensor::from_data(&[1, 2], vec![5, 6]);
    let result = Tensor::concat(0, &[a, b]).unwrap();
    assert_eq!(result.shape(), &[3, 2]);
    assert_eq!(result.to_vec(), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_apply() {
    let data = vec![1., 2., 3., 4.];

    // Contiguous tensor.
    let mut tensor = NdTensor::from_data([2, 2], data);
    tensor.apply(|x| *x * 2.);
    assert_eq!(tensor.to_vec(), &[2., 4., 6., 8.]);

    // Non-contiguous tensor
    tensor.transpose();
    tensor.apply(|x| *x / 2.);
    assert_eq!(tensor.to_vec(), &[1., 3., 2., 4.]);
}

#[test]
fn test_arange() {
    let x = Tensor::arange(2, 6, None);
    let y = NdTensor::arange(2, 6, None);
    assert_eq!(x.data(), Some([2, 3, 4, 5].as_slice()));
    assert_eq!(y.data(), Some([2, 3, 4, 5].as_slice()));
}

#[test]
fn test_arc_tensor() {
    let data: Vec<_> = (0..5i32).collect();
    let tensor_a = ArcTensor::from_data(&[data.len()], Arc::new(data));
    let tensor_b = tensor_a.clone();
    assert_eq!(tensor_a, tensor_b);
    assert_eq!(tensor_a, NdTensorView::from_data([5], &[0, 1, 2, 3, 4]));

    // Verify that cloned tensor shares the data.
    assert_eq!(
        tensor_a.data().unwrap().as_ptr(),
        tensor_b.data().unwrap().as_ptr()
    );
}

#[test]
fn test_as_cow_into_cow() {
    for copy in [true, false] {
        let x = Tensor::arange(0, 4, None).into_shape([2, 2]);
        let cow_x = if copy { x.into_cow() } else { x.as_cow() };
        assert_eq!(cow_x.shape(), [2, 2]);
        assert_eq!(cow_x.data().unwrap(), &[0, 1, 2, 3]);
    }
}

#[test]
fn test_as_dyn() {
    let data = vec![1., 2., 3., 4.];
    let tensor = NdTensor::from_data([2, 2], data);
    let dyn_view = tensor.as_dyn();
    assert_eq!(dyn_view.shape(), tensor.shape().as_ref());
    assert_eq!(dyn_view.to_vec(), tensor.to_vec());
}

#[test]
fn test_as_dyn_mut() {
    let data = vec![1., 2., 3., 4.];
    let mut tensor = NdTensor::from_data([2, 2], data);
    let mut dyn_view = tensor.as_dyn_mut();

    dyn_view[[0, 0]] = 9.;

    assert_eq!(tensor[[0, 0]], 9.);
}

#[test]
fn test_assign_array() {
    let mut tensor = NdTensor::zeros([2, 2]);
    let mut transposed = tensor.view_mut();

    transposed.permute([1, 0]);
    transposed.slice_mut(0).assign_array([1, 2]);
    transposed.slice_mut(1).assign_array([3, 4]);

    assert_eq!(tensor.iter().copied().collect::<Vec<_>>(), [1, 3, 2, 4]);
}

#[test]
fn test_axis_chunks() {
    let tensor = NdTensor::arange(0, 8, None).into_shape([4, 2]);
    let mut row_chunks = tensor.axis_chunks(0, 2);

    let chunk = row_chunks.next().unwrap();
    assert_eq!(chunk.shape(), [2, 2]);
    assert_eq!(chunk.to_vec(), &[0, 1, 2, 3]);

    let chunk = row_chunks.next().unwrap();
    assert_eq!(chunk.shape(), [2, 2]);
    assert_eq!(chunk.to_vec(), &[4, 5, 6, 7]);

    assert!(row_chunks.next().is_none());
}

#[test]
fn test_axis_chunks_mut() {
    let mut tensor = NdTensor::arange(1, 9, None).into_shape([4, 2]);
    let mut row_chunks = tensor.axis_chunks_mut(0, 2);

    let mut chunk = row_chunks.next().unwrap();
    chunk.apply(|x| x * 2);

    let mut chunk = row_chunks.next().unwrap();
    chunk.apply(|x| x * -2);

    assert!(row_chunks.next().is_none());
    assert_eq!(tensor.to_vec(), [2, 4, 6, 8, -10, -12, -14, -16]);
}

#[test]
fn test_axis_iter() {
    let tensor = NdTensor::arange(0, 4, None).into_shape([2, 2]);
    let mut rows = tensor.axis_iter(0);

    let row = rows.next().unwrap();
    assert_eq!(row.shape(), [2]);
    assert_eq!(row.to_vec(), &[0, 1]);

    let row = rows.next().unwrap();
    assert_eq!(row.shape(), [2]);
    assert_eq!(row.to_vec(), &[2, 3]);

    assert!(rows.next().is_none());
}

#[test]
fn test_axis_iter_mut() {
    let mut tensor = NdTensor::arange(1, 5, None).into_shape([2, 2]);
    let mut rows = tensor.axis_iter_mut(0);

    let mut row = rows.next().unwrap();
    row.apply(|x| x * 2);

    let mut row = rows.next().unwrap();
    row.apply(|x| x * -2);

    assert!(rows.next().is_none());
    assert_eq!(tensor.to_vec(), [2, 4, -6, -8]);
}

#[test]
fn test_broadcast() {
    let data = vec![1., 2., 3., 4.];
    let dest_shape = [3, 1, 2, 2];
    let expected_data: Vec<_> = data.iter().copied().cycle().take(data.len() * 3).collect();
    let ndtensor = NdTensor::from_data([2, 2], data);

    // Broadcast static -> static.
    let view = ndtensor.broadcast(dest_shape);
    assert_eq!(view.shape(), dest_shape);
    assert_eq!(view.to_vec(), expected_data);

    // Broadcast static -> dynamic.
    let view = ndtensor.broadcast(dest_shape.as_slice());
    assert_eq!(view.shape(), dest_shape);
    assert_eq!(view.to_vec(), expected_data);

    // Broadcast dynamic -> static.
    let tensor = ndtensor.as_dyn();
    let view = tensor.broadcast(dest_shape);
    assert_eq!(view.shape(), dest_shape);
    assert_eq!(view.to_vec(), expected_data);

    // Broadcast dynamic -> dynamic.
    let view = tensor.broadcast(dest_shape.as_slice());
    assert_eq!(view.shape(), dest_shape);
    assert_eq!(view.to_vec(), expected_data);
}

#[test]
fn test_clip_dim() {
    let mut tensor = NdTensor::arange(0, 9, None).into_shape([3, 3]);
    tensor.clip_dim(0, 0..3); // No-op
    assert_eq!(tensor.shape(), [3, 3]);

    tensor.clip_dim(0, 1..2); // Remove first and last rows
    assert_eq!(tensor.shape(), [1, 3]);
    assert_eq!(tensor.data(), Some([3, 4, 5].as_slice()));

    // Clip empty tensor
    let mut tensor = NdTensor::<f32, 2>::zeros([0, 10]);
    tensor.clip_dim(1, 2..5);
    assert_eq!(tensor.shape(), [0, 3]);
}

#[test]
fn test_clone() {
    let data = vec![1., 2., 3., 4.];
    let tensor = NdTensor::from_data([2, 2], data);
    let cloned = tensor.clone();
    assert_eq!(tensor.shape(), cloned.shape());
    assert_eq!(tensor.to_vec(), cloned.to_vec());
}

#[test]
fn test_copy_view() {
    let data = &[1., 2., 3., 4.];
    let view = NdTensorView::from_data([2, 2], data);

    // Verify that views are copyable, if their layout is.
    let view2 = view;

    assert_eq!(view.shape(), view2.shape());
}

#[test]
fn test_copy_from() {
    let mut dest = Tensor::zeros(&[2, 2]);
    let src = Tensor::from_data(&[2, 2], vec![1., 2., 3., 4.]);
    dest.copy_from(&src);
    assert_eq!(dest.to_vec(), &[1., 2., 3., 4.]);
}

#[test]
fn test_copy_into_slice() {
    let src = NdTensor::from([[1, 2], [3, 4], [5, 6]]);
    let mut buf = Vec::with_capacity(src.len());
    let buf_uninit = &mut buf.spare_capacity_mut()[..src.len()];

    // Contiguous case.
    let elts = src.copy_into_slice(buf_uninit);
    assert_eq!(elts, &[1, 2, 3, 4, 5, 6]);

    // Non-contiguous case.
    let transposed_elts = src.transposed().copy_into_slice(buf_uninit);
    assert_eq!(transposed_elts, &[1, 3, 5, 2, 4, 6]);
}

#[test]
fn test_data() {
    let data = &[1., 2., 3., 4., 5., 6.];
    let tensor = NdTensorView::from_data([2, 3], data);
    assert_eq!(tensor.data(), Some(data.as_slice()));

    let permuted = tensor.permuted([1, 0]);
    assert_eq!(permuted.shape(), [3, 2]);
    assert_eq!(permuted.data(), None);
}

#[test]
fn test_data_mut() {
    let mut data = vec![1., 2., 3., 4., 5., 6.];
    let mut tensor = NdTensor::from_data([2, 3], data.clone());
    assert_eq!(tensor.data_mut(), Some(data.as_mut_slice()));

    let mut permuted = tensor.permuted_mut([1, 0]);
    assert_eq!(permuted.shape(), [3, 2]);
    assert_eq!(permuted.data_mut(), None);
}

// See https://github.com/robertknight/rten/pull/861
#[test]
fn test_data_truncates_slice() {
    // Manually create a contiguous mutable tensor view where the storage length
    // is greater than the minimum required by the layout.
    let mut data = [0, 1, 2, 3, 4];
    let layout = NdLayout::from_shape([3]);
    let mut tensor =
        NdTensorViewMut::from_storage_and_layout(data.as_mut_slice().into_storage(), layout);

    // Check that slice-returning methods of the tensor view only return the
    // part of the storage that matches the layout.
    assert_eq!(tensor.data(), Some([0, 1, 2].as_slice()));
    assert_eq!(tensor.data_mut(), Some([0, 1, 2].as_mut_slice()));
    assert_eq!(tensor.into_slice_mut(), Some([0, 1, 2].as_mut_slice()));
}

#[test]
fn test_fill() {
    let data = vec![1., 2., 3., 4.];
    let mut tensor = NdTensor::from_data([2, 2], data);
    tensor.fill(9.);
    assert_eq!(tensor.to_vec(), &[9., 9., 9., 9.]);
}

#[test]
fn test_from_fn() {
    // Static rank
    let x = NdTensor::from_fn([], |_| 5);
    assert_eq!(x.data(), Some([5].as_slice()));

    let x = NdTensor::from_fn([5], |i| i[0]);
    assert_eq!(x.data(), Some([0, 1, 2, 3, 4].as_slice()));

    let x = NdTensor::from_fn([2, 2], |[y, x]| y * 10 + x);
    assert_eq!(x.data(), Some([0, 1, 10, 11].as_slice()));

    // Dynamic rank
    let x = Tensor::from_fn(&[], |_| 6);
    assert_eq!(x.data(), Some([6].as_slice()));

    let x = Tensor::from_fn(&[2, 2], |index| index[0] * 10 + index[1]);
    assert_eq!(x.data(), Some([0, 1, 10, 11].as_slice()));
}

#[test]
fn test_from_nested_array() {
    // Scalar
    let x = NdTensor::from(5);
    assert!(x.shape().is_empty());
    assert_eq!(x.data(), Some([5].as_slice()));

    // 1D
    let x = NdTensor::from([1, 2, 3]);
    assert_eq!(x.shape(), [3]);
    assert_eq!(x.data(), Some([1, 2, 3].as_slice()));

    // 2D
    let x = NdTensor::from([[1, 2], [3, 4]]);
    assert_eq!(x.shape(), [2, 2]);
    assert_eq!(x.data(), Some([1, 2, 3, 4].as_slice()));

    // 3D
    let x = NdTensor::from([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    assert_eq!(x.shape(), [2, 2, 2]);
    assert_eq!(x.data(), Some([1, 2, 3, 4, 5, 6, 7, 8].as_slice()));

    // Float
    let x = NdTensor::from([1., 2., 3.]);
    assert_eq!(x.shape(), [3]);
    assert_eq!(x.data(), Some([1., 2., 3.].as_slice()));

    // Bool
    let x = NdTensor::from([true, false]);
    assert_eq!(x.shape(), [2]);
    assert_eq!(x.data(), Some([true, false].as_slice()));
}

#[test]
fn test_from_simple_fn() {
    let mut next_val = 0;
    let mut gen_int = || {
        let curr = next_val;
        next_val += 1;
        curr
    };

    // Static rank
    let x = NdTensor::from_simple_fn([2, 2], &mut gen_int);
    assert_eq!(x.data(), Some([0, 1, 2, 3].as_slice()));

    let x = NdTensor::from_simple_fn([], &mut gen_int);
    assert_eq!(x.data(), Some([4].as_slice()));

    // Dynamic rank
    let x = Tensor::from_simple_fn(&[2, 2], gen_int);
    assert_eq!(x.data(), Some([5, 6, 7, 8].as_slice()));
}

#[test]
fn test_from_vec_or_slice() {
    let x = NdTensor::from(vec![1, 2, 3, 4]);
    assert_eq!(x.shape(), [4]);
    assert_eq!(x.data(), Some([1, 2, 3, 4].as_slice()));

    let x = NdTensorView::from(&[1, 2, 3]);
    assert_eq!(x.shape(), [3]);
    assert_eq!(x.data(), Some([1, 2, 3].as_slice()));
}

#[test]
fn test_dyn_tensor_from_nd_tensor() {
    let x = NdTensor::from_data([2, 2], vec![1, 2, 3, 4]);
    let y: Tensor<i32> = x.into();
    assert_eq!(y.data(), Some([1, 2, 3, 4].as_slice()));
    assert_eq!(y.shape(), &[2, 2]);
}

#[test]
fn test_nd_tensor_from_dyn_tensor() {
    let x = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
    let y: NdTensor<i32, 2> = x.try_into().unwrap();
    assert_eq!(y.data(), Some([1, 2, 3, 4].as_slice()));
    assert_eq!(y.shape(), [2, 2]);

    let x = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
    let y: Result<NdTensor<i32, 3>, _> = x.try_into();
    assert!(y.is_err());
}

#[test]
fn test_from_data() {
    let x = NdTensor::from_data([1, 2, 2], vec![1, 2, 3, 4]);
    assert_eq!(x.shape(), [1, 2, 2]);
    assert_eq!(x.strides(), [4, 2, 1]);
    assert_eq!(x.to_vec(), [1, 2, 3, 4]);
}

#[test]
#[should_panic(expected = "data length 4 does not match shape [2, 2, 2]")]
fn test_from_data_shape_mismatch() {
    NdTensor::from_data([2, 2, 2], vec![1, 2, 3, 4]);
}

#[test]
fn test_from_data_with_strides() {
    let x = NdTensor::from_data_with_strides([2, 2, 1], vec![1, 2, 3, 4], [1, 2, 4]).unwrap();
    assert_eq!(x.shape(), [2, 2, 1]);
    assert_eq!(x.strides(), [1, 2, 4]);
    assert_eq!(x.to_vec(), [1, 3, 2, 4]);

    // Invalid (wrong storage length)
    let x = NdTensor::from_data_with_strides([2, 2, 2], vec![1, 2, 3, 4], [1, 2, 4]);
    assert_eq!(x, Err(FromDataError::StorageTooShort));

    // Invalid strides (overlapping)
    let x = NdTensor::from_data_with_strides([2, 2], vec![1, 2], [0, 1]);
    assert_eq!(x, Err(FromDataError::MayOverlap));
}

#[test]
fn test_from_slice_with_strides() {
    // The strides here are overlapping, but `from_slice_with_strides`
    // allows this since it is a read-only view.
    let data = [1, 2];
    let x = NdTensorView::from_slice_with_strides([2, 2], &data, [0, 1]).unwrap();
    assert_eq!(x.to_vec(), [1, 2, 1, 2]);
}

#[test]
fn test_from_storage_and_layout() {
    let layout = DynLayout::from_shape(&[3, 3]);
    let storage = Vec::from([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    let tensor = Tensor::from_storage_and_layout(storage, layout);
    assert_eq!(tensor.shape(), &[3, 3]);
    assert_eq!(tensor.data(), Some([0, 1, 2, 3, 4, 5, 6, 7, 8].as_slice()));
}

#[test]
fn test_from_iter() {
    let x: Tensor = [1., 2., 3., 4.].into_iter().collect();
    assert_eq!(x.shape(), &[4]);
    assert_eq!(x.data(), Some([1., 2., 3., 4.].as_slice()));

    let y: NdTensor<_, 1> = [1., 2., 3., 4.].into_iter().collect();
    assert_eq!(y.shape(), [4]);
    assert_eq!(y.data(), Some([1., 2., 3., 4.].as_slice()));
}

#[test]
fn test_from_scalar() {
    let x = Tensor::from_scalar(5.);
    let y = NdTensor::from_scalar(6.);
    assert_eq!(x.item(), Some(&5.));
    assert_eq!(y.item(), Some(&6.));
}

#[test]
fn test_from_vec() {
    let x = NdTensor::from_vec(vec![1, 2, 3, 4]);
    assert_eq!(x.shape(), [4]);
    assert_eq!(x.data(), Some([1, 2, 3, 4].as_slice()));
}

#[test]
fn test_full() {
    let tensor = NdTensor::full([2, 2], 2.);
    assert_eq!(tensor.shape(), [2, 2]);
    assert_eq!(tensor.data(), Some([2., 2., 2., 2.].as_slice()));
}

#[test]
fn test_full_in() {
    let pool = FakeAlloc::new();
    NdTensor::<_, 2>::full_in(&pool, [2, 2], 5.);
    assert_eq!(pool.count(), 1);
}

#[test]
fn test_get() {
    // NdLayout
    let data = vec![1., 2., 3., 4.];
    let tensor: NdTensor<f32, 2> = NdTensor::from_data([2, 2], data);

    // Impl for tensors
    assert_eq!(tensor.get([1, 1]), Some(&4.));
    assert_eq!(tensor.get([2, 1]), None);

    // Impl for views
    assert_eq!(tensor.view().get([1, 1]), Some(&4.));
    assert_eq!(tensor.view().get([2, 1]), None);

    // DynLayout
    let data = vec![1., 2., 3., 4.];
    let tensor: Tensor<f32> = Tensor::from_data(&[2, 2], data);

    // Impl for tensors
    assert_eq!(tensor.get([1, 1]), Some(&4.));
    assert_eq!(tensor.get([2, 1]), None); // Invalid index
    assert_eq!(tensor.get([1, 2, 3]), None); // Incorrect dim count

    // Impl for views
    assert_eq!(tensor.view().get([1, 1]), Some(&4.));
    assert_eq!(tensor.view().get([2, 1]), None); // Invalid index
    assert_eq!(tensor.view().get([1, 2, 3]), None); // Incorrect dim count
}

#[test]
fn test_get_array() {
    let tensor = NdTensor::arange(1, 17, None).into_shape([4, 2, 2]);

    // First dim, zero base.
    let values: [i32; 4] = tensor.get_array([0, 0, 0], 0);
    assert_eq!(values, [1, 5, 9, 13]);

    // First dim, different base.
    let values: [i32; 4] = tensor.get_array([0, 1, 1], 0);
    assert_eq!(values, [4, 8, 12, 16]);

    // Last dim, zero base.
    let values: [i32; 2] = tensor.get_array([0, 0, 0], 2);
    assert_eq!(values, [1, 2]);
}

#[test]
fn test_get_mut() {
    let data = vec![1., 2., 3., 4.];
    let mut tensor: NdTensor<f32, 2> = NdTensor::from_data([2, 2], data);
    if let Some(elem) = tensor.get_mut([1, 1]) {
        *elem = 9.;
    }
    assert_eq!(tensor[[1, 1]], 9.);
    assert_eq!(tensor.get_mut([2, 1]), None);
}

#[test]
fn test_get_unchecked() {
    let ndtensor = NdTensor::arange(1, 5, None);
    for i in 0..ndtensor.size(0) {
        // Called on a tensor.
        assert_eq!(unsafe { ndtensor.get_unchecked([i]) }, &ndtensor[[i]]);

        // Called on a view.
        assert_eq!(
            unsafe { ndtensor.view().get_unchecked([i]) },
            &ndtensor[[i]]
        );
    }

    let tensor = Tensor::arange(1, 5, None);
    for i in 0..tensor.size(0) {
        // Called on a tensor.
        assert_eq!(unsafe { tensor.get_unchecked([i]) }, &ndtensor[[i]]);
        // Called on a view.
        assert_eq!(unsafe { tensor.view().get_unchecked([i]) }, &ndtensor[[i]]);
    }
}

#[test]
fn test_get_unchecked_mut() {
    let mut ndtensor = NdTensor::arange(1, 5, None);
    for i in 0..ndtensor.size(0) {
        unsafe { *ndtensor.get_unchecked_mut([i]) += 1 }
    }
    assert_eq!(ndtensor.to_vec(), &[2, 3, 4, 5]);

    let mut tensor = Tensor::arange(1, 5, None);
    for i in 0..tensor.size(0) {
        unsafe { *tensor.get_unchecked_mut([i]) += 1 }
    }
    assert_eq!(tensor.to_vec(), &[2, 3, 4, 5]);
}

#[test]
fn test_has_capacity() {
    let tensor = NdTensor::<f32, 3>::with_capacity([2, 3, 4], 1);
    assert_eq!(tensor.shape(), [2, 0, 4]);
    for i in 0..=3 {
        assert!(tensor.has_capacity(1, i));
    }
    assert!(!tensor.has_capacity(1, 4));
}

#[test]
fn test_index_and_index_mut() {
    // NdLayout
    let data = vec![1., 2., 3., 4.];
    let mut tensor: NdTensor<f32, 2> = NdTensor::from_data([2, 2], data);
    assert_eq!(tensor[[1, 1]], 4.);
    tensor[[1, 1]] = 9.;
    assert_eq!(tensor[[1, 1]], 9.);

    // DynLayout
    let data = vec![1., 2., 3., 4.];
    let mut tensor: Tensor<f32> = Tensor::from_data(&[2, 2], data);
    assert_eq!(tensor[[1, 1]], 4.);
    tensor[&[1, 1]] = 9.;
    assert_eq!(tensor[[1, 1]], 9.);
}

#[test]
fn test_index_axis() {
    let mut tensor = NdTensor::arange(0, 8, None).into_shape([4, 2]);
    let slice = tensor.index_axis(0, 2);
    assert_eq!(slice.shape(), [2]);
    assert_eq!(slice.data().unwrap(), [4, 5]);

    let mut slice = tensor.index_axis_mut(0, 3);
    assert_eq!(slice.shape(), [2]);
    assert_eq!(slice.data_mut().unwrap(), [6, 7]);
}

#[test]
fn test_init_from() {
    // Contiguous case
    let src = NdTensor::arange(0, 4, None).into_shape([2, 2]);
    let dest = NdTensor::uninit([2, 2]);
    let dest = dest.init_from(&src);
    assert_eq!(dest.to_vec(), &[0, 1, 2, 3]);

    // Non-contigous
    let dest = NdTensor::uninit([2, 2]);
    let dest = dest.init_from(&src.transposed());
    assert_eq!(dest.to_vec(), &[0, 2, 1, 3]);
}

#[test]
#[should_panic(expected = "shape mismatch")]
fn test_init_from_shape_mismatch() {
    let src = NdTensor::arange(0, 4, None).into_shape([2, 2]);
    let dest = NdTensor::uninit([2, 3]);
    let dest = dest.init_from(&src);
    assert_eq!(dest.to_vec(), &[0, 1, 2, 3]);
}

#[test]
fn test_into_arc() {
    let tensor = NdTensor::from([2., 3.]);
    let arc_tensor = tensor.into_arc();
    assert_eq!(arc_tensor.data().unwrap(), [2., 3.]);
}

#[test]
fn test_into_data() {
    let tensor = NdTensor::from([2., 3.]);
    assert_eq!(tensor.into_data(), vec![2., 3.]);

    let mut tensor = NdTensor::from_data([2, 2], vec![1., 2., 3., 4.]);
    tensor.transpose();
    assert_eq!(tensor.into_data(), vec![1., 3., 2., 4.]);
}

#[test]
fn test_into_data_truncates_vec() {
    // Manually create a tensor where the storage length is greater than the
    // minimum required by the layout.
    let vec = vec![0, 1, 2, 3, 4];
    let layout = NdLayout::from_shape([3]);
    let tensor = NdTensor::from_storage_and_layout(vec.into_storage(), layout);

    // Extracting the vec should truncate it to match the layout.
    let data_vec = tensor.into_data();
    assert_eq!(data_vec, [0, 1, 2]);
}

#[test]
fn test_into_slice_mut() {
    let mut tensor = NdTensor::from([[1, 2], [3, 4]]);
    let contiguous = tensor.view_mut();
    assert_eq!(
        contiguous.into_slice_mut(),
        Some([1, 2, 3, 4].as_mut_slice())
    );

    let non_contiguous = tensor.slice_mut((.., 0));
    assert_eq!(non_contiguous.into_slice_mut(), None);
}

#[test]
fn test_into_non_contiguous_data() {
    let mut tensor = NdTensor::from_data([2, 2], vec![1., 2., 3., 4.]);
    tensor.transpose();
    assert_eq!(tensor.into_non_contiguous_data(), vec![1., 2., 3., 4.]);
}

#[test]
fn test_cow_into_data_truncates_vec() {
    // Manually create a tensor where the storage length is greater than the
    // minimum required by the layout.
    let vec = vec![0, 1, 2, 3, 4];
    let layout = NdLayout::from_shape([3]);
    let tensor = NdTensor::from_storage_and_layout(vec.into_storage(), layout).into_cow();

    // Extracting the vec should truncate it to match the layout.
    let data_vec = tensor.into_non_contiguous_data().unwrap();
    assert_eq!(data_vec, [0, 1, 2]);
}

#[test]
fn test_into_dyn() {
    let tensor = NdTensor::from_data([2, 2], vec![1., 2., 3., 4.]);
    let dyn_tensor = tensor.into_dyn();
    assert_eq!(dyn_tensor.shape(), &[2, 2]);
    assert_eq!(dyn_tensor.data(), Some([1., 2., 3., 4.].as_slice()));
}

#[test]
fn test_into_shape() {
    // Contiguous tensor.
    let tensor = NdTensor::from_data([2, 2], vec![1., 2., 3., 4.]);
    let reshaped = tensor.into_shape([4]);
    assert_eq!(reshaped.shape(), [4]);
    assert_eq!(reshaped.data(), Some([1., 2., 3., 4.].as_slice()));

    // Non-contiguous tensor.
    let mut tensor = NdTensor::from_data([2, 2], vec![1., 2., 3., 4.]);
    tensor.transpose();
    let reshaped = tensor.into_shape([4]);
    assert_eq!(reshaped.shape(), [4]);
    assert_eq!(reshaped.data(), Some([1., 3., 2., 4.].as_slice()));
}

#[test]
#[should_panic(expected = "element count mismatch reshaping [16] to [2, 2]")]
fn test_into_shape_invalid() {
    NdTensor::arange(0, 16, None).into_shape([2, 2]);
}

#[test]
fn test_inner_iter() {
    let tensor = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
    let mut rows = tensor.inner_iter::<1>();

    let row = rows.next().unwrap();
    assert_eq!(row.shape(), [2]);
    assert_eq!(row.to_vec(), &[1, 2]);

    let row = rows.next().unwrap();
    assert_eq!(row.shape(), [2]);
    assert_eq!(row.to_vec(), &[3, 4]);

    assert_eq!(rows.next(), None);
}

#[test]
fn test_inner_iter_dyn() {
    let tensor = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
    let mut rows = tensor.inner_iter_dyn(1);

    let row = rows.next().unwrap();
    assert_eq!(row, Tensor::from([1, 2]));

    let row = rows.next().unwrap();
    assert_eq!(row, Tensor::from([3, 4]));

    assert_eq!(rows.next(), None);
}

#[test]
fn test_inner_iter_mut() {
    let mut tensor = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
    let mut rows = tensor.inner_iter_mut::<1>();

    let mut row = rows.next().unwrap();
    assert_eq!(row.shape(), [2]);
    row.apply(|x| x * 2);

    let mut row = rows.next().unwrap();
    assert_eq!(row.shape(), [2]);
    row.apply(|x| x * 2);

    assert_eq!(rows.next(), None);

    assert_eq!(tensor.to_vec(), &[2, 4, 6, 8]);
}

#[test]
fn test_inner_iter_dyn_mut() {
    let mut tensor = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
    let mut rows = tensor.inner_iter_dyn_mut(1);

    let mut row = rows.next().unwrap();
    assert_eq!(row.shape(), [2]);
    row.apply(|x| x * 2);

    let mut row = rows.next().unwrap();
    assert_eq!(row.shape(), [2]);
    row.apply(|x| x * 2);

    assert_eq!(rows.next(), None);

    assert_eq!(tensor.to_vec(), &[2, 4, 6, 8]);
}

#[test]
fn test_insert_axis() {
    let mut tensor = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
    tensor.insert_axis(0);
    assert_eq!(tensor.shape(), &[1, 2, 2]);
    tensor.insert_axis(3);
    assert_eq!(tensor.shape(), &[1, 2, 2, 1]);
}

#[test]
fn test_item() {
    let tensor = NdTensor::from(5.);
    assert_eq!(tensor.item(), Some(&5.));
    let tensor = NdTensor::from([6.]);
    assert_eq!(tensor.item(), Some(&6.));
    let tensor = NdTensor::from([2., 3.]);
    assert_eq!(tensor.item(), None);

    let tensor = Tensor::from(5.);
    assert_eq!(tensor.item(), Some(&5.));
    let tensor = Tensor::from([6.]);
    assert_eq!(tensor.item(), Some(&6.));
    let tensor = Tensor::from([2., 3.]);
    assert_eq!(tensor.item(), None);
}

#[test]
fn test_iter() {
    let data = vec![1., 2., 3., 4.];
    let tensor = NdTensor::from_data([2, 2], data);
    assert_eq!(
        tensor.iter().copied().collect::<Vec<_>>(),
        &[1., 2., 3., 4.]
    );
    let transposed = tensor.transposed();
    assert_eq!(
        transposed.iter().copied().collect::<Vec<_>>(),
        &[1., 3., 2., 4.]
    );

    let data = vec![1., 2., 3., 4.];
    let tensor = Tensor::from_data(&[2, 2], data);
    assert_eq!(
        tensor.iter().copied().collect::<Vec<_>>(),
        &[1., 2., 3., 4.]
    );
    let transposed = tensor.transposed();
    assert_eq!(
        transposed.iter().copied().collect::<Vec<_>>(),
        &[1., 3., 2., 4.]
    );
}

#[test]
fn test_iter_mut() {
    let data = vec![1., 2., 3., 4.];
    let mut tensor = NdTensor::from_data([2, 2], data);
    tensor.iter_mut().for_each(|x| *x *= 2.);
    assert_eq!(tensor.to_vec(), &[2., 4., 6., 8.]);
}

#[test]
fn test_lanes() {
    let data = vec![1., 2., 3., 4.];
    let tensor = NdTensor::from_data([2, 2], data);
    let mut lanes = tensor.lanes(1);
    assert_eq!(
        lanes.next().unwrap().copied().collect::<Vec<_>>(),
        &[1., 2.]
    );
    assert_eq!(
        lanes.next().unwrap().copied().collect::<Vec<_>>(),
        &[3., 4.]
    );
}

#[test]
fn test_lanes_mut() {
    let data = vec![1., 2., 3., 4.];
    let mut tensor = NdTensor::from_data([2, 2], data);
    let mut lanes = tensor.lanes_mut(1);
    assert_eq!(lanes.next().unwrap().collect::<Vec<_>>(), &[&1., &2.]);
    assert_eq!(lanes.next().unwrap().collect::<Vec<_>>(), &[&3., &4.]);
}

#[test]
fn test_make_contiguous() {
    let mut tensor = NdTensor::from_data([2, 2], vec![1., 2., 3., 4.]);
    assert!(tensor.is_contiguous());

    // No-op, since tensor is already contiguous.
    tensor.make_contiguous();
    assert!(tensor.is_contiguous());

    // On a non-contiguous tensor, the data should be shuffled.
    tensor.transpose();
    assert!(!tensor.is_contiguous());
    tensor.make_contiguous();
    assert!(tensor.is_contiguous());
    assert_eq!(tensor.data(), Some([1., 3., 2., 4.].as_slice()));
}

#[test]
fn test_map() {
    let data = vec![1., 2., 3., 4.];
    let tensor = NdTensor::from_data([2, 2], data);

    // Contiguous tensor
    let doubled = tensor.map(|x| x * 2.);
    assert_eq!(doubled.to_vec(), &[2., 4., 6., 8.]);

    // Non-contiguous tensor
    let halved = doubled.transposed().map(|x| x / 2.);
    assert_eq!(halved.to_vec(), &[1., 3., 2., 4.]);
}

#[test]
fn test_map_in() {
    let alloc = FakeAlloc::new();
    let tensor = NdTensor::arange(0, 4, None);

    let doubled = tensor.map_in(&alloc, |x| x * 2);
    assert_eq!(doubled.to_vec(), &[0, 2, 4, 6]);
    assert_eq!(alloc.count(), 1);
}

#[test]
fn test_matrix_layout() {
    let data = &[1., 2., 3., 4., 5., 6.];
    let tensor = NdTensorView::from_data([2, 3], data);
    assert_eq!(tensor.rows(), 2);
    assert_eq!(tensor.row_stride(), 3);
    assert_eq!(tensor.cols(), 3);
    assert_eq!(tensor.col_stride(), 1);
}

#[test]
fn test_merge_axes() {
    let mut tensor = Tensor::from_data(&[2, 2], vec![1, 2, 3, 4]);
    tensor.insert_axis(1);
    tensor.insert_axis(1);
    assert_eq!(tensor.shape(), &[2, 1, 1, 2]);
    assert_eq!(tensor.strides(), &[2, 4, 4, 1]);

    tensor.merge_axes();
    assert_eq!(tensor.shape(), &[4]);
}

#[test]
fn test_move_axis() {
    let data = &[1., 2., 3., 4., 5., 6.];
    let mut tensor = NdTensorView::from_data([2, 3], data);

    tensor.move_axis(1, 0);
    assert_eq!(tensor.shape(), [3, 2]);
    assert_eq!(tensor.to_vec(), &[1., 4., 2., 5., 3., 6.]);

    tensor.move_axis(0, 1);
    assert_eq!(tensor.shape(), [2, 3]);
    assert_eq!(tensor.to_vec(), &[1., 2., 3., 4., 5., 6.]);
}

#[test]
fn test_nd_view() {
    let tensor: Tensor<f32> = Tensor::zeros(&[1, 4, 5]);

    // Dynamic -> static rank conversion.
    let nd_view = tensor.nd_view::<3>();
    assert_eq!(nd_view.shape(), [1, 4, 5]);
    assert_eq!(nd_view.strides().as_ref(), tensor.strides());

    // Static -> static rank conversion. Pointless, but it should compile.
    let nd_view_2 = nd_view.nd_view::<3>();
    assert_eq!(nd_view_2.shape(), nd_view.shape());
}

#[test]
fn test_nd_view_mut() {
    let mut tensor: Tensor<f32> = Tensor::zeros(&[1, 4, 5]);
    let mut nd_view = tensor.nd_view_mut::<3>();
    assert_eq!(nd_view.shape(), [1, 4, 5]);

    nd_view[[0, 0, 0]] = 9.;

    assert_eq!(tensor[[0, 0, 0]], 9.);
}

#[test]
fn test_rand() {
    let mut rng = XorShiftRng::new(1234);
    let tensor = NdTensor::<f32, 2>::rand([2, 2], &mut rng);
    assert_eq!(tensor.shape(), [2, 2]);
    for &x in tensor.iter() {
        assert!(x >= 0. && x <= 1.);
    }
}

#[test]
fn test_permute() {
    let data = &[1., 2., 3., 4., 5., 6.];
    let mut tensor = NdTensorView::from_data([2, 3], data);

    tensor.permute([1, 0]);

    assert_eq!(tensor.shape(), [3, 2]);
    assert_eq!(tensor.to_vec(), &[1., 4., 2., 5., 3., 6.]);
}

#[test]
fn test_permuted() {
    let data = &[1., 2., 3., 4., 5., 6.];
    let tensor = NdTensorView::from_data([2, 3], data);

    let permuted = tensor.permuted([1, 0]);

    assert_eq!(permuted.shape(), [3, 2]);
    assert_eq!(permuted.to_vec(), &[1., 4., 2., 5., 3., 6.]);
}

#[test]
fn test_permuted_mut() {
    let data = vec![1., 2., 3., 4., 5., 6.];
    let mut tensor = NdTensor::from_data([2, 3], data);

    let mut permuted = tensor.permuted_mut([1, 0]);
    permuted[[2, 1]] = 8.;

    assert_eq!(permuted.shape(), [3, 2]);
    assert_eq!(permuted.to_vec(), &[1., 4., 2., 5., 3., 8.]);
}

#[test]
fn test_remove_axis() {
    let mut tensor = Tensor::arange(0., 16., None).into_shape([1, 2, 1, 8, 1].as_slice());
    tensor.remove_axis(0);
    tensor.remove_axis(1);
    tensor.remove_axis(2);
    assert_eq!(tensor.shape(), [2, 8]);
}

#[test]
fn test_reshape() {
    let mut tensor = Tensor::<f32>::from_data(&[2, 2], vec![1., 2., 3., 4.]);
    tensor.transpose();
    tensor.reshape(&[4]);
    assert_eq!(tensor.shape(), &[4]);
    assert_eq!(tensor.to_vec(), &[1., 3., 2., 4.]);
}

#[test]
#[should_panic(expected = "element count mismatch reshaping [16] to [2, 2]")]
fn test_reshape_invalid() {
    let mut tensor = Tensor::arange(0, 16, None);
    tensor.reshape(&[2, 2]);
}

#[test]
fn test_reshaped() {
    let data = &[1., 2., 3., 4., 5., 6.];
    let tensor = NdTensorView::from_data([2, 3], data);

    // Non-copying reshape to static dim count
    let reshaped = tensor.reshaped([6]);
    assert_eq!(reshaped.shape(), [6]);
    assert_eq!(
        reshaped.view().storage().as_ptr(),
        tensor.view().storage().as_ptr()
    );

    // Copying reshape to static dim count
    let reshaped = tensor.transposed().reshaped([6]);
    assert_eq!(reshaped.shape(), [6]);
    assert_ne!(
        reshaped.view().storage().as_ptr(),
        tensor.view().storage().as_ptr()
    );
    assert_eq!(reshaped.to_vec(), &[1., 4., 2., 5., 3., 6.]);

    // Non-copying reshape to dynamic dim count
    let reshaped = tensor.reshaped([6].as_slice());
    assert_eq!(reshaped.shape(), &[6]);
    assert_eq!(
        reshaped.view().storage().as_ptr(),
        tensor.view().storage().as_ptr()
    );
}

#[test]
#[should_panic(expected = "element count mismatch reshaping [16] to [2, 2]")]
fn test_reshaped_invalid() {
    let tensor = NdTensor::arange(0, 16, None);
    tensor.reshaped([2, 2]);
}

#[test]
fn test_reshaped_mut() {
    let data = vec![1., 2., 3., 4., 5., 6.];
    let mut tensor = NdTensor::from_data([1, 1, 2, 1, 3], data);

    let mut reshaped = tensor.reshaped_mut([6]).unwrap();
    reshaped[[0]] = 0.;
    reshaped[[5]] = 0.;

    assert_eq!(tensor.data(), Some([0., 2., 3., 4., 5., 0.].as_slice()));
}

#[test]
fn test_set_array() {
    let mut tensor = NdTensor::arange(1, 17, None).into_shape([4, 2, 2]);
    tensor.set_array([0, 0, 0], 0, [-1, -2, -3, -4]);
    assert_eq!(
        tensor.iter().copied().collect::<Vec<_>>(),
        &[-1, 2, 3, 4, -2, 6, 7, 8, -3, 10, 11, 12, -4, 14, 15, 16]
    );
}

// nb. In addition to the tests here, see also tests for the `Slice` op
// in the rten crate.
#[test]
fn test_slice_copy() {
    struct Case<'a> {
        shape: &'a [usize],
        slice_range: &'a [SliceItem],
        expected: Tensor<i32>,
    }

    let cases = [
        // No-op slice.
        Case {
            shape: &[4, 4],
            slice_range: &[],
            expected: Tensor::<i32>::arange(0, 16, None).into_shape([4, 4].as_slice()),
        },
        // Positive step and endpoints.
        Case {
            shape: &[4, 4],
            slice_range: &[
                // Every row
                SliceItem::Range(SliceRange::new(0, None, 1)),
                // Every other column
                SliceItem::Range(SliceRange::new(0, None, 2)),
            ],
            expected: Tensor::from([[0, 2], [4, 6], [8, 10], [12, 14]]),
        },
        // Negative step and endpoints.
        Case {
            shape: &[4, 4],
            slice_range: &[
                // Every row, reversed
                SliceItem::Range(SliceRange::new(-1, None, -1)),
                // Every other column, reversed
                SliceItem::Range(SliceRange::new(-1, None, -2)),
            ],
            expected: Tensor::from([[15, 13], [11, 9], [7, 5], [3, 1]]),
        },
    ];

    for Case {
        shape,
        slice_range,
        expected,
    } in cases
    {
        let len = shape.iter().product::<usize>() as i32;
        let tensor = Tensor::<i32>::arange(0, len as i32, None).into_shape(shape);
        let sliced = tensor.slice_copy(slice_range);
        assert_eq!(sliced, expected);
    }
}

#[test]
fn test_slice() {
    // Slice static-rank array. The rank of the slice is inferred.
    let data = NdTensor::from([[[1, 2, 3], [4, 5, 6]]]);
    let row = data.slice((0, 0));
    assert_eq!(row.shape(), [3usize]);
    assert_eq!(row.data().unwrap(), &[1, 2, 3]);

    // Slice dynamic-rank array. The rank of the slice is dynamic.
    let data = Tensor::from([[[1, 2, 3], [4, 5, 6]]]);
    let row = data.slice((0, 0));
    assert_eq!(row.shape(), [3usize]);
    assert_eq!(row.data().unwrap(), &[1, 2, 3]);
}

#[test]
fn test_slice_axis() {
    let data = NdTensor::from([[1, 2, 3], [4, 5, 6]]);
    let row = data.slice_axis(0, 0..1);
    let col = data.slice_axis(1, 1..2);
    assert_eq!(row, data.slice((0..1, ..)));
    assert_eq!(col, data.slice((.., 1..2)));
}

#[test]
fn test_slice_axis_mut() {
    let mut data = NdTensor::from([[1, 2, 3], [4, 5, 6]]);
    let mut row = data.slice_axis_mut(0, 0..1);
    row.fill(8);
    let mut col = data.slice_axis_mut(1, 1..2);
    col.fill(9);
    assert_eq!(data, NdTensor::from([[8, 9, 8], [4, 9, 6]]));
}

#[test]
fn test_slice_mut() {
    // Slice static-rank array. The rank of the slice is inferred.
    let mut data = NdTensor::from([[[1, 2, 3], [4, 5, 6]]]);
    let mut row = data.slice_mut((0, 0));
    row[[0usize]] = 5;
    assert_eq!(row.shape(), [3usize]);
    assert_eq!(row.data().unwrap(), &[5, 2, 3]);

    // Slice dynamic-rank array. The rank of the slice is dynamic.
    let mut data = Tensor::from([[[1, 2, 3], [4, 5, 6]]]);
    let mut row = data.slice_mut((0, 0));
    row[[0usize]] = 10;
    assert_eq!(row.shape(), [3usize]);
    assert_eq!(row.data().unwrap(), &[10, 2, 3]);
}

#[test]
fn test_squeezed() {
    let data = &[1., 2., 3., 4., 5., 6.];
    let tensor = NdTensorView::from_data([1, 1, 2, 1, 3], data);

    let squeezed = tensor.squeezed();

    assert_eq!(squeezed.shape(), &[2, 3]);
}

#[test]
fn test_split_at() {
    struct Case {
        shape: [usize; 2],
        axis: usize,
        mid: usize,
        expected_left: NdTensor<i32, 2>,
        expected_right: NdTensor<i32, 2>,
    }

    let cases = [
        // Split first dim.
        Case {
            shape: [4, 2],
            axis: 0,
            mid: 1,
            expected_left: NdTensor::from([[0, 1]]),
            expected_right: NdTensor::from([[2, 3], [4, 5], [6, 7]]),
        },
        // Split last dim.
        Case {
            shape: [4, 2],
            axis: 1,
            mid: 1,
            expected_left: NdTensor::from([[0], [2], [4], [6]]),
            expected_right: NdTensor::from([[1], [3], [5], [7]]),
        },
        // Split last dim such that left split is empty.
        Case {
            shape: [4, 2],
            axis: 1,
            mid: 0,
            expected_left: NdTensor::from([[], [], [], []]),
            expected_right: NdTensor::from([[0, 1], [2, 3], [4, 5], [6, 7]]),
        },
        // Split last dim such that right split is empty.
        Case {
            shape: [4, 2],
            axis: 1,
            mid: 2,
            expected_left: NdTensor::from([[0, 1], [2, 3], [4, 5], [6, 7]]),
            expected_right: NdTensor::from([[], [], [], []]),
        },
    ];

    for Case {
        shape,
        axis,
        mid,
        expected_left,
        expected_right,
    } in cases
    {
        // For each case, test all combinations of (static/dynamic,
        // immutable/mutable).
        let len: usize = shape.iter().product();
        let mut tensor = NdTensor::arange(0, len as i32, None).into_shape(shape);
        let (left, right) = tensor.view().split_at(axis, mid);
        assert_eq!(left, expected_left);
        assert_eq!(right, expected_right);

        let (left, right) = tensor.view_mut().split_at_mut(axis, mid);
        assert_eq!(left, expected_left);
        assert_eq!(right, expected_right);

        let mut tensor = tensor.into_dyn();
        let (left, right) = tensor.view().split_at(axis, mid);
        assert_eq!(left, expected_left);
        assert_eq!(right, expected_right);

        let (left, right) = tensor.view_mut().split_at_mut(axis, mid);
        assert_eq!(left, expected_left.as_dyn());
        assert_eq!(right, expected_right.as_dyn());
    }
}

#[test]
fn test_storage() {
    let data = &[1, 2, 3, 4];
    let tensor = NdTensorView::from_data([2, 2], data);
    let storage = tensor.storage();
    assert_eq!(storage.len(), 4);
    assert_eq!(storage.as_ptr(), data.as_ptr());
}

#[test]
fn test_storage_mut() {
    let data = &mut [1, 2, 3, 4];
    let ptr = data.as_mut_ptr();
    let mut tensor = NdTensorViewMut::from_data([2, 2], data.as_mut_slice());
    let storage = tensor.storage_mut();
    assert_eq!(storage.len(), 4);
    assert_eq!(storage.as_ptr(), ptr);
}

#[test]
fn test_to_array() {
    let tensor = NdTensor::arange(1., 5., None).into_shape([2, 2]);
    let col0: [f32; 2] = tensor.view().transposed().slice(0).to_array();
    let col1: [f32; 2] = tensor.view().transposed().slice(1).to_array();
    assert_eq!(col0, [1., 3.]);
    assert_eq!(col1, [2., 4.]);
}

#[test]
fn test_to_contiguous() {
    let data = vec![1., 2., 3., 4.];
    let tensor = NdTensor::from_data([2, 2], data);

    // Tensor is already contiguous, so this is a no-op.
    let tensor = tensor.to_contiguous();
    assert_eq!(tensor.to_vec(), &[1., 2., 3., 4.]);

    // Swap strides to make tensor non-contiguous.
    let mut tensor = tensor.into_inner();
    tensor.transpose();
    assert!(!tensor.is_contiguous());
    assert_eq!(tensor.to_vec(), &[1., 3., 2., 4.]);

    // Create a new contiguous copy.
    let tensor = tensor.to_contiguous();
    assert!(tensor.is_contiguous());
    assert_eq!(tensor.to_vec(), &[1., 3., 2., 4.]);
}

#[test]
fn test_to_shape() {
    let tensor = NdTensor::from_data([2, 2], vec![1, 2, 3, 4]);
    let flat = tensor.to_shape([4]);
    assert_eq!(flat.shape(), [4]);
    assert_eq!(flat.data(), Some([1, 2, 3, 4].as_slice()));
}

#[test]
#[should_panic(expected = "reshape failed")]
fn test_to_shape_invalid() {
    NdTensor::arange(0, 16, None).to_shape([2, 2]);
}

#[test]
fn test_to_vec() {
    // Contiguous case
    let tensor = NdTensor::from_data([2, 2], vec![1, 2, 3, 4]);
    assert_eq!(tensor.to_vec(), &[1, 2, 3, 4]);

    // Non-contiguous case
    let mut tensor = tensor.clone();
    tensor.transpose();
    assert_eq!(tensor.to_vec(), &[1, 3, 2, 4]);
}

#[test]
fn test_to_vec_in() {
    let alloc = FakeAlloc::new();
    let tensor = NdTensor::arange(0, 4, None);
    let vec = tensor.to_vec_in(&alloc);

    assert_eq!(vec, &[0, 1, 2, 3]);
    assert_eq!(alloc.count(), 1);
}

#[test]
fn test_to_slice() {
    let tensor = NdTensor::arange(0, 4, None).into_shape([2, 2]);
    assert_eq!(tensor.to_slice(), Cow::Borrowed(&[0, 1, 2, 3]));
    assert_eq!(
        tensor.transposed().to_slice(),
        Cow::<[i32]>::Owned(vec![0, 2, 1, 3])
    );
}

#[test]
fn test_to_tensor() {
    let data = &[1., 2., 3., 4.];
    let view = NdTensorView::from_data([2, 2], data);
    let tensor = view.to_tensor();
    assert_eq!(tensor.shape(), view.shape());
    assert_eq!(tensor.to_vec(), view.to_vec());
}

#[test]
fn test_to_tensor_in() {
    let alloc = FakeAlloc::new();
    let tensor = NdTensor::arange(0, 4, None).into_shape([2, 2]);

    // Contiguous case.
    let cloned = tensor.to_tensor_in(&alloc);
    assert_eq!(cloned.to_vec(), &[0, 1, 2, 3]);
    assert_eq!(alloc.count(), 1);

    // Non-contigous case.
    let cloned = tensor.transposed().to_tensor_in(&alloc);
    assert_eq!(cloned.to_vec(), &[0, 2, 1, 3]);
    assert_eq!(alloc.count(), 2);
}

#[test]
fn test_transpose() {
    let data = &[1., 2., 3., 4., 5., 6.];
    let mut tensor = NdTensorView::from_data([2, 3], data);

    tensor.transpose();

    assert_eq!(tensor.shape(), [3, 2]);
    assert_eq!(tensor.to_vec(), &[1., 4., 2., 5., 3., 6.]);
}

#[test]
fn test_transposed() {
    let data = &[1., 2., 3., 4., 5., 6.];
    let tensor = NdTensorView::from_data([2, 3], data);

    let permuted = tensor.transposed();

    assert_eq!(permuted.shape(), [3, 2]);
    assert_eq!(permuted.to_vec(), &[1., 4., 2., 5., 3., 6.]);
}

#[test]
fn test_try_from_data() {
    let x = NdTensor::try_from_data([1, 2, 2], vec![1, 2, 3, 4]);
    assert!(x.is_ok());
    if let Ok(x) = x {
        assert_eq!(x.shape(), [1, 2, 2]);
        assert_eq!(x.strides(), [4, 2, 1]);
        assert_eq!(x.to_vec(), [1, 2, 3, 4]);
    }

    let x = NdTensor::try_from_data([1, 2, 2], vec![1]);
    assert_eq!(x, Err(FromDataError::StorageLengthMismatch));
}

#[test]
fn test_try_slice() {
    let data = vec![1., 2., 3., 4.];
    let tensor = Tensor::from_data(&[2, 2], data);

    let row = tensor.try_slice(0);
    assert!(row.is_ok());
    assert_eq!(row.unwrap().data(), Some([1., 2.].as_slice()));

    let row = tensor.try_slice(1);
    assert!(row.is_ok());

    let row = tensor.try_slice(2);
    assert!(row.is_err());
}

#[test]
fn test_try_slice_mut() {
    let data = vec![1., 2., 3., 4.];
    let mut tensor = Tensor::from_data(&[2, 2], data);

    let mut row = tensor.try_slice_mut(0).unwrap();
    row[[0]] += 1.;
    row[[1]] += 1.;
    assert_eq!(row.data(), Some([2., 3.].as_slice()));

    let row = tensor.try_slice_mut(1);
    assert!(row.is_ok());

    let row = tensor.try_slice(2);
    assert!(row.is_err());
}

#[test]
fn test_uninit() {
    let mut tensor = NdTensor::uninit([2, 2]);
    for (i, x) in tensor.iter_mut().enumerate() {
        x.write(i);
    }

    let view = unsafe { tensor.view().assume_init() };
    assert_eq!(view, NdTensorView::from_data([2, 2], &[0, 1, 2, 3]));

    let mut_view = unsafe { tensor.view_mut().assume_init() };
    assert_eq!(mut_view, NdTensorView::from_data([2, 2], &[0, 1, 2, 3]));

    let tensor = unsafe { tensor.assume_init() };
    assert_eq!(tensor, NdTensor::from_data([2, 2], vec![0, 1, 2, 3]));
}

#[test]
fn test_uninit_in() {
    let pool = FakeAlloc::new();
    NdTensor::<f32, 2>::uninit_in(&pool, [2, 2]);
    assert_eq!(pool.count(), 1);
}

#[test]
fn test_view() {
    let tensor = NdTensor::from_data([2, 2], vec![1, 2, 3, 4]);
    let view = tensor.view();
    assert_eq!(view.data(), Some([1, 2, 3, 4].as_slice()));
}

#[test]
fn test_view_mut() {
    let mut tensor = NdTensor::from_data([2, 2], vec![1, 2, 3, 4]);
    let mut view = tensor.view_mut();
    view[[0, 0]] = 0;
    view[[1, 1]] = 0;
    assert_eq!(tensor.data(), Some([0, 2, 3, 0].as_slice()));
}

#[test]
fn test_weakly_checked_view() {
    let tensor = NdTensor::from_data([2, 2], vec![1, 2, 3, 4]);
    let view = tensor.weakly_checked_view();

    // Valid indexing should work the same as a normal view.
    for y in 0..tensor.size(0) {
        for x in 0..tensor.size(1) {
            assert_eq!(view[[y, x]], tensor[[y, x]]);
        }
    }

    // Indexes that are invalid, but lead to an in-bounds offset, won't
    // trigger a panic, unlike a normal view.
    assert_eq!(view[[0, 2]], 3);
}

#[test]
fn test_weakly_checked_view_mut() {
    let mut tensor = NdTensor::from_data([2, 2], vec![1, 2, 3, 4]);
    let mut view = tensor.weakly_checked_view_mut();

    // Valid indices
    view[[0, 0]] = 5;
    view[[1, 1]] = 6;

    // Indices that are invalid, but lead to an in-bounds offset, won't
    // trigger a panic, unlike a normal view.
    view[[0, 2]] = 7;

    assert_eq!(tensor.data(), Some([5, 2, 7, 6].as_slice()));
}

#[test]
fn test_zeros() {
    let tensor = NdTensor::zeros([2, 2]);
    assert_eq!(tensor.shape(), [2, 2]);
    assert_eq!(tensor.data(), Some([0, 0, 0, 0].as_slice()));
}

#[test]
fn test_zeros_in() {
    let pool = FakeAlloc::new();
    NdTensor::<f32, 2>::zeros_in(&pool, [2, 2]);
    assert_eq!(pool.count(), 1);
}
