use std::fmt::Debug;

use rten_tensor::prelude::*;
use rten_tensor::{MutLayout, NdTensorView, Storage, Tensor, TensorBase, TensorView};

use crate::number::{Identities, IsInt};
use crate::ops::OpError;
use crate::ops::{
    arg_max, div, matmul, mul, pad, reduce_l2, reduce_max, reduce_mean, reduce_min, reduce_sum,
    resize_image, softmax, topk,
};
use crate::tensor_pool::TensorPool;
use crate::threading::thread_pool;

/// Trait which exposes ONNX operators as methods of tensors.
///
/// This trait provides methods which are available on all tensor types. See
/// [FloatOperators] for additional operators which are only available on float
/// tensors.
pub trait Operators {
    type Elem;

    fn arg_max(&self, axis: isize, keep_dims: bool) -> Result<Tensor<i32>, OpError>
    where
        Self::Elem: Copy + PartialOrd;

    fn div(&self, other: TensorView<Self::Elem>) -> Result<Tensor<Self::Elem>, OpError>
    where
        Self::Elem: Copy
            + Debug
            + Default
            + std::ops::Mul<Output = Self::Elem>
            + std::ops::Div<Output = Self::Elem>
            + IsInt
            + Identities;

    fn mul(&self, other: TensorView<Self::Elem>) -> Result<Tensor<Self::Elem>, OpError>
    where
        Self::Elem: Copy + Debug + Default + std::ops::Mul<Output = Self::Elem>;

    fn reduce_max(
        &self,
        axes: Option<&[i32]>,
        keep_dims: bool,
    ) -> Result<Tensor<Self::Elem>, OpError>
    where
        Self::Elem: Copy + PartialOrd;

    fn reduce_min(
        &self,
        axes: Option<&[i32]>,
        keep_dims: bool,
    ) -> Result<Tensor<Self::Elem>, OpError>
    where
        Self::Elem: Copy + PartialOrd;

    fn reduce_sum(
        &self,
        axes: Option<&[i32]>,
        keep_dims: bool,
    ) -> Result<Tensor<Self::Elem>, OpError>
    where
        Self::Elem: Copy + Default + std::ops::Add<Self::Elem, Output = Self::Elem>;

    fn pad(
        &self,
        padding: NdTensorView<i32, 1>,
        val: Self::Elem,
    ) -> Result<Tensor<Self::Elem>, OpError>
    where
        Self::Elem: Copy;

    fn topk(
        &self,
        k: usize,
        axis: Option<isize>,
        largest: bool,
        sorted: bool,
    ) -> Result<(Tensor<Self::Elem>, Tensor<i32>), OpError>
    where
        Self::Elem: Copy + Default + PartialOrd;
}

/// Trait which exposes ONNX operators as methods of tensors.
///
/// This trait provides methods which are only available on float tensors.
pub trait FloatOperators {
    fn matmul(&self, other: TensorView) -> Result<Tensor, OpError>;

    fn reduce_l2(&self, axes: Option<&[i32]>, keep_dims: bool) -> Result<Tensor, OpError>;
    fn reduce_mean(&self, axes: Option<&[i32]>, keep_dims: bool) -> Result<Tensor, OpError>;

    /// Resize an NCHW image tensor to a given `[height, width]` using bilinear
    /// interpolation.
    fn resize_image(&self, size: [usize; 2]) -> Result<Tensor, OpError>;
    fn softmax(&self, axis: isize) -> Result<Tensor, OpError>;
}

/// Run `op` in this library's global thread pool.
///
/// Ideally this would run the task on the current thread, but cause any
/// parallel tasks to be spawned in the thread pool.
/// `rayon::ThreadPool::in_place_scope` looks like the ideal API for this, but
/// it does not change which thread pool is used by parallel iterators. See
/// https://github.com/rayon-rs/rayon/issues/1165.
fn use_thread_pool<R: Send, F: Send + FnOnce() -> R>(op: F) -> R {
    thread_pool().run(op)
}

impl<T: Send, S: Storage<Elem = T>, L: MutLayout> Operators for TensorBase<S, L> {
    type Elem = T;

    fn arg_max(&self, axis: isize, keep_dims: bool) -> Result<Tensor<i32>, OpError>
    where
        T: Copy + PartialOrd,
    {
        let view = self.as_dyn();
        use_thread_pool(|| arg_max(&TensorPool::new(), view, axis, keep_dims))
    }

    fn div(&self, other: TensorView<Self::Elem>) -> Result<Tensor<Self::Elem>, OpError>
    where
        Self::Elem: Copy
            + Debug
            + Default
            + std::ops::Mul<Output = Self::Elem>
            + std::ops::Div<Output = Self::Elem>
            + IsInt
            + Identities,
    {
        let view = self.as_dyn();
        use_thread_pool(|| div(&TensorPool::new(), view, other))
    }

    fn mul(&self, other: TensorView<T>) -> Result<Tensor<T>, OpError>
    where
        T: Copy + Debug + Default + std::ops::Mul<Output = T>,
    {
        let view = self.as_dyn();
        use_thread_pool(|| mul(&TensorPool::new(), view, other))
    }

    fn reduce_max(&self, axes: Option<&[i32]>, keep_dims: bool) -> Result<Tensor<T>, OpError>
    where
        T: Copy + PartialOrd,
    {
        let view = self.as_dyn();
        use_thread_pool(|| reduce_max(&TensorPool::new(), view, axes, keep_dims))
    }

    fn reduce_min(&self, axes: Option<&[i32]>, keep_dims: bool) -> Result<Tensor<T>, OpError>
    where
        T: Copy + PartialOrd,
    {
        let view = self.as_dyn();
        use_thread_pool(|| reduce_min(&TensorPool::new(), view, axes, keep_dims))
    }

    fn reduce_sum(
        &self,
        axes: Option<&[i32]>,
        keep_dims: bool,
    ) -> Result<Tensor<Self::Elem>, OpError>
    where
        Self::Elem: Copy + Default + std::ops::Add<Self::Elem, Output = Self::Elem>,
    {
        let view = self.as_dyn();
        use_thread_pool(|| reduce_sum(&TensorPool::new(), view, axes, keep_dims))
    }

    fn pad(&self, padding: NdTensorView<i32, 1>, val: T) -> Result<Tensor<Self::Elem>, OpError>
    where
        Self::Elem: Copy,
    {
        let view = self.as_dyn();
        use_thread_pool(move || pad(&TensorPool::new(), view, &padding, val))
    }

    fn topk(
        &self,
        k: usize,
        axis: Option<isize>,
        largest: bool,
        sorted: bool,
    ) -> Result<(Tensor<Self::Elem>, Tensor<i32>), OpError>
    where
        T: Copy + Default + PartialOrd,
    {
        let view = self.as_dyn();
        use_thread_pool(|| topk(&TensorPool::new(), view, k, axis, largest, sorted))
    }
}

impl<S: Storage<Elem = f32>, L: MutLayout> FloatOperators for TensorBase<S, L> {
    fn matmul(&self, other: TensorView) -> Result<Tensor, OpError> {
        let view = self.as_dyn();
        use_thread_pool(|| matmul(&TensorPool::new(), view, other))
    }

    fn reduce_l2(&self, axes: Option<&[i32]>, keep_dims: bool) -> Result<Tensor, OpError> {
        let view = self.as_dyn();
        use_thread_pool(|| reduce_l2(&TensorPool::new(), view, axes, keep_dims))
    }

    fn reduce_mean(&self, axes: Option<&[i32]>, keep_dims: bool) -> Result<Tensor, OpError> {
        let view = self.as_dyn();
        use_thread_pool(|| reduce_mean(&TensorPool::new(), view, axes, keep_dims))
    }

    fn resize_image(&self, size: [usize; 2]) -> Result<Tensor, OpError> {
        let view = self.as_dyn();
        use_thread_pool(|| resize_image(view, size))
    }

    fn softmax(&self, axis: isize) -> Result<Tensor, OpError> {
        let view = self.as_dyn();
        use_thread_pool(|| softmax(&TensorPool::new(), view, axis))
    }
}
