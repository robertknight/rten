use std::fmt::Debug;

use rten_base::num::{Identities, IsInt, IsNaN, MinMax};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensorView, Storage, Tensor, TensorBase, TensorView};

use crate::buffer_pool::BufferPool;
use crate::ops::OpError;
use crate::ops::{
    PadMode, arg_max, div, matmul, mul, pad, reduce_l2, reduce_max, reduce_mean, reduce_min,
    reduce_sum, resize_image, softmax, topk,
};
use crate::threading::thread_pool;

/// Trait which exposes ONNX operators as methods of tensors.
///
/// This trait provides methods which are available on all tensor types. See
/// [`FloatOperators`] for additional operators which are only available on float
/// tensors.
pub trait Operators {
    type Elem;

    fn arg_max(&self, axis: isize, keep_dims: bool) -> Result<Tensor<i32>, OpError>
    where
        Self::Elem: Copy + PartialOrd + IsNaN;

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
        Self::Elem: Copy + PartialOrd + IsNaN + MinMax;

    fn reduce_min(
        &self,
        axes: Option<&[i32]>,
        keep_dims: bool,
    ) -> Result<Tensor<Self::Elem>, OpError>
    where
        Self::Elem: Copy + PartialOrd + IsNaN + MinMax;

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
        Self::Elem: Copy + Default + PartialEq;

    fn topk(
        &self,
        k: usize,
        axis: Option<isize>,
        largest: bool,
        sorted: bool,
    ) -> Result<(Tensor<Self::Elem>, Tensor<i32>), OpError>
    where
        Self::Elem: Copy + Default + PartialOrd + IsNaN;
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

/// Set up the environment to run an operation and dispatch it.
///
/// This runs `op` in the same thread pool as model inference using `Model::run`
/// and passes a `BufferPool` for allocating outputs and temporary buffers.
///
/// Ideally this would run the task on the current thread, but cause any
/// parallel tasks to be spawned in the thread pool.
/// `rayon::ThreadPool::in_place_scope` looks like the ideal API for this, but
/// it does not change which thread pool is used by parallel iterators. See
/// https://github.com/rayon-rs/rayon/issues/1165.
fn run_operator<R: Send, F: Send + FnOnce(&BufferPool) -> R>(op: F) -> R {
    let pool = BufferPool::new();
    thread_pool().run(|| op(&pool))
}

impl<T: Send, S: Storage<Elem = T> + Sync, L: Layout + Clone + Sync> Operators
    for TensorBase<S, L>
{
    type Elem = T;

    fn arg_max(&self, axis: isize, keep_dims: bool) -> Result<Tensor<i32>, OpError>
    where
        T: Copy + PartialOrd + IsNaN,
    {
        run_operator(|pool| arg_max(pool, self.as_dyn(), axis, keep_dims))
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
        run_operator(|pool| div(pool, self.as_dyn(), other))
    }

    fn mul(&self, other: TensorView<T>) -> Result<Tensor<T>, OpError>
    where
        T: Copy + Debug + Default + std::ops::Mul<Output = T>,
    {
        run_operator(|pool| mul(pool, self.as_dyn(), other))
    }

    fn reduce_max(&self, axes: Option<&[i32]>, keep_dims: bool) -> Result<Tensor<T>, OpError>
    where
        T: Copy + PartialOrd + IsNaN + MinMax,
    {
        run_operator(|pool| reduce_max(pool, self.as_dyn(), axes, keep_dims))
    }

    fn reduce_min(&self, axes: Option<&[i32]>, keep_dims: bool) -> Result<Tensor<T>, OpError>
    where
        T: Copy + PartialOrd + IsNaN + MinMax,
    {
        run_operator(|pool| reduce_min(pool, self.as_dyn(), axes, keep_dims))
    }

    fn reduce_sum(
        &self,
        axes: Option<&[i32]>,
        keep_dims: bool,
    ) -> Result<Tensor<Self::Elem>, OpError>
    where
        Self::Elem: Copy + Default + std::ops::Add<Self::Elem, Output = Self::Elem>,
    {
        run_operator(|pool| reduce_sum(pool, self.as_dyn(), axes, keep_dims))
    }

    fn pad(&self, padding: NdTensorView<i32, 1>, val: T) -> Result<Tensor<Self::Elem>, OpError>
    where
        Self::Elem: Copy + Default + PartialEq,
    {
        run_operator(move |pool| pad(pool, self.as_dyn(), &padding, PadMode::Constant, val))
    }

    fn topk(
        &self,
        k: usize,
        axis: Option<isize>,
        largest: bool,
        sorted: bool,
    ) -> Result<(Tensor<Self::Elem>, Tensor<i32>), OpError>
    where
        T: Copy + Default + PartialOrd + IsNaN,
    {
        run_operator(|pool| topk(pool, self.as_dyn(), k, axis, largest, sorted))
    }
}

impl<S: Storage<Elem = f32> + Sync, L: Layout + Clone + Sync> FloatOperators for TensorBase<S, L> {
    fn matmul(&self, other: TensorView) -> Result<Tensor, OpError> {
        run_operator(|pool| matmul(pool, self.as_dyn(), other, None))
    }

    fn reduce_l2(&self, axes: Option<&[i32]>, keep_dims: bool) -> Result<Tensor, OpError> {
        run_operator(|pool| reduce_l2(pool, self.as_dyn(), axes, keep_dims))
    }

    fn reduce_mean(&self, axes: Option<&[i32]>, keep_dims: bool) -> Result<Tensor, OpError> {
        run_operator(|pool| reduce_mean(pool, self.as_dyn(), axes, keep_dims))
    }

    fn resize_image(&self, size: [usize; 2]) -> Result<Tensor, OpError> {
        run_operator(|_pool| resize_image(self.as_dyn(), size))
    }

    fn softmax(&self, axis: isize) -> Result<Tensor, OpError> {
        run_operator(|pool| softmax(pool, self.as_dyn(), axis))
    }
}
