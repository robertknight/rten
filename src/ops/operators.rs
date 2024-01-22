use std::fmt::Debug;

use rten_tensor::prelude::*;
use rten_tensor::{DynLayout, NdTensorBase, NdTensorView, Tensor, TensorBase, TensorView};

use crate::number::{Identities, IsInt};
use crate::ops::OpError;
use crate::ops::{
    arg_max, div, matmul, mul, pad, reduce_l2, reduce_mean, resize_image, softmax, topk,
};

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

impl<T, S: AsRef<[T]>> Operators for TensorBase<T, S, DynLayout> {
    type Elem = T;

    fn arg_max(&self, axis: isize, keep_dims: bool) -> Result<Tensor<i32>, OpError>
    where
        T: Copy + PartialOrd,
    {
        arg_max(self.view(), axis, keep_dims)
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
        div(self.view(), other)
    }

    fn mul(&self, other: TensorView<T>) -> Result<Tensor<T>, OpError>
    where
        T: Copy + Debug + Default + std::ops::Mul<Output = T>,
    {
        mul(self.view(), other)
    }

    fn pad(&self, padding: NdTensorView<i32, 1>, val: T) -> Result<Tensor<Self::Elem>, OpError>
    where
        Self::Elem: Copy,
    {
        pad(self.view(), &padding, val)
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
        topk(self.view(), k, axis, largest, sorted)
    }
}

impl<T, S: AsRef<[T]>, const N: usize> Operators for NdTensorBase<T, S, N> {
    type Elem = T;

    fn arg_max(&self, axis: isize, keep_dims: bool) -> Result<Tensor<i32>, OpError>
    where
        T: Copy + PartialOrd,
    {
        arg_max(self.as_dyn(), axis, keep_dims)
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
        div(self.as_dyn(), other)
    }

    fn mul(&self, other: TensorView<T>) -> Result<Tensor<T>, OpError>
    where
        T: Copy + Debug + Default + std::ops::Mul<Output = T>,
    {
        mul(self.as_dyn(), other)
    }

    fn pad(&self, padding: NdTensorView<i32, 1>, val: T) -> Result<Tensor<Self::Elem>, OpError>
    where
        Self::Elem: Copy,
    {
        pad(self.as_dyn(), &padding, val)
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
        topk(self.as_dyn(), k, axis, largest, sorted)
    }
}

impl<S: AsRef<[f32]>> FloatOperators for TensorBase<f32, S, DynLayout> {
    fn matmul(&self, other: TensorView) -> Result<Tensor, OpError> {
        matmul(self.view(), other)
    }

    fn reduce_l2(&self, axes: Option<&[i32]>, keep_dims: bool) -> Result<Tensor, OpError> {
        reduce_l2(self.view(), axes, keep_dims)
    }

    fn reduce_mean(&self, axes: Option<&[i32]>, keep_dims: bool) -> Result<Tensor, OpError> {
        reduce_mean(self.view(), axes, keep_dims)
    }

    fn resize_image(&self, size: [usize; 2]) -> Result<Tensor, OpError> {
        resize_image(self.view(), size)
    }

    fn softmax(&self, axis: isize) -> Result<Tensor, OpError> {
        softmax(self.view(), axis)
    }
}

impl<S: AsRef<[f32]>, const N: usize> FloatOperators for NdTensorBase<f32, S, N> {
    fn matmul(&self, other: TensorView) -> Result<Tensor, OpError> {
        matmul(self.as_dyn(), other)
    }

    fn reduce_l2(&self, axes: Option<&[i32]>, keep_dims: bool) -> Result<Tensor, OpError> {
        reduce_l2(self.as_dyn(), axes, keep_dims)
    }

    fn reduce_mean(&self, axes: Option<&[i32]>, keep_dims: bool) -> Result<Tensor, OpError> {
        reduce_mean(self.as_dyn(), axes, keep_dims)
    }

    fn resize_image(&self, size: [usize; 2]) -> Result<Tensor, OpError> {
        resize_image(self.as_dyn(), size)
    }

    fn softmax(&self, axis: isize) -> Result<Tensor, OpError> {
        softmax(self.as_dyn(), axis)
    }
}
