use smallvec::SmallVec;

use crate::infer_shapes::{InferShapes, InferShapesError};
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::{SymElem, SymTensor};

/// Return the output size for a spatial dimension in a convolution or pooling
/// operation.
///
/// The formulae are given in the ONNX docs for convolution and pooling
/// operators, but expressed more clearly in the [PyTorch
/// docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html).
fn output_size(
    in_size: SymElem,
    kernel_size: SymElem,
    stride: usize,
    dilation: usize,
    padding: DimPadding,
) -> SymElem {
    let stride = SymElem::from(stride as i32);

    match padding {
        DimPadding::Fixed {
            start: pad_start,
            end: pad_end,
        } => {
            let dilation = SymElem::from(dilation as i32);
            let one = SymElem::from(1);
            let padded_in_size =
                in_size + SymElem::from(pad_start as i32) + SymElem::from(pad_end as i32);
            (padded_in_size - dilation * (kernel_size - one.clone()) - one.clone()) / stride
                + one.clone()
        }
        DimPadding::Same => in_size.div_ceil(&stride),
    }
}

/// Specifies the padding mode used by a convolution or pooling operator.
///
/// This is derived from the `pads` and `auto_pad` operator attributes.
pub enum Padding<'a> {
    /// Pad the input so that the size of each output spatial dimension is
    /// `ceil(input_size / stride)`.
    ///
    /// ONNX supports two "same" padding modes, SAME_UPPER and SAME_LOWER.
    /// Both produce the same result for shape inference.
    Same,

    /// Fixed padding specified as `[starts..., ends...]` where `starts`
    /// and `ends` are the start/end padding along each spatial dim.
    Fixed(&'a [usize]),
}

impl Padding<'_> {
    /// Get the padding for a single spatial dimension.
    fn dim(&self, dim: usize, spatial_dim_count: usize) -> Option<DimPadding> {
        match self {
            Padding::Same => Some(DimPadding::Same),
            Padding::Fixed(padding) => {
                let start = *padding.get(dim)?;
                let end = *padding.get(spatial_dim_count + dim)?;
                Some(DimPadding::Fixed { start, end })
            }
        }
    }
}

/// Padding for a single spatial dimension.
enum DimPadding {
    Same,
    Fixed { start: usize, end: usize },
}

/// Conv operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Conv.html>.
pub struct Conv<'a> {
    pub dilations: &'a [usize],
    pub padding: Padding<'a>,
    pub strides: &'a [usize],
}

impl InferShapes for Conv<'_> {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let [data, weights, ..] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };

        let Some(data_dims) = data.shape() else {
            return Ok([SymTensor::unknown("unknown input shape")].into());
        };
        let Some(weight_dims) = weights.shape() else {
            return Ok([SymTensor::unknown("unknown weights shape")].into());
        };

        if data_dims.len() < 3 {
            return Err(InferShapesError::IncorrectRank);
        }

        // First dim of weights and data have a different meaning (batch size
        // and output channels respectively), but the ranks should always be
        // equal.
        if weight_dims.len() != data_dims.len() {
            return Err(InferShapesError::IncorrectRank);
        }

        let data_shape: SmallVec<[_; 4]> = data_dims.collect();
        let weight_shape: SmallVec<[_; 4]> = weight_dims.collect();
        let spatial_dims = data_shape.len() - 2;

        let pad_h = self
            .padding
            .dim(0, spatial_dims)
            .ok_or(InferShapesError::InvalidValue)?;
        let out_h = output_size(
            data_shape[2].clone(),
            weight_shape[2].clone(),
            *self.strides.first().ok_or(InferShapesError::InvalidValue)?,
            *self
                .dilations
                .first()
                .ok_or(InferShapesError::InvalidValue)?,
            pad_h,
        );

        let mut out_shape = Vec::with_capacity(data_shape.len());
        out_shape.push(data_shape[0].clone());
        out_shape.push(weight_shape[0].clone());
        out_shape.push(out_h);

        if let Some(in_w) = data_shape.get(3).cloned()
            && let Some(k_w) = weight_shape.get(3).cloned()
        {
            let pad_w = self
                .padding
                .dim(1, spatial_dims)
                .ok_or(InferShapesError::InvalidValue)?;
            let out_w = output_size(
                in_w,
                k_w,
                *self.strides.get(1).ok_or(InferShapesError::InvalidValue)?,
                *self
                    .dilations
                    .get(1)
                    .ok_or(InferShapesError::InvalidValue)?,
                pad_w,
            );
            out_shape.push(out_w);
        }

        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

/// Local pooling operators (MaxPool, AveragePool etc.)
///
/// See <https://onnx.ai/onnx/operators/onnx__MaxPool.html>.
pub struct Pool<'a> {
    pub dilations: &'a [usize],
    pub kernel_size: &'a [usize],
    pub padding: Padding<'a>,
    pub strides: &'a [usize],
}

impl InferShapes for Pool<'_> {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let [data, ..] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };

        let Some(data_dims) = data.shape() else {
            return Ok([SymTensor::unknown("unknown input shape")].into());
        };

        if data_dims.len() < 3 {
            return Err(InferShapesError::IncorrectRank);
        }

        let spatial_dims = data_dims.len() - 2;

        let data_shape: SmallVec<[_; 4]> = data_dims.collect();

        let pad_h = self
            .padding
            .dim(0, spatial_dims)
            .ok_or(InferShapesError::InvalidValue)?;
        let kernel_h = *self
            .kernel_size
            .first()
            .ok_or(InferShapesError::InvalidValue)?;
        let out_h = output_size(
            data_shape[2].clone(),
            SymElem::from(kernel_h as i32),
            *self.strides.first().ok_or(InferShapesError::InvalidValue)?,
            *self
                .dilations
                .first()
                .ok_or(InferShapesError::InvalidValue)?,
            pad_h,
        );

        let mut out_shape = Vec::with_capacity(data_shape.len());
        out_shape.push(data_shape[0].clone());
        out_shape.push(data_shape[1].clone());
        out_shape.push(out_h);

        if let Some(in_w) = data_shape.get(3).cloned() {
            let pad_w = self
                .padding
                .dim(1, spatial_dims)
                .ok_or(InferShapesError::InvalidValue)?;
            let kernel_w = *self
                .kernel_size
                .get(1)
                .ok_or(InferShapesError::InvalidValue)?;

            let out_w = output_size(
                in_w,
                SymElem::from(kernel_w as i32),
                *self.strides.get(1).ok_or(InferShapesError::InvalidValue)?,
                *self
                    .dilations
                    .get(1)
                    .ok_or(InferShapesError::InvalidValue)?,
                pad_w,
            );
            out_shape.push(out_w);
        }

        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

/// Global pooling operators (GlobalAveragePool, GlobalMaxPool etc.)
///
/// See <https://onnx.ai/onnx/operators/onnx__GlobalAveragePool.html>.
pub struct GlobalPool;

impl InferShapes for GlobalPool {
    fn infer_shapes(
        &self,
        inputs: &[SymTensor],
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let [data, ..] = inputs else {
            return Err(InferShapesError::IncorrectInputCount);
        };
        let Some(dims) = data.shape() else {
            return Ok([SymTensor::unknown("unknown input shape")].into());
        };
        let spatial_dims = dims.len().saturating_sub(2);

        // Given input (N, C, D1, D2 ...) the output is (N, C, 1, 1 ...)
        let shape: Vec<_> = dims
            .take(2)
            .chain(std::iter::repeat_n(SymElem::from(1), spatial_dims))
            .collect();
        Ok([SymTensor::from_shape(shape)].into())
    }
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::InferShapes;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymElem, SymTensor, sym_shape};

    use super::{Conv, GlobalPool, Padding, Pool};

    #[test]
    fn test_conv() {
        let mut sym_gen = SymbolGen::new();

        // 1D conv
        let data = sym_shape!("batch", "in_c", "len");
        let weights = sym_shape!(768, 3, 32);
        let op = Conv {
            padding: Padding::Fixed(&[0, 2]),
            dilations: &[4],
            strides: &[16],
        };
        let result = op
            .infer_shapes(&[data.clone(), weights.clone()], &mut sym_gen)
            .unwrap();
        assert_eq!(
            result[0],
            sym_shape!(
                "batch",
                768,
                (SymElem::from("len") + SymElem::from(0) + SymElem::from(2)
                    - SymElem::from(4) * (SymElem::from(32) - SymElem::from(1))
                    - SymElem::from(1))
                    / SymElem::from(16)
                    + SymElem::from(1),
            )
        );

        // 1D conv with "same" padding.
        let op = Conv {
            padding: Padding::Same,
            dilations: &[4],
            strides: &[16],
        };
        let result = op.infer_shapes(&[data, weights], &mut sym_gen).unwrap();
        assert_eq!(
            result[0],
            sym_shape!(
                "batch",
                768,
                SymElem::from("len").div_ceil(&SymElem::from(16))
            )
        );

        // 2D conv
        let data = sym_shape!("batch", "in_c", "height", "width");
        let weights = sym_shape!(768, 3, 32, 32);
        let op = Conv {
            padding: Padding::Fixed(&[0, 1, 2, 3]),
            dilations: &[4, 5],
            strides: &[16, 32],
        };
        let result = op.infer_shapes(&[data, weights], &mut sym_gen).unwrap();
        assert_eq!(
            result[0],
            sym_shape!(
                "batch",
                768,
                (SymElem::from("height") + SymElem::from(0) + SymElem::from(2)
                    - SymElem::from(4) * (SymElem::from(32) - SymElem::from(1))
                    - SymElem::from(1))
                    / SymElem::from(16)
                    + SymElem::from(1),
                (SymElem::from("width") + SymElem::from(1) + SymElem::from(3)
                    - SymElem::from(5) * (SymElem::from(32) - SymElem::from(1))
                    - SymElem::from(1))
                    / SymElem::from(32)
                    + SymElem::from(1),
            )
        );
    }

    #[test]
    fn test_pool() {
        let mut sym_gen = SymbolGen::new();

        // 1D pool
        let data = sym_shape!("batch", "in_c", "seq");
        let op = Pool {
            kernel_size: &[32],
            padding: Padding::Fixed(&[0, 2]),
            dilations: &[4],
            strides: &[16],
        };
        let result = op.infer_shapes(&[data.clone()], &mut sym_gen).unwrap();
        assert_eq!(
            result[0],
            sym_shape!(
                "batch",
                "in_c",
                (SymElem::from("seq") + SymElem::from(0) + SymElem::from(2)
                    - SymElem::from(4) * (SymElem::from(32) - SymElem::from(1))
                    - SymElem::from(1))
                    / SymElem::from(16)
                    + SymElem::from(1),
            )
        );

        // 1D pool with "same" padding
        let op = Pool {
            kernel_size: &[32],
            padding: Padding::Same,
            dilations: &[4],
            strides: &[16],
        };
        let result = op.infer_shapes(&[data], &mut sym_gen).unwrap();
        assert_eq!(
            result[0],
            sym_shape!(
                "batch",
                "in_c",
                SymElem::from("seq").div_ceil(&SymElem::from(16)),
            )
        );

        // 2D pool
        let data = sym_shape!("batch", "in_c", "height", "width");
        let op = Pool {
            kernel_size: &[32, 32],
            padding: Padding::Fixed(&[0, 1, 2, 3]),
            dilations: &[4, 5],
            strides: &[16, 32],
        };
        let result = op.infer_shapes(&[data], &mut sym_gen).unwrap();
        assert_eq!(
            result[0],
            sym_shape!(
                "batch",
                "in_c",
                (SymElem::from("height") + SymElem::from(0) + SymElem::from(2)
                    - SymElem::from(4) * (SymElem::from(32) - SymElem::from(1))
                    - SymElem::from(1))
                    / SymElem::from(16)
                    + SymElem::from(1),
                (SymElem::from("width") + SymElem::from(1) + SymElem::from(3)
                    - SymElem::from(5) * (SymElem::from(32) - SymElem::from(1))
                    - SymElem::from(1))
                    / SymElem::from(32)
                    + SymElem::from(1),
            )
        );
    }

    #[test]
    fn test_global_pool() {
        let mut sym_gen = SymbolGen::new();

        // 1D global pool
        let data = sym_shape!("batch", "in_c", "height", "width");
        let result = GlobalPool.infer_shapes(&[data], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!("batch", "in_c", 1, 1,));

        // 2D global pool
        let data = sym_shape!("batch", "in_c", "height", "width");
        let result = GlobalPool.infer_shapes(&[data], &mut sym_gen).unwrap();
        assert_eq!(result[0], sym_shape!("batch", "in_c", 1, 1,));
    }
}
