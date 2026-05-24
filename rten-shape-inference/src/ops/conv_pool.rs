use smallvec::SmallVec;

use crate::infer_shapes::{InferShapes, InferShapesError};
use crate::sym_expr::SymExpr;
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::SymTensor;

/// Return the output size for a spatial dimension in a convolution or pooling
/// operation.
///
/// The formulae are given in the ONNX docs for convolution and pooling
/// operators, but expressed more clearly in the [PyTorch
/// docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html).
///
/// The ONNX-spec formula for Fixed padding is `(padded - dil*(k-1) - 1) /
/// stride + 1`, where `/` is truncating integer division. We emit
/// `DivCeil(padded - dil*(k-1), stride)` instead. The two agree whenever
/// `padded - dil*(k-1) >= 1` — the only case where the kernel fits the input.
/// They diverge at the boundary `padded - dil*(k-1) == 0`: DivCeil yields the
/// correct zero valid kernel positions, while the ONNX formula yields a
/// spurious one. Emitting `DivCeil` also keeps the simplifier-friendly form
/// rather than relying on pattern recognition to undo the C-style emulation.
fn output_size(
    in_size: SymExpr,
    kernel_size: SymExpr,
    stride: usize,
    dilation: usize,
    padding: DimPadding,
) -> SymExpr {
    let stride = SymExpr::from(stride as i32);

    match padding {
        DimPadding::Fixed {
            start: pad_start,
            end: pad_end,
        } => {
            let dilation = SymExpr::from(dilation as i32);
            let one = SymExpr::from(1);
            let padded_in_size =
                in_size + SymExpr::from(pad_start as i32) + SymExpr::from(pad_end as i32);
            (padded_in_size - dilation * (kernel_size - one)).div_ceil(&stride)
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

/// Return the output size for a spatial dimension in a transposed
/// convolution.
///
/// See the output_shape formula in
/// <https://onnx.ai/onnx/operators/onnx__ConvTranspose.html>.
fn conv_transpose_output_size(
    in_size: SymExpr,
    kernel_size: SymExpr,
    stride: usize,
    dilation: usize,
    output_padding: usize,
    padding: DimPadding,
) -> SymExpr {
    let stride = SymExpr::from(stride as i32);
    match padding {
        DimPadding::Same => in_size * stride,
        DimPadding::Fixed {
            start: pad_start,
            end: pad_end,
        } => {
            let one = SymExpr::from(1);
            let dilated_kernel =
                (kernel_size - one.clone()) * SymExpr::from(dilation as i32) + one.clone();
            (in_size - one) * stride + SymExpr::from(output_padding as i32) + dilated_kernel
                - SymExpr::from(pad_start as i32)
                - SymExpr::from(pad_end as i32)
        }
    }
}

/// ConvTranspose operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__ConvTranspose.html>.
pub struct ConvTranspose<'a> {
    pub groups: usize,
    pub padding: Padding<'a>,
    pub strides: &'a [usize],
    pub dilations: &'a [usize],
    pub output_padding: Option<&'a [usize]>,
}

impl InferShapes for ConvTranspose<'_> {
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
        if weight_dims.len() != data_dims.len() {
            return Err(InferShapesError::IncorrectRank);
        }

        let data_shape: SmallVec<[_; 4]> = data_dims.collect();
        let weight_shape: SmallVec<[_; 4]> = weight_dims.collect();
        let spatial_dims = data_shape.len() - 2;

        // Output channels = weights.size(1) * groups. Weights are laid out as
        // (in_channels, out_channels / groups, ...spatial).
        let out_channels = weight_shape[1].clone() * SymExpr::from(self.groups as i32);

        let mut out_shape = Vec::with_capacity(data_shape.len());
        out_shape.push(data_shape[0].clone());
        out_shape.push(out_channels);

        for d in 0..spatial_dims {
            let pad = self
                .padding
                .dim(d, spatial_dims)
                .ok_or(InferShapesError::InvalidValue)?;
            let stride = *self.strides.get(d).ok_or(InferShapesError::InvalidValue)?;
            let dilation = *self
                .dilations
                .get(d)
                .ok_or(InferShapesError::InvalidValue)?;
            let out_pad = match self.output_padding {
                Some(out_pad) => *out_pad.get(d).ok_or(InferShapesError::InvalidValue)?,
                None => 0,
            };

            out_shape.push(conv_transpose_output_size(
                data_shape[2 + d].clone(),
                weight_shape[2 + d].clone(),
                stride,
                dilation,
                out_pad,
                pad,
            ));
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
            SymExpr::from(kernel_h as i32),
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
                SymExpr::from(kernel_w as i32),
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
            .chain(std::iter::repeat_n(SymExpr::from(1), spatial_dims))
            .collect();
        Ok([SymTensor::from_shape(shape)].into())
    }
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::InferShapes;
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_shape};

    use super::{Conv, ConvTranspose, GlobalPool, Padding, Pool};

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
                (SymExpr::from("len") + SymExpr::from(0) + SymExpr::from(2)
                    - SymExpr::from(4) * (SymExpr::from(32) - SymExpr::from(1)))
                    .div_ceil(&SymExpr::from(16)),
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
                SymExpr::from("len").div_ceil(&SymExpr::from(16))
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
                (SymExpr::from("height") + SymExpr::from(0) + SymExpr::from(2)
                    - SymExpr::from(4) * (SymExpr::from(32) - SymExpr::from(1)))
                    .div_ceil(&SymExpr::from(16)),
                (SymExpr::from("width") + SymExpr::from(1) + SymExpr::from(3)
                    - SymExpr::from(5) * (SymExpr::from(32) - SymExpr::from(1)))
                    .div_ceil(&SymExpr::from(32)),
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
                (SymExpr::from("seq") + SymExpr::from(0) + SymExpr::from(2)
                    - SymExpr::from(4) * (SymExpr::from(32) - SymExpr::from(1)))
                    .div_ceil(&SymExpr::from(16)),
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
                SymExpr::from("seq").div_ceil(&SymExpr::from(16)),
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
                (SymExpr::from("height") + SymExpr::from(0) + SymExpr::from(2)
                    - SymExpr::from(4) * (SymExpr::from(32) - SymExpr::from(1)))
                    .div_ceil(&SymExpr::from(16)),
                (SymExpr::from("width") + SymExpr::from(1) + SymExpr::from(3)
                    - SymExpr::from(5) * (SymExpr::from(32) - SymExpr::from(1)))
                    .div_ceil(&SymExpr::from(32)),
            )
        );
    }

    #[test]
    fn test_conv_transpose() {
        let mut sym_gen = SymbolGen::new();

        // 2D transposed conv with stride 1, no padding, no output padding.
        // Expected: out = (in - 1)*1 + 0 + k - 0 - 0 = in + k - 1, so
        // in=4, k=3 -> 6.
        let data = sym_shape!(1, "in_c", 4, 4);
        let weights = sym_shape!("in_c", 16, 3, 3);
        let op = ConvTranspose {
            groups: 1,
            padding: Padding::Fixed(&[0, 0, 0, 0]),
            strides: &[1, 1],
            dilations: &[1, 1],
            output_padding: None,
        };
        let result = op.infer_shapes(&[data, weights], &mut sym_gen).unwrap();
        assert_eq!(result[0].clone().simplify(), sym_shape!(1, 16, 6, 6));

        // 2D transposed conv with output padding.
        // Reference values from `test_conv_transpose_output_size_and_padding`:
        // in=[5,5], k=[3,3], output_padding=[1,0], stride 1, no pad -> [8,7].
        let data = sym_shape!(1, "in_c", 5, 5);
        let weights = sym_shape!("in_c", 16, 3, 3);
        let op = ConvTranspose {
            groups: 1,
            padding: Padding::Fixed(&[0, 0, 0, 0]),
            strides: &[1, 1],
            dilations: &[1, 1],
            output_padding: Some(&[1, 0]),
        };
        let result = op.infer_shapes(&[data, weights], &mut sym_gen).unwrap();
        assert_eq!(result[0].clone().simplify(), sym_shape!(1, 16, 8, 7));

        // 2D transposed conv with Same padding. Expected: out = in * stride.
        let data = sym_shape!("batch", "in_c", "height", "width");
        let weights = sym_shape!("in_c", 16, 3, 3);
        let op = ConvTranspose {
            groups: 1,
            padding: Padding::Same,
            strides: &[2, 2],
            dilations: &[1, 1],
            output_padding: None,
        };
        let result = op.infer_shapes(&[data, weights], &mut sym_gen).unwrap();
        assert_eq!(
            result[0].clone().simplify(),
            sym_shape!(
                "batch",
                16,
                SymExpr::from("height") * SymExpr::from(2),
                SymExpr::from("width") * SymExpr::from(2),
            )
        );

        // Grouped transposed conv: out_channels = weights.size(1) * groups.
        let data = sym_shape!(1, 4, 5, 5);
        let weights = sym_shape!(4, 2, 3, 3);
        let op = ConvTranspose {
            groups: 2,
            padding: Padding::Fixed(&[0, 0, 0, 0]),
            strides: &[1, 1],
            dilations: &[1, 1],
            output_padding: None,
        };
        let result = op.infer_shapes(&[data, weights], &mut sym_gen).unwrap();
        assert_eq!(result[0].clone().simplify(), sym_shape!(1, 4, 7, 7));

        // 1D transposed conv. Expected:
        //   (in-1)*stride + output_padding + ((k-1)*dilation + 1) - pad_start - pad_end
        //   = (4-1)*2 + 0 + 4 - 1 - 1 = 8.
        let data = sym_shape!(1, "in_c", 4);
        let weights = sym_shape!("in_c", 8, 4);
        let op = ConvTranspose {
            groups: 1,
            padding: Padding::Fixed(&[1, 1]),
            strides: &[2],
            dilations: &[1],
            output_padding: None,
        };
        let result = op.infer_shapes(&[data, weights], &mut sym_gen).unwrap();
        assert_eq!(result[0].clone().simplify(), sym_shape!(1, 8, 8));

        // Unknown input shape.
        let result = ConvTranspose {
            groups: 1,
            padding: Padding::Fixed(&[0, 0, 0, 0]),
            strides: &[1, 1],
            dilations: &[1, 1],
            output_padding: None,
        }
        .infer_shapes(
            &[SymTensor::unknown("?"), sym_shape!("in_c", 16, 3, 3)],
            &mut sym_gen,
        )
        .unwrap();
        assert_eq!(result[0], SymTensor::unknown("unknown input shape"));

        // Dilation > 1: with in=5, k=3, dilation=2, no pad, stride=1 the
        // dilated kernel covers (3-1)*2 + 1 = 5 input positions, so
        // out = (5-1)*1 + 0 + 5 - 0 - 0 = 9.
        let data = sym_shape!(1, "in_c", 5, 5);
        let weights = sym_shape!("in_c", 16, 3, 3);
        let op = ConvTranspose {
            groups: 1,
            padding: Padding::Fixed(&[0, 0, 0, 0]),
            strides: &[1, 1],
            dilations: &[2, 2],
            output_padding: None,
        };
        let result = op.infer_shapes(&[data, weights], &mut sym_gen).unwrap();
        assert_eq!(result[0].clone().simplify(), sym_shape!(1, 16, 9, 9));
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
