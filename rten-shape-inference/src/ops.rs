//! Shape inference for various ONNX operators.
//!
//! See the [ONNX operator reference](https://onnx.ai/onnx/operators/index.html)
//! for operator details.

use crate::infer_shapes::{
    BinaryOp, InferShapes, InferShapesContext, InferShapesError, resolve_axis,
};
use crate::sym_expr::SymExpr;
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::SymTensor;

mod attention;
mod binary;
mod concat;
mod conv_pool;
mod einsum;
mod fft;
mod gather;
mod generate;
mod layout;
mod matmul;
mod pad;
mod quantize;
mod random;
mod reduce;
mod resize;
mod rnn;
mod slice;
mod split;
mod unary;

pub use attention::{Attention, GroupQueryAttention, MultiHeadAttention};
pub use binary::{Add, Div, Equal, Mul, Sub};
pub use concat::{Concat, Tile};
pub use conv_pool::{Conv, ConvTranspose, GlobalPool, Padding, Pool};
pub use einsum::Einsum;
pub use fft::{DFT, STFT};
pub use gather::{Gather, GatherElements, GatherND};
pub use generate::{ConstantOfShape, OneHot, Range};
pub use layout::{
    DepthToSpace, Expand, Flatten, Reshape, Shape, Size, Squeeze, Transpose, Unsqueeze,
};
pub use matmul::{Gemm, MatMul, MatMulNBits};
pub use pad::Pad;
pub use quantize::DynamicQuantizeLinear;
pub use random::{Dropout, Multinomial};
pub use reduce::TopK;
pub use resize::{Resize, Upsample};
pub use rnn::{Direction, GRU, LSTM};
pub use slice::Slice;
pub use split::Split;
pub use unary::Neg;

/// GridSample operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__GridSample.html>.
pub struct GridSample;

impl InferShapes for GridSample {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let data = inputs.require(0)?;
        let grid = inputs.require(1)?;

        let Some(data_dims) = data.shape() else {
            return Ok([SymTensor::unknown("unknown input shape")].into());
        };
        let Some(grid_dims) = grid.shape() else {
            return Ok([SymTensor::unknown("unknown grid shape")].into());
        };

        // data is (N, C, D1, D2) and grid is (N, D1_out, D2_out, ..., r) where
        // D1..Dn are the spatial dims.
        let data_shape: Vec<_> = data_dims.collect();
        let grid_shape: Vec<_> = grid_dims.collect();
        if data_shape.len() < 3 || data_shape.len() != grid_shape.len() {
            return Err(InferShapesError::IncorrectRank);
        }

        // Output is (N, C, D1_out, D2_out, ...).
        let spatial_dims = data_shape.len() - 2;
        let out_shape = data_shape
            .into_iter()
            .take(2) // (N, C)
            .chain(grid_shape.into_iter().skip(1).take(spatial_dims))
            .collect();

        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

/// Identity operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Identity.html>.
///
/// Unlike [`UnaryOp`](crate::UnaryOp), this copies the input's values (when
/// known) to the output, not just its shape.
pub struct Identity;

impl InferShapes for Identity {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let data = inputs.require(0)?;
        Ok([data.clone()].into())
    }
}

/// NonZero operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__NonZero.html>.
pub struct NonZero;

impl InferShapes for NonZero {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let data = inputs.require(0)?;

        // Output is a 2D tensor of shape `(input.ndim(), num_nonzero)`.
        let first_dim = data
            .ndim()
            .map(|n| SymExpr::Value(n as i32))
            .unwrap_or_else(|| sym_gen.gen_positive());
        let out_shape = vec![first_dim, sym_gen.gen_positive()];

        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

/// Operator which produces a tensor of a fixed shape.
pub struct FixedShape<'a> {
    pub shape: &'a [usize],
}

impl InferShapes for FixedShape<'_> {
    fn infer_shapes(
        &self,
        _inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        Ok([SymTensor::from_fixed_shape(self.shape)].into())
    }
}

/// NonMaxSuppression operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__NonMaxSuppression.html>.
pub struct NonMaxSuppression;

impl InferShapes for NonMaxSuppression {
    fn infer_shapes(
        &self,
        _inputs: InferShapesContext,
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        // Output is `(num_selected, 3)`. `num_selected` is data-dependent.
        let out_shape = vec![sym_gen.gen_positive(), SymExpr::Value(3)];
        Ok([SymTensor::from_shape(out_shape)].into())
    }
}

/// SkipLayerNormalization / SkipSimplifiedLayerNormalization operators.
///
/// See <https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.SkipLayerNormalization>.
pub struct SkipLayerNormalization;

impl InferShapes for SkipLayerNormalization {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let data = inputs.require(0)?;

        let shape = if let Some(dims) = data.shape() {
            SymTensor::from_shape(dims.collect())
        } else {
            SymTensor::unknown("unknown input shape")
        };

        // Outputs are `output`, `mean`, `inv_std_var` and `input_skip_bias_sum`.
        //
        // `output` and `input_skip_bias_sum` have the same shape as the input.
        // `mean` and `inv_std_var` are not computed by the rten implementation
        // and are returned as empty placeholders.
        let placeholder = SymTensor::from_fixed_shape(&[0]);
        Ok([shape.clone(), placeholder.clone(), placeholder, shape].into())
    }
}

/// Where operator.
///
/// See <https://onnx.ai/onnx/operators/onnx__Where.html>.
pub struct Where;

impl InferShapes for Where {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let cond = inputs.require(0)?;
        let x = inputs.require(1)?;
        let y = inputs.require(2)?;

        if let Some(cond_vals) = cond.values()
            && let Some(x_vals) = x.values()
            && let Some(y_vals) = y.values()
        {
            let len = cond_vals.len().max(x_vals.len()).max(y_vals.len());

            let cs = cond_vals.iter().cycle().take(len);
            let xs = x_vals.iter().cycle().take(len);
            let ys = y_vals.iter().cycle().take(len);

            let vals: Option<Vec<SymExpr>> = cs
                .zip(xs.zip(ys))
                .map(|(cond, (x, y))| {
                    let cond_bool = match cond {
                        SymExpr::Value(v) => Some(*v == 1),
                        SymExpr::Var(_)
                        | SymExpr::Neg(_)
                        | SymExpr::Add(..)
                        | SymExpr::Sub(..)
                        | SymExpr::Mul(..)
                        | SymExpr::Div(..)
                        | SymExpr::DivCeil(..)
                        | SymExpr::Max(..)
                        | SymExpr::Min(..)
                        | SymExpr::Broadcast(..) => None,
                    }?;
                    if cond_bool {
                        Some(x.clone())
                    } else {
                        Some(y.clone())
                    }
                })
                .collect();
            if let Some(vals) = vals {
                return Ok([SymTensor::from_vec(vals)].into());
            }
        }

        // Broadcast the first two inputs together, then broadcast the result
        // against the last input.
        let cond_x = BinaryOp
            .infer_shapes([cond.clone(), x.clone()].into(), sym_gen)?
            .remove(0);
        BinaryOp.infer_shapes([cond_x, y.clone()].into(), sym_gen)
    }
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::{InferShapes, InferShapesContext, InferShapesError};
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_shape, sym_vec};

    use super::{
        FixedShape, GridSample, Identity, NonMaxSuppression, NonZero, SkipLayerNormalization, Where,
    };

    #[test]
    fn test_identity() {
        let mut sym_gen = SymbolGen::new();

        let input = sym_vec!("batch", 16, "seq", 24);
        let result = Identity
            .infer_shapes([input.clone()].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result, &[input]);

        let err = Identity
            .infer_shapes(InferShapesContext::new(&[]), &mut sym_gen)
            .err()
            .unwrap();
        assert_eq!(err, InferShapesError::IncorrectInputCount);
    }

    #[test]
    fn test_skip_layer_normalization() {
        let mut sym_gen = SymbolGen::new();

        // `output` and `input_skip_bias_sum` match the input shape, while
        // `mean` and `inv_std_var` are empty placeholders.
        let data = sym_shape!("batch", "seq", 32);
        let result = SkipLayerNormalization
            .infer_shapes([data].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(
            result,
            &[
                sym_shape!("batch", "seq", 32),
                sym_shape!(0),
                sym_shape!(0),
                sym_shape!("batch", "seq", 32),
            ]
        );

        // Unknown input shape.
        let data = SymTensor::unknown("unknown");
        let result = SkipLayerNormalization
            .infer_shapes([data].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].ndim(), None);
        assert_eq!(result[1], sym_shape!(0));
        assert_eq!(result[2], sym_shape!(0));
        assert_eq!(result[3].ndim(), None);

        // Missing input.
        let err = SkipLayerNormalization
            .infer_shapes(InferShapesContext::new(&[]), &mut sym_gen)
            .err();
        assert_eq!(err, Some(InferShapesError::IncorrectInputCount));
    }

    #[test]
    fn test_fixed_shape() {
        let mut sym_gen = SymbolGen::new();
        let op = FixedShape { shape: &[2, 3, 4] };
        let result = op
            .infer_shapes(InferShapesContext::new(&[]), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!(2, 3, 4));

        // Zero-dim shape produces a scalar tensor.
        let op = FixedShape { shape: &[] };
        let result = op
            .infer_shapes(InferShapesContext::new(&[]), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!());
    }

    #[test]
    fn test_grid_sample() {
        let mut sym_gen = SymbolGen::new();

        // 2D sampling: data is (N, C, H, W), grid is (N, H_out, W_out, 2).
        let data = sym_shape!("batch", 3, 224, 224);
        let grid = sym_shape!("batch", 32, 64, 2);
        let result = GridSample
            .infer_shapes([data, grid].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("batch", 3, 32, 64));

        // 1D sampling: data is (N, C, W), grid is (N, W_out, 1).
        let data = sym_shape!("batch", 3, 224);
        let grid = sym_shape!("batch", 32, 1);
        let result = GridSample
            .infer_shapes([data, grid].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!("batch", 3, 32));

        // Unknown input shape.
        let data = SymTensor::unknown("unknown");
        let grid = sym_shape!(1, 32, 64, 2);
        let result = GridSample
            .infer_shapes([data, grid].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0].ndim(), None);

        // Wrong rank.
        let data = sym_shape!(1, 3, 224);
        let grid = sym_shape!(1, 32, 64, 2);
        let err = GridSample
            .infer_shapes([data, grid].into(), &mut sym_gen)
            .unwrap_err();
        assert_eq!(err, InferShapesError::IncorrectRank);
    }

    #[test]
    fn test_non_zero() {
        let mut sym_gen = SymbolGen::new();

        // Known input shape, output is 2D with first dim = ndim.
        let data = sym_shape!("batch", 16, 32);
        let result = NonZero.infer_shapes([data].into(), &mut sym_gen).unwrap();
        let shape: Vec<_> = result[0].shape().unwrap().collect();
        assert_eq!(shape.len(), 2);
        assert_eq!(shape[0], SymExpr::Value(3));
        assert!(matches!(shape[1], SymExpr::Var(_)));

        // Unknown input shape, output is still 2D.
        let data = SymTensor::unknown("unknown");
        let result = NonZero.infer_shapes([data].into(), &mut sym_gen).unwrap();
        let shape: Vec<_> = result[0].shape().unwrap().collect();
        assert_eq!(shape.len(), 2);
    }

    #[test]
    fn test_non_max_suppression() {
        let mut sym_gen = SymbolGen::new();

        // Output is `(num_selected, 3)` with a symbolic first dim.
        let boxes = sym_shape!(1, 100, 4);
        let scores = sym_shape!(1, 80, 100);
        let result = NonMaxSuppression
            .infer_shapes([boxes, scores].into(), &mut sym_gen)
            .unwrap();
        let shape: Vec<_> = result[0].shape().unwrap().collect();
        assert_eq!(shape.len(), 2);
        assert!(matches!(shape[0], SymExpr::Var(_)));
        assert_eq!(shape[1], SymExpr::Value(3));
    }

    #[test]
    fn test_where() {
        let mut sym_gen = SymbolGen::new();

        // Where op with symbolic vectors.
        let cond = sym_vec!(0, 1, 0, 1);
        let x = sym_vec!(1, 2, 3, 4);
        let y = sym_vec!("foo", "bar", "baz", "meep");
        let result = Where
            .infer_shapes([cond, x, y].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_vec!("foo", 2, "baz", 4));

        // Where op with shapes.
        //
        // This broadcasts the three inputs together.
        let cond = sym_shape!(1, 16, 1);
        let x = sym_shape!(8, 16, 1);
        let y = sym_shape!(1, 16, 24);
        let result = Where
            .infer_shapes([cond, x, y].into(), &mut sym_gen)
            .unwrap();
        assert_eq!(result[0], sym_shape!(8, 16, 24));
    }
}
