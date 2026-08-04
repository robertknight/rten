//! Shape inference for various ONNX operators.
//!
//! See the [ONNX operator reference](https://onnx.ai/onnx/operators/index.html)
//! for operator details.

use crate::infer_shapes::{InferShapes, InferShapesContext, InferShapesError, resolve_axis};
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
mod grid_sample;
mod identity;
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
pub use binary::{Add, Div, Equal, Mul, Sub, Where};
pub use concat::{Concat, Tile};
pub use conv_pool::{Conv, ConvTranspose, GlobalPool, Padding, Pool};
pub use einsum::Einsum;
pub use fft::{DFT, STFT};
pub use gather::{Gather, GatherElements, GatherND};
pub use generate::{ConstantOfShape, OneHot, Range};
pub use grid_sample::GridSample;
pub use identity::Identity;
pub use layout::{
    DepthToSpace, Expand, Flatten, Reshape, Shape, Size, Squeeze, Transpose, Unsqueeze,
};
pub use matmul::{Gemm, MatMul, MatMulNBits};
pub use pad::Pad;
pub use quantize::DynamicQuantizeLinear;
pub use random::{Dropout, Multinomial};
pub use reduce::{NonZero, TopK};
pub use resize::{Resize, Upsample};
pub use rnn::{Direction, GRU, LSTM};
pub use slice::Slice;
pub use split::Split;
pub use unary::Neg;

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

#[cfg(test)]
mod tests {
    use crate::infer_shapes::{InferShapes, InferShapesContext, InferShapesError};
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_shape};

    use super::{FixedShape, NonMaxSuppression, SkipLayerNormalization};

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
}
