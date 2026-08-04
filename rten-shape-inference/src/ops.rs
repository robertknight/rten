//! Shape inference for various ONNX operators.
//!
//! See the [ONNX operator reference](https://onnx.ai/onnx/operators/index.html)
//! for operator details.

use crate::infer_shapes::{InferShapes, InferShapesContext, InferShapesError};
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
mod non_max_suppression;
mod norm;
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
pub use non_max_suppression::NonMaxSuppression;
pub use norm::SkipLayerNormalization;
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

#[cfg(test)]
mod tests {
    use crate::infer_shapes::{InferShapes, InferShapesContext};
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_shape};

    use super::FixedShape;

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
}
