//! ONNX Runtime contrib embedding operators.

use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, TensorView};

use crate::{
    infer_shapes::{InferShapes, UnaryOp},
    operator::{
        IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
        OutputTypesContext,
    },
};

use super::rotary_embedding;

#[derive(Debug)]
pub struct RotaryEmbeddingMicrosoft {
    pub interleaved: bool,
    pub num_heads: Option<usize>,
    pub rotary_embedding_dim: usize,
}

impl Operator for RotaryEmbeddingMicrosoft {
    fn name(&self) -> &str {
        "com.microsoft.RotaryEmbedding"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(4)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();

        let input: TensorView<f32> = inputs.require_as(0)?;
        let position_ids: TensorView<i32> = inputs.require_as(1)?;
        let cos: TensorView<f32> = inputs.require_as(2)?;
        let sin = inputs.require_as(3)?;

        // The sequence length is needed to expand the offset form of
        // `position_ids` below. The shared implementation re-validates the full
        // input shape.
        let (batch, seq_len) = match *input.shape() {
            [batch, seq_len, _] => (batch, seq_len),
            [batch, _, seq_len, _] => (batch, seq_len),
            _ => {
                return Err(OpError::IncompatibleInputShapes(
                    "Input processed needs 3-4 dimensions",
                ));
            }
        };

        // `position_ids` has two forms, matching ONNX Runtime's contrib op:
        //
        //  - Format 1, shape `(batch, seq_len)`: the explicit position of each
        //    token.
        //  - Format 0, a scalar or 1-element tensor holding an offset `k`:
        //    token `i` uses position `k + i`. This is used during generation,
        //    where only the position of the sequence's first token is passed.
        let offset_positions;
        let position_ids = match (position_ids.ndim(), position_ids.item().copied()) {
            (0 | 1, Some(offset)) => {
                offset_positions =
                    NdTensor::from_fn([batch, seq_len], |[_, pos]| offset + pos as i32);
                offset_positions.view()
            }
            (2, _) => position_ids.nd_view(),
            _ => {
                return Err(OpError::InvalidValue(
                    "position_ids must be a scalar, a 1-element vector or have 2 dims",
                ));
            }
        };

        let num_heads = match input.shape() {
            &[_, _, hidden_size] => match self.num_heads {
                // The ONNX spec requires `num_heads` for rank-3 input. The
                // Microsoft contrib op gives this attribute a default of 0, so
                // infer it from the cos cache before calling the shared
                // implementation. The cache's last dimension is
                // `rotary_embedding_dim / 2`, which only equals `head_size / 2`
                // for full rotation. With partial rotary embeddings the head
                // size cannot be recovered from the cache, so `num_heads` must
                // be provided explicitly.
                Some(0) | None => {
                    if self.rotary_embedding_dim != 0 {
                        return Err(OpError::InvalidValue(
                            "num_heads must be specified when rotary_embedding_dim is set",
                        ));
                    }
                    let head_size_half = cos
                        .shape()
                        .last()
                        .copied()
                        .ok_or(OpError::InvalidValue("cos cache must not be scalar"))?;
                    if head_size_half == 0 {
                        return Err(OpError::InvalidValue(
                            "Last dimension of cos cache must not be 0",
                        ));
                    }
                    let head_size = head_size_half * 2;
                    if hidden_size % head_size != 0 {
                        return Err(OpError::InvalidValue(
                            "hidden_size must be divisible by head size",
                        ));
                    }
                    hidden_size / head_size
                }
                Some(num_heads) => num_heads,
            },
            _ => self.num_heads.unwrap_or_default(),
        };

        let output = rotary_embedding(
            ctx.pool(),
            input,
            cos,
            sin,
            Some(position_ids),
            self.interleaved,
            num_heads,
            self.rotary_embedding_dim,
        )?;

        output.into_op_result()
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(&UnaryOp)
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::{Tensor, test_util::expect_equal_with_tolerance};

    use super::RotaryEmbeddingMicrosoft;
    use crate::operator::{OpError, OperatorExt};

    #[test]
    fn test_infer_num_heads_from_cache_dim() {
        let op = RotaryEmbeddingMicrosoft {
            interleaved: false,
            num_heads: Some(0),
            rotary_embedding_dim: 0,
        };

        let input = Tensor::from([[[1., 2., 3., 4.]]]);
        let position_ids = Tensor::from([[0i32]]);
        let cos_cache = Tensor::from([[1.0, 1.0]]);
        let sin_cache = Tensor::from([[0.0, 0.0]]);

        let result: Tensor<f32> = op
            .run_simple((
                input.view(),
                position_ids.view(),
                cos_cache.view(),
                sin_cache.view(),
            ))
            .unwrap();

        expect_equal_with_tolerance(&input.view(), &result.view(), 1e-4, 0.0).unwrap();
    }

    #[test]
    fn test_reject_inferring_num_heads_with_partial_rotary_dim() {
        // `num_heads` cannot be inferred from the cache when partial rotary
        // embeddings are used, since the cache only encodes
        // `rotary_embedding_dim / 2`, not `head_size / 2`.
        let op = RotaryEmbeddingMicrosoft {
            interleaved: false,
            num_heads: None,
            rotary_embedding_dim: 2,
        };

        let input = Tensor::from([[[1., 2., 3., 4.]]]);
        let position_ids = Tensor::from([[0i32]]);
        let cos_cache = Tensor::from([[1.0]]);
        let sin_cache = Tensor::from([[0.0]]);

        let result = op.run_simple::<_, Tensor<f32>>((
            input.view(),
            position_ids.view(),
            cos_cache.view(),
            sin_cache.view(),
        ));

        assert_eq!(
            result,
            Err(OpError::InvalidValue(
                "num_heads must be specified when rotary_embedding_dim is set"
            ))
        );
    }

    #[test]
    fn test_reject_zero_cache_dim_when_inferring_num_heads() {
        let op = RotaryEmbeddingMicrosoft {
            interleaved: false,
            num_heads: None,
            rotary_embedding_dim: 0,
        };

        let input = Tensor::from([[[1., 2., 3., 4.]]]);
        let position_ids = Tensor::from([[0i32]]);
        let cos_cache = Tensor::<f32>::zeros(&[1, 0]);
        let sin_cache = Tensor::<f32>::zeros(&[1, 0]);

        let result = op.run_simple::<_, Tensor<f32>>((
            input.view(),
            position_ids.view(),
            cos_cache.view(),
            sin_cache.view(),
        ));

        assert_eq!(
            result,
            Err(OpError::InvalidValue(
                "Last dimension of cos cache must not be 0"
            ))
        );
    }

    // Exercises the Microsoft variant's distinctive paths: input ordering
    // (input, position_ids, cos, sin) and explicit `(batch, seq_len)`
    // position_ids. Input data ported from the
    // `RotaryEmbedding_CustomRotaryDim_SmallData_Phi` case in
    // `test_rotary_embedding`.
    #[test]
    fn test_rotary_embedding_microsoft() {
        let op = RotaryEmbeddingMicrosoft {
            interleaved: false,
            num_heads: Some(1),
            rotary_embedding_dim: 4,
        };

        let input = Tensor::from([[
            [-1.0408, 0.9166, -1.3042, -1.1097, -1.2188, 1.1676],
            [1.0076, -0.7529, -0.2250, -0.4327, -1.5071, -0.4586],
        ]]);
        let position_ids = Tensor::from([[0i32, 1]]);
        let cos_cache = Tensor::from([[1.0000, 1.0000], [1.0000, 0.5403]]);
        let sin_cache = Tensor::from([[0.0000, 0.0000], [0.0000, 0.8415]]);
        let expected = Tensor::from([[
            [-1.0408, 0.9166, -1.3042, -1.1097, -1.2188, 1.1676],
            [1.0076, -0.0427, -0.2250, -0.8673, -1.5071, -0.4586],
        ]]);

        let result: Tensor<f32> = op
            .run_simple((
                input.view(),
                position_ids.view(),
                cos_cache.view(),
                sin_cache.view(),
            ))
            .unwrap();

        expect_equal_with_tolerance(&expected.view(), &result.view(), 1e-4, 0.0).unwrap();
    }

    // The Microsoft contrib op accepts `position_ids` either as an explicit
    // `(batch, seq_len)` tensor or as a scalar / 1-element "offset" `k`, in
    // which case token `i` uses position `k + i`. Verify the offset form is
    // equivalent to the corresponding explicit positions, as ONNX Runtime does.
    #[test]
    fn test_rotary_embedding_microsoft_position_offset() {
        let op = RotaryEmbeddingMicrosoft {
            interleaved: false,
            num_heads: Some(1),
            rotary_embedding_dim: 0,
        };

        // Input shape [batch=1, seq_len=3, hidden=2] (head_size=2).
        let input = Tensor::from([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]);
        // Cache rows for positions 0..=4, distinct so the position matters.
        let cos_cache = Tensor::from([[0.1], [0.2], [0.3], [0.4], [0.5]]);
        let sin_cache = Tensor::from([[0.9], [0.8], [0.7], [0.6], [0.5]]);

        let offset = 2i32;
        // The offset form should expand to these explicit positions.
        let explicit = Tensor::from([[2i32, 3, 4]]);
        let expected: Tensor<f32> = op
            .run_simple((
                input.view(),
                explicit.view(),
                cos_cache.view(),
                sin_cache.view(),
            ))
            .unwrap();

        // 1-element vector `[2]` (shape `(1,)`).
        let offset_1d = Tensor::from([offset]);
        let result_1d: Tensor<f32> = op
            .run_simple((
                input.view(),
                offset_1d.view(),
                cos_cache.view(),
                sin_cache.view(),
            ))
            .unwrap();
        expect_equal_with_tolerance(&expected.view(), &result_1d.view(), 1e-5, 0.0).unwrap();

        // Scalar `2` (shape `()`).
        let offset_scalar = Tensor::from_scalar(offset);
        let result_scalar: Tensor<f32> = op
            .run_simple((
                input.view(),
                offset_scalar.view(),
                cos_cache.view(),
                sin_cache.view(),
            ))
            .unwrap();
        expect_equal_with_tolerance(&expected.view(), &result_scalar.view(), 1e-5, 0.0).unwrap();
    }

    #[test]
    fn test_reject_multi_element_1d_position_ids() {
        // A 1D `position_ids` with more than one element is rejected, matching
        // ONNX Runtime. The scalar / 1-element form is an offset, and explicit
        // per-token positions must be 2D.
        let op = RotaryEmbeddingMicrosoft {
            interleaved: false,
            num_heads: Some(1),
            rotary_embedding_dim: 0,
        };
        let input = Tensor::from([[[1.0, 2.0], [3.0, 4.0]]]);
        let position_ids = Tensor::from([0i32, 1]);
        let cos_cache = Tensor::from([[0.1], [0.2]]);
        let sin_cache = Tensor::from([[0.9], [0.8]]);

        let result = op.run_simple::<_, Tensor<f32>>((
            input.view(),
            position_ids.view(),
            cos_cache.view(),
            sin_cache.view(),
        ));
        assert_eq!(
            result,
            Err(OpError::InvalidValue(
                "position_ids must be a scalar, a 1-element vector or have 2 dims"
            ))
        );
    }
}
