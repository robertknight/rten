use rten_tensor::{AsView, Layout, NdTensorView, SliceRange, Tensor, TensorView};

use crate::{
    buffer_pool::{AutoReturn, BufferPool},
    infer_shapes::{InferShapes, UnaryOp},
    operator::{
        IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
        OutputTypesContext,
    },
    ops::{
        binary_elementwise::{add, mul, sub},
        concat, gather,
    },
};

fn rotary_embedding(
    pool: &BufferPool,
    input: TensorView<f32>,
    cos: TensorView<f32>,
    sin: TensorView<f32>,
    position_ids: Option<NdTensorView<i32, 2>>,
    interleaved: bool,
    num_heads: usize,
    rotary_embedding_dim: usize,
) -> Result<Tensor, OpError> {
    let reshaped_input = match input.shape() {
        &[batch, seq_len, hidden_size] => {
            if num_heads == 0 {
                return Err(OpError::InvalidValue(
                    "num_heads must not be 0 for 3 dimensioned input",
                ));
            }
            if hidden_size % num_heads != 0 {
                // The reference implementation also adds "or rank-3 input", but not sure fully
                // what that means in this context so excluded it = after all input is rank-3
                // here.
                //
                // Note without this check this becomes a panic as the resize will fail - maybe
                // acceptable?
                return Err(OpError::InvalidValue(
                    "hidden_size must be divisible by num_heads",
                ));
            }

            let head_size = hidden_size / num_heads;
            input.reshaped([batch, seq_len, num_heads, head_size])
        }
        [_batch, _num_heads, _seq_len, _head_size] => {
            input.nd_view().permuted([0, 2, 1, 3]).as_cow()
        }
        _ => {
            return Err(OpError::IncompatibleInputShapes(
                "Input processed needs 3-4 dimensions",
            ));
        }
    };

    let head_size = reshaped_input.shape()[3];
    let rotary_embedding_dim = if rotary_embedding_dim == 0 {
        head_size
    } else {
        rotary_embedding_dim
    };
    if rotary_embedding_dim == 0 || rotary_embedding_dim % 2 != 0 {
        return Err(OpError::InvalidValue(
            "rotary_embedding_dim must be a positive even number",
        ));
    }
    if rotary_embedding_dim > head_size {
        return Err(OpError::InvalidValue(
            "rotary_embedding_dim must not exceed head size",
        ));
    }

    let x_rotate = reshaped_input.slice((.., .., .., ..rotary_embedding_dim));
    let x_not_rotate = reshaped_input.slice((.., .., .., rotary_embedding_dim..));

    let rotary_embedding_dim_half = rotary_embedding_dim / 2;

    let (cos_cache, sin_cache) = if let Some(position_ids) = position_ids {
        let cos_subset = gather(pool, cos, 0, position_ids.as_dyn())?.into_cow();
        let sin_subset = gather(pool, sin, 0, position_ids.as_dyn())?.into_cow();
        (cos_subset, sin_subset)
    } else {
        (cos.as_cow(), sin.as_cow())
    };

    if cos_cache.shape().last() != Some(&rotary_embedding_dim_half) {
        return Err(OpError::InvalidValue(
            "Last dimension of cos cache does not match rotary_embedding_dim/2",
        ));
    }

    if sin_cache.shape().last() != Some(&rotary_embedding_dim_half) {
        return Err(OpError::InvalidValue(
            "Last dimension of sin cache does not match rotary_embedding_dim/2",
        ));
    }

    let cos_cache = cos_cache.view().with_new_axis(2);
    let sin_cache = sin_cache.view().with_new_axis(2);

    let (x1, x2) = if interleaved {
        let x1 = x_rotate.slice((.., .., .., SliceRange::new(0, None, 2)));
        let x2 = x_rotate.slice((.., .., .., SliceRange::new(1, None, 2)));
        (x1, x2)
    } else {
        x_rotate.split_at(3, rotary_embedding_dim_half)
    };

    let cos_x1 = mul(pool, cos_cache.view(), x1.as_dyn())?.auto_return(pool);
    let sin_x2 = mul(pool, sin_cache.view(), x2.as_dyn())?.auto_return(pool);
    let real = sub(pool, cos_x1.view(), sin_x2.view())?.auto_return(pool);

    let sin_x1 = mul(pool, sin_cache.view(), x1.as_dyn())?.auto_return(pool);
    let cos_x2 = mul(pool, cos_cache.view(), x2.as_dyn())?.auto_return(pool);
    let imag = add(pool, sin_x1.view(), cos_x2.view())?.auto_return(pool);

    let x_rotate = if interleaved {
        let insert_axis = real.ndim();
        let real = real.view().with_new_axis(insert_axis);
        let imag = imag.view().with_new_axis(insert_axis);

        let mut x_rotate_concat = concat(pool, &[real, imag], -1)?;
        x_rotate_concat.reshape(&x_rotate.shape());
        x_rotate_concat
    } else {
        concat(pool, &[real.view(), imag.view()], -1)?
    }
    .auto_return(pool);

    let mut output = concat(pool, &[x_rotate.view(), x_not_rotate.as_dyn()], -1)?;

    if input.ndim() == 3 {
        output.reshape(input.shape());
    } else {
        output.permute(&[0, 2, 1, 3])
    };

    Ok(output)
}

#[derive(Debug)]
pub struct RotaryEmbedding {
    pub interleaved: bool,
    pub num_heads: usize,
    pub rotary_embedding_dim: usize,
}

impl Operator for RotaryEmbedding {
    fn name(&self) -> &str {
        "RotaryEmbedding"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(4)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input = ctx.inputs().require_as(0)?;
        let cos = ctx.inputs().require_as(1)?;
        let sin = ctx.inputs().require_as(2)?;
        let position_ids = ctx.inputs().get_as(3)?;

        let output = rotary_embedding(
            ctx.pool(),
            input,
            cos,
            sin,
            position_ids,
            self.interleaved,
            self.num_heads,
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
        let position_ids = match position_ids.ndim() {
            1 => position_ids.with_new_axis(0).nd_view(),
            2 => position_ids.nd_view(),
            _ => {
                return Err(OpError::InvalidValue("position_ids must have 1 or 2 dims"));
            }
        };
        let cos: TensorView<f32> = inputs.require_as(2)?;
        let sin = inputs.require_as(3)?;

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
    use crate::{
        BufferPool,
        operator::{InputList, OperatorExt},
    };

    use super::*;
    use rten_tensor::{Tensor, test_util::expect_equal_with_tolerance};
    use rten_testing::TestCases;

    // Test rotary embedding using ported test cases from
    // https://github.com/microsoft/onnxruntime/blob/e3c34da40639669f3dbb7ae95db0662afbec8cc9/onnxruntime/test/providers/cpu/llm/rotary_embedding_op_test.cc#L509
    #[test]
    fn test_rotary_embedding() {
        #[derive(Debug)]
        struct Case {
            input: Tensor<f32>,
            position_ids: Option<Tensor<i32>>,
            cos_cache: Tensor<f32>,
            sin_cache: Tensor<f32>,
            expected: Tensor<f32>,
            op: RotaryEmbedding,
        }

        let cases = [
            // RotaryEmbedding_Interleaved_SmallData_LlamaMSFT_4D_Input
            // Input shape [batch=1, num_heads=2, seq=3, head_size=4]. The
            // nested literal is `[num_heads, seq, head_size]`; `with_new_axis`
            // prepends the batch dim.
            Case {
                input: Tensor::from([
                    [
                        [-1.0408, 0.9166, -1.3042, -1.1097],
                        [-1.2188, 1.1676, -1.0574, -0.1188],
                        [-0.8110, 0.6737, -1.1233, -0.0919],
                    ],
                    [
                        [-0.1320, -0.2751, -0.2350, 0.0937],
                        [-0.7396, -1.2425, -0.1752, 0.6990],
                        [-0.6861, 0.7202, 0.1963, 0.6142],
                    ],
                ])
                .with_new_axis(0),
                position_ids: Some(Tensor::from([[0i32, 1, 2]])),
                cos_cache: Tensor::from([
                    [1.0000, 1.0000],
                    [0.5403, 0.9999],
                    [-0.4161, 0.9998],
                    [-0.9900, 0.9996],
                    [-0.6536, 0.9992],
                    [0.2837, 0.9988],
                    [0.9602, 0.9982],
                    [0.7539, 0.9976],
                ]),
                sin_cache: Tensor::from([
                    [0.0000, 0.0000],
                    [0.8415, 0.0100],
                    [0.9093, 0.0200],
                    [0.1411, 0.0300],
                    [-0.7568, 0.0400],
                    [-0.9589, 0.0500],
                    [-0.2794, 0.0600],
                    [0.6570, 0.0699],
                ]),
                expected: Tensor::from([
                    [
                        [-1.0408, 0.9166, -1.3042, -1.1097],
                        [-1.6411, -0.3948, -1.0561, -0.1294],
                        [-0.2751, -1.0178, -1.1212, -0.1143],
                    ],
                    [
                        [-0.1320, -0.2751, -0.2350, 0.0937],
                        [0.6460, -1.2937, -0.1822, 0.6972],
                        [-0.3694, -0.9235, 0.1840, 0.6180],
                    ],
                ])
                .with_new_axis(0),
                op: RotaryEmbedding {
                    interleaved: true,
                    num_heads: 2,
                    rotary_embedding_dim: 0,
                },
            },
            // RotaryEmbedding_NotInterleaved_SmallData_LlamaMSFT
            // Input shape [batch=1, seq=2, hidden=18] (num_heads=3, head_size=6).
            Case {
                input: Tensor::from([[
                    [
                        -1.0408, 0.9166, -1.3042, -1.1097, -1.2188, 1.1676, 1.0076, -0.7529,
                        -0.2250, -0.4327, -1.5071, -0.4586, -0.8663, -0.2656, 0.1665, 0.7911,
                        -0.9320, -0.8579,
                    ],
                    [
                        -1.0574, -0.1188, -0.9078, 0.3452, -0.5713, -0.2351, -0.8480, 0.5266,
                        -1.2944, -0.0243, -0.2354, -0.7087, -0.9647, -0.0991, -0.2994, -0.0650,
                        -1.5720, -1.3211,
                    ],
                ]]),
                position_ids: Some(Tensor::from([[0i32, 1]])),
                cos_cache: Tensor::from([
                    [1.0000, 1.0000, 1.0000],
                    [0.5403, 0.9989, 1.0000],
                    [-0.4161, 0.9957, 1.0000],
                    [-0.9900, 0.9903, 1.0000],
                ]),
                sin_cache: Tensor::from([
                    [0.0000, 0.0000, 0.0000],
                    [0.8415, 0.0464, 0.0022],
                    [0.9093, 0.0927, 0.0043],
                    [0.1411, 0.1388, 0.0065],
                ]),
                expected: Tensor::from([[
                    [
                        -1.0408, 0.9166, -1.3042, -1.1097, -1.2188, 1.1676, 1.0076, -0.7529,
                        -0.2250, -0.4327, -1.5071, -0.4586, -0.8663, -0.2656, 0.1665, 0.7911,
                        -0.9320, -0.8579,
                    ],
                    [
                        -0.8618, -0.0922, -0.9073, -0.7032, -0.5762, -0.2371, -0.4377, 0.5370,
                        -1.2929, -0.7267, -0.2107, -0.7115, -0.4666, -0.0261, -0.2965, -0.8469,
                        -1.5749, -1.3217,
                    ],
                ]]),
                op: RotaryEmbedding {
                    interleaved: false,
                    num_heads: 3,
                    rotary_embedding_dim: 0,
                },
            },
            // RotaryEmbedding_CustomRotaryDim_SmallData_Phi
            // Input shape [batch=1, seq=2, hidden=6] (num_heads=1, head_size=6).
            Case {
                input: Tensor::from([[
                    [-1.0408, 0.9166, -1.3042, -1.1097, -1.2188, 1.1676],
                    [1.0076, -0.7529, -0.2250, -0.4327, -1.5071, -0.4586],
                ]]),
                position_ids: Some(Tensor::from([[0i32, 1]])),
                cos_cache: Tensor::from([[1.0000, 1.0000], [1.0000, 0.5403]]),
                sin_cache: Tensor::from([[0.0000, 0.0000], [0.0000, 0.8415]]),
                expected: Tensor::from([[
                    [-1.0408, 0.9166, -1.3042, -1.1097, -1.2188, 1.1676],
                    [1.0076, -0.0427, -0.2250, -0.8673, -1.5071, -0.4586],
                ]]),
                op: RotaryEmbedding {
                    interleaved: false,
                    num_heads: 1,
                    rotary_embedding_dim: 4,
                },
            },
            // RotaryEmbedding_NotInterleaved_NoPosIds_SmallData_LlamaMSFT
            // Input shape [batch=1, seq=2, hidden=18]; cache [batch=1, seq=2, dim/2=3].
            Case {
                input: Tensor::from([[
                    [
                        -1.0408, 0.9166, -1.3042, -1.1097, -1.2188, 1.1676, 1.0076, -0.7529,
                        -0.2250, -0.4327, -1.5071, -0.4586, -0.8663, -0.2656, 0.1665, 0.7911,
                        -0.9320, -0.8579,
                    ],
                    [
                        -1.0574, -0.1188, -0.9078, 0.3452, -0.5713, -0.2351, -0.8480, 0.5266,
                        -1.2944, -0.0243, -0.2354, -0.7087, -0.9647, -0.0991, -0.2994, -0.0650,
                        -1.5720, -1.3211,
                    ],
                ]]),
                position_ids: None,
                cos_cache: Tensor::from([[[1.0000, 1.0000, 1.0000], [0.5403, 0.9989, 1.0000]]]),
                sin_cache: Tensor::from([[[0.0000, 0.0000, 0.0000], [0.8415, 0.0464, 0.0022]]]),
                expected: Tensor::from([[
                    [
                        -1.0408, 0.9166, -1.3042, -1.1097, -1.2188, 1.1676, 1.0076, -0.7529,
                        -0.2250, -0.4327, -1.5071, -0.4586, -0.8663, -0.2656, 0.1665, 0.7911,
                        -0.9320, -0.8579,
                    ],
                    [
                        -0.8618, -0.0922, -0.9073, -0.7032, -0.5762, -0.2371, -0.4377, 0.5370,
                        -1.2929, -0.7267, -0.2107, -0.7115, -0.4666, -0.0261, -0.2965, -0.8469,
                        -1.5749, -1.3217,
                    ],
                ]]),
                op: RotaryEmbedding {
                    interleaved: false,
                    num_heads: 3,
                    rotary_embedding_dim: 0,
                },
            },
            // RotaryEmbedding_Interleaved_NoPosIds_SmallData_LlamaMSFT
            // Input shape [batch=1, seq=3, hidden=8] (num_heads=2, head_size=4).
            Case {
                input: Tensor::from([[
                    [
                        -1.0408, 0.9166, -1.3042, -1.1097, -0.1320, -0.2751, -0.2350, 0.0937,
                    ],
                    [
                        -1.2188, 1.1676, -1.0574, -0.1188, -0.7396, -1.2425, -0.1752, 0.6990,
                    ],
                    [
                        -0.8110, 0.6737, -1.1233, -0.0919, -0.6861, 0.7202, 0.1963, 0.6142,
                    ],
                ]]),
                position_ids: None,
                cos_cache: Tensor::from([[[1.0000, 1.0000], [0.5403, 0.9999], [-0.4161, 0.9998]]]),
                sin_cache: Tensor::from([[[0.0000, 0.0000], [0.8415, 0.0100], [0.9093, 0.0200]]]),
                expected: Tensor::from([[
                    [
                        -1.0408, 0.9166, -1.3042, -1.1097, -0.1320, -0.2751, -0.2350, 0.0937,
                    ],
                    [
                        -1.6411, -0.3948, -1.0561, -0.1294, 0.6460, -1.2937, -0.1822, 0.6972,
                    ],
                    [
                        -0.2751, -1.0178, -1.1212, -0.1143, -0.3694, -0.9235, 0.1840, 0.6180,
                    ],
                ]]),
                op: RotaryEmbedding {
                    interleaved: true,
                    num_heads: 2,
                    rotary_embedding_dim: 0,
                },
            },
        ];

        cases.test_each(|case| {
            let Case {
                input,
                position_ids,
                cos_cache,
                sin_cache,
                expected,
                op,
            } = case;

            let result: Tensor<f32> = if let Some(pos_ids) = position_ids.as_ref() {
                op.run_simple((
                    input.view(),
                    cos_cache.view(),
                    sin_cache.view(),
                    pos_ids.view(),
                ))
            } else {
                op.run_simple((input.view(), cos_cache.view(), sin_cache.view()))
            }
            .unwrap();

            expect_equal_with_tolerance(&expected.view(), &result.view(), 1e-4, 0.0).unwrap();
        });
    }

    #[test]
    fn test_reject_indivisible_hidden_size() {
        let op = RotaryEmbedding {
            interleaved: false,
            num_heads: 2,
            rotary_embedding_dim: 0,
        };

        let input_data = Tensor::from([[0., 0., 0., 0., 0.]]).with_new_axis(0);
        let cos_cache = Tensor::from([[[1.0]]]);
        let sin_cache = Tensor::from([[[0.0]]]);
        let mut input_list = InputList::new();
        input_list.push(input_data.view());
        input_list.push(cos_cache.view());
        input_list.push(sin_cache.view());

        let pool = BufferPool::new();
        let ctx = OpRunContext::new(&pool, &input_list);

        assert!(op.run(&ctx).is_err());
    }

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

    #[test]
    fn test_reject_zero_or_odd_rotary_embedding_dim() {
        for rotary_embedding_dim in [0, 1] {
            let op = RotaryEmbedding {
                interleaved: false,
                num_heads: 1,
                rotary_embedding_dim,
            };

            let input = Tensor::from([[[1.]]]);
            let cos_cache = Tensor::<f32>::zeros(&[1, 0]);
            let sin_cache = Tensor::<f32>::zeros(&[1, 0]);

            let result =
                op.run_simple::<_, Tensor<f32>>((input.view(), cos_cache.view(), sin_cache.view()));

            assert_eq!(
                result,
                Err(OpError::InvalidValue(
                    "rotary_embedding_dim must be a positive even number"
                ))
            );
        }
    }

    // Exercises the Microsoft variant's distinctive paths: input ordering
    // (input, position_ids, cos, sin) and 1D → 2D position_ids reshape. Input
    // data ported from the `RotaryEmbedding_CustomRotaryDim_SmallData_Phi`
    // case above.
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
        let position_ids = Tensor::from([0i32, 1]);
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
}
