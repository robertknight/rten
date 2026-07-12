use rayon::prelude::*;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensorView, Tensor, TensorView};

use crate::{
    buffer_pool::{AutoReturn, BufferPool},
    infer_shapes::{InferShapes, UnaryOp},
    operator::{
        IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
        OutputTypesContext,
    },
    ops::gather,
};

/// Validate a cos/sin cache and return its `(batch, seq_len)` dimensions.
///
/// The cache must have shape `[batch, seq_len, rotary_embedding_dim / 2]`, where
/// the batch and sequence dimensions may each be 1 to broadcast across the
/// corresponding input dimension.
fn rotary_cache_dims(
    cache_shape: &[usize],
    batch: usize,
    seq_len: usize,
    half: usize,
    bad_last_dim: &'static str,
) -> Result<(usize, usize), OpError> {
    let &[cache_batch, cache_seq, cache_half] = cache_shape else {
        return Err(OpError::InvalidValue("cos/sin cache must be a 3D tensor"));
    };
    if cache_half != half {
        return Err(OpError::InvalidValue(bad_last_dim));
    }
    if cache_seq != 1 && cache_seq != seq_len {
        return Err(OpError::InvalidValue(
            "cos/sin cache sequence length must be 1 or match the input",
        ));
    }
    if cache_batch != 1 && cache_batch != batch {
        return Err(OpError::InvalidValue(
            "cos/sin cache batch size must be 1 or match the input",
        ));
    }
    Ok((cache_batch, cache_seq))
}

pub(crate) fn rotary_embedding(
    pool: &BufferPool,
    input: TensorView<f32>,
    cos: TensorView<f32>,
    sin: TensorView<f32>,
    position_ids: Option<NdTensorView<i32, 2>>,
    interleaved: bool,
    num_heads: usize,
    rotary_embedding_dim: usize,
) -> Result<Tensor, OpError> {
    // Reshape input to `[batch, seq_len, num_heads, head_size]`.
    let reshaped_input = match input.shape() {
        &[batch, seq_len, hidden_size] => {
            if num_heads == 0 {
                return Err(OpError::InvalidValue(
                    "num_heads must not be 0 for 3 dimensioned input",
                ));
            }
            if hidden_size % num_heads != 0 {
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
    let [batch, seq_len, num_heads, head_size] = reshaped_input.shape();

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
    let half = rotary_embedding_dim / 2;

    // Resolve the cos/sin caches to a `[batch, seq_len, half]` layout. When
    // `position_ids` is provided the caches are gathered by position, otherwise
    // they are indexed by `(batch, seq_len)` directly.
    let (cos_cache, sin_cache) = if let Some(position_ids) = position_ids {
        let cos = gather(pool, cos, 0, position_ids.as_dyn())?.into_cow();
        let sin = gather(pool, sin, 0, position_ids.as_dyn())?.into_cow();
        (cos, sin)
    } else {
        (cos.as_cow(), sin.as_cow())
    };

    let (cos_batch, cos_seq) = rotary_cache_dims(
        cos_cache.shape(),
        batch,
        seq_len,
        half,
        "Last dimension of cos cache does not match rotary_embedding_dim/2",
    )?;
    let (sin_batch, sin_seq) = rotary_cache_dims(
        sin_cache.shape(),
        batch,
        seq_len,
        half,
        "Last dimension of sin cache does not match rotary_embedding_dim/2",
    )?;

    // Make the inputs contiguous so each head vector and cache row is a
    // contiguous slice that the kernel can index directly.
    let input_contig = reshaped_input.to_contiguous_in(pool).auto_return(pool);
    let cos_contig = cos_cache.to_contiguous_in(pool).auto_return(pool);
    let sin_contig = sin_cache.to_contiguous_in(pool).auto_return(pool);
    let in_data = input_contig.data();
    let cos_data = cos_contig.data();
    let sin_data = sin_contig.data();

    let n_rows = batch * seq_len * num_heads;
    let out_len = n_rows * head_size;
    let mut out_data = pool.alloc::<f32>(out_len);

    // For each `(batch, seq, head)` row, apply the rotation to the first
    // `rotary_embedding_dim` elements and copy the remainder.
    let out_uninit = &mut out_data.spare_capacity_mut()[..out_len];
    in_data
        .par_chunks(head_size)
        .zip(out_uninit.par_chunks_mut(head_size))
        .enumerate()
        .for_each(|(row, (x, y))| {
            let bs = row / num_heads;
            let b = bs / seq_len;
            let s = bs % seq_len;

            let broadcast_idx = |dim_size: usize, index: usize| {
                if dim_size == 1 { 0 } else { index }
            };

            let cos_off =
                (broadcast_idx(cos_batch, b) * cos_seq + broadcast_idx(cos_seq, s)) * half;
            let sin_off =
                (broadcast_idx(sin_batch, b) * sin_seq + broadcast_idx(sin_seq, s)) * half;
            let cos_row = &cos_data[cos_off..cos_off + half];
            let sin_row = &sin_data[sin_off..sin_off + half];

            if interleaved {
                for i in 0..half {
                    let (x1, x2) = (x[2 * i], x[2 * i + 1]);
                    let (cos_i, sin_i) = (cos_row[i], sin_row[i]);
                    y[2 * i].write(x1 * cos_i - x2 * sin_i);
                    y[2 * i + 1].write(x1 * sin_i + x2 * cos_i);
                }
            } else {
                for i in 0..half {
                    let (x1, x2) = (x[i], x[half + i]);
                    let (cos_i, sin_i) = (cos_row[i], sin_row[i]);
                    y[i].write(x1 * cos_i - x2 * sin_i);
                    y[half + i].write(x1 * sin_i + x2 * cos_i);
                }
            }

            // Copy the elements that are not rotated.
            for (yj, &xj) in y[rotary_embedding_dim..]
                .iter_mut()
                .zip(&x[rotary_embedding_dim..])
            {
                yj.write(xj);
            }
        });

    // Safety: every element of `out_data[..out_len]` was initialized above.
    unsafe { out_data.set_len(out_len) };

    let mut output = Tensor::from_data(&[batch, seq_len, num_heads, head_size], out_data);
    if input.ndim() == 3 {
        output.reshape(input.shape());
    } else {
        output.permute(&[0, 2, 1, 3]);
    }

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

#[cfg(feature = "onnx_format")]
pub use contrib::RotaryEmbeddingMicrosoft;

#[cfg(feature = "onnx_format")]
mod contrib;

#[cfg(test)]
mod tests {
    use rten_base::bit_set::BitSet;
    use rten_tensor::{Tensor, test_util::expect_equal_with_tolerance};
    use rten_testing::TestCases;

    use crate::{
        BufferPool,
        operator::{InputList, OperatorExt},
    };

    use super::*;

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

    // Exercises batch sizes > 1.
    #[test]
    fn test_rotary_embedding_batched() {
        #[derive(Debug)]
        struct Case {
            input: Tensor<f32>,
            cos_cache: Tensor<f32>,
            sin_cache: Tensor<f32>,
            expected: Tensor<f32>,
        }

        // All cases use `num_heads=1`, `head_size=2` and non-interleaved rotation,
        // so with `half=1` the rotation reduces to:
        //   y[0] = x[0]*cos - x[1]*sin
        //   y[1] = x[0]*sin + x[1]*cos
        let cases = [
            // Distinct cache row per batch (cos/sin shape [batch=2, seq=1, 1]).
            // Batch 0 uses cos=1,sin=0 (identity); batch 1 uses cos=0,sin=1
            // (90-degree rotation).
            Case {
                input: Tensor::from([[[1.0, 2.0]], [[3.0, 4.0]]]),
                cos_cache: Tensor::from([[[1.0]], [[0.0]]]),
                sin_cache: Tensor::from([[[0.0]], [[1.0]]]),
                expected: Tensor::from([[[1.0, 2.0]], [[-4.0, 3.0]]]),
            },
            // Single-batch cache (shape [1, seq=1, 1]) broadcast across batch=2.
            // Both batches use cos=0,sin=1.
            Case {
                input: Tensor::from([[[1.0, 2.0]], [[3.0, 4.0]]]),
                cos_cache: Tensor::from([[[0.0]]]),
                sin_cache: Tensor::from([[[1.0]]]),
                expected: Tensor::from([[[-2.0, 1.0]], [[-4.0, 3.0]]]),
            },
        ];

        cases.test_each(|case| {
            let Case {
                input,
                cos_cache,
                sin_cache,
                expected,
            } = case;

            let op = RotaryEmbedding {
                interleaved: false,
                num_heads: 1,
                rotary_embedding_dim: 0,
            };
            let result: Tensor<f32> = op
                .run_simple((input.view(), cos_cache.view(), sin_cache.view()))
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
        let ctx = OpRunContext::new(&pool, &input_list, BitSet::ones(1));

        assert!(op.run(&ctx).is_err());
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
}
