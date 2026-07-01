//! Attention-related operations.

use rayon::prelude::*;
use rten_gemm::{GemmExecutor, GemmInputA, GemmInputB, GemmUninitOptions};
use rten_simd::SimdOp;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensorView, Tensor, TensorView};
use rten_vecmath::Softmax;

use crate::buffer_pool::{AutoReturn, BufferPool};
use crate::infer_shapes::InferShapes;
use crate::operator::{
    IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
    OutputTypesContext,
};
use crate::ops::{
    binary_elementwise::broadcast_shapes, layout::expand_to, norm::NanHandling, resolve_axis,
};
use crate::value::Value;

const BROADCAST_ERROR: OpError = OpError::IncompatibleInputShapes("Cannot broadcast inputs");

/// Perform lanewise `Add + Softmax` on tensors `qk` and `m`.
///
/// `m` must be broadcastable to the shape of `qk`.
fn add_softmax_in_place(
    pool: &BufferPool,
    qk: Tensor<f32>,
    m: TensorView<f32>,
    nan_handling: NanHandling,
) -> Result<Tensor, OpError> {
    let axis = resolve_axis(qk.ndim(), -1)?;
    let m = m.try_broadcast(qk.shape()).map_err(|_| BROADCAST_ERROR)?;

    // We assume `qk` is likely already contiguous and `m` is likely contiguous
    // along the `axis` dim, so this will be a no-op, but handle the case where
    // they are not.
    let mut qk = if qk.stride(axis) == 1 {
        qk
    } else {
        qk.auto_return(pool).to_tensor_in(pool)
    };

    let flush_nans = match nan_handling {
        NanHandling::KeepNans => false,
        NanHandling::FlushToZero => true,
    };

    qk.lanes_mut(axis)
        .into_par_iter()
        .zip(m.lanes(axis).into_par_iter())
        .for_each(|(mut qk_inner, m_inner)| {
            // OK, as we made the lanes contiguous above.
            let qk_inner = qk_inner.as_slice_mut().unwrap();
            for (qk, m) in qk_inner.iter_mut().zip(m_inner) {
                *qk += m;
            }
            Softmax::new_mut(qk_inner)
                .flush_nans_to_zero(flush_nans)
                .dispatch();
        });

    Ok(qk)
}

/// Operation which fuses Add(QK, M) -> Softmax(axis = -1).
///
/// This sequence is common in attention operations where `QK` is the query-key
/// product and `M` is a mask matrix.
///
/// The fusion takes advantage of the fact that we can perform Add + Softmax
/// on each lane separately, and get better cache efficiency by having
/// the lane already be in a higher cache level when the Softmax step runs.
#[derive(Debug)]
pub struct AddSoftmax {
    /// See `flush_nans_to_zero` on Softmax operator.
    pub flush_nans_to_zero: bool,
}

impl AddSoftmax {
    fn nan_handling(&self) -> NanHandling {
        if self.flush_nans_to_zero {
            NanHandling::FlushToZero
        } else {
            NanHandling::KeepNans
        }
    }
}

impl Operator for AddSoftmax {
    fn name(&self) -> &str {
        "AddSoftmax"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(2)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let x: TensorView = ctx.inputs().require_as(0)?;
        let y: TensorView = ctx.inputs().require_as(1)?;

        let (qk, m) = if x.len() > y.len() { (x, y) } else { (y, x) };

        let out_shape = broadcast_shapes(qk.shape(), m.shape());
        let qk = match out_shape.as_deref() {
            // Create a copy and run this operator in-place, on the assumption
            // that the operator will usually run via `run_in_place`.
            Some(shape) => qk.broadcast(shape).to_tensor_in(ctx.pool()),
            None => {
                return Err(BROADCAST_ERROR);
            }
        };

        add_softmax_in_place(ctx.pool(), qk, m, self.nan_handling()).into_op_result()
    }

    fn is_commutative(&self) -> bool {
        true
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Value, ctx: &OpRunContext) -> Result<Value, OpError> {
        let qk: Tensor = input.try_into()?;
        let m: TensorView = ctx.inputs().require_as(0)?;

        let out_shape = broadcast_shapes(qk.shape(), m.shape());
        let qk = match out_shape.as_deref() {
            // We expect to always use this path, as commutative ops always
            // receive the largest input as the in-place input.
            Some(shape) if shape == qk.shape() => qk,

            // However, the `Add` operation allows for broadcasting _both_
            // inputs to a larger size, in which case fall back to a copy.
            Some(shape) => qk.broadcast(shape).to_tensor_in(ctx.pool()),

            None => {
                return Err(BROADCAST_ERROR);
            }
        };

        add_softmax_in_place(ctx.pool(), qk, m, self.nan_handling()).map(|qk| qk.into())
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        None
    }
}

fn repeat_interleave<T: Copy>(
    pool: &BufferPool,
    mut input: TensorView<T>,
    axis: usize,
    repeats: usize,
) -> Result<Tensor<T>, OpError> {
    if input.ndim() <= axis {
        return Err(OpError::InvalidValue("Input has too few dims"));
    }

    // Insert temporary 1-sized axis and use broadcasting to repeat along
    // that axis.
    //
    // This is effectively a combination of Unsqueeze + Expand + Reshape.
    input.insert_axis(axis + 1);
    let mut target_shape = input.shape().to_vec();
    target_shape[axis + 1] *= repeats;
    let mut expanded = expand_to(pool, input, &target_shape);
    target_shape.remove(axis + 1);
    target_shape[axis] *= repeats;
    expanded.reshape(&target_shape);

    Ok(expanded)
}

/// Repeat elements of a tensor.
///
/// This differs from the `Tile` ONNX operator in that it repeats individual
/// elements rather than a whole axis, eg. `[1, 2]` -> `[1, 1, 2, 2]` rather
/// than `[1, 2]` -> `[1, 2, 1, 2]`.
///
/// This operation has limited value as a fusion by itself, since it doesn't
/// eliminate the expensive step of materializing the expanded tensor, but
/// it acts as a building block for higher-level fusions.
///
/// See https://docs.pytorch.org/docs/stable/generated/torch.repeat_interleave.html.
#[derive(Debug)]
pub struct RepeatInterleave {
    pub axis: usize,
    pub repeats: usize,
}

impl Operator for RepeatInterleave {
    fn name(&self) -> &str {
        "RepeatInterleave"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(1)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input: TensorView<f32> = ctx.inputs().require_as(0)?;
        repeat_interleave(ctx.pool(), input, self.axis, self.repeats).into_op_result()
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        None
    }
}

/// A fusion of `MatMul(Q, RepeatInterleave(K))` where Q and K are 4D tensors
/// and the second dimension of K is repeated.
///
/// This fusion is used in Grouped-query Attention operators, where K represents
/// either the key or value tensor.
#[derive(Debug)]
pub struct GroupedQueryAttentionMatMul {
    /// Number of times to repeat the second dimension of the RHS input.
    pub repeats: usize,
    /// Alpha value for the matmul.
    pub alpha: Option<f32>,
    /// True if the last two dimensions of the RHS input should be transposed.
    pub transpose_rhs: bool,
}

impl Operator for GroupedQueryAttentionMatMul {
    fn name(&self) -> &str {
        "GroupedQueryAttentionMatMul"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(2)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let lhs: NdTensorView<f32, 4> = ctx.inputs().require_as(0)?;
        let mut rhs: NdTensorView<f32, 4> = ctx.inputs().require_as(1)?;
        if self.transpose_rhs {
            rhs.permute([0, 1, 3, 2]);
        }

        let [batch, heads, seq, k] = lhs.shape();
        let [rhs_batch, rhs_heads, rhs_k, rhs_n] = rhs.shape();

        if batch != rhs_batch {
            return Err(OpError::IncompatibleInputShapes("Batch size mismatch"));
        }
        if k != rhs_k {
            return Err(OpError::IncompatibleInputShapes("K size mismatch"));
        }
        if rhs_heads * self.repeats != heads {
            return Err(OpError::IncompatibleInputShapes(
                "Repeated axis size mismatch",
            ));
        }

        let chunk_size = self.repeats * seq * rhs_n;
        let out_size = batch * (heads / self.repeats) * chunk_size;
        let mut out_data = ctx.pool().alloc(out_size);
        let out_uninit = &mut out_data.spare_capacity_mut()[..out_size];

        let gemm = GemmExecutor::default();
        let lhs_mats = lhs.reshaped_in(
            ctx.pool(),
            [batch, heads / self.repeats, self.repeats * seq, k],
        );
        let opts = GemmUninitOptions {
            alpha: self.alpha.unwrap_or(1.0),
            ..Default::default()
        };

        lhs_mats
            .inner_iter::<2>()
            .into_par_iter()
            .zip(rhs.inner_iter::<2>())
            .zip(out_uninit.par_chunks_mut(chunk_size))
            .for_each(|((lhs, rhs), out)| {
                gemm.gemm_uninit(
                    out,
                    GemmInputA::Unpacked(lhs),
                    GemmInputB::Unpacked(rhs),
                    opts.clone(),
                )
                .unwrap();
            });

        // Safety: gemm_uninit initialized the full output data.
        unsafe { out_data.set_len(out_size) };

        Tensor::from_data(&[batch, heads, seq, rhs_n], out_data).into_op_result()
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        None
    }
}

#[cfg(feature = "onnx_format")]
pub use contrib::{GroupQueryAttention, MultiHeadAttention};

#[cfg(feature = "onnx_format")]
mod contrib {
    use rayon::prelude::*;
    use std::mem::MaybeUninit;

    use rten_gemm::{GemmExecutor, GemmInputA, GemmInputB, GemmUninitOptions};
    use rten_shape_inference::ops as shape_ops;
    use rten_simd::SimdOp;
    use rten_tensor::prelude::*;
    use rten_tensor::{CowNdTensor, NdTensor, NdTensorView, NdTensorViewMut, TensorView};
    use rten_vecmath::Softmax;

    use crate::buffer_pool::{AutoReturn, BufferPool};
    use crate::infer_shapes::{InferShapes, impl_infer_shapes};
    use crate::operator::{
        OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList, OutputTypesContext,
    };
    use crate::ops::{binary_elementwise::add, concat::concat, embedding::rotary_embedding};

    use super::BROADCAST_ERROR;

    /// Reshape (batch, seq, hidden) to (batch, num_heads, seq, head).
    fn split_attention_heads<'a>(
        pool: &BufferPool,
        input: CowNdTensor<'a, f32, 3>,
        num_heads: usize,
        head_size: usize,
    ) -> Result<CowNdTensor<'a, f32, 4>, OpError> {
        let [batch_size, seq_len, hidden] = input.shape();
        if hidden != num_heads * head_size {
            return Err(OpError::IncompatibleInputShapes(
                "Hidden size does not match number of attention heads",
            ));
        }

        Ok(input
            .into_shape_in(pool, [batch_size, seq_len, num_heads, head_size])
            .into_permuted([0, 2, 1, 3]))
    }

    /// Compute scaled dot-product attention for a single (batch, head):
    ///
    /// `out = softmax(score_mod(scale · Q Kᵀ)) · V`
    ///
    /// `query` is `[q_seq, head_size]`, `key` is `[kv_seq, head_size]`, `value`
    /// is `[kv_seq, v_head_size]` and `out` is `[q_seq, v_head_size]`.
    /// `score_mod(row, query_index)` modifies a single score row (of length
    /// `kv_seq`).
    pub(super) fn sdpa_head(
        pool: &BufferPool,
        gemm: &GemmExecutor<f32>,
        scale: f32,
        query: NdTensorView<f32, 2>,
        key: NdTensorView<f32, 2>,
        value: NdTensorView<f32, 2>,
        mut out: NdTensorViewMut<MaybeUninit<f32>, 2>,
        score_mod: impl Fn(&mut [f32], usize),
    ) {
        let q_seq = query.size(0);
        let kv_seq = key.size(0);

        // scores = scale · Q Kᵀ
        let mut scores = NdTensor::uninit_in(pool, [q_seq, kv_seq]);
        gemm.gemm_uninit(
            scores.data_mut().unwrap(),
            GemmInputA::Unpacked(query),
            GemmInputB::Unpacked(key.transposed()),
            GemmUninitOptions {
                alpha: scale,
                ..Default::default()
            },
        )
        .unwrap();
        // Safety: `gemm_uninit` initializes every element.
        let mut scores = unsafe { scores.assume_init() };

        for (s, mut row) in scores.lanes_mut(1).enumerate() {
            let row = row.as_slice_mut().unwrap();
            score_mod(row, s);
            Softmax::new_mut(row).flush_nans_to_zero(true).dispatch();
        }

        // out = scores · V
        gemm.gemm_uninit(
            out.data_mut().unwrap(),
            GemmInputA::Unpacked(scores.view()),
            GemmInputB::Unpacked(value),
            GemmUninitOptions::default(),
        )
        .unwrap();
    }

    /// `query` input for MultiHeadAttention which can be either the query tensor
    /// or packed QKV tensors.
    enum MhaQuery<'a> {
        /// Tensor of shape (batch, kv_seq, num_heads, 3, head_dim)
        Packed(NdTensorView<'a, f32, 5>),
        /// Tensor of shape (batch, seq, hidden) where hidden = num_heads * head_dim
        Unpacked(NdTensorView<'a, f32, 3>),
    }

    impl<'a> MhaQuery<'a> {
        fn new(query: TensorView<'a, f32>) -> Result<Self, OpError> {
            match query.ndim() {
                5 => Ok(Self::Packed(query.nd_view())),
                3 => Ok(Self::Unpacked(query.nd_view())),
                _ => Err(OpError::InvalidValue("query must have 3 or 5 dims")),
            }
        }
    }

    /// Fused multi-head attention contrib operator.
    ///
    /// See
    /// <https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftmultiheadattention>.
    #[derive(Debug)]
    pub struct MultiHeadAttention {
        pub mask_filter_value: f32,
        pub num_heads: u32,
        pub scale: Option<f32>,
        pub unidirectional: bool,
    }

    impl Operator for MultiHeadAttention {
        fn name(&self) -> &str {
            "MultiHeadAttention"
        }

        fn max_inputs(&self) -> Option<usize> {
            Some(10)
        }

        fn max_outputs(&self) -> Option<usize> {
            // Spec defines 4 outputs: output, present_key, present_value, qk.
            // The `qk` output is not yet implemented.
            Some(3)
        }

        fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
            // (batch, seq, hidden) or (batch, kv_seq, num_heads, 3, head_size)
            let query: TensorView<f32> = ctx.inputs().require_as(0)?;
            let query = MhaQuery::new(query)?;

            // (batch, kv_seq, hidden). Spec says it can be 4D or 5D as well, but
            // this is not supported.
            let key: Option<NdTensorView<f32, 3>> = ctx.inputs().get_as(1)?;

            // (batch, kv_seq, hidden). Spec says it can be 4D as well, but this is
            // not supported.
            let value: Option<NdTensorView<f32, 3>> = ctx.inputs().get_as(2)?;
            let bias: Option<NdTensorView<f32, 1>> = ctx.inputs().get_as(3)?;

            // (batch, kv_seq). Spec says it can be 1D or 3D, but this is not supported.
            let key_padding_mask: Option<NdTensorView<i32, 2>> = ctx.inputs().get_as(4)?;

            // (batch or 1, num_heads or 1, seq, total_seq)
            let attention_bias: Option<NdTensorView<f32, 4>> = ctx.inputs().get_as(5)?;

            // (batch, num_heads, past_seq, head_size)
            let past_key: Option<NdTensorView<f32, 4>> = ctx.inputs().get_as(6)?;
            // (batch, num_heads, past_seq, head_size)
            let past_value: Option<NdTensorView<f32, 4>> = ctx.inputs().get_as(7)?;

            let past_seq_len: Option<NdTensorView<i32, 0>> = ctx.inputs().get_as(8)?;
            if past_seq_len.is_some() {
                return Err(OpError::UnsupportedValue("past_seq_len is not supported"));
            }

            let cache_indirection: Option<NdTensorView<i32, 3>> = ctx.inputs().get_as(9)?;
            if cache_indirection.is_some() {
                return Err(OpError::UnsupportedValue(
                    "cache_indirection is not supported",
                ));
            }

            let num_heads = self.num_heads as usize;
            if num_heads == 0 {
                return Err(OpError::InvalidValue("num_heads must be positive"));
            }

            let (query, key, value, batch_size, seq_len, head_size, v_head_size, v_hidden) =
                match query {
                    MhaQuery::Packed(query) => {
                        let [batch_size, kv_seq_len, q_num_heads, three, head_size] = query.shape();

                        if key.is_some() {
                            return Err(OpError::InvalidValue(
                                "key must be None when query is packed",
                            ));
                        }
                        if value.is_some() {
                            return Err(OpError::InvalidValue(
                                "value must be None when query is packed",
                            ));
                        }
                        if bias.is_some() {
                            return Err(OpError::InvalidValue(
                                "bias is not supported with packed QKV format",
                            ));
                        }
                        if three != 3 {
                            return Err(OpError::InvalidValue(
                                "4th dimension of packed qkv input must be 3",
                            ));
                        }
                        if q_num_heads != num_heads {
                            return Err(OpError::InvalidValue(
                                "2nd dimension of packed qkv input must be equal to number of attention heads",
                            ));
                        }
                        let q = query.slice((.., .., .., 0, ..));
                        let k = query.slice((.., .., .., 1, ..));
                        let v = query.slice((.., .., .., 2, ..));

                        (
                            // (batch, kv_seq, num_heads, head) => (batch, num_heads, kv_seq, head)
                            q.permuted([0, 2, 1, 3]).as_cow(),
                            k.permuted([0, 2, 1, 3]).as_cow(),
                            v.permuted([0, 2, 1, 3]).as_cow(),
                            batch_size,
                            kv_seq_len,
                            head_size,
                            head_size,
                            num_heads * head_size,
                        )
                    }
                    MhaQuery::Unpacked(query) => {
                        let [batch_size, seq_len, hidden] = query.shape();
                        if hidden % num_heads != 0 {
                            return Err(OpError::IncompatibleInputShapes(
                                "Hidden size must be divisible by number of attention heads",
                            ));
                        }
                        let head_size = hidden / num_heads;

                        let (key, value) = match (key, value) {
                            (None, _) => (query, query), // Reference impl ignores if value is some
                            (Some(key), Some(value)) => (key, value),
                            (Some(_), None) => {
                                return Err(OpError::InvalidValue(
                                    "value input must be set if key input is present",
                                ));
                            }
                        };

                        let [key_batch, key_seq_len, key_hidden] = key.shape();
                        let [value_batch, value_seq_len, v_hidden] = value.shape();
                        if key_batch != batch_size
                            || value_batch != batch_size
                            || value_seq_len != key_seq_len
                        {
                            return Err(OpError::IncompatibleInputShapes(
                                "Key and value batch or sequence lengths do not match",
                            ));
                        }
                        if key_hidden != hidden {
                            return Err(OpError::IncompatibleInputShapes(
                                "Key hidden size does not match query hidden size",
                            ));
                        }
                        if v_hidden % num_heads != 0 {
                            return Err(OpError::IncompatibleInputShapes(
                                "Value hidden size must be divisible by number of attention heads",
                            ));
                        }
                        let v_head_size = v_hidden / num_heads;

                        let (query, key, value) = if let Some(bias) = bias {
                            if bias.shape() != [hidden * 2 + v_hidden] {
                                return Err(OpError::IncompatibleInputShapes(
                                    "Bias shape does not match QKV hidden sizes",
                                ));
                            }
                            let q_bias = bias.slice(..hidden);
                            let k_bias = bias.slice(hidden..(hidden * 2));
                            let v_bias = bias.slice((hidden * 2)..);

                            let query = add(ctx.pool(), query.as_dyn(), q_bias.as_dyn())?
                                .into_rank::<3>()
                                .unwrap();
                            let key = add(ctx.pool(), key.as_dyn(), k_bias.as_dyn())?
                                .into_rank::<3>()
                                .unwrap();
                            let value = add(ctx.pool(), value.as_dyn(), v_bias.as_dyn())?
                                .into_rank::<3>()
                                .unwrap();
                            (
                                split_attention_heads(
                                    ctx.pool(),
                                    query.into_cow(),
                                    num_heads,
                                    head_size,
                                )?,
                                split_attention_heads(
                                    ctx.pool(),
                                    key.into_cow(),
                                    num_heads,
                                    head_size,
                                )?,
                                split_attention_heads(
                                    ctx.pool(),
                                    value.into_cow(),
                                    num_heads,
                                    v_head_size,
                                )?,
                            )
                        } else {
                            (
                                split_attention_heads(
                                    ctx.pool(),
                                    query.as_cow(),
                                    num_heads,
                                    head_size,
                                )?,
                                split_attention_heads(
                                    ctx.pool(),
                                    key.as_cow(),
                                    num_heads,
                                    head_size,
                                )?,
                                split_attention_heads(
                                    ctx.pool(),
                                    value.as_cow(),
                                    num_heads,
                                    v_head_size,
                                )?,
                            )
                        };

                        (
                            query,
                            key,
                            value,
                            batch_size,
                            seq_len,
                            head_size,
                            v_head_size,
                            v_hidden,
                        )
                    }
                };

            let query = query.auto_return(ctx.pool());
            let mut key = key.auto_return(ctx.pool());
            let mut value = value.auto_return(ctx.pool());

            // Concatenate present and past KV
            match (past_key, past_value) {
                (Some(past_key), Some(past_value)) => {
                    let [past_batch, past_heads, _past_seq, past_head_size] = past_key.shape();
                    if past_batch != batch_size
                        || past_heads != num_heads
                        || past_head_size != head_size
                    {
                        return Err(OpError::IncompatibleInputShapes(
                            "past_key shape does not match query/key shape",
                        ));
                    }
                    let [past_batch, past_heads, _past_seq, past_v_head_size] = past_value.shape();
                    if past_batch != batch_size
                        || past_heads != num_heads
                        || past_v_head_size != v_head_size
                    {
                        return Err(OpError::IncompatibleInputShapes(
                            "past_value shape does not match query/value shape",
                        ));
                    }
                    key = concat(ctx.pool(), &[past_key.as_dyn(), key.as_dyn()], 2)?
                        .into_rank::<4>()
                        .unwrap()
                        .into_cow()
                        .auto_return(ctx.pool());
                    value = concat(ctx.pool(), &[past_value.as_dyn(), value.as_dyn()], 2)?
                        .into_rank::<4>()
                        .unwrap()
                        .into_cow()
                        .auto_return(ctx.pool());
                }
                (Some(_), None) | (None, Some(_)) => {
                    return Err(OpError::InvalidValue(
                        "past_key and past_value must either both be present or both be absent",
                    ));
                }
                (None, None) => {}
            }

            // Compute attention output per (batch, head):
            //   out = softmax(scale · Q Kᵀ + bias, masked) · V
            let total_seq_len = key.size(2);
            let scale = self
                .scale
                .unwrap_or_else(|| 1.0 / (head_size as f32).sqrt());

            // Validate attention bias.
            let attention_bias = attention_bias
                .map(|ab| ab.try_broadcast([batch_size, num_heads, seq_len, total_seq_len]))
                .transpose()
                .map_err(|_| BROADCAST_ERROR)?;

            // Validate key padding mask.
            if let Some(key_padding_mask) = key_padding_mask
                && key_padding_mask.shape() != [batch_size, total_seq_len]
            {
                return Err(OpError::IncompatibleInputShapes(
                    "key_padding_mask shape does not match key sequence length",
                ));
            }

            let gemm = GemmExecutor::new();
            let pool = ctx.pool();
            let past_len = total_seq_len - seq_len;
            let mut attn_out =
                NdTensor::uninit_in(pool, [batch_size, num_heads, seq_len, v_head_size]);
            attn_out
                .inner_iter_mut::<2>()
                .into_par_iter()
                .enumerate()
                .for_each(|(i, out)| {
                    let b = i / num_heads;
                    let h = i % num_heads;
                    let q_head = query.slice([b, h]);
                    let k_head = key.slice([b, h]);
                    let v_head = value.slice([b, h]);
                    let head_attn_bias = attention_bias.as_ref().map(|bs| bs.slice([b, h]));

                    sdpa_head(
                        pool,
                        &gemm,
                        scale,
                        q_head,
                        k_head,
                        v_head,
                        out,
                        |row, q_idx| {
                            // Add the broadcast attention bias for this query row.
                            if let Some(bias) = head_attn_bias.as_ref() {
                                let bias = bias.slice(q_idx);
                                row.iter_mut().zip(bias.iter()).for_each(|(x, b)| {
                                    *x += b;
                                });
                            }

                            // Mask future positions for causal attention.
                            if self.unidirectional {
                                row[past_len + q_idx + 1..].fill(self.mask_filter_value);
                            }

                            // Mask padded key positions.
                            if let Some(key_padding_mask) = key_padding_mask {
                                for (key_idx, x) in row.iter_mut().enumerate() {
                                    if key_padding_mask[[b, key_idx]] == 0 {
                                        *x = self.mask_filter_value;
                                    }
                                }
                            }
                        },
                    );
                });

            // Safety: every (seq, v_head_size) block was written by sdpa_head.
            let attn_out = unsafe { attn_out.assume_init() };
            let output = attn_out
                .permuted([0, 2, 1, 3])
                .to_tensor_in(pool)
                .into_shape([batch_size, seq_len, v_hidden]);

            let mut outputs: OutputList = [output.into()].into();
            if ctx.outputs().get(1) || ctx.outputs().get(2) {
                outputs.push(key.take().into_owned().into());
                outputs.push(value.take().into_owned().into());
            }
            Ok(outputs)
        }

        fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
            Some([OutputType::CopyFromInput(0)].into())
        }

        fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
            Some(self)
        }
    }

    impl_infer_shapes!(
        MultiHeadAttention,
        op,
        shape_ops::MultiHeadAttention {
            num_heads: op.num_heads,
        }
    );

    /// Fused grouped-query attention contrib operator.
    ///
    /// See
    /// <https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GroupQueryAttention>.
    #[derive(Debug)]
    pub struct GroupQueryAttention {
        pub num_heads: u32,
        pub kv_num_heads: u32,
        pub scale: Option<f32>,
        pub do_rotary: bool,
        pub rotary_interleaved: bool,
        /// Left window size for local (sliding-window) attention, or `None` if
        /// unused.
        pub local_window_size: Option<u32>,
        pub softcap: f32,
        pub smooth_softmax: bool,
    }

    impl Operator for GroupQueryAttention {
        fn name(&self) -> &str {
            "GroupQueryAttention"
        }

        fn max_inputs(&self) -> Option<usize> {
            // Spec defines up to 16 inputs. Quantization scale and Q/K norm weight
            // inputs (12-15) are not supported.
            Some(12)
        }

        fn max_outputs(&self) -> Option<usize> {
            // Spec defines 4 outputs: output, present_key, present_value, output_qk.
            // The `output_qk` output is not implemented.
            Some(3)
        }

        fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
            let inputs = ctx.inputs();

            // (batch, seq, q_hidden)
            let query: NdTensorView<f32, 3> = inputs.require_as(0)?;
            // (batch, seq, kv_hidden). Packed QKV (key absent) is not supported.
            let key: NdTensorView<f32, 3> = inputs.require_as(1)?;
            let value: NdTensorView<f32, 3> = inputs.require_as(2)?;
            // (batch, kv_num_heads, past_seq, head_size)
            let past_key: Option<NdTensorView<f32, 4>> = inputs.get_as(3)?;
            let past_value: Option<NdTensorView<f32, 4>> = inputs.get_as(4)?;
            // (batch,). Equal to total_sequence_lengths - 1.
            let seqlens_k: NdTensorView<i32, 1> = inputs.require_as(5)?;
            // Scalar. Maximum total sequence length (past + new) across the batch.
            let total_seqlen: NdTensorView<i32, 0> = inputs.require_as(6)?;
            // (max_seq, head_size / 2)
            let cos_cache: Option<NdTensorView<f32, 2>> = inputs.get_as(7)?;
            let sin_cache: Option<NdTensorView<f32, 2>> = inputs.get_as(8)?;
            // (batch, seq)
            let position_ids: Option<NdTensorView<i32, 2>> = inputs.get_as(9)?;
            // (batch or 1, num_heads or 1, seq, total_seq)
            let attention_bias: Option<NdTensorView<f32, 4>> = inputs.get_as(10)?;
            let head_sink: Option<NdTensorView<f32, 1>> = inputs.get_as(11)?;

            if head_sink.is_some() {
                return Err(OpError::UnsupportedValue("head_sink is not supported"));
            }
            if self.smooth_softmax {
                return Err(OpError::UnsupportedValue("smooth_softmax is not supported"));
            }

            let num_heads = self.num_heads as usize;
            let kv_num_heads = self.kv_num_heads as usize;
            if num_heads == 0 || kv_num_heads == 0 {
                return Err(OpError::InvalidValue(
                    "num_heads and kv_num_heads must be positive",
                ));
            }
            if !num_heads.is_multiple_of(kv_num_heads) {
                return Err(OpError::InvalidValue(
                    "num_heads must be a multiple of kv_num_heads",
                ));
            }

            let [batch, seq, q_hidden] = query.shape();
            if !q_hidden.is_multiple_of(num_heads) {
                return Err(OpError::IncompatibleInputShapes(
                    "query hidden size must be divisible by num_heads",
                ));
            }
            let head_size = q_hidden / num_heads;

            let [key_batch, kv_seq, kv_hidden] = key.shape();
            let [value_batch, value_seq, value_hidden] = value.shape();
            if key_batch != batch || value_batch != batch {
                return Err(OpError::IncompatibleInputShapes(
                    "key and value batch size must match query",
                ));
            }
            if kv_seq != value_seq || kv_hidden != value_hidden {
                return Err(OpError::IncompatibleInputShapes(
                    "key and value must have the same shape",
                ));
            }
            if kv_hidden != kv_num_heads * head_size {
                return Err(OpError::IncompatibleInputShapes(
                    "key hidden size must equal kv_num_heads * head_size",
                ));
            }
            // Shared KV buffer mode (kv_seq == 0) and cross attention are not
            // supported. New key/value sequence length must match the query.
            if kv_seq != seq {
                return Err(OpError::UnsupportedValue(
                    "key sequence length must match query sequence length",
                ));
            }

            // Resolve per-batch sequence lengths.
            if seqlens_k.len() != batch {
                return Err(OpError::IncompatibleInputShapes(
                    "seqlens_k must have batch_size elements",
                ));
            }

            let total_sequence_length = total_seqlen.item().copied().unwrap();
            if total_sequence_length <= 0 {
                return Err(OpError::InvalidValue(
                    "total_sequence_length must be positive",
                ));
            }
            let total_sequence_length = total_sequence_length as usize;

            let past_seq = match (past_key, past_value) {
                (Some(past_key), Some(past_value)) => {
                    let [pk_batch, pk_heads, pk_seq, pk_head_size] = past_key.shape();
                    let [pv_batch, pv_heads, pv_seq, pv_head_size] = past_value.shape();
                    if pk_batch != batch
                        || pv_batch != batch
                        || pk_heads != kv_num_heads
                        || pv_heads != kv_num_heads
                        || pk_head_size != head_size
                        || pv_head_size != head_size
                        || pk_seq != pv_seq
                    {
                        return Err(OpError::IncompatibleInputShapes(
                            "past_key/past_value shape does not match",
                        ));
                    }
                    pk_seq
                }
                (None, None) => 0,
                _ => {
                    return Err(OpError::InvalidValue(
                        "past_key and past_value must both be present or both absent",
                    ));
                }
            };

            let present_seq = past_seq + seq;
            let is_first_prompt = seq == total_sequence_length;
            let is_subsequent_prompt = seq > 1 && seq != total_sequence_length;

            if is_subsequent_prompt && batch != 1 {
                return Err(OpError::UnsupportedValue(
                    "batch size must be 1 when sequence_length > 1 and a past context is given",
                ));
            }
            if !is_first_prompt && !is_subsequent_prompt && seq != 1 {
                return Err(OpError::InvalidValue(
                    "sequence_length must be 1 when query is not a prompt",
                ));
            }

            for &len in seqlens_k.iter() {
                if len < 0 || len as usize >= present_seq {
                    return Err(OpError::InvalidValue("seqlens_k entry is out of range"));
                }
                if (len as usize + 1) < seq {
                    return Err(OpError::InvalidValue(
                        "seqlens_k entry is too small for the query sequence length",
                    ));
                }
            }

            // Validate attention bias.
            if let Some(bias) = attention_bias.as_ref() {
                let [bias_batch, bias_heads, bias_seq, bias_total] = bias.shape();
                if (bias_batch != 1 && bias_batch != batch)
                    || (bias_heads != 1 && bias_heads != num_heads)
                    || bias_seq < seq
                    || bias_total < present_seq
                {
                    return Err(OpError::IncompatibleInputShapes(
                        "attention_bias shape is incompatible with query/key shapes",
                    ));
                }
            }

            let past_len = |batch_idx: usize| -> usize {
                if is_first_prompt {
                    0
                } else {
                    (seqlens_k[batch_idx] as usize + 1) - seq
                }
            };

            let scale = self
                .scale
                .unwrap_or_else(|| 1.0 / (head_size as f32).sqrt());

            // Apply rotary embeddings to Q and K
            let (rotary_q, rotary_k) = if self.do_rotary {
                let (Some(cos), Some(sin)) = (cos_cache, sin_cache) else {
                    return Err(OpError::InvalidValue(
                        "cos_cache and sin_cache are required when do_rotary is set",
                    ));
                };
                let rotary_dim = cos.size(1) * 2;

                // Position of each token.
                let pos_ids = if let Some(position_ids) = position_ids {
                    position_ids.as_cow()
                } else {
                    NdTensor::from_fn([batch, seq], |[b, s]| (past_len(b) + s) as i32).into_cow()
                };

                // TODO - These two calls each gather the cos/sin caches with the
                // same `pos_ids`. Gather once and reuse.
                let q = rotary_embedding(
                    ctx.pool(),
                    query.as_dyn(),
                    cos.as_dyn(),
                    sin.as_dyn(),
                    Some(pos_ids.view()),
                    self.rotary_interleaved,
                    num_heads,
                    rotary_dim,
                )?;
                let k = rotary_embedding(
                    ctx.pool(),
                    key.as_dyn(),
                    cos.as_dyn(),
                    sin.as_dyn(),
                    Some(pos_ids.view()),
                    self.rotary_interleaved,
                    kv_num_heads,
                    rotary_dim,
                )?;
                (Some(q), Some(k))
            } else {
                (None, None)
            };

            // `rotary_q` and `rotary_k` have same rank as `query` and `key`
            // respectively (ie. 3).
            let query = rotary_q.as_ref().map(|q| q.nd_view::<3>()).unwrap_or(query);
            let key = rotary_k.as_ref().map(|k| k.nd_view::<3>()).unwrap_or(key);

            // Reshape Q to (batch, num_heads, seq, head_size).
            let query = query.reshaped([batch, seq, num_heads, head_size]);
            let query = query.permuted([0, 2, 1, 3]);

            // Build the present key/value caches in BNSH layout by concatenating the
            // past cache with the new key/value tokens.
            let key = key.reshaped([batch, seq, kv_num_heads, head_size]);
            let value = value.reshaped([batch, seq, kv_num_heads, head_size]);

            // TODO - The copy loop below overwrites the `[0, past_b + seq)` rows of
            // every head, so zero-initializing them here is wasted work. Allocate
            // uninitialized and zero only the unfilled tail rows per batch item.
            let mut present_key =
                NdTensor::zeros_in(ctx.pool(), [batch, kv_num_heads, present_seq, head_size]);
            let mut present_value =
                NdTensor::zeros_in(ctx.pool(), [batch, kv_num_heads, present_seq, head_size]);

            for b in 0..batch {
                let past_b = past_len(b);
                for h in 0..kv_num_heads {
                    if let Some(past_key) = past_key {
                        present_key
                            .slice_mut((b, h, ..past_b))
                            .copy_from(&past_key.slice((b, h, ..past_b)));
                    }
                    if let Some(past_value) = past_value {
                        present_value
                            .slice_mut((b, h, ..past_b))
                            .copy_from(&past_value.slice((b, h, ..past_b)));
                    }
                    present_key
                        .slice_mut((b, h, past_b..past_b + seq))
                        .copy_from(&key.slice((b, .., h)));
                    present_value
                        .slice_mut((b, h, past_b..past_b + seq))
                        .copy_from(&value.slice((b, .., h)));
                }
            }

            // Compute attention output
            let kv_factor = num_heads / kv_num_heads;
            let mut attn_out = NdTensor::uninit_in(ctx.pool(), [batch, num_heads, seq, head_size]);
            let attention_bias = attention_bias.map(|b| b.to_contiguous_in(ctx.pool()));

            let gemm = GemmExecutor::<f32>::new();
            let pool = ctx.pool();
            attn_out
                .inner_iter_mut::<2>()
                .into_par_iter()
                .enumerate()
                .for_each(|(i, out)| {
                    let b = i / num_heads;
                    let h = i % num_heads;
                    let kv_head = h / kv_factor;
                    let kv_len = (seqlens_k[b] + 1) as usize;
                    let causal_past = past_len(b);

                    let q_head = query.slice([b, h]);
                    let k_head = present_key.slice((b, kv_head, ..kv_len));
                    let v_head = present_value.slice((b, kv_head, ..kv_len));
                    let head_bias = attention_bias.as_ref().map(|bias| {
                        let bb = if bias.size(0) == 1 { 0 } else { b };
                        let hh = if bias.size(1) == 1 { 0 } else { h };
                        bias.slice([bb, hh])
                    });

                    sdpa_head(pool, &gemm, scale, q_head, k_head, v_head, out, |row, s| {
                        let bias = head_bias.as_ref().map(|bs| bs.slice(s).data().unwrap());
                        let seq_causal = causal_past + s + 1;
                        let (start, window) = match self.local_window_size {
                            Some(local) if seq_causal > local as usize => {
                                let local = local as usize;
                                (seq_causal - local, local)
                            }
                            _ => (0, seq_causal),
                        };

                        let attended = &mut row[start..start + window];

                        if self.softcap > 0.0 {
                            let softcap = self.softcap;
                            for x in attended.iter_mut() {
                                *x = softcap * (*x / softcap).tanh();
                            }
                        }

                        if let Some(bias) = bias {
                            for (x, b) in attended.iter_mut().zip(&bias[start..start + window]) {
                                *x += b;
                            }
                        }

                        for x in &mut row[..start] {
                            *x = f32::NEG_INFINITY;
                        }
                        for x in &mut row[seq_causal..] {
                            *x = f32::NEG_INFINITY;
                        }
                    });
                });

            // Safety: every (seq, head_size) block was written by the GEMM above.
            let attn_out = unsafe { attn_out.assume_init() };

            // (batch, num_heads, seq, head_size) -> (batch, seq, num_heads * head_size).
            let output = attn_out
                .permuted([0, 2, 1, 3])
                .to_tensor_in(ctx.pool())
                .into_shape([batch, seq, q_hidden]);

            let mut outputs: OutputList = [output.into()].into();
            if ctx.outputs().get(1) || ctx.outputs().get(2) {
                outputs.push(present_key.into());
                outputs.push(present_value.into());
            }
            Ok(outputs)
        }

        fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
            Some(
                [
                    OutputType::CopyFromInput(0),
                    OutputType::CopyFromInput(0),
                    OutputType::CopyFromInput(0),
                ]
                .into_iter()
                .collect(),
            )
        }

        fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
            Some(self)
        }
    }

    impl_infer_shapes!(
        GroupQueryAttention,
        op,
        shape_ops::GroupQueryAttention {
            num_heads: op.num_heads,
            kv_num_heads: op.kv_num_heads,
        }
    );
}

#[cfg(test)]
mod tests {
    use rten_base::bit_set::BitSet;
    use rten_gemm::GemmExecutor;
    use rten_simd::SimdOp;
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::expect_equal;
    use rten_tensor::{NdTensor, Tensor, TensorView};
    use rten_testing::TestCases;
    use rten_vecmath::Softmax as SoftmaxSimd;

    use super::contrib::sdpa_head;
    use super::{
        AddSoftmax, BROADCAST_ERROR, GroupQueryAttention, GroupedQueryAttentionMatMul,
        MultiHeadAttention, RepeatInterleave,
    };
    use crate::buffer_pool::BufferPool;
    use crate::operator::{InputList, OpError, OpRunContext, Operator, OperatorExt};
    use crate::ops::{Add, Softmax};
    use crate::value::ValueView;

    /// Reference implementation of causal grouped-query attention.
    #[allow(clippy::too_many_arguments)]
    fn reference_gqa(
        query: &NdTensor<f32, 3>,
        key: &NdTensor<f32, 3>,
        value: &NdTensor<f32, 3>,
        past_key: Option<&NdTensor<f32, 4>>,
        past_value: Option<&NdTensor<f32, 4>>,
        num_heads: usize,
        kv_num_heads: usize,
        scale: f32,
    ) -> NdTensor<f32, 3> {
        let [batch, seq, q_hidden] = query.shape();
        let head_size = q_hidden / num_heads;
        let past_seq = past_key.map(|p| p.size(2)).unwrap_or(0);
        let total = past_seq + seq;
        let kv_factor = num_heads / kv_num_heads;

        // Build the full key/value caches: (batch, kv_num_heads, total, head_size).
        let gather = |new: &NdTensor<f32, 3>, past: Option<&NdTensor<f32, 4>>| {
            NdTensor::from_fn([batch, kv_num_heads, total, head_size], |[b, h, t, d]| {
                if t < past_seq {
                    past.unwrap()[[b, h, t, d]]
                } else {
                    new[[b, t - past_seq, h * head_size + d]]
                }
            })
        };
        let k_full = gather(key, past_key);
        let v_full = gather(value, past_value);

        let mut out = NdTensor::zeros([batch, seq, q_hidden]);
        for b in 0..batch {
            for n in 0..num_heads {
                let kv_head = n / kv_factor;
                for s in 0..seq {
                    // Causal limit: query at position `past_seq + s` attends to
                    // key positions `0..=past_seq + s`.
                    let limit = past_seq + s + 1;
                    let mut scores = vec![0.0f32; limit];
                    for (t, score) in scores.iter_mut().enumerate() {
                        let mut dot = 0.0;
                        for d in 0..head_size {
                            dot += query[[b, s, n * head_size + d]] * k_full[[b, kv_head, t, d]];
                        }
                        *score = dot * scale;
                    }

                    SoftmaxSimd::new_mut(&mut scores).dispatch();

                    for d in 0..head_size {
                        let mut acc = 0.0;
                        for (t, score) in scores.iter().enumerate() {
                            acc += score * v_full[[b, kv_head, t, d]];
                        }
                        out[[b, s, n * head_size + d]] = acc;
                    }
                }
            }
        }
        out
    }

    /// Run [`GroupQueryAttention`] with the given inputs, returning all outputs.
    fn run_gqa(
        op: &GroupQueryAttention,
        query: &NdTensor<f32, 3>,
        key: &NdTensor<f32, 3>,
        value: &NdTensor<f32, 3>,
        past_key: Option<&NdTensor<f32, 4>>,
        past_value: Option<&NdTensor<f32, 4>>,
        seqlens_k: &NdTensor<i32, 1>,
        total_seqlen: i32,
    ) -> Result<Vec<Tensor>, OpError> {
        let total = NdTensor::from_scalar(total_seqlen);
        let inputs = [
            Some(ValueView::from(query.view())),
            Some(ValueView::from(key.view())),
            Some(ValueView::from(value.view())),
            past_key.map(|p| ValueView::from(p.view())),
            past_value.map(|p| ValueView::from(p.view())),
            Some(ValueView::from(seqlens_k.view())),
            Some(ValueView::from(total.view())),
        ];
        let input_list = InputList::from_optional(&inputs);
        let pool = BufferPool::new();
        let ctx = OpRunContext::new(&pool, &input_list, BitSet::ones(3));
        op.run(&ctx)
            .map(|outputs| outputs.into_iter().map(|o| o.try_into().unwrap()).collect())
    }

    fn default_gqa(num_heads: u32, kv_num_heads: u32, scale: Option<f32>) -> GroupQueryAttention {
        GroupQueryAttention {
            num_heads,
            kv_num_heads,
            scale,
            do_rotary: false,
            rotary_interleaved: false,
            local_window_size: None,
            softcap: 0.0,
            smooth_softmax: false,
        }
    }

    fn reference_add_softmax(x: TensorView, y: TensorView) -> Result<Tensor, OpError> {
        let add = Add {};
        let softmax = Softmax {
            axis: -1,
            flush_nans_to_zero: false,
        };
        let sum: Tensor = add.run_simple((x, y))?;
        softmax.run_simple(sum.view())
    }

    #[test]
    fn test_add_softmax() {
        #[derive(Debug)]
        struct Case {
            qk_shape: Vec<usize>,
            m_shape: Vec<usize>,
            expected_err: Option<OpError>,
            in_place: bool,
        }

        let cases = [
            // Standard attention inputs where QK has shape (batch, n_heads,
            // sequence_len, head_size) and M has shape (batch, 1, sequence_len,
            // head_size).
            Case {
                qk_shape: [1, 8, 32, 32].into(),
                m_shape: [1, 1, 32, 32].into(),
                expected_err: None,
                in_place: true,
            },
            // In-place execution where broadcasting fails
            Case {
                qk_shape: [1, 8, 32, 32].into(),
                m_shape: [1, 2, 32, 32].into(),
                expected_err: Some(BROADCAST_ERROR),
                in_place: true,
            },
            // Non in-place execution where broadcasting fails
            Case {
                qk_shape: [1, 8, 32, 32].into(),
                m_shape: [1, 2, 32, 32].into(),
                expected_err: Some(BROADCAST_ERROR),
                in_place: false,
            },
            // In-place execution where both QK and M should be broadcast.
            Case {
                qk_shape: [1, 8, 16].into(),
                m_shape: [8, 1, 16].into(),
                expected_err: None,
                in_place: true,
            },
            // Non in-place execution where QK and M are swapped.
            Case {
                qk_shape: [1, 1, 32, 32].into(),
                m_shape: [1, 8, 32, 32].into(),
                expected_err: None,
                in_place: false,
            },
        ];

        cases.test_each(|case| {
            let mut rng = XorShiftRng::new(1234);
            let op = AddSoftmax {
                flush_nans_to_zero: false,
            };
            let qk = Tensor::rand(&case.qk_shape, &mut rng);
            let m = Tensor::rand(&case.m_shape, &mut rng);

            let result: Result<Tensor, _> = if case.in_place {
                op.run_simple_in_place(qk.clone(), m.view())
            } else {
                op.run_simple((qk.view(), m.view()))
            };
            if let Some(expected_err) = &case.expected_err {
                assert_eq!(result.as_ref().err().unwrap(), expected_err);
            } else {
                let expected = reference_add_softmax(qk.view(), m.view()).unwrap();
                expect_equal(&result.unwrap(), &expected).unwrap();
            }
        });
    }

    // Test that flush_nans_to_zero behavior works correctly when all inputs are
    // negative infinity after the add operation.
    #[test]
    fn test_add_softmax_flush_nans_to_zero() {
        // When all inputs are -inf after addition, normal softmax produces NaN.
        let qk = Tensor::from([f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY]);
        let m = Tensor::from([0., 0., 0.]);
        let op = AddSoftmax {
            flush_nans_to_zero: false,
        };
        let result: Tensor = op.run_simple((qk.view(), m.view())).unwrap();
        assert!(result.iter().all(|x| x.is_nan()));

        // With flush_nans_to_zero, output should be all zeros.
        let qk = Tensor::from([f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY]);
        let m = Tensor::from([0., 0., 0.]);
        let op = AddSoftmax {
            flush_nans_to_zero: true,
        };
        let result: Tensor = op.run_simple((qk.view(), m.view())).unwrap();
        assert_eq!(result.to_vec(), vec![0., 0., 0.]);
    }

    #[test]
    fn test_repeat_interleave() {
        let input = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);

        let op = RepeatInterleave {
            axis: 1,
            repeats: 2,
        };
        let repeated: Tensor = op.run_simple(input.view()).unwrap();

        assert_eq!(repeated, Tensor::from([[1., 1., 2., 2.], [3., 3., 4., 4.]]));
    }

    #[test]
    fn test_group_query_attention() {
        #[derive(Debug)]
        struct Case {
            num_heads: u32,
            kv_num_heads: u32,
            seq: usize,
            past_seq: usize,
            batch: usize,
            scale: Option<f32>,
        }

        let cases = [
            // Multi-head self-attention prompt (no past, no grouping).
            Case {
                num_heads: 2,
                kv_num_heads: 2,
                seq: 4,
                past_seq: 0,
                batch: 1,
                scale: None,
            },
            // Grouped-query prompt: 4 query heads share 2 KV heads.
            Case {
                num_heads: 4,
                kv_num_heads: 2,
                seq: 3,
                past_seq: 0,
                batch: 1,
                scale: Some(0.3),
            },
            // Multi-query decode: 1 new token, several past tokens, 1 KV head.
            Case {
                num_heads: 4,
                kv_num_heads: 1,
                seq: 1,
                past_seq: 5,
                batch: 1,
                scale: None,
            },
            // Subsequent prompt: new tokens appended to an existing cache.
            Case {
                num_heads: 4,
                kv_num_heads: 2,
                seq: 3,
                past_seq: 2,
                batch: 1,
                scale: None,
            },
            // Multiple batch items.
            Case {
                num_heads: 2,
                kv_num_heads: 1,
                seq: 3,
                past_seq: 0,
                batch: 2,
                scale: None,
            },
        ];

        cases.test_each(|case| {
            let &Case {
                num_heads,
                kv_num_heads,
                seq,
                past_seq,
                batch,
                scale,
            } = case;
            let head_size = 8;
            let q_hidden = num_heads as usize * head_size;
            let kv_hidden = kv_num_heads as usize * head_size;
            let total = past_seq + seq;

            let mut rng = XorShiftRng::new(1234);
            let query = NdTensor::<f32, 3>::rand([batch, seq, q_hidden], &mut rng);
            let key = NdTensor::<f32, 3>::rand([batch, seq, kv_hidden], &mut rng);
            let value = NdTensor::<f32, 3>::rand([batch, seq, kv_hidden], &mut rng);
            let (past_key, past_value) = if past_seq > 0 {
                (
                    Some(NdTensor::<f32, 4>::rand(
                        [batch, kv_num_heads as usize, past_seq, head_size],
                        &mut rng,
                    )),
                    Some(NdTensor::<f32, 4>::rand(
                        [batch, kv_num_heads as usize, past_seq, head_size],
                        &mut rng,
                    )),
                )
            } else {
                (None, None)
            };
            let seqlens_k = NdTensor::<i32, 1>::full([batch], total as i32 - 1);

            let op = default_gqa(num_heads, kv_num_heads, scale);
            let resolved_scale = scale.unwrap_or(1.0 / (head_size as f32).sqrt());
            let outputs = run_gqa(
                &op,
                &query,
                &key,
                &value,
                past_key.as_ref(),
                past_value.as_ref(),
                &seqlens_k,
                total as i32,
            )
            .unwrap();

            let expected = reference_gqa(
                &query,
                &key,
                &value,
                past_key.as_ref(),
                past_value.as_ref(),
                num_heads as usize,
                kv_num_heads as usize,
                resolved_scale,
            );
            expect_equal(&outputs[0].nd_view::<3>(), &expected.view()).unwrap();

            assert_eq!(outputs[0].shape(), [batch, seq, q_hidden]);
            assert_eq!(
                outputs[1].shape(),
                [batch, kv_num_heads as usize, total, head_size]
            );
            assert_eq!(
                outputs[2].shape(),
                [batch, kv_num_heads as usize, total, head_size]
            );
        });
    }

    #[test]
    fn test_group_query_attention_present_cache() {
        let op = default_gqa(2, 1, Some(1.0));
        let query = NdTensor::<f32, 3>::zeros([1, 1, 16]);
        let key = NdTensor::from([[[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]]);
        let value = NdTensor::from([[[8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]]);
        let past_key = NdTensor::<f32, 4>::zeros([1, 1, 2, 8]);
        let past_value =
            NdTensor::<f32, 4>::from_fn([1, 1, 2, 8], |[_, _, t, d]| (t * 8 + d) as f32);
        let seqlens_k = NdTensor::from([2i32]); // total - 1 = 2

        let outputs = run_gqa(
            &op,
            &query,
            &key,
            &value,
            Some(&past_key),
            Some(&past_value),
            &seqlens_k,
            3,
        )
        .unwrap();

        let present_key = outputs[1].nd_view::<4>();

        // The present cache should contain the past cache followed by the new
        // key/value tokens.
        assert_eq!(present_key.slice((0, 0, 0)).to_vec(), vec![0.0; 8]);
        assert_eq!(present_key.slice((0, 0, 1)).to_vec(), vec![0.0; 8]);
        assert_eq!(
            present_key.slice((0, 0, 2)).to_vec(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn test_group_query_attention_rejects_unsupported() {
        let query = NdTensor::<f32, 3>::zeros([1, 2, 16]);
        let key = NdTensor::<f32, 3>::zeros([1, 2, 8]);
        let value = NdTensor::<f32, 3>::zeros([1, 2, 8]);
        let seqlens_k = NdTensor::from([1i32]);

        // smooth_softmax is not supported.
        let mut op = default_gqa(2, 1, None);
        op.smooth_softmax = true;
        let result = run_gqa(&op, &query, &key, &value, None, None, &seqlens_k, 2);
        assert_eq!(
            result.err().unwrap(),
            OpError::UnsupportedValue("smooth_softmax is not supported")
        );

        // num_heads must be a multiple of kv_num_heads.
        let op = default_gqa(3, 2, None);
        let result = run_gqa(&op, &query, &key, &value, None, None, &seqlens_k, 2);
        assert_eq!(
            result.err().unwrap(),
            OpError::InvalidValue("num_heads must be a multiple of kv_num_heads")
        );

        // Inconsistent seqlens_k / total_sequence_length
        let op = default_gqa(2, 1, None);
        let result = run_gqa(
            &op,
            &query,
            &key,
            &value,
            None,
            None,
            &NdTensor::from([0i32]),
            1,
        );
        assert_eq!(
            result.err().unwrap(),
            OpError::InvalidValue("seqlens_k entry is too small for the query sequence length")
        );

        // First-prompt query (seq == total_sequence_length) whose seqlens_k entry
        // is smaller than the sequence length.
        let op = default_gqa(2, 1, None);
        let result = run_gqa(
            &op,
            &query,
            &key,
            &value,
            None,
            None,
            &NdTensor::from([0i32]),
            2,
        );
        assert_eq!(
            result.err().unwrap(),
            OpError::InvalidValue("seqlens_k entry is too small for the query sequence length")
        );

        // seqlens_k implies a past context larger than the supplied past_key
        // cache (past_seq = 2 but seqlens_k = 9 => 8 past tokens for a 1-token
        // decode query).
        let op = default_gqa(2, 1, None);
        let q1 = NdTensor::<f32, 3>::zeros([1, 1, 16]);
        let k1 = NdTensor::<f32, 3>::zeros([1, 1, 8]);
        let v1 = NdTensor::<f32, 3>::zeros([1, 1, 8]);
        let past_key = NdTensor::<f32, 4>::zeros([1, 1, 2, 8]);
        let past_value = NdTensor::<f32, 4>::zeros([1, 1, 2, 8]);
        let result = run_gqa(
            &op,
            &q1,
            &k1,
            &v1,
            Some(&past_key),
            Some(&past_value),
            &NdTensor::from([9i32]),
            10,
        );
        assert_eq!(
            result.err().unwrap(),
            OpError::InvalidValue("seqlens_k entry is out of range")
        );
    }

    #[test]
    fn test_group_query_attention_omits_unrequested_kv_cache() {
        let op = default_gqa(2, 1, None);
        let query = NdTensor::<f32, 3>::zeros([1, 2, 16]);
        let key = NdTensor::<f32, 3>::zeros([1, 2, 8]);
        let value = NdTensor::<f32, 3>::zeros([1, 2, 8]);
        let seqlens_k = NdTensor::from([1i32]);
        let total = NdTensor::from_scalar(2i32);
        let input_vec = [
            Some(ValueView::from(query.view())),
            Some(ValueView::from(key.view())),
            Some(ValueView::from(value.view())),
            None,
            None,
            Some(ValueView::from(seqlens_k.view())),
            Some(ValueView::from(total.view())),
        ];
        let input_list = InputList::from_optional(&input_vec);
        let pool = BufferPool::new();

        // When only the first output is requested, the present KV caches are not
        // materialized as outputs.
        let ctx = OpRunContext::new(&pool, &input_list, BitSet::from_indices([0]));
        let outputs = op.run(&ctx).unwrap();
        assert_eq!(outputs.len(), 1);

        // When a KV-cache output is requested, all three outputs are returned.
        let ctx = OpRunContext::new(&pool, &input_list, BitSet::from_indices([0, 1]));
        let outputs = op.run(&ctx).unwrap();
        assert_eq!(outputs.len(), 3);
    }

    #[test]
    fn test_group_query_attention_validates_attention_bias() {
        // Run GQA with an `attention_bias` input (index 10).
        let run_with_bias = |bias: &NdTensor<f32, 4>| -> Result<Vec<Tensor>, OpError> {
            let op = default_gqa(2, 1, None);
            let query = NdTensor::<f32, 3>::zeros([1, 2, 16]);
            let key = NdTensor::<f32, 3>::zeros([1, 2, 8]);
            let value = NdTensor::<f32, 3>::zeros([1, 2, 8]);
            let seqlens_k = NdTensor::from([1i32]);
            let total = NdTensor::from_scalar(2i32);
            let inputs = [
                Some(ValueView::from(query.view())),
                Some(ValueView::from(key.view())),
                Some(ValueView::from(value.view())),
                None, // past_key
                None, // past_value
                Some(ValueView::from(seqlens_k.view())),
                Some(ValueView::from(total.view())),
                None, // cos_cache
                None, // sin_cache
                None, // position_ids
                Some(ValueView::from(bias.view())),
            ];
            let input_list = InputList::from_optional(&inputs);
            let pool = BufferPool::new();
            let ctx = OpRunContext::new(&pool, &input_list, BitSet::ones(3));
            op.run(&ctx)
                .map(|outputs| outputs.into_iter().map(|o| o.try_into().unwrap()).collect())
        };

        // A bias covering (1, 1, seq, total_seq) broadcasts over batch and heads.
        let bias = NdTensor::<f32, 4>::zeros([1, 1, 2, 2]);
        assert!(run_with_bias(&bias).is_ok());

        // Bias with incorrect `seq` length.
        let bias = NdTensor::<f32, 4>::zeros([1, 1, 1, 2]);
        assert_eq!(
            run_with_bias(&bias).err().unwrap(),
            OpError::IncompatibleInputShapes(
                "attention_bias shape is incompatible with query/key shapes"
            )
        );

        // Bias with incorrect `total_seq` length.
        let bias = NdTensor::<f32, 4>::zeros([1, 1, 2, 1]);
        assert_eq!(
            run_with_bias(&bias).err().unwrap(),
            OpError::IncompatibleInputShapes(
                "attention_bias shape is incompatible with query/key shapes"
            )
        );

        // Bias with batch/head dims that cannot broadcast to actual size.
        let bias = NdTensor::<f32, 4>::zeros([1, 3, 2, 2]);
        assert_eq!(
            run_with_bias(&bias).err().unwrap(),
            OpError::IncompatibleInputShapes(
                "attention_bias shape is incompatible with query/key shapes"
            )
        );
    }

    #[test]
    fn test_grouped_query_attention_matmul() {
        let batch = 1;
        let query_heads = 8;
        let kv_heads = 2;
        let seq = 3;
        let d_model = 8;

        let query = NdTensor::<f32, 4>::zeros([batch, query_heads, seq, d_model]);
        let key = NdTensor::<f32, 4>::zeros([batch, kv_heads, seq, d_model]);
        let value = NdTensor::<f32, 4>::zeros([batch, kv_heads, seq, d_model]);

        // Query-key matmul.
        let op = GroupedQueryAttentionMatMul {
            repeats: query_heads / kv_heads,
            // In the QK matmul, we have a scale and transposed RHS.
            alpha: Some(0.5),
            transpose_rhs: true,
        };

        let query_key: NdTensor<f32, 4> = op.run_simple((query.view(), key.view())).unwrap();
        assert_eq!(query_key.shape(), [batch, query_heads, seq, seq]);

        // Query-value matmul
        let op = GroupedQueryAttentionMatMul {
            repeats: query_heads / kv_heads,
            // In the QK @ V matmul, we typically have no scale and the RHS is
            // not transposed.
            alpha: None,
            transpose_rhs: false,
        };

        let qkv: NdTensor<f32, 4> = op.run_simple((query_key.view(), value.view())).unwrap();
        assert_eq!(qkv.shape(), [batch, query_heads, seq, d_model]);
    }

    #[test]
    fn test_multihead_attention_self_attention() {
        let op = MultiHeadAttention {
            mask_filter_value: -10000.0,
            num_heads: 1,
            scale: Some(1.0),
            unidirectional: false,
        };
        let query = Tensor::from_data(&[1, 2, 2], vec![1., 0., 0., 1.]);

        let result: Tensor = op.run_simple(query.view()).unwrap();
        let e = std::f32::consts::E;
        let expected = Tensor::from_data(
            &[1, 2, 2],
            vec![
                e / (e + 1.0),
                1.0 / (e + 1.0),
                1.0 / (e + 1.0),
                e / (e + 1.0),
            ],
        );
        expect_equal(&result, &expected).unwrap();
    }

    #[test]
    fn test_multihead_attention_packed_qkv() {
        let op = MultiHeadAttention {
            mask_filter_value: -10000.0,
            num_heads: 1,
            scale: Some(1.0),
            unidirectional: false,
        };
        let packed = Tensor::from_data(
            &[1, 2, 1, 3, 2],
            vec![1., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 1.],
        );

        let result: Tensor = op.run_simple(packed.view()).unwrap();
        let e = std::f32::consts::E;
        let expected = Tensor::from_data(
            &[1, 2, 2],
            vec![
                e / (e + 1.0),
                1.0 / (e + 1.0),
                1.0 / (e + 1.0),
                e / (e + 1.0),
            ],
        );
        expect_equal(&result, &expected).unwrap();
    }

    #[test]
    fn test_multihead_attention_key_padding_mask() {
        let op = MultiHeadAttention {
            mask_filter_value: -10000.0,
            num_heads: 1,
            scale: Some(1.0),
            unidirectional: false,
        };
        let query = Tensor::from_data(&[1, 2, 2], vec![1., 0., 0., 1.]);
        // Key position 0 is masked out, so both queries attend only to
        // key/value position 1.
        let key_padding_mask = Tensor::from_data(&[1, 2], vec![0i32, 1]);
        let inputs = [
            Some(ValueView::from(query.view())),
            None,
            None,
            None,
            Some(ValueView::from(key_padding_mask.view())),
        ];
        let input_list = InputList::from_optional(&inputs);
        let pool = BufferPool::new();
        let ctx = OpRunContext::new(&pool, &input_list, BitSet::ones(1));

        let mut outputs = op.run(&ctx).unwrap();
        let result: Tensor = outputs.remove(0).try_into().unwrap();
        let expected = Tensor::from_data(&[1, 2, 2], vec![0., 1., 0., 1.]);
        expect_equal(&result, &expected).unwrap();
    }

    #[test]
    fn test_multihead_attention_attention_bias() {
        let op = MultiHeadAttention {
            mask_filter_value: -10000.0,
            num_heads: 1,
            scale: Some(1.0),
            unidirectional: false,
        };
        let query = Tensor::from_data(&[1, 2, 2], vec![1., 0., 0., 1.]);

        let run_with_bias = |bias: &Tensor| -> Tensor {
            let inputs = [
                Some(ValueView::from(query.view())),
                None, // key
                None, // value
                None, // bias
                None, // key_padding_mask
                Some(ValueView::from(bias.view())),
            ];
            let input_list = InputList::from_optional(&inputs);
            let pool = BufferPool::new();
            let ctx = OpRunContext::new(&pool, &input_list, BitSet::ones(1));
            let mut outputs = op.run(&ctx).unwrap();
            outputs.remove(0).try_into().unwrap()
        };

        // A full (1, 1, seq, total_seq) bias is added per (query, key) position.
        // The self-attention scores are the identity matrix; adding
        // [[0, 1], [1, 0]] makes both rows uniform, so each query attends
        // equally to both values.
        let bias = Tensor::from_data(&[1, 1, 2, 2], vec![0., 1., 1., 0.]);
        let result = run_with_bias(&bias);
        let expected = Tensor::from_data(&[1, 2, 2], vec![0.5, 0.5, 0.5, 0.5]);
        expect_equal(&result, &expected).unwrap();

        // A bias that broadcasts over the key dimension (last dim == 1) adds a
        // constant to each score row, which softmax leaves unchanged, so the
        // result matches plain self-attention.
        let bias = Tensor::from_data(&[1, 1, 2, 1], vec![3., -2.]);
        let result = run_with_bias(&bias);
        let e = std::f32::consts::E;
        let expected = Tensor::from_data(
            &[1, 2, 2],
            vec![
                e / (e + 1.0),
                1.0 / (e + 1.0),
                1.0 / (e + 1.0),
                e / (e + 1.0),
            ],
        );
        expect_equal(&result, &expected).unwrap();
    }

    #[test]
    fn test_multihead_attention_ort_cross_attention_head_size_8() {
        // Ported from ONNX Runtime's MultiHeadAttentionTest.CrossAttention_Batch1_HeadSize8
        // fixture in testdata/attention/attention_test_data.txt.
        let op = MultiHeadAttention {
            mask_filter_value: -10000.0,
            num_heads: 2,
            scale: None,
            unidirectional: false,
        };
        let query = Tensor::from_data(
            &[1, 2, 16],
            vec![
                0.74714613,
                -2.49789214,
                -0.11628322,
                1.33038604,
                0.82568336,
                0.07685500,
                2.47562003,
                2.61135578,
                1.55278158,
                -1.85635769,
                0.36962336,
                0.87219834,
                0.69827259,
                0.95257485,
                -0.77894646,
                1.46218395,
                1.29534733,
                2.14051294,
                1.09895217,
                1.39164531,
                -0.01471180,
                -1.40148544,
                -0.50825417,
                0.26134527,
                -0.70491123,
                0.63738143,
                2.13708138,
                0.05667466,
                -0.44220763,
                0.85254443,
                2.00844359,
                -1.23413038,
            ],
        );
        let key = Tensor::from_data(
            &[1, 3, 16],
            vec![
                1.70455408,
                0.07344571,
                0.18893155,
                -1.48390186,
                -0.86155319,
                0.10993601,
                -0.29869685,
                0.73800445,
                0.94670546,
                -1.36712539,
                -0.41328859,
                0.88237023,
                1.62447476,
                0.80396229,
                -1.38206959,
                1.62546301,
                -1.61546838,
                -0.56213129,
                -0.23501799,
                0.89255226,
                -1.95987988,
                0.85192877,
                -0.06520678,
                -1.32849765,
                2.07457638,
                -0.08192353,
                -2.03260493,
                0.58190948,
                2.22535419,
                -0.60754669,
                1.14538383,
                0.22928622,
                -0.11596665,
                -0.57144678,
                -0.23428933,
                -0.68404931,
                -1.46875453,
                1.32763886,
                0.28525546,
                -0.11347114,
                1.63199806,
                -1.44967401,
                -2.54707336,
                0.78083873,
                -0.19109090,
                0.59508920,
                0.58886564,
                0.81380880,
            ],
        );
        let value = Tensor::from_data(
            &[1, 3, 16],
            vec![
                0.20429733,
                -0.57036293,
                0.22116289,
                0.07601038,
                1.79898310,
                0.62182522,
                0.48815370,
                -1.59284389,
                0.33195397,
                0.34822315,
                0.54315579,
                1.06468117,
                1.34500551,
                -0.09528533,
                -1.30459058,
                -0.07034321,
                -1.34877563,
                1.58868146,
                -1.44948101,
                0.74792957,
                0.91922742,
                -0.56811053,
                0.59939134,
                -1.10749292,
                1.36371183,
                -0.89673072,
                -0.28341034,
                0.93497890,
                1.62986696,
                -0.83026254,
                -0.20963377,
                -2.14284325,
                -0.95242530,
                0.37379366,
                1.17815948,
                -0.55676895,
                0.74420613,
                0.58715403,
                -0.43127203,
                0.62706453,
                0.50881875,
                2.14387321,
                0.85787302,
                2.32273459,
                -0.04902139,
                -0.04061748,
                1.55004728,
                -0.25090796,
            ],
        );
        let bias = Tensor::from_data(
            &[48],
            vec![
                -0.38124341,
                0.02696526,
                -0.11914945,
                -0.43795273,
                -0.34948170,
                -0.19608477,
                0.19725692,
                0.39987487,
                0.04772711,
                -0.03419551,
                -0.30606642,
                0.42656231,
                -0.23178342,
                -0.13692456,
                -0.04889601,
                0.48739988,
                0.27079183,
                0.42074734,
                -0.40314156,
                -0.43726659,
                0.27376485,
                -0.38174152,
                -0.43700469,
                0.38040614,
                -0.40546918,
                0.06927037,
                0.16979086,
                0.41458064,
                0.07120579,
                -0.08055863,
                0.12095112,
                -0.27988660,
                -0.10567203,
                0.26791072,
                -0.08976898,
                0.31341976,
                0.06027532,
                0.14307594,
                0.31587386,
                0.16180152,
                0.34785229,
                0.00531715,
                -0.35168743,
                -0.11641458,
                0.39196932,
                0.44535065,
                0.43545735,
                0.15593112,
            ],
        );

        let result: Tensor = op
            .run_simple((query.view(), key.view(), value.view(), bias.view()))
            .unwrap();
        let expected = Tensor::from_data(
            &[1, 2, 16],
            vec![
                -0.61998826,
                0.38731366,
                0.38371456,
                0.17248757,
                1.26609111,
                0.61097330,
                0.38864893,
                -0.34083632,
                0.78583258,
                0.67860925,
                0.20943914,
                1.22361767,
                1.44091177,
                0.31527188,
                -0.15526980,
                -0.08799548,
                -0.25185302,
                0.10573119,
                0.01646931,
                0.40613887,
                1.61315691,
                0.59776157,
                0.70979917,
                -1.10025024,
                1.16315329,
                0.47766802,
                -0.03506046,
                1.33826876,
                1.36242199,
                0.06935713,
                0.58279711,
                -0.82380491,
            ],
        );
        expect_equal(&result, &expected).unwrap();

        let result: Tensor = op
            .run_simple((query.view(), key.view(), value.view()))
            .unwrap();
        let expected = Tensor::from_data(
            &[1, 2, 16],
            vec![
                -0.51569921,
                0.13232709,
                0.43551767,
                -0.12155488,
                1.21165323,
                0.45272583,
                0.08948315,
                -0.53300208,
                0.44346270,
                0.59271330,
                0.53993183,
                1.29220927,
                1.10357487,
                -0.14063509,
                -0.68309224,
                -0.26137090,
                -0.15928616,
                -0.13984840,
                0.07850466,
                0.10540886,
                1.54793286,
                0.43936923,
                0.40107274,
                -1.26946867,
                0.86807090,
                0.27874026,
                0.24483341,
                1.36524665,
                1.07833946,
                -0.42526853,
                0.03085684,
                -1.09703445,
            ],
        );
        expect_equal(&result, &expected).unwrap();
    }

    #[test]
    fn test_sdpa_head_zeros_fully_masked_rows() {
        let pool = BufferPool::new();
        let gemm = GemmExecutor::<f32>::new();

        let q_seq = 3;
        let kv_seq = 4;
        let head_size = 8;
        let mut rng = XorShiftRng::new(9999);
        let query = NdTensor::<f32, 2>::rand([q_seq, head_size], &mut rng);
        let key = NdTensor::<f32, 2>::rand([kv_seq, head_size], &mut rng);
        let value = NdTensor::<f32, 2>::rand([kv_seq, head_size], &mut rng);

        let mut out = NdTensor::<f32, 2>::uninit_in(&pool, [q_seq, head_size]);
        sdpa_head(
            &pool,
            &gemm,
            1.0 / (head_size as f32).sqrt(),
            query.view(),
            key.view(),
            value.view(),
            out.view_mut(),
            // Fully mask query row 1: every key position is disallowed.
            |row, q_idx| {
                if q_idx == 1 {
                    row.fill(f32::NEG_INFINITY);
                }
            },
        );
        // Safety: `sdpa_head` initializes every element of `out`.
        let out = unsafe { out.assume_init() };

        // The fully-masked row is finite (all zeros), not NaN.
        assert!(out.iter().all(|x| x.is_finite()));
        assert_eq!(out.slice([1]).to_vec(), vec![0.0; head_size]);
        // Unmasked rows still receive normal (non-zero) attention output.
        assert!(out.slice([0]).iter().any(|&x| x != 0.0));
        assert!(out.slice([2]).iter().any(|&x| x != 0.0));
    }
}
