//! Attention-related operations.

use rayon::prelude::*;
use rten_gemm::{GemmExecutor, GemmInputA, GemmInputB, GemmUninitOptions};
use rten_simd::SimdOp;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, Tensor, TensorView};
use rten_vecmath::Softmax;

use crate::buffer_pool::{AutoReturn, BufferPool};
use crate::infer_shapes::InferShapes;
use crate::operator::{
    IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
    OutputTypesContext,
};
use crate::ops::{
    binary_elementwise::{add, broadcast_shapes},
    concat::concat,
    layout::expand_to,
    matmul::matmul,
    norm::NanHandling,
    resolve_axis,
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

/// Reshape (batch, seq, hidden) to (batch, num_heads, seq, head).
fn split_attention_heads(
    pool: &BufferPool,
    input: NdTensorView<f32, 3>,
    num_heads: usize,
    head_size: usize,
) -> Result<NdTensor<f32, 4>, OpError> {
    let [batch_size, seq_len, hidden] = input.shape();
    if hidden != num_heads * head_size {
        return Err(OpError::IncompatibleInputShapes(
            "Hidden size does not match number of attention heads",
        ));
    }

    let reshaped = input.reshaped_in(pool, [batch_size, seq_len, num_heads, head_size]);
    Ok(reshaped.permuted([0, 2, 1, 3]).to_tensor_in(pool))
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

        let (query, mut key, mut value, batch_size, seq_len, head_size, v_head_size, v_hidden) =
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
                        q.permuted([0, 2, 1, 3]).to_tensor_in(ctx.pool()),
                        k.permuted([0, 2, 1, 3]).to_tensor_in(ctx.pool()),
                        v.permuted([0, 2, 1, 3]).to_tensor_in(ctx.pool()),
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
                            .auto_return(ctx.pool());
                        let key =
                            add(ctx.pool(), key.as_dyn(), k_bias.as_dyn())?.auto_return(ctx.pool());
                        let value = add(ctx.pool(), value.as_dyn(), v_bias.as_dyn())?
                            .auto_return(ctx.pool());
                        (
                            split_attention_heads(
                                ctx.pool(),
                                query.nd_view(),
                                num_heads,
                                head_size,
                            )?,
                            split_attention_heads(ctx.pool(), key.nd_view(), num_heads, head_size)?,
                            split_attention_heads(
                                ctx.pool(),
                                value.nd_view(),
                                num_heads,
                                v_head_size,
                            )?,
                        )
                    } else {
                        (
                            split_attention_heads(ctx.pool(), query, num_heads, head_size)?,
                            split_attention_heads(ctx.pool(), key, num_heads, head_size)?,
                            split_attention_heads(ctx.pool(), value, num_heads, v_head_size)?,
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
                    .try_into()
                    .expect("concat of rank-4 tensors has rank 4");
                value = concat(ctx.pool(), &[past_value.as_dyn(), value.as_dyn()], 2)?
                    .try_into()
                    .expect("concat of rank-4 tensors has rank 4");
            }
            (Some(_), None) | (None, Some(_)) => {
                return Err(OpError::InvalidValue(
                    "past_key and past_value must either both be present or both be absent",
                ));
            }
            (None, None) => {}
        }

        // Compute attention scores - (Q ^ K^T) * scale + bias
        let total_seq_len = key.size(2);
        let scale = self
            .scale
            .unwrap_or_else(|| 1.0 / (head_size as f32).sqrt());
        let key_t = key.permuted([0, 1, 3, 2]);

        let mut scores = NdTensor::uninit_in(
            ctx.pool(),
            [batch_size, num_heads, query.size(2), key_t.size(3)],
        );
        let gemm = GemmExecutor::new();
        scores
            .inner_iter_mut::<2>()
            .into_par_iter()
            .zip(
                query
                    .inner_iter::<2>()
                    .into_par_iter()
                    .zip(key_t.inner_iter::<2>().into_par_iter()),
            )
            .for_each(|(mut out, (query, key))| {
                gemm.gemm_uninit(
                    out.data_mut().unwrap(),
                    GemmInputA::Unpacked(query),
                    GemmInputB::Unpacked(key),
                    GemmUninitOptions {
                        alpha: scale,
                        ..Default::default()
                    },
                )
                .unwrap();
            });

        // Safety: Score matrices initialized.
        let mut scores = unsafe { scores.assume_init() };

        if let Some(attention_bias) = attention_bias {
            scores = add(ctx.pool(), scores.as_dyn(), attention_bias.as_dyn())?
                .try_into()
                .expect("attention scores have rank 4");
        }

        debug_assert_eq!(
            scores.shape(),
            [batch_size, num_heads, seq_len, total_seq_len]
        );

        // Apply masks to attention scores
        if self.unidirectional {
            let past_len = total_seq_len - seq_len;
            for mut head_scores in scores.inner_iter_mut::<2>() {
                for q_idx in 0..head_scores.size(0) {
                    for k_idx in (past_len + q_idx + 1)..head_scores.size(1) {
                        head_scores[[q_idx, k_idx]] = self.mask_filter_value;
                    }
                }
            }
        }

        if let Some(key_padding_mask) = key_padding_mask {
            if key_padding_mask.shape() != [batch_size, total_seq_len] {
                return Err(OpError::IncompatibleInputShapes(
                    "key_padding_mask shape does not match key sequence length",
                ));
            }
            for batch in 0..scores.size(0) {
                for head in 0..scores.size(1) {
                    for key_idx in 0..scores.size(2) {
                        if key_padding_mask[[batch, key_idx]] == 0 {
                            for query_idx in 0..seq_len {
                                scores[[batch, head, query_idx, key_idx]] = self.mask_filter_value;
                            }
                        }
                    }
                }
            }
        }

        // Normalize scores to probabilities.
        for mut lane in scores.lanes_mut(3) {
            Softmax::new_mut(lane.as_slice_mut().unwrap()).dispatch();
        }

        // Compute attention outputs.
        let context: NdTensor<f32, 4> = matmul(ctx.pool(), scores.as_dyn(), value.as_dyn(), None)?
            .try_into()
            .expect("matmul of rank-4 tensors has rank 4");
        let output = context
            .permuted([0, 2, 1, 3])
            .to_tensor_in(ctx.pool())
            .into_shape([batch_size, seq_len, v_hidden]);

        let mut outputs: OutputList = [output.into()].into();
        let return_present = past_key.is_some() || ctx.outputs().count_true() > 1;
        if return_present {
            outputs.push(key.into());
            outputs.push(value.into());
        }
        Ok(outputs)
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        None
    }
}

#[cfg(test)]
mod tests {
    use rten_base::bit_set::BitSet;
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::expect_equal;
    use rten_tensor::{NdTensor, Tensor, TensorView};
    use rten_testing::TestCases;

    use super::{
        AddSoftmax, BROADCAST_ERROR, GroupedQueryAttentionMatMul, MultiHeadAttention,
        RepeatInterleave,
    };
    use crate::buffer_pool::BufferPool;
    use crate::operator::{InputList, OpError, OpRunContext, Operator, OperatorExt};
    use crate::ops::{Add, Softmax};
    use crate::value::ValueView;

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
}
