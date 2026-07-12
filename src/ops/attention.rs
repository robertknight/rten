//! Attention-related operations.

use std::mem::MaybeUninit;

use rayon::prelude::*;
use rten_base::bit_set::BitSet;
use rten_gemm::{GemmExecutor, GemmInputA, GemmInputB, GemmUninitOptions};
use rten_shape_inference::ops as shape_ops;
use rten_simd::SimdOp;
use rten_tensor::prelude::*;
use rten_tensor::{CowNdTensor, NdTensor, NdTensorView, NdTensorViewMut, Tensor, TensorView};
use rten_vecmath::Softmax;

use crate::buffer_pool::{AutoReturn, BufferPool, PoolRef};
use crate::infer_shapes::{InferShapes, impl_infer_shapes};
use crate::operator::{
    InPlaceInputs, IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType,
    OutputTypeList, OutputTypesContext,
};
use crate::ops::{
    binary_elementwise::broadcast_shapes, layout::expand_to, norm::NanHandling, resolve_axis,
};
use crate::value::{Value, ValueView};

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

    fn in_place_inputs(&self) -> BitSet<u16> {
        BitSet::from_indices([0])
    }

    fn run_in_place(
        &self,
        in_place: InPlaceInputs,
        ctx: &OpRunContext,
    ) -> Result<OutputList, OpError> {
        let qk: Tensor = in_place.into_single().try_into()?;
        // This operator is commutative, so the other input may be at either
        // position depending on which was selected for in-place execution.
        let m: TensorView = ctx.inputs().require_first_present_as()?;

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

        add_softmax_in_place(ctx.pool(), qk, m, self.nan_handling()).into_op_result()
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
pub fn split_attention_heads<'a>(
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

/// Reshape (batch, num_heads, seq, head) to (batch, seq, num_heads * head).
///
/// This is the inverse of [`split_attention_heads`].
pub fn merge_attention_heads(pool: &BufferPool, input: NdTensor<f32, 4>) -> NdTensor<f32, 3> {
    let [batch, num_heads, seq, head_size] = input.shape();
    input
        .permuted([0, 2, 1, 3])
        .to_tensor_in(pool)
        .into_shape([batch, seq, num_heads * head_size])
}

/// Concatenate a past KV cache and the current key or value along the
/// sequence dimension.
///
/// `past` has shape `(batch, kv_heads, past_seq, head_size)` and `current`
/// has shape `(batch, kv_heads, current_seq, head_size)`.
fn concat_kv_cache(
    pool: &BufferPool,
    past: NdTensorView<f32, 4>,
    current: NdTensorView<f32, 4>,
) -> NdTensor<f32, 4> {
    let [batch, kv_heads, past_seq, head_size] = past.shape();
    let total_seq = past_seq + current.size(2);

    let mut out = NdTensor::uninit_in(pool, [batch, kv_heads, total_seq, head_size]);
    out.inner_iter_mut::<2>()
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut block)| {
            let b = i / kv_heads;
            let h = i % kv_heads;
            block.slice_mut(..past_seq).init_from(&past.slice([b, h]));
            block
                .slice_mut(past_seq..)
                .init_from(&current.slice([b, h]));
        });

    // Safety: every (total_seq, head_size) block was fully initialized above.
    unsafe { out.assume_init() }
}

/// Extend an owned past KV cache with the current key or value along the
/// sequence dimension (axis 2), reusing spare capacity in the past cache's
/// buffer if the caller reserved any.
///
/// This is the in-place equivalent of [`concat_kv_cache`] and produces an
/// identical result. It makes decoding much more efficient, as the KV cache no
/// longer needs to be reallocated and copied on every step. See
/// <https://github.com/robertknight/rten/issues/1305>.
fn concat_kv_cache_in_place(
    pool: &BufferPool,
    mut past: NdTensor<f32, 4>,
    current: NdTensorView<f32, 4>,
) -> NdTensor<f32, 4> {
    let total_seq = past.size(2) + current.size(2);
    if past.has_capacity(2, total_seq) {
        past.append(2, &current).expect("cache has capacity");
        past
    } else {
        // Not enough reserved capacity, so fall back to allocating a new cache.
        let past = past.auto_return(pool);
        concat_kv_cache(pool, past.view(), current)
    }
}

/// A past key or value cache passed to [`concat_past_kv`].
enum PastCache<'a> {
    View(NdTensorView<'a, f32, 4>),
    Owned(NdTensor<f32, 4>),
}

impl PastCache<'_> {
    fn view(&self) -> NdTensorView<'_, f32, 4> {
        match self {
            PastCache::View(view) => view.view(),
            PastCache::Owned(tensor) => tensor.view(),
        }
    }

    fn shape(&self) -> [usize; 4] {
        self.view().shape()
    }
}

/// Concatenate a past KV cache and the current key or value along the sequence
/// dimension, extending the past cache in-place if possible.
fn extend_kv_cache(
    pool: &BufferPool,
    past: PastCache,
    current: NdTensorView<f32, 4>,
) -> NdTensor<f32, 4> {
    match past {
        PastCache::View(past) => concat_kv_cache(pool, past, current),
        PastCache::Owned(past) => concat_kv_cache_in_place(pool, past, current),
    }
}

/// Extract the owned `past_key` and `past_value` caches from the in-place
/// inputs of an attention operator.
fn take_past_kv(
    in_place: InPlaceInputs,
    past_key_index: usize,
    past_value_index: usize,
) -> Result<(Option<PastCache<'static>>, Option<PastCache<'static>>), OpError> {
    let mut past_key = None;
    let mut past_value = None;
    for (index, value) in in_place {
        let cache = PastCache::Owned(value.try_into()?);
        if index == past_key_index {
            past_key = Some(cache);
        } else if index == past_value_index {
            past_value = Some(cache);
        } else {
            return Err(OpError::InvalidValue("unexpected in-place input"));
        }
    }
    Ok((past_key, past_value))
}

/// Validate the shapes of past key/value caches against the current key and
/// value, then concatenate past and current along the sequence dimension.
///
/// `key` and `value` have shape `(batch, kv_heads, kv_seq, {head_size,
/// v_head_size})` and the past tensors, if present, must have shape
/// `(batch, kv_heads, past_seq, {head_size, v_head_size})`.
///
/// Returns the number of cached key/value positions (zero when there is no
/// cache).
fn concat_past_kv<'a, 'b>(
    pool: &'a BufferPool,
    past_key: Option<PastCache>,
    past_value: Option<PastCache>,
    key: &mut PoolRef<'a, CowNdTensor<'b, f32, 4>>,
    value: &mut PoolRef<'a, CowNdTensor<'b, f32, 4>>,
) -> Result<usize, OpError> {
    match (past_key, past_value) {
        (Some(past_key), Some(past_value)) => {
            let [batch, kv_heads, _kv_seq, head_size] = key.shape();
            let v_head_size = value.size(3);
            let [pk_batch, pk_heads, pk_seq, pk_head_size] = past_key.shape();
            let [pv_batch, pv_heads, pv_seq, pv_head_size] = past_value.shape();
            if pk_batch != batch
                || pv_batch != batch
                || pk_heads != kv_heads
                || pv_heads != kv_heads
                || pk_head_size != head_size
                || pv_head_size != v_head_size
                || pk_seq != pv_seq
            {
                return Err(OpError::IncompatibleInputShapes(
                    "past_key/past_value shape does not match key/value shape",
                ));
            }
            *key = extend_kv_cache(pool, past_key, key.view())
                .into_cow()
                .auto_return(pool);
            *value = extend_kv_cache(pool, past_value, value.view())
                .into_cow()
                .auto_return(pool);
            Ok(pk_seq)
        }
        (None, None) => Ok(0),
        _ => Err(OpError::InvalidValue(
            "past_key and past_value must either both be present or both be absent",
        )),
    }
}

/// Compute scaled dot-product attention for a single (batch, head):
///
/// `out = softmax(score_mod(scale · Q Kᵀ)) · V`
///
/// `query` is `[q_seq, head_size]`, `key` is `[kv_seq, head_size]`, `value`
/// is `[kv_seq, v_head_size]` and `out` is `[q_seq, v_head_size]`.
/// `score_mod(row, query_index)` modifies a single score row (of length
/// `kv_seq`).
pub fn sdpa_head(
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
        // Flush NaNs to zero so that fully-masked rows (all scores -inf)
        // produces zeros rather than NaN outputs.
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

/// Softcap a row of attention scores in place via `softcap · tanh(x / softcap)`.
pub fn apply_softcap(row: &mut [f32], softcap: f32) {
    if softcap > 0.0 {
        for x in row.iter_mut() {
            *x = softcap * (*x / softcap).tanh();
        }
    }
}

/// Apply causal masking to a row of attention scores.
fn causal_mask_row(row: &mut [f32], offset: isize, q_idx: usize, fill: f32) {
    let masked_from = (offset + q_idx as isize + 1).clamp(0, row.len() as isize) as usize;
    row[masked_from..].fill(fill);
}

/// Compute scaled dot-product attention for every (batch, head) of a query.
///
/// `query` has shape (batch, q_heads, q_seq, head_size); `key` and `value`
/// have shape (batch, kv_heads, kv_seq, {head_size, v_head_size}), where
/// `q_heads` must be a multiple of `kv_heads` (grouped-query attention).
/// `score_mod(b, h, row, q_idx)` modifies a single score row (of length
/// `kv_seq`) for query index `q_idx` of batch `b` and query head `h`.
///
/// Returns the attention output with shape (batch, q_heads, q_seq,
/// v_head_size).
fn sdpa_multi_head(
    pool: &BufferPool,
    gemm: &GemmExecutor<f32>,
    scale: f32,
    query: NdTensorView<f32, 4>,
    key: NdTensorView<f32, 4>,
    value: NdTensorView<f32, 4>,
    score_mod: impl Fn(usize, usize, &mut [f32], usize) + Sync,
) -> NdTensor<f32, 4> {
    let [batch, q_heads, q_seq, _head_size] = query.shape();
    let kv_factor = q_heads / key.size(1);
    let v_head_size = value.size(3);

    let mut attn_out = NdTensor::uninit_in(pool, [batch, q_heads, q_seq, v_head_size]);
    attn_out
        .inner_iter_mut::<2>()
        .into_par_iter()
        .enumerate()
        .for_each(|(i, out)| {
            let b = i / q_heads;
            let h = i % q_heads;
            let kv_head = h / kv_factor;

            sdpa_head(
                pool,
                gemm,
                scale,
                query.slice([b, h]),
                key.slice([b, kv_head]),
                value.slice([b, kv_head]),
                out,
                |row, q_idx| score_mod(b, h, row, q_idx),
            );
        });

    // Safety: every (q_seq, v_head_size) block was written by `sdpa_head`.
    unsafe { attn_out.assume_init() }
}

/// Attention mask, broadcast to (batch, q_heads, q_seq, total_seq).
///
/// A boolean mask allows attending to a key position where the value is
/// non-zero. A float mask is added to the pre-softmax attention scores.
#[derive(Clone, Copy)]
enum Mask<'a> {
    Bool(NdTensorView<'a, i32, 4>),
    Float(NdTensorView<'a, f32, 4>),
}

/// Scaled dot-product attention operator from the ONNX standard.
///
/// This computes `softmax(softcap(scale · Q Kᵀ) + mask) · V` with optional
/// causal masking, attention softcapping and grouped-query attention.
///
/// See <https://onnx.ai/onnx/operators/onnx__Attention.html>.
#[derive(Debug)]
pub struct Attention {
    /// Enable causal masking.
    pub is_causal: bool,
    /// Number of key/value heads. Required when Q/K/V are 3D.
    pub kv_num_heads: Option<u32>,
    /// Number of query heads. Required when Q/K/V are 3D.
    pub q_num_heads: Option<u32>,
    /// Scale applied to `Q Kᵀ`. Defaults to `1 / sqrt(head_size)`.
    pub scale: Option<f32>,
    /// If `> 0`, softcap the attention scores via `softcap · tanh(x / softcap)`.
    pub softcap: f32,
}

impl Attention {
    fn run_impl(
        &self,
        ctx: &OpRunContext,
        past_key: Option<PastCache>,
        past_value: Option<PastCache>,
    ) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();

        // (batch, q_seq, q_hidden) or (batch, q_heads, q_seq, head_size)
        let query: TensorView<f32> = inputs.require_as(0)?;
        let key: TensorView<f32> = inputs.require_as(1)?;
        let value: TensorView<f32> = inputs.require_as(2)?;
        // Broadcastable to (batch, q_heads, q_seq, total_seq).
        let attn_mask = inputs.get(3);
        // (batch,). Number of valid (non-padding) key/value positions per
        // batch row, when the caller manages the KV cache externally and the
        // key/value inputs are right-padded.
        let nonpad_kv_seqlen: Option<NdTensorView<i32, 1>> = inputs.get_as(6)?;

        if ctx.outputs().get(3) {
            return Err(OpError::UnsupportedValue(
                "qk_matmul_output output is not supported",
            ));
        }

        let input_3d = match query.ndim() {
            3 => true,
            4 => false,
            _ => {
                return Err(OpError::InvalidValue("query must have 3 or 4 dimensions"));
            }
        };
        if key.ndim() != query.ndim() || value.ndim() != query.ndim() {
            return Err(OpError::IncompatibleInputShapes(
                "query, key and value must have the same rank",
            ));
        }

        let pool = ctx.pool();

        // Reshape inputs to (batch, heads, seq, head_size) layout.
        let (query, key, value) = if input_3d {
            let q_num_heads = self.q_num_heads.ok_or(OpError::InvalidValue(
                "q_num_heads is required for 3D inputs",
            ))? as usize;
            let kv_num_heads = self.kv_num_heads.ok_or(OpError::InvalidValue(
                "kv_num_heads is required for 3D inputs",
            ))? as usize;
            if q_num_heads == 0 || kv_num_heads == 0 {
                return Err(OpError::InvalidValue(
                    "q_num_heads and kv_num_heads must be positive",
                ));
            }

            let query = query.nd_view::<3>();
            let key = key.nd_view::<3>();
            let value = value.nd_view::<3>();

            let [batch, _q_seq, q_hidden] = query.shape();
            let [key_batch, kv_seq, k_hidden] = key.shape();
            let [value_batch, v_seq, v_hidden] = value.shape();
            if key_batch != batch || value_batch != batch {
                return Err(OpError::IncompatibleInputShapes(
                    "query, key and value must have the same batch size",
                ));
            }
            if kv_seq != v_seq {
                return Err(OpError::IncompatibleInputShapes(
                    "key and value must have the same sequence length",
                ));
            }
            if q_hidden % q_num_heads != 0 {
                return Err(OpError::IncompatibleInputShapes(
                    "query hidden size must be divisible by q_num_heads",
                ));
            }
            let head_size = q_hidden / q_num_heads;
            if k_hidden % kv_num_heads != 0 || v_hidden % kv_num_heads != 0 {
                return Err(OpError::IncompatibleInputShapes(
                    "key/value hidden size must be divisible by kv_num_heads",
                ));
            }
            if k_hidden / kv_num_heads != head_size {
                return Err(OpError::IncompatibleInputShapes(
                    "key head size must match query head size",
                ));
            }
            let v_head_size = v_hidden / kv_num_heads;

            (
                split_attention_heads(pool, query.as_cow(), q_num_heads, head_size)?,
                split_attention_heads(pool, key.as_cow(), kv_num_heads, head_size)?,
                split_attention_heads(pool, value.as_cow(), kv_num_heads, v_head_size)?,
            )
        } else {
            let query = query.nd_view::<4>();
            let key = key.nd_view::<4>();
            let value = value.nd_view::<4>();

            let [batch, _q_heads, _q_seq, head_size] = query.shape();
            let [key_batch, kv_heads, kv_seq, k_head_size] = key.shape();
            let [value_batch, v_kv_heads, value_seq, _v_head_size] = value.shape();
            if key_batch != batch || value_batch != batch {
                return Err(OpError::IncompatibleInputShapes(
                    "query, key and value must have the same batch size",
                ));
            }
            if kv_heads != v_kv_heads || kv_seq != value_seq {
                return Err(OpError::IncompatibleInputShapes(
                    "key and value must have the same number of heads and sequence length",
                ));
            }
            if k_head_size != head_size {
                return Err(OpError::IncompatibleInputShapes(
                    "key head size must match query head size",
                ));
            }

            (query.as_cow(), key.as_cow(), value.as_cow())
        };

        let query = query.auto_return(pool);
        let mut key = key.auto_return(pool);
        let mut value = value.auto_return(pool);

        let [batch, q_num_heads, q_seq, head_size] = query.shape();
        let kv_num_heads = key.size(1);

        if q_num_heads == 0 || kv_num_heads == 0 || !q_num_heads.is_multiple_of(kv_num_heads) {
            return Err(OpError::IncompatibleInputShapes(
                "q_num_heads must be a positive multiple of kv_num_heads",
            ));
        }

        // Concatenate past and present key/value caches along the sequence
        // dimension to form the full (batch, kv_heads, total_seq, head)
        // tensors.
        let has_past_kv = past_key.is_some();
        let past_len = concat_past_kv(pool, past_key, past_value, &mut key, &mut value)?;

        let total_seq = key.size(2);

        if let Some(nonpad) = nonpad_kv_seqlen.as_ref() {
            if has_past_kv {
                return Err(OpError::InvalidValue(
                    "nonpad_kv_seqlen cannot be combined with past_key/past_value",
                ));
            }
            if nonpad.len() != batch {
                return Err(OpError::IncompatibleInputShapes(
                    "nonpad_kv_seqlen must have batch_size elements",
                ));
            }
            if nonpad
                .iter()
                .any(|&len| len < 0 || len as usize > total_seq)
            {
                return Err(OpError::InvalidValue(
                    "nonpad_kv_seqlen entry is out of range",
                ));
            }
        }
        let scale = self
            .scale
            .unwrap_or_else(|| 1.0 / (head_size as f32).sqrt());

        // Validate and broadcast the attention mask.
        let target = [batch, q_num_heads, q_seq, total_seq];
        let mask = match attn_mask {
            None => None,
            Some(ValueView::FloatTensor(mask)) => Some(Mask::Float(
                mask.try_broadcast(target).map_err(|_| BROADCAST_ERROR)?,
            )),
            Some(ValueView::Int32Tensor(mask)) => Some(Mask::Bool(
                mask.try_broadcast(target).map_err(|_| BROADCAST_ERROR)?,
            )),
            Some(_) => {
                return Err(OpError::InvalidValue(
                    "attn_mask must have a float or bool (int32) type",
                ));
            }
        };

        let gemm = GemmExecutor::<f32>::new();
        let attn_out = sdpa_multi_head(
            pool,
            &gemm,
            scale,
            query.view(),
            key.view(),
            value.view(),
            |b, h, row, q_idx| {
                apply_softcap(row, self.softcap);

                // Apply the attention mask. A float mask is added to the
                // scores; a boolean mask disallows attending to positions
                // where the value is zero.
                match mask {
                    Some(Mask::Float(m)) => {
                        for (x, m) in row.iter_mut().zip(m.slice([b, h, q_idx]).iter()) {
                            *x += m;
                        }
                    }
                    Some(Mask::Bool(m)) => {
                        for (x, keep) in row.iter_mut().zip(m.slice([b, h, q_idx]).iter()) {
                            if *keep == 0 {
                                *x = f32::NEG_INFINITY;
                            }
                        }
                    }
                    None => {}
                }

                // Mask future positions for causal attention. With an
                // external KV cache the causal window is anchored by the
                // per-batch valid key/value length instead of the internal
                // cache length.
                if self.is_causal {
                    let offset = match nonpad_kv_seqlen.as_ref() {
                        Some(nonpad) => nonpad[b] as isize - q_seq as isize,
                        None => past_len as isize,
                    };
                    causal_mask_row(row, offset, q_idx, f32::NEG_INFINITY);
                } else if let Some(nonpad) = nonpad_kv_seqlen.as_ref() {
                    // Mask padded key positions.
                    row[nonpad[b] as usize..].fill(f32::NEG_INFINITY);
                }
            },
        );

        // For 3D inputs, reshape the output back to (batch, q_seq, hidden).
        let output: Value = if input_3d {
            merge_attention_heads(pool, attn_out).into()
        } else {
            attn_out.into()
        };

        let mut outputs: OutputList = [output].into();
        if ctx.outputs().get(1) || ctx.outputs().get(2) {
            outputs.push(key.take().into_owned_in(pool).into());
        }
        if ctx.outputs().get(2) {
            outputs.push(value.take().into_owned_in(pool).into());
        }
        Ok(outputs)
    }
}

impl Operator for Attention {
    fn name(&self) -> &str {
        "Attention"
    }

    fn max_inputs(&self) -> Option<usize> {
        // Q, K, V, attn_mask, past_key, past_value, nonpad_kv_seqlen.
        Some(7)
    }

    fn max_outputs(&self) -> Option<usize> {
        // Spec defines 4 outputs: Y, present_key, present_value,
        // qk_matmul_output. The `qk_matmul_output` output is not implemented.
        Some(4)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        // (batch, kv_heads, past_seq, head_size)
        let past_key: Option<NdTensorView<f32, 4>> = ctx.inputs().get_as(4)?;
        // (batch, kv_heads, past_seq, v_head_size)
        let past_value: Option<NdTensorView<f32, 4>> = ctx.inputs().get_as(5)?;
        self.run_impl(
            ctx,
            past_key.map(PastCache::View),
            past_value.map(PastCache::View),
        )
    }

    fn in_place_inputs(&self) -> BitSet<u16> {
        // past_key and past_value.
        BitSet::from_indices([4, 5])
    }

    fn run_in_place(
        &self,
        in_place: InPlaceInputs,
        ctx: &OpRunContext,
    ) -> Result<OutputList, OpError> {
        let (past_key, past_value) = take_past_kv(in_place, 4, 5)?;
        self.run_impl(ctx, past_key, past_value)
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
    Attention,
    op,
    shape_ops::Attention {
        q_num_heads: op.q_num_heads,
        kv_num_heads: op.kv_num_heads,
    }
);

#[cfg(feature = "contrib")]
pub use contrib::{GroupQueryAttention, MultiHeadAttention};

#[cfg(feature = "contrib")]
mod contrib;

#[cfg(test)]
mod tests {
    use rten_base::bit_set::BitSet;
    use rten_gemm::GemmExecutor;
    use rten_simd::SimdOp;
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::expect_equal;
    use rten_tensor::{NdTensor, NdTensorView, Tensor, TensorView};
    use rten_testing::TestCases;
    use rten_vecmath::Softmax as SoftmaxSimd;

    use super::{
        AddSoftmax, Attention, BROADCAST_ERROR, GroupedQueryAttentionMatMul, Mask,
        RepeatInterleave, apply_softcap, sdpa_head,
    };
    use crate::buffer_pool::BufferPool;
    use crate::operator::{InPlaceInputs, InputList, OpError, OpRunContext, Operator, OperatorExt};
    use crate::ops::{Add, Softmax};
    use crate::value::{Value, ValueView};

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

        assert_eq!(out.size(0), 3);
        // The fully-masked row is all-zeros (not NaNs).
        assert_eq!(out.slice(1).to_vec(), vec![0.0; head_size]);
        // Unmasked rows still receive normal (non-zero) attention output.
        assert!(out.slice(0).iter().any(|&x| x != 0.0));
        assert!(out.slice(2).iter().any(|&x| x != 0.0));
    }

    /// Reshape (batch, seq, num_heads * head_size) to (batch, num_heads, seq,
    /// head_size).
    fn split_heads(x: &NdTensor<f32, 3>, num_heads: usize) -> NdTensor<f32, 4> {
        let [batch, seq, hidden] = x.shape();
        let head_size = hidden / num_heads;
        NdTensor::from_fn([batch, num_heads, seq, head_size], |[b, h, s, d]| {
            x[[b, s, h * head_size + d]]
        })
    }

    /// Reshape (batch, num_heads, seq, head_size) to (batch, seq, num_heads *
    /// head_size).
    fn merge_heads(x: &NdTensor<f32, 4>) -> NdTensor<f32, 3> {
        let [batch, num_heads, seq, head_size] = x.shape();
        NdTensor::from_fn([batch, seq, num_heads * head_size], |[b, s, c]| {
            x[[b, c / head_size, s, c % head_size]]
        })
    }

    /// Reference implementation of the ONNX Attention operator operating on
    /// inputs in (batch, num_heads, seq, head_size) layout.
    #[allow(clippy::too_many_arguments)]
    fn reference_attention(
        query: NdTensorView<f32, 4>,
        key: NdTensorView<f32, 4>,
        value: NdTensorView<f32, 4>,
        past_key: Option<NdTensorView<f32, 4>>,
        past_value: Option<NdTensorView<f32, 4>>,
        mask: Option<Mask>,
        nonpad_kv_seqlen: Option<&[i32]>,
        scale: f32,
        softcap: f32,
        is_causal: bool,
    ) -> NdTensor<f32, 4> {
        let [batch, q_heads, q_seq, head_size] = query.shape();
        let [_, kv_heads, kv_seq, _] = key.shape();
        let v_head_size = value.size(3);
        let past_seq = past_key.map(|p| p.size(2)).unwrap_or(0);
        let total = past_seq + kv_seq;
        let kv_factor = q_heads / kv_heads;

        let k_at = |b, h, t: usize, d| {
            if t < past_seq {
                past_key.unwrap()[[b, h, t, d]]
            } else {
                key[[b, h, t - past_seq, d]]
            }
        };
        let v_at = |b, h, t: usize, d| {
            if t < past_seq {
                past_value.unwrap()[[b, h, t, d]]
            } else {
                value[[b, h, t - past_seq, d]]
            }
        };

        let mut out = NdTensor::zeros([batch, q_heads, q_seq, v_head_size]);
        for b in 0..batch {
            for n in 0..q_heads {
                let kvh = n / kv_factor;
                for s in 0..q_seq {
                    let mut scores = vec![0.0f32; total];
                    for (t, score) in scores.iter_mut().enumerate() {
                        let mut dot = 0.0;
                        for d in 0..head_size {
                            dot += query[[b, n, s, d]] * k_at(b, kvh, t, d);
                        }
                        *score = dot * scale;
                    }

                    apply_softcap(&mut scores, softcap);

                    match mask {
                        Some(Mask::Float(m)) => {
                            for (t, score) in scores.iter_mut().enumerate() {
                                *score += m[[b, n, s, t]];
                            }
                        }
                        Some(Mask::Bool(m)) => {
                            for (t, score) in scores.iter_mut().enumerate() {
                                if m[[b, n, s, t]] == 0 {
                                    *score = f32::NEG_INFINITY;
                                }
                            }
                        }
                        None => {}
                    }

                    // Mask padded key positions.
                    if let Some(nonpad) = nonpad_kv_seqlen {
                        for score in scores.iter_mut().skip(nonpad[b] as usize) {
                            *score = f32::NEG_INFINITY;
                        }
                    }

                    // Causal alignment: query `s` attends to `0..=s + offset`,
                    // where the offset is anchored by the internal cache length
                    // or the per-batch valid key/value length.
                    if is_causal {
                        let offset = match nonpad_kv_seqlen {
                            Some(nonpad) => nonpad[b] as isize - q_seq as isize,
                            None => past_seq as isize,
                        };
                        for (t, score) in scores.iter_mut().enumerate() {
                            if t as isize > s as isize + offset {
                                *score = f32::NEG_INFINITY;
                            }
                        }
                    }

                    SoftmaxSimd::new_mut(&mut scores)
                        .flush_nans_to_zero(true)
                        .dispatch();

                    for d in 0..v_head_size {
                        let mut acc = 0.0;
                        for (t, score) in scores.iter().enumerate() {
                            acc += score * v_at(b, kvh, t, d);
                        }
                        out[[b, n, s, d]] = acc;
                    }
                }
            }
        }
        out
    }

    /// Run [`Attention`] with the given inputs, returning all outputs.
    fn run_attention(op: &Attention, inputs: &[Option<ValueView>]) -> Result<Vec<Tensor>, OpError> {
        let input_list = InputList::from_optional(inputs);
        let pool = BufferPool::new();
        let ctx = OpRunContext::new(&pool, &input_list, BitSet::ones(3));
        op.run(&ctx)
            .map(|outputs| outputs.into_iter().map(|o| o.try_into().unwrap()).collect())
    }

    #[test]
    fn test_attention() {
        #[derive(Debug)]
        enum MaskKind {
            None,
            Float,
            Bool,
        }

        #[derive(Debug)]
        struct Case {
            q_num_heads: u32,
            kv_num_heads: u32,
            q_seq: usize,
            kv_seq: usize,
            past_seq: usize,
            batch: usize,
            scale: Option<f32>,
            softcap: f32,
            is_causal: bool,
            mask: MaskKind,
        }

        let cases = [
            // Multi-head self-attention prompt (no past, no grouping).
            Case {
                q_num_heads: 2,
                kv_num_heads: 2,
                q_seq: 4,
                kv_seq: 4,
                past_seq: 0,
                batch: 1,
                scale: None,
                softcap: 0.0,
                is_causal: false,
                mask: MaskKind::None,
            },
            // Grouped-query attention with a custom scale.
            Case {
                q_num_heads: 4,
                kv_num_heads: 2,
                q_seq: 3,
                kv_seq: 3,
                past_seq: 0,
                batch: 1,
                scale: Some(0.3),
                softcap: 0.0,
                is_causal: false,
                mask: MaskKind::None,
            },
            // Multi-query attention.
            Case {
                q_num_heads: 4,
                kv_num_heads: 1,
                q_seq: 3,
                kv_seq: 3,
                past_seq: 0,
                batch: 2,
                scale: None,
                softcap: 0.0,
                is_causal: false,
                mask: MaskKind::None,
            },
            // Causal masking.
            Case {
                q_num_heads: 2,
                kv_num_heads: 2,
                q_seq: 5,
                kv_seq: 5,
                past_seq: 0,
                batch: 1,
                scale: None,
                softcap: 0.0,
                is_causal: true,
                mask: MaskKind::None,
            },
            // Causal masking with a past key/value cache.
            Case {
                q_num_heads: 2,
                kv_num_heads: 1,
                q_seq: 2,
                kv_seq: 2,
                past_seq: 3,
                batch: 1,
                scale: None,
                softcap: 0.0,
                is_causal: true,
                mask: MaskKind::None,
            },
            // Cross-attention where the key/value sequence is shorter than the
            // query sequence.
            Case {
                q_num_heads: 2,
                kv_num_heads: 2,
                q_seq: 5,
                kv_seq: 3,
                past_seq: 0,
                batch: 1,
                scale: None,
                softcap: 0.0,
                is_causal: false,
                mask: MaskKind::None,
            },
            // Cross attention where query sequence length is longer than
            // key/value sequence length. Causal masking is enabled, which is
            // not a sensible configuration for cross-attention, but at least
            // it shouldn't panic.
            Case {
                q_num_heads: 2,
                kv_num_heads: 2,
                q_seq: 5,
                kv_seq: 3,
                past_seq: 0,
                batch: 1,
                scale: None,
                softcap: 0.0,
                is_causal: true,
                mask: MaskKind::None,
            },
            // Cross-attention where the key/value sequence is longer than the
            // query sequence.
            Case {
                q_num_heads: 2,
                kv_num_heads: 2,
                q_seq: 2,
                kv_seq: 4,
                past_seq: 0,
                batch: 1,
                scale: None,
                softcap: 0.0,
                is_causal: false,
                mask: MaskKind::None,
            },
            // Additive float mask.
            Case {
                q_num_heads: 2,
                kv_num_heads: 2,
                q_seq: 3,
                kv_seq: 3,
                past_seq: 0,
                batch: 1,
                scale: None,
                softcap: 0.0,
                is_causal: false,
                mask: MaskKind::Float,
            },
            // Boolean mask.
            Case {
                q_num_heads: 2,
                kv_num_heads: 2,
                q_seq: 3,
                kv_seq: 3,
                past_seq: 0,
                batch: 1,
                scale: None,
                softcap: 0.0,
                is_causal: false,
                mask: MaskKind::Bool,
            },
            // Float mask combined with causal masking.
            Case {
                q_num_heads: 2,
                kv_num_heads: 2,
                q_seq: 4,
                kv_seq: 4,
                past_seq: 0,
                batch: 1,
                scale: None,
                softcap: 0.0,
                is_causal: true,
                mask: MaskKind::Float,
            },
            // Boolean mask with a past KV cache.
            Case {
                q_num_heads: 2,
                kv_num_heads: 1,
                q_seq: 2,
                kv_seq: 2,
                past_seq: 3,
                batch: 1,
                scale: None,
                softcap: 0.0,
                is_causal: false,
                mask: MaskKind::Bool,
            },
            // Float mask with a past KV cache and causal masking.
            Case {
                q_num_heads: 2,
                kv_num_heads: 2,
                q_seq: 2,
                kv_seq: 2,
                past_seq: 2,
                batch: 1,
                scale: None,
                softcap: 0.0,
                is_causal: true,
                mask: MaskKind::Float,
            },
            // Softcapping.
            Case {
                q_num_heads: 2,
                kv_num_heads: 2,
                q_seq: 4,
                kv_seq: 4,
                past_seq: 0,
                batch: 1,
                scale: None,
                softcap: 20.0,
                is_causal: false,
                mask: MaskKind::None,
            },
        ];

        cases.test_each(|case| {
            let &Case {
                q_num_heads,
                kv_num_heads,
                q_seq,
                kv_seq,
                past_seq,
                batch,
                scale,
                softcap,
                is_causal,
                ref mask,
            } = case;
            let head_size = 8;
            let q_hidden = q_num_heads as usize * head_size;
            let kv_hidden = kv_num_heads as usize * head_size;
            let total = past_seq + kv_seq;

            let mut rng = XorShiftRng::new(1234);
            let query = NdTensor::<f32, 3>::rand([batch, q_seq, q_hidden], &mut rng);
            let key = NdTensor::<f32, 3>::rand([batch, kv_seq, kv_hidden], &mut rng);
            let value = NdTensor::<f32, 3>::rand([batch, kv_seq, kv_hidden], &mut rng);
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

            let (float_mask, bool_mask) = match mask {
                MaskKind::None => (None, None),
                MaskKind::Float => (
                    Some(NdTensor::<f32, 4>::rand(
                        [batch, q_num_heads as usize, q_seq, total],
                        &mut rng,
                    )),
                    None,
                ),
                MaskKind::Bool => {
                    // Generate a boolean mask, forcing the first key position to
                    // be attended so that no query row is fully masked.
                    let mut m = NdTensor::<i32, 4>::from_fn(
                        [batch, q_num_heads as usize, q_seq, total],
                        |[_, _, _, t]| if t % 2 == 0 { 1 } else { 0 },
                    );
                    m.slice_mut((.., .., .., 0)).fill(1);
                    (None, Some(m))
                }
            };
            let ref_mask = match (&float_mask, &bool_mask) {
                (Some(m), _) => Some(Mask::Float(m.view())),
                (_, Some(m)) => Some(Mask::Bool(m.view())),
                _ => None,
            };

            let op = Attention {
                is_causal,
                kv_num_heads: Some(kv_num_heads),
                q_num_heads: Some(q_num_heads),
                scale,
                softcap,
            };
            let resolved_scale = scale.unwrap_or(1.0 / (head_size as f32).sqrt());

            // Build expected output using the reference implementation.
            let expected = reference_attention(
                split_heads(&query, q_num_heads as usize).view(),
                split_heads(&key, kv_num_heads as usize).view(),
                split_heads(&value, kv_num_heads as usize).view(),
                past_key.as_ref().map(|p| p.view()),
                past_value.as_ref().map(|p| p.view()),
                ref_mask,
                None,
                resolved_scale,
                softcap,
                is_causal,
            );
            let expected = merge_heads(&expected);

            let mask_input = match (&float_mask, &bool_mask) {
                (Some(m), _) => Some(ValueView::from(m.view())),
                (_, Some(m)) => Some(ValueView::from(m.view())),
                _ => None,
            };
            let inputs = [
                Some(ValueView::from(query.view())),
                Some(ValueView::from(key.view())),
                Some(ValueView::from(value.view())),
                mask_input,
                past_key.as_ref().map(|p| ValueView::from(p.view())),
                past_value.as_ref().map(|p| ValueView::from(p.view())),
            ];
            let outputs = run_attention(&op, &inputs).unwrap();

            expect_equal(&outputs[0], &expected.into_dyn()).unwrap();

            // Check the present key/value cache outputs.
            assert_eq!(outputs.len(), 3);
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
    fn test_attention_4d_inputs() {
        let batch = 1;
        let q_heads = 4;
        let kv_heads = 2;
        let q_seq = 3;
        let head_size = 8;

        let mut rng = XorShiftRng::new(5678);
        let query = NdTensor::<f32, 4>::rand([batch, q_heads, q_seq, head_size], &mut rng);
        let key = NdTensor::<f32, 4>::rand([batch, kv_heads, q_seq, head_size], &mut rng);
        let value = NdTensor::<f32, 4>::rand([batch, kv_heads, q_seq, head_size], &mut rng);

        let op = Attention {
            is_causal: true,
            kv_num_heads: None,
            q_num_heads: None,
            scale: None,
            softcap: 0.0,
        };
        let scale = 1.0 / (head_size as f32).sqrt();
        let expected = reference_attention(
            query.view(),
            key.view(),
            value.view(),
            None,
            None,
            None,
            None,
            scale,
            0.0,
            true,
        );

        let inputs = [
            Some(ValueView::from(query.view())),
            Some(ValueView::from(key.view())),
            Some(ValueView::from(value.view())),
        ];
        let outputs = run_attention(&op, &inputs).unwrap();

        // 4D inputs produce a 4D output with the same layout.
        expect_equal(&outputs[0], &expected.into_dyn()).unwrap();
    }

    #[test]
    fn test_attention_key_value_seq_mismatch() {
        let mut rng = XorShiftRng::new(9012);
        let query = NdTensor::<f32, 3>::rand([1, 3, 16], &mut rng);
        let key = NdTensor::<f32, 3>::rand([1, 4, 16], &mut rng);
        let value = NdTensor::<f32, 3>::rand([1, 3, 16], &mut rng);

        let op = Attention {
            is_causal: false,
            kv_num_heads: Some(2),
            q_num_heads: Some(2),
            scale: None,
            softcap: 0.0,
        };
        let inputs = [
            Some(ValueView::from(query.view())),
            Some(ValueView::from(key.view())),
            Some(ValueView::from(value.view())),
        ];
        let err = run_attention(&op, &inputs).unwrap_err();
        assert_eq!(
            err,
            OpError::IncompatibleInputShapes("key and value must have the same sequence length")
        );
    }

    #[test]
    fn test_attention_nonpad_kv_seqlen() {
        #[derive(Debug)]
        struct Case {
            /// Valid key/value length per batch row. Batch size is the length
            /// of this vec.
            valid_lens: Vec<i32>,
            q_seq: usize,
            is_causal: bool,
        }

        let kv_seq = 5;
        let cases = [
            // Right-padded keys/values, bidirectional attention.
            Case {
                valid_lens: vec![5, 3],
                q_seq: 4,
                is_causal: false,
            },
            // Causal masking anchored per batch row: query `s` attends to key
            // positions `0..=s + valid_len - q_seq`.
            Case {
                valid_lens: vec![5, 3],
                q_seq: 3,
                is_causal: true,
            },
            // Valid length shorter than the query sequence gives a negative
            // causal offset, fully masking the leading query rows (which
            // produce zero output).
            Case {
                valid_lens: vec![1],
                q_seq: 3,
                is_causal: true,
            },
        ];

        cases.test_each(|case| {
            let &Case {
                ref valid_lens,
                q_seq,
                is_causal,
            } = case;
            let batch = valid_lens.len();
            let num_heads = 2;
            let head_size = 8;
            let hidden = num_heads * head_size;

            let mut rng = XorShiftRng::new(3456);
            let query = NdTensor::<f32, 3>::rand([batch, q_seq, hidden], &mut rng);
            let key = NdTensor::<f32, 3>::rand([batch, kv_seq, hidden], &mut rng);
            let value = NdTensor::<f32, 3>::rand([batch, kv_seq, hidden], &mut rng);
            let nonpad = NdTensor::<i32, 1>::from(valid_lens.clone());

            let op = Attention {
                is_causal,
                kv_num_heads: Some(num_heads as u32),
                q_num_heads: Some(num_heads as u32),
                scale: None,
                softcap: 0.0,
            };
            let scale = 1.0 / (head_size as f32).sqrt();
            let expected = reference_attention(
                split_heads(&query, num_heads).view(),
                split_heads(&key, num_heads).view(),
                split_heads(&value, num_heads).view(),
                None,
                None,
                None,
                Some(valid_lens),
                scale,
                0.0,
                is_causal,
            );
            let expected = merge_heads(&expected);

            let inputs = [
                Some(ValueView::from(query.view())),
                Some(ValueView::from(key.view())),
                Some(ValueView::from(value.view())),
                None,
                None,
                None,
                Some(ValueView::from(nonpad.view())),
            ];
            let outputs = run_attention(&op, &inputs).unwrap();
            expect_equal(&outputs[0], &expected.into_dyn()).unwrap();
        });
    }

    #[test]
    fn test_attention_nonpad_kv_seqlen_invalid() {
        #[derive(Debug)]
        struct Case {
            nonpad: Vec<i32>,
            with_past: bool,
            expected: OpError,
        }

        let cases = [
            Case {
                nonpad: vec![3, 3],
                with_past: true,
                expected: OpError::InvalidValue(
                    "nonpad_kv_seqlen cannot be combined with past_key/past_value",
                ),
            },
            Case {
                nonpad: vec![3],
                with_past: false,
                expected: OpError::IncompatibleInputShapes(
                    "nonpad_kv_seqlen must have batch_size elements",
                ),
            },
            Case {
                nonpad: vec![-1, 3],
                with_past: false,
                expected: OpError::InvalidValue("nonpad_kv_seqlen entry is out of range"),
            },
            Case {
                nonpad: vec![4, 3],
                with_past: false,
                expected: OpError::InvalidValue("nonpad_kv_seqlen entry is out of range"),
            },
        ];

        cases.test_each(|case| {
            let batch = 2;
            let kv_seq = 3;
            let num_heads = 2;
            let head_size = 8;
            let hidden = num_heads * head_size;

            let mut rng = XorShiftRng::new(7890);
            let query = NdTensor::<f32, 3>::rand([batch, kv_seq, hidden], &mut rng);
            let key = NdTensor::<f32, 3>::rand([batch, kv_seq, hidden], &mut rng);
            let value = NdTensor::<f32, 3>::rand([batch, kv_seq, hidden], &mut rng);
            let past = NdTensor::<f32, 4>::rand([batch, num_heads, 2, head_size], &mut rng);
            let nonpad = NdTensor::<i32, 1>::from(case.nonpad.clone());

            let op = Attention {
                is_causal: false,
                kv_num_heads: Some(num_heads as u32),
                q_num_heads: Some(num_heads as u32),
                scale: None,
                softcap: 0.0,
            };
            let past_input = case.with_past.then(|| ValueView::from(past.view()));
            let inputs = [
                Some(ValueView::from(query.view())),
                Some(ValueView::from(key.view())),
                Some(ValueView::from(value.view())),
                None,
                past_input.clone(),
                past_input,
                Some(ValueView::from(nonpad.view())),
            ];
            let err = run_attention(&op, &inputs).unwrap_err();
            assert_eq!(err, case.expected);
        });
    }

    /// Check that running an attention operator in-place produces the same
    /// outputs as a normal run.
    pub(super) fn check_in_place_kv_cache(
        op: &dyn Operator,
        inputs: &[Option<ValueView>],
        past_key_index: usize,
        past_value_index: usize,
        max_seq: Option<usize>,
    ) {
        let pool = BufferPool::new();

        // Reference outputs from a normal run with the caches passed as views.
        let input_list = InputList::from_optional(inputs);
        let ctx = OpRunContext::new(&pool, &input_list, BitSet::ones(3));
        let expected: Vec<Tensor> = op
            .run(&ctx)
            .unwrap()
            .into_iter()
            .map(|o| o.try_into().unwrap())
            .collect();

        // Convert a past cache into an owned buffer, reserving spare capacity to
        // grow the sequence dimension (axis 2) up to `max_seq` when it is set.
        let owned_cache = |index: usize| -> NdTensor<f32, 4> {
            let data: NdTensorView<f32, 4> = inputs[index].clone().unwrap().try_into().unwrap();
            if let Some(max_seq) = max_seq {
                let [batch, heads, _seq, head_size] = data.shape();
                let mut cache = NdTensor::with_capacity([batch, heads, max_seq, head_size], 2);
                cache.append(2, &data).unwrap();
                cache
            } else {
                data.to_tensor()
            }
        };

        // Convert the past caches into owned buffers and remove them from the
        // input list, so they are passed as in-place inputs instead.
        let past_key = owned_cache(past_key_index);
        let past_value = owned_cache(past_value_index);
        let past_key_ptr = past_key.data_ptr();
        let past_value_ptr = past_value.data_ptr();

        let mut inputs = inputs.to_vec();
        inputs[past_key_index] = None;
        inputs[past_value_index] = None;
        let input_list = InputList::from_optional(&inputs);
        let ctx = OpRunContext::new(&pool, &input_list, BitSet::ones(3));
        let in_place = InPlaceInputs::from_iter([
            (past_key_index, Value::from(past_key)),
            (past_value_index, Value::from(past_value)),
        ]);
        let mut outputs = op.run_in_place(in_place, &ctx).unwrap();

        let present_value: NdTensor<f32, 4> = outputs.remove(2).try_into().unwrap();
        let present_key: NdTensor<f32, 4> = outputs.remove(1).try_into().unwrap();
        let output: Tensor = outputs.remove(0).try_into().unwrap();

        // The present caches reuse the past buffers exactly when capacity was
        // reserved, otherwise a new buffer is allocated.
        assert_eq!(present_key.data_ptr() == past_key_ptr, max_seq.is_some());
        assert_eq!(
            present_value.data_ptr() == past_value_ptr,
            max_seq.is_some()
        );

        expect_equal(&output, &expected[0]).unwrap();
        expect_equal(&present_key.into_dyn(), &expected[1]).unwrap();
        expect_equal(&present_value.into_dyn(), &expected[2]).unwrap();
    }

    #[test]
    fn test_attention_in_place_kv_cache() {
        let batch = 1;
        let q_heads = 2;
        let kv_heads = 2;
        let head_size = 4;
        let past_seq = 3;
        let seq = 1;

        let mut rng = XorShiftRng::new(1234);
        let query = NdTensor::<f32, 4>::rand([batch, q_heads, seq, head_size], &mut rng);
        let key = NdTensor::<f32, 4>::rand([batch, kv_heads, seq, head_size], &mut rng);
        let value = NdTensor::<f32, 4>::rand([batch, kv_heads, seq, head_size], &mut rng);
        let past_key = NdTensor::<f32, 4>::rand([batch, kv_heads, past_seq, head_size], &mut rng);
        let past_value = NdTensor::<f32, 4>::rand([batch, kv_heads, past_seq, head_size], &mut rng);

        let op = Attention {
            is_causal: true,
            kv_num_heads: None,
            q_num_heads: None,
            scale: None,
            softcap: 0.0,
        };

        // past_key and past_value occupy inputs 4 and 5.
        let inputs = [
            Some(ValueView::from(query.view())),
            Some(ValueView::from(key.view())),
            Some(ValueView::from(value.view())),
            None,
            Some(ValueView::from(past_key.view())),
            Some(ValueView::from(past_value.view())),
        ];

        // Test both with and without reserved capacity in the past caches.
        for max_seq in [Some(past_seq + seq), None] {
            check_in_place_kv_cache(&op, &inputs, 4, 5, max_seq);
        }
    }
}
