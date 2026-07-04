use crate::infer_shapes::{InferShapes, InferShapesContext, InferShapesError};
use crate::sym_expr::SymExpr;
use crate::sym_gen::SymbolGen;
use crate::sym_tensor::SymTensor;

/// Placeholder shapes for the three attention outputs (attention output,
/// present_key, present_value) when the input shapes are insufficient to
/// infer the real shapes.
fn unknown_outputs() -> Vec<SymTensor> {
    (0..3)
        .map(|_| SymTensor::unknown("unknown attention shape"))
        .collect()
}

/// Total present key/value sequence length: the new key/value length plus any
/// past KV cache.
fn total_kv_seq(past_key: Option<&SymTensor>, kv_seq: SymExpr) -> SymExpr {
    match past_key.and_then(|p| p.size(2)) {
        Some(past_seq) => past_seq + kv_seq,
        None => kv_seq,
    }
}

/// MultiHeadAttention operator.
///
/// See <https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MultiHeadAttention>.
pub struct MultiHeadAttention {
    pub num_heads: u32,
}

impl InferShapes for MultiHeadAttention {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        if self.num_heads == 0 || self.num_heads > i32::MAX as u32 {
            return Err(InferShapesError::InvalidValue);
        }
        let num_heads = SymExpr::Value(self.num_heads as i32);

        // Inputs: query (0), key (1), value (2), bias (3), key_padding_mask (4),
        // attention_bias (5), past_key (6), past_value (7), ...
        let query = inputs.require(0)?;
        let key = inputs.get(1);
        let value = inputs.get(2);
        let past_key = inputs.get(6);

        let Some(query_ndim) = query.ndim() else {
            return Ok(unknown_outputs());
        };

        // Batch and query sequence length are the first two dims in both the
        // separate `[batch, seq, hidden]` and packed
        // `[batch, kv_seq, num_heads, 3, head_size]` query formats.
        let (Some(batch), Some(seq)) = (query.size(0), query.size(1)) else {
            return Ok(unknown_outputs());
        };

        // Resolve the output hidden size, the key/value sequence length and the
        // per-head sizes used by the present key/value outputs.
        let resolved = match query_ndim {
            // Packed QKV. Query shape is (batch, kv_seq, num_heads, 3, head_dim).
            // KV sequence length is same as query.
            5 => query.size(4).map(|head_size| {
                let out_hidden = num_heads.clone() * head_size.clone();
                (out_hidden, seq.clone(), head_size.clone(), head_size)
            }),

            // Separate Q/K/V. Shape of query, key and value are (batch, seq,
            // hidden) where hidden = num_heads * head_size.
            3 => {
                // TODO: The `[SymTensor]` type used to represent inputs in
                // shape inference cannot distinguish between a missing input
                // and an input with unknown shape. Here we treat unknown key/value
                // rank as a missing input, but this could be incorrect.
                let key = key.filter(|k| k.ndim().is_some()).unwrap_or(query);
                let value = value.filter(|v| v.ndim().is_some()).unwrap_or(query);
                match (key.size(1), query.size(2), value.size(2)) {
                    (Some(kv_seq), Some(q_hidden), Some(v_hidden)) => {
                        let qk_head_size = q_hidden / num_heads.clone();
                        let v_head_size = v_hidden.clone() / num_heads.clone();
                        Some((v_hidden, kv_seq, qk_head_size, v_head_size))
                    }
                    _ => None,
                }
            }

            _ => return Err(InferShapesError::IncorrectRank),
        };

        let Some((v_hidden, kv_seq, qk_head_size, v_head_size)) = resolved else {
            return Ok(unknown_outputs());
        };

        // Output 0: attention output `[batch, seq, v_hidden]`.
        let output = SymTensor::from_shape(vec![batch.clone(), seq, v_hidden]);

        // Outputs 1 & 2: present key/value
        // `[batch, num_heads, kv_seq + past_seq, head_size]`. The total sequence
        // length includes any past KV cache that was passed in.
        let total_seq = total_kv_seq(past_key, kv_seq);
        let present_key = SymTensor::from_shape(vec![
            batch.clone(),
            num_heads.clone(),
            total_seq.clone(),
            qk_head_size,
        ]);
        let present_value = SymTensor::from_shape(vec![batch, num_heads, total_seq, v_head_size]);

        Ok([output, present_key, present_value].into())
    }
}

/// GroupQueryAttention operator.
///
/// See <https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GroupQueryAttention>.
pub struct GroupQueryAttention {
    pub num_heads: u32,
    pub kv_num_heads: u32,
}

impl InferShapes for GroupQueryAttention {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        if self.num_heads == 0
            || self.kv_num_heads == 0
            || self.num_heads > i32::MAX as u32
            || self.kv_num_heads > i32::MAX as u32
        {
            return Err(InferShapesError::InvalidValue);
        }
        let num_heads = SymExpr::Value(self.num_heads as i32);
        let kv_num_heads = SymExpr::Value(self.kv_num_heads as i32);

        // Inputs: query (0), key (1), value (2), past_key (3), past_value (4),
        // seqlens_k (5), total_sequence_length (6), ...
        let query = inputs.require(0)?;
        let past_key = inputs.get(3);

        let Some(query_ndim) = query.ndim() else {
            return Ok(unknown_outputs());
        };
        // Only the separate `[batch, seq, hidden]` query format is supported.
        if query_ndim != 3 {
            return Err(InferShapesError::IncorrectRank);
        }

        let (Some(batch), Some(seq), Some(q_hidden)) =
            (query.size(0), query.size(1), query.size(2))
        else {
            return Ok(unknown_outputs());
        };

        // `q_hidden = num_heads * head_size`, so the per-head size follows from
        // the query hidden size.
        let head_size = q_hidden.clone() / num_heads;

        // The present KV cache holds the past cache plus the new key/value
        // tokens. New KV length equals the query sequence length.
        let total_seq = total_kv_seq(past_key, seq.clone());

        // Output 0: attention output `[batch, seq, q_hidden]`.
        let output = SymTensor::from_shape(vec![batch.clone(), seq, q_hidden]);

        // Outputs 1 & 2: present key/value
        // `[batch, kv_num_heads, past_seq + seq, head_size]`.
        let present_key = SymTensor::from_shape(vec![
            batch.clone(),
            kv_num_heads.clone(),
            total_seq.clone(),
            head_size.clone(),
        ]);
        let present_value = SymTensor::from_shape(vec![batch, kv_num_heads, total_seq, head_size]);

        Ok([output, present_key, present_value].into())
    }
}

/// Attention operator (ai.onnx).
///
/// See <https://onnx.ai/onnx/operators/onnx__Attention.html>.
pub struct Attention {
    /// Number of query heads. Required (and only used) when Q/K/V are 3D.
    pub q_num_heads: Option<u32>,
    /// Number of key/value heads. Required (and only used) when Q/K/V are 3D.
    pub kv_num_heads: Option<u32>,
}

impl InferShapes for Attention {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        // Validate the head-count attributes that are set.
        for n in [self.q_num_heads, self.kv_num_heads].into_iter().flatten() {
            if n == 0 || n > i32::MAX as u32 {
                return Err(InferShapesError::InvalidValue);
            }
        }

        // Inputs: query (0), key (1), value (2), attn_mask (3), past_key (4),
        // past_value (5), nonpad_kv_seqlen (6).
        let query = inputs.require(0)?;
        let (Some(key), Some(value)) = (inputs.get(1), inputs.get(2)) else {
            return Err(InferShapesError::IncorrectInputCount);
        };
        let past_key = inputs.get(4);

        // Three outputs: attention output, present_key, present_value. (The
        // optional `qk_matmul_output` output is not implemented.)
        let Some(query_ndim) = query.ndim() else {
            return Ok(unknown_outputs());
        };
        if !matches!(query_ndim, 3 | 4) {
            return Err(InferShapesError::IncorrectRank);
        }
        let Some(batch) = query.size(0) else {
            return Ok(unknown_outputs());
        };

        // Resolve output 0 plus the KV head count, per-head sizes and new
        // key/value length used by the present key/value outputs.
        let (output, kv_heads, kv_seq, head_size, v_head_size) = if query_ndim == 3 {
            // Separate Q/K/V where Q is `[batch, q_seq, q_hidden]`, K is
            // `[batch, kv_seq, k_hidden]` and V is `[batch, kv_seq, v_hidden]`.
            // The head counts come from the operator attributes.
            let (Some(q_num_heads), Some(kv_num_heads)) = (self.q_num_heads, self.kv_num_heads)
            else {
                // Head counts are required to infer the per-head sizes.
                return Ok(unknown_outputs());
            };
            let q_num_heads = SymExpr::Value(q_num_heads as i32);
            let kv_num_heads = SymExpr::Value(kv_num_heads as i32);

            let (Some(q_seq), Some(q_hidden)) = (query.size(1), query.size(2)) else {
                return Ok(unknown_outputs());
            };
            let (Some(kv_seq), Some(v_hidden)) = (key.size(1), value.size(2)) else {
                return Ok(unknown_outputs());
            };

            let head_size = q_hidden / q_num_heads.clone();
            let v_head_size = v_hidden / kv_num_heads.clone();
            // Output hidden size is `q_num_heads * v_head_size`.
            let out_hidden = q_num_heads * v_head_size.clone();

            // Output 0: `[batch, q_seq, q_num_heads * v_head_size]`.
            let output = SymTensor::from_shape(vec![batch.clone(), q_seq, out_hidden]);
            (output, kv_num_heads, kv_seq, head_size, v_head_size)
        } else {
            // Q is `[batch, q_heads, q_seq, head_size]`, K is
            // `[batch, kv_heads, kv_seq, head_size]` and V is
            // `[batch, kv_heads, kv_seq, v_head_size]`. The head counts come
            // from the input shapes.
            let (Some(q_heads), Some(q_seq), Some(head_size)) =
                (query.size(1), query.size(2), query.size(3))
            else {
                return Ok(unknown_outputs());
            };
            let (Some(kv_heads), Some(kv_seq), Some(v_head_size)) =
                (key.size(1), key.size(2), value.size(3))
            else {
                return Ok(unknown_outputs());
            };

            // Output 0: `[batch, q_heads, q_seq, v_head_size]`.
            let output =
                SymTensor::from_shape(vec![batch.clone(), q_heads, q_seq, v_head_size.clone()]);
            (output, kv_heads, kv_seq, head_size, v_head_size)
        };

        // Outputs 1 & 2: present key/value
        // `[batch, kv_heads, total_seq, {head_size, v_head_size}]`.
        let total_seq = total_kv_seq(past_key, kv_seq);
        let present_key = SymTensor::from_shape(vec![
            batch.clone(),
            kv_heads.clone(),
            total_seq.clone(),
            head_size,
        ]);
        let present_value = SymTensor::from_shape(vec![batch, kv_heads, total_seq, v_head_size]);
        Ok([output, present_key, present_value].into())
    }
}

#[cfg(test)]
mod tests {
    use crate::infer_shapes::{InferShapes, InferShapesError};
    use crate::sym_expr::SymExpr;
    use crate::sym_gen::SymbolGen;
    use crate::sym_tensor::{SymTensor, sym_shape};

    use super::{Attention, GroupQueryAttention, MultiHeadAttention};

    fn infer(num_heads: u32, inputs: &[SymTensor]) -> Vec<SymTensor> {
        let mut sym_gen = SymbolGen::new();
        let op = MultiHeadAttention { num_heads };
        op.infer_shapes(inputs.to_vec().into(), &mut sym_gen)
            .unwrap()
            .into_iter()
            .map(SymTensor::simplify)
            .collect()
    }

    #[test]
    fn test_self_attention() {
        // Self-attention: only the query is provided, so key == value == query.
        let query = sym_shape!("batch", "seq", 768);
        let result = infer(12, &[query]);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], sym_shape!("batch", "seq", 768));
        assert_eq!(result[1], sym_shape!("batch", 12, "seq", 64));
        assert_eq!(result[2], sym_shape!("batch", 12, "seq", 64));
    }

    #[test]
    fn test_cross_attention() {
        // Cross-attention with distinct key/value sequence length and a value
        // hidden size that differs from the query/key hidden size.
        let query = sym_shape!("batch", "q_seq", 768);
        let key = sym_shape!("batch", "kv_seq", 768);
        let value = sym_shape!("batch", "kv_seq", 1024);
        let result = infer(16, &[query, key, value]);
        assert_eq!(result[0], sym_shape!("batch", "q_seq", 1024));
        assert_eq!(result[1], sym_shape!("batch", 16, "kv_seq", 48));
        assert_eq!(result[2], sym_shape!("batch", 16, "kv_seq", 64));
    }

    #[test]
    fn test_past_kv_cache() {
        // present_key/value sequence length includes the past KV cache.
        let query = sym_shape!("batch", "seq", 768);
        let key = sym_shape!("batch", "seq", 768);
        let value = sym_shape!("batch", "seq", 768);
        let bias = SymTensor::unknown("bias");
        let key_padding_mask = SymTensor::unknown("mask");
        let attention_bias = SymTensor::unknown("attn_bias");
        let past_key = sym_shape!("batch", 12, "past_seq", 64);
        let past_value = sym_shape!("batch", 12, "past_seq", 64);
        let result = infer(
            12,
            &[
                query,
                key,
                value,
                bias,
                key_padding_mask,
                attention_bias,
                past_key,
                past_value,
            ],
        );
        let total_seq = SymExpr::from("seq") + SymExpr::from("past_seq");
        assert_eq!(result[0], sym_shape!("batch", "seq", 768));
        assert_eq!(result[1], sym_shape!("batch", 12, total_seq.clone(), 64));
        assert_eq!(result[2], sym_shape!("batch", 12, total_seq, 64));
    }

    #[test]
    fn test_packed_qkv() {
        // Packed QKV query: `[batch, kv_seq, num_heads, 3, head_size]`.
        let query = sym_shape!("batch", "kv_seq", 12, 3, 64);
        let result = infer(12, &[query]);
        assert_eq!(result[0], sym_shape!("batch", "kv_seq", 768));
        assert_eq!(result[1], sym_shape!("batch", 12, "kv_seq", 64));
        assert_eq!(result[2], sym_shape!("batch", 12, "kv_seq", 64));
    }

    #[test]
    fn test_unknown_query_shape() {
        let result = infer(12, &[SymTensor::unknown("query")]);
        assert_eq!(result.len(), 3);
        for shape in &result {
            assert!(shape.ndim().is_none());
        }
    }

    #[test]
    fn test_invalid_num_heads() {
        let mut sym_gen = SymbolGen::new();
        let op = MultiHeadAttention { num_heads: 0 };
        let err = op
            .infer_shapes([sym_shape!("batch", "seq", 768)].into(), &mut sym_gen)
            .unwrap_err();
        assert_eq!(err, InferShapesError::InvalidValue);
    }

    fn infer_gqa(num_heads: u32, kv_num_heads: u32, inputs: &[SymTensor]) -> Vec<SymTensor> {
        let mut sym_gen = SymbolGen::new();
        let op = GroupQueryAttention {
            num_heads,
            kv_num_heads,
        };
        op.infer_shapes(inputs.to_vec().into(), &mut sym_gen)
            .unwrap()
            .into_iter()
            .map(SymTensor::simplify)
            .collect()
    }

    #[test]
    fn test_gqa_prompt() {
        // Prompt with no past KV cache. 16 query heads share 4 KV heads, so the
        // per-head size is 1024 / 16 = 64 and the KV hidden size is 4 * 64 = 256.
        let query = sym_shape!("batch", "seq", 1024);
        let key = sym_shape!("batch", "seq", 256);
        let value = sym_shape!("batch", "seq", 256);
        let result = infer_gqa(16, 4, &[query, key, value]);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], sym_shape!("batch", "seq", 1024));
        assert_eq!(result[1], sym_shape!("batch", 4, "seq", 64));
        assert_eq!(result[2], sym_shape!("batch", 4, "seq", 64));
    }

    #[test]
    fn test_gqa_with_past_cache() {
        // present_key/value sequence length is past_seq + seq.
        let query = sym_shape!("batch", "seq", 1024);
        let key = sym_shape!("batch", "seq", 256);
        let value = sym_shape!("batch", "seq", 256);
        let past_key = sym_shape!("batch", 4, "past_seq", 64);
        let past_value = sym_shape!("batch", 4, "past_seq", 64);
        let result = infer_gqa(16, 4, &[query, key, value, past_key, past_value]);
        let total_seq = SymExpr::from("past_seq") + SymExpr::from("seq");
        assert_eq!(result[0], sym_shape!("batch", "seq", 1024));
        assert_eq!(result[1], sym_shape!("batch", 4, total_seq.clone(), 64));
        assert_eq!(result[2], sym_shape!("batch", 4, total_seq, 64));
    }

    #[test]
    fn test_gqa_unknown_query_shape() {
        let result = infer_gqa(16, 4, &[SymTensor::unknown("query")]);
        assert_eq!(result.len(), 3);
        for shape in &result {
            assert!(shape.ndim().is_none());
        }
    }

    #[test]
    fn test_gqa_invalid_num_heads() {
        let mut sym_gen = SymbolGen::new();
        let op = GroupQueryAttention {
            num_heads: 16,
            kv_num_heads: 0,
        };
        let err = op
            .infer_shapes([sym_shape!("batch", "seq", 1024)].into(), &mut sym_gen)
            .unwrap_err();
        assert_eq!(err, InferShapesError::InvalidValue);
    }

    fn infer_attention(
        q_num_heads: Option<u32>,
        kv_num_heads: Option<u32>,
        inputs: &[SymTensor],
    ) -> Vec<SymTensor> {
        let mut sym_gen = SymbolGen::new();
        let op = Attention {
            q_num_heads,
            kv_num_heads,
        };
        op.infer_shapes(inputs.to_vec().into(), &mut sym_gen)
            .unwrap()
            .into_iter()
            .map(SymTensor::simplify)
            .collect()
    }

    #[test]
    fn test_attention_3d_self_attention() {
        // 3D multi-head self-attention.
        let query = sym_shape!("batch", "seq", 768);
        let key = sym_shape!("batch", "seq", 768);
        let value = sym_shape!("batch", "seq", 768);
        let result = infer_attention(Some(12), Some(12), &[query, key, value]);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], sym_shape!("batch", "seq", 768));
        assert_eq!(result[1], sym_shape!("batch", 12, "seq", 64));
        assert_eq!(result[2], sym_shape!("batch", 12, "seq", 64));
    }

    #[test]
    fn test_attention_3d_grouped_query() {
        // 3D grouped-query attention: 16 query heads share 4 KV heads, so the
        // per-head size is 1024 / 16 = 64 and the KV hidden size is 4 * 64 = 256.
        let query = sym_shape!("batch", "q_seq", 1024);
        let key = sym_shape!("batch", "kv_seq", 256);
        let value = sym_shape!("batch", "kv_seq", 256);
        let result = infer_attention(Some(16), Some(4), &[query, key, value]);
        assert_eq!(result[0], sym_shape!("batch", "q_seq", 1024));
        assert_eq!(result[1], sym_shape!("batch", 4, "kv_seq", 64));
        assert_eq!(result[2], sym_shape!("batch", 4, "kv_seq", 64));
    }

    #[test]
    fn test_attention_3d_cross_attention() {
        // Distinct key/value sequence length and a value hidden size that differs
        // from the query/key hidden size.
        let query = sym_shape!("batch", "q_seq", 768);
        let key = sym_shape!("batch", "kv_seq", 768);
        let value = sym_shape!("batch", "kv_seq", 1024);
        let result = infer_attention(Some(16), Some(16), &[query, key, value]);
        // Output hidden = q_num_heads * v_head_size = 16 * (1024 / 16) = 1024.
        assert_eq!(result[0], sym_shape!("batch", "q_seq", 1024));
        assert_eq!(result[1], sym_shape!("batch", 16, "kv_seq", 48));
        assert_eq!(result[2], sym_shape!("batch", 16, "kv_seq", 64));
    }

    #[test]
    fn test_attention_3d_past_cache() {
        // present_key/value sequence length includes the past KV cache.
        let query = sym_shape!("batch", "seq", 768);
        let key = sym_shape!("batch", "seq", 768);
        let value = sym_shape!("batch", "seq", 768);
        let attn_mask = SymTensor::unknown("mask");
        let past_key = sym_shape!("batch", 12, "past_seq", 64);
        let past_value = sym_shape!("batch", 12, "past_seq", 64);
        let result = infer_attention(
            Some(12),
            Some(12),
            &[query, key, value, attn_mask, past_key, past_value],
        );
        let total_seq = SymExpr::from("past_seq") + SymExpr::from("seq");
        assert_eq!(result[0], sym_shape!("batch", "seq", 768));
        assert_eq!(result[1], sym_shape!("batch", 12, total_seq.clone(), 64));
        assert_eq!(result[2], sym_shape!("batch", 12, total_seq, 64));
    }

    #[test]
    fn test_attention_4d_inputs() {
        // 4D inputs: head counts come from the shapes, not the attributes.
        let query = sym_shape!("batch", 16, "q_seq", 64);
        let key = sym_shape!("batch", 4, "kv_seq", 64);
        let value = sym_shape!("batch", 4, "kv_seq", 128);
        let result = infer_attention(None, None, &[query, key, value]);
        // 4D inputs produce a 4D output `[batch, q_heads, q_seq, v_head_size]`.
        assert_eq!(result[0], sym_shape!("batch", 16, "q_seq", 128));
        assert_eq!(result[1], sym_shape!("batch", 4, "kv_seq", 64));
        assert_eq!(result[2], sym_shape!("batch", 4, "kv_seq", 128));
    }

    #[test]
    fn test_attention_3d_missing_head_counts() {
        // Without head counts, 3D shapes cannot be inferred.
        let query = sym_shape!("batch", "seq", 768);
        let key = sym_shape!("batch", "seq", 768);
        let value = sym_shape!("batch", "seq", 768);
        let result = infer_attention(None, None, &[query, key, value]);
        assert_eq!(result.len(), 3);
        for shape in &result {
            assert!(shape.ndim().is_none());
        }
    }

    #[test]
    fn test_attention_unknown_query_shape() {
        let result = infer_attention(
            Some(12),
            Some(12),
            &[
                SymTensor::unknown("query"),
                SymTensor::unknown("key"),
                SymTensor::unknown("value"),
            ],
        );
        assert_eq!(result.len(), 3);
        for shape in &result {
            assert!(shape.ndim().is_none());
        }
    }

    #[test]
    fn test_attention_invalid_num_heads() {
        let mut sym_gen = SymbolGen::new();
        let op = Attention {
            q_num_heads: Some(0),
            kv_num_heads: Some(12),
        };
        let err = op
            .infer_shapes([sym_shape!("batch", "seq", 768)].into(), &mut sym_gen)
            .unwrap_err();
        assert_eq!(err, InferShapesError::InvalidValue);
    }
}
