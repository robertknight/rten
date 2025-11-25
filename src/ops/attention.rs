//! Attention-related operations.

use rayon::prelude::*;
use rten_gemm::{GemmExecutor, GemmInputA, GemmInputB, GemmUninitOptions};
use rten_simd::SimdOp;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensorView, Tensor, TensorView};
use rten_vecmath::Softmax;

use crate::buffer_pool::{AutoReturn, BufferPool};
use crate::operator::{IntoOpResult, OpError, OpRunContext, Operator, OutputList};
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
        "QueryKeyValueMatMul"
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
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::expect_equal;
    use rten_tensor::{NdTensor, Tensor, TensorView};
    use rten_testing::TestCases;

    use super::{AddSoftmax, BROADCAST_ERROR, GroupedQueryAttentionMatMul, RepeatInterleave};
    use crate::operator::{OpError, OperatorExt};
    use crate::ops::{Add, Softmax};

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
}
