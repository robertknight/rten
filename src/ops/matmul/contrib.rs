//! ONNX Runtime contrib matmul operators.

use rten_gemm::{
    BlockQuantizedError, BlockQuantizedGemm, BlockQuantizedMatrix, ComputeMode, GemmExecutor,
    GemmInputA, GemmInputB, GemmUninitOptions,
};
use rten_shape_inference::ops as shape_ops;
use rten_tensor::prelude::*;
use rten_tensor::{CowNdTensor, NdTensorView, Tensor, TensorView};
use rten_vecmath::ExtendInit;
use smallvec::SmallVec;

use crate::buffer_pool::{AutoReturn, BufferPool};
use crate::infer_shapes::InferShapes;
use crate::operator::{
    IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
    OutputTypesContext,
};
use crate::value::{DataType, ValueType};

fn matmul_nbits(
    pool: &BufferPool,
    lhs: TensorView<f32>,
    rhs: NdTensorView<u8, 3>,
    scales: NdTensorView<f32, 2>,
    bits: u8,
    accuracy: AccuracyLevel,
) -> Result<Tensor<f32>, OpError> {
    if lhs.ndim() < 2 {
        return Err(OpError::InvalidValue("A input must have at least 2 dims"));
    }

    let batch_dims = &lhs.shape()[..lhs.ndim() - 2];
    let rows = lhs.size(lhs.ndim() - 2);
    let lhs_cols = lhs.size(lhs.ndim() - 1);

    let rhs = rhs.to_contiguous_in(pool);
    let scales = scales.to_contiguous_in(pool);

    let b_mat = BlockQuantizedMatrix::new(rhs.view(), scales.view(), bits).map_err(|err| {
        OpError::UnsupportedValue(match err {
            BlockQuantizedError::UnsupportedBlockSize => "Unsupported K block size",
            BlockQuantizedError::UnsupportedElementSize => "Unsupported bits-per-element",
        })
    })?;

    let batch_len = batch_dims.iter().product();
    let out_shape: SmallVec<[usize; 4]> = batch_dims
        .iter()
        .copied()
        .chain([rows, b_mat.cols()])
        .collect();
    let out_len = out_shape.iter().product();
    let mut out_data = pool.alloc(out_len);

    if lhs_cols != b_mat.rows() {
        return Err(OpError::IncompatibleInputShapes(
            "Columns of first matrix does not match rows of second matrix",
        ));
    }

    // For vector-matrix products use an optimized implementation. Otherwise use
    // the standard GEMM implementation which handles block-quantized inputs via
    // a custom packing function, but otherwise uses the same logic as for
    // regular matmuls.
    if rows == 1 {
        let compute = match accuracy {
            AccuracyLevel::Int8 => ComputeMode::Int8,
            AccuracyLevel::Float => ComputeMode::Float,
        };
        let gemm = if BlockQuantizedGemm::is_compute_optimized(compute) {
            BlockQuantizedGemm::new().with_compute(compute)
        } else {
            BlockQuantizedGemm::new()
        };

        let lhs = lhs
            .reshaped_in(pool, [batch_len, rows, lhs_cols])
            .auto_return(pool);
        out_data.extend_init(|uninit_out_data| {
            gemm.batched_gemm_uninit(&mut uninit_out_data[..out_len], lhs.view(), b_mat)
                .unwrap()
        });
    } else {
        let lhs = lhs
            .reshaped_in(pool, [batch_len * rows, lhs_cols])
            .auto_return(pool);

        let gemm = GemmExecutor::default();
        out_data.extend_init(|uninit_out_data| {
            gemm.gemm_uninit(
                &mut uninit_out_data[..out_len],
                GemmInputA::Unpacked(lhs.view()),
                GemmInputB::BlockQuantized(b_mat),
                GemmUninitOptions::default(),
            )
            .unwrap()
        });
    }

    Ok(Tensor::from_data(out_shape.as_slice(), out_data))
}

/// Specifies whether the LHS input may be quantized.
///
/// Using [`Int8`](Self::Int8) quantization can significantly improve
/// performance but may reduce accuracy. The accuracy level that is used may
/// be higher than requested if an optimized implementation of the requested
/// level is not available on the current platform.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AccuracyLevel {
    /// Do not quantize the LHS input.
    Float,
    /// Quantize the LHS to 8-bit integers using blockwise quantization with
    /// the same quantization as the RHS.
    Int8,
}

/// Matrix multiplication of un-quantized LHS by block-quantized RHS.
///
/// See https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MatMulNBits.
#[derive(Debug)]
pub struct MatMulNBits {
    pub bits: u8,
    pub block_size: usize,
    pub accuracy_level: AccuracyLevel,
}

impl Operator for MatMulNBits {
    fn name(&self) -> &str {
        "MatMulNBits"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(3)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let lhs: TensorView<f32> = ctx.inputs().require_as(0)?;
        let rhs: NdTensorView<u8, 3> = ctx.inputs().require_as(1)?;

        // Current spec requires scales to be 2D, but earlier versions used 1D
        // scales. See https://github.com/microsoft/onnxruntime/pull/24828.
        let scales: TensorView<f32> = ctx.inputs().require_as(2)?;

        let scales: CowNdTensor<f32, 2> = match scales.ndim() {
            2 => scales
                .to_contiguous_in(ctx.pool())
                .into_inner()
                .try_into()
                .unwrap(),
            1 => {
                let k = lhs.ndim().checked_sub(1).map(|d| lhs.size(d)).unwrap_or(1);
                let k_blocks = k.checked_div(self.block_size).unwrap_or(0);
                let rhs_cols = rhs.size(0);

                if scales.len() != rhs_cols * k_blocks {
                    return Err(OpError::InvalidValue(
                        "Expected 1D `scales` size to match columns * block_size",
                    ));
                }
                scales.reshaped([rhs_cols, k_blocks])
            }
            _ => {
                return Err(OpError::InvalidValue(
                    "Expected `scales` to have one or two dims",
                ));
            }
        };

        if ctx.inputs().len() > 3 {
            return Err(OpError::UnsupportedValue(
                "zero_points, g_idx and bias inputs are unsupported",
            ));
        }

        matmul_nbits(
            ctx.pool(),
            lhs,
            rhs,
            scales.view(),
            self.bits,
            self.accuracy_level,
        )
        .into_op_result()
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::Fixed(ValueType::Tensor(DataType::Float))].into())
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(&shape_ops::MatMulNBits)
    }
}

#[cfg(test)]
mod tests {
    use rten_gemm::{BlockQuantizedMatrix, GemmExecutor, GemmInputA, GemmInputB, GemmOptions};
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::{expect_equal, expect_equal_with_tolerance};
    use rten_tensor::{NdTensor, NdTensorView, Tensor, TensorView};
    use rten_testing::TestCases;

    use super::{AccuracyLevel, MatMulNBits};
    use crate::operator::{OpError, OperatorExt};

    fn reference_matmul_nbits(
        lhs: TensorView<f32>,
        rhs: NdTensorView<u8, 3>,
        scales: NdTensorView<f32, 2>,
        n_bits: u8,
    ) -> Tensor<f32> {
        let batch_dims = &lhs.shape()[..lhs.ndim() - 2];
        let batch_len: usize = batch_dims.iter().product();

        let m = lhs.size(lhs.ndim() - 2);
        let k = lhs.size(lhs.ndim() - 1);
        let [n, _k_blocks, _block_size] = rhs.shape();

        let rhs = rhs.to_contiguous();
        let scales = scales.to_contiguous();
        let bqm = BlockQuantizedMatrix::new(rhs.view(), scales.view(), n_bits).unwrap();

        let out_shape: Vec<usize> = batch_dims.iter().copied().chain([m, n]).collect();
        let mut out = Tensor::zeros(&out_shape);
        let gemm = GemmExecutor::default();

        gemm.gemm(
            out.data_mut().unwrap(),
            GemmInputA::Unpacked(lhs.reshaped([batch_len * m, k]).view()),
            GemmInputB::BlockQuantized(bqm),
            GemmOptions::default(),
        )
        .unwrap();

        out
    }

    #[test]
    fn test_matmul_nbits() {
        #[derive(Clone, Debug)]
        struct Case {
            batch_dims: Vec<usize>,
            m: usize,
            accuracy_level: AccuracyLevel,
            tolerance: Option<f32>,
        }

        let cases = [
            // Vector-matrix product
            Case {
                batch_dims: [2].into(),
                m: 1,
                accuracy_level: AccuracyLevel::Float,
                tolerance: None,
            },
            Case {
                batch_dims: [2].into(),
                m: 1,
                accuracy_level: AccuracyLevel::Int8,
                tolerance: Some(0.1),
            },
            // Matrix-matrix product
            Case {
                batch_dims: [2].into(),
                m: 4,
                accuracy_level: AccuracyLevel::Float,
                tolerance: None,
            },
            Case {
                batch_dims: [2].into(),
                m: 4,
                accuracy_level: AccuracyLevel::Int8,
                // MatMulNBits currently falls back to float compute for
                // matrix-matrix products, so no tolerance is required.
                tolerance: None,
            },
            // Matrix-matrix product with 2 batch dims.
            Case {
                batch_dims: [2, 2].into(),
                m: 4,
                accuracy_level: AccuracyLevel::Float,
                tolerance: None,
            },
            // Matrix-matrix product with 0 batch dims.
            Case {
                batch_dims: [].into(),
                m: 4,
                accuracy_level: AccuracyLevel::Float,
                tolerance: None,
            },
        ];

        cases.test_each_clone(|case| {
            let Case {
                batch_dims,
                m,
                accuracy_level,
                tolerance,
            } = case;

            let mut rng = XorShiftRng::new(1234);

            let block_size = 16;
            let block_bytes = block_size / 2;
            let k = block_size * 2;
            let n = 8;
            let n_bits = 4;

            let lhs_shape: Vec<usize> = batch_dims.iter().copied().chain([m, k]).collect();
            let lhs = Tensor::rand(&lhs_shape, &mut rng);
            let rhs = NdTensor::<u8, 3>::rand([n, k / block_size, block_bytes], &mut rng);
            let scales = NdTensor::<f32, 2>::rand([n, k / block_size], &mut rng);

            // nb. Reference result is always computed in full precision.
            let expected = reference_matmul_nbits(lhs.as_dyn(), rhs.view(), scales.view(), n_bits);

            let op = MatMulNBits {
                bits: n_bits,
                block_size,
                accuracy_level,
            };

            // With 2D scales.
            let result: Tensor<f32> = op
                .run_simple((lhs.view(), rhs.view(), scales.view()))
                .unwrap();
            if let Some(atol) = tolerance {
                let rtol = 0.;
                expect_equal_with_tolerance(&result, &expected, atol, rtol).unwrap();
            } else {
                expect_equal(&result, &expected).unwrap();
            }

            // With 1D scales (older models)
            let result: Tensor<f32> = op
                .run_simple((
                    lhs.view(),
                    rhs.view(),
                    scales.reshaped([scales.len()]).view(),
                ))
                .unwrap();
            if let Some(atol) = tolerance {
                let rtol = 0.;
                expect_equal_with_tolerance(&result, &expected, atol, rtol).unwrap();
            } else {
                expect_equal(&result, &expected).unwrap();
            }
        });
    }

    #[test]
    fn test_matmul_nbits_invalid() {
        #[derive(Debug)]
        struct Case {
            lhs_shape: Vec<usize>,
            rhs_shape: [usize; 3],
            scales_shape: Vec<usize>,
            expected: OpError,
        }

        let cases = [Case {
            lhs_shape: [1].into(),
            rhs_shape: [1, 1, 16],
            scales_shape: [1, 1].into(),
            expected: OpError::InvalidValue("A input must have at least 2 dims"),
        }];

        cases.test_each(|case| {
            let op = MatMulNBits {
                bits: 4,
                block_size: 32,
                accuracy_level: AccuracyLevel::Float,
            };
            let lhs = Tensor::<f32>::zeros(&case.lhs_shape);
            let rhs = Tensor::<u8>::zeros(&case.rhs_shape);
            let scales = Tensor::<f32>::zeros(&case.scales_shape);
            let result: Result<Tensor<f32>, _> =
                op.run_simple((lhs.view(), rhs.view(), scales.view()));
            assert_eq!(result.err().unwrap(), case.expected);
        });
    }
}
