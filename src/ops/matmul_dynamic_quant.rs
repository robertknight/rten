//! Matrix multiplication with int8 weights and dynamically quantized
//! activations.

use rayon::prelude::*;
use rten_gemm::GemmExecutor;
use rten_shape_inference::ops as shape_ops;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, Tensor, TensorView};
use smallvec::SmallVec;

use crate::buffer_pool::{AutoReturn, BufferPool};
use crate::infer_shapes::InferShapes;
use crate::operator::{
    IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
    OutputTypesContext, PrepackedInput, check_eq,
};
use crate::ops::matmul::{matmul_integer, matmul_prepack_b};
use crate::value::{DataType, ValueType, ValueView};

/// Quantize each row of `a`, a `[M, K]` matrix, to `u8` with a separate
/// scale and zero point for each row.
///
/// Values are quantized to `[0, levels]`. The range of each row includes zero,
/// so that zero is exactly representable. Returns the quantized matrix and the
/// scale and zero point of each row.
fn quantize_rows(
    pool: &BufferPool,
    a: NdTensorView<f32, 2>,
    levels: f32,
) -> (NdTensor<u8, 2>, Vec<f32>, Vec<u8>) {
    let [rows, cols] = a.shape();
    let a = a.to_contiguous_in(pool).auto_return(pool);
    let mut quantized = NdTensor::<u8, 2>::uninit_in(pool, [rows, cols]);

    let params: Vec<(f32, u8)> = quantized
        .data_mut()
        .unwrap()
        .par_chunks_mut(cols.max(1))
        .zip(a.data().par_chunks(cols.max(1)))
        .map(|(q_row, row)| {
            let (lo, hi) = row
                .iter()
                .fold((0f32, 0f32), |(lo, hi), &x| (lo.min(x), hi.max(x)));
            let scale = if hi > lo { (hi - lo) / levels } else { 1. };
            let inv_scale = 1. / scale;
            let zero_point = (-lo * inv_scale).round_ties_even().clamp(0., levels);
            for (q, &x) in q_row.iter_mut().zip(row) {
                let q_val = (x * inv_scale + zero_point).round_ties_even() as i32;
                q.write(q_val.clamp(0, levels as i32) as u8);
            }
            (scale, zero_point as u8)
        })
        .collect();

    // Safety: Every row of `quantized` was initialized.
    let quantized = unsafe { quantized.assume_init() };
    let (scales, zero_points) = params.into_iter().unzip();
    (quantized, scales, zero_points)
}

/// Matrix multiplication with int8 weights and dynamically quantized
/// activations.
///
/// Inputs are `A, B, B_scale, bias`, where `A` is an f32 tensor of shape
/// `[..., M, K]`, `B` is an int8 matrix of shape `[K, N]`, `B_scale` is a
/// per-column scale of shape `[N]` and `bias` is an optional per-column bias
/// of shape `[N]`. The output has shape `[..., M, N]`.
///
/// Each row of `A` is quantized with a separate scale and zero point. This is
/// not an ONNX operator. It is created from `MatMul` operators with
/// `DequantizeLinear` weights when dynamic quantization of matrix
/// multiplications is enabled.
#[derive(Debug)]
pub struct MatMulDynamicQuant {}

impl Operator for MatMulDynamicQuant {
    fn name(&self) -> &str {
        "MatMulDynamicQuant"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(4)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let pool = ctx.pool();
        let inputs = ctx.inputs();
        let a: TensorView<f32> = inputs.require_as(0)?;
        let b: NdTensorView<i8, 2> = inputs.require_as(1)?;
        let b_scale: NdTensorView<f32, 1> = inputs.require_as(2)?;
        let bias: Option<NdTensorView<f32, 1>> = inputs.get_as(3)?;
        let packed_b = match inputs.get_prepacked(1) {
            Some(PrepackedInput::Int8BMatrix(packed)) => Some(packed),
            _ => None,
        };

        let [k, n] = b.shape();
        check_eq!(b_scale.size(0), n, "scale length should match column count")?;
        if let Some(bias) = &bias {
            check_eq!(bias.size(0), n, "bias length should match column count")?;
        }
        let Some(&a_cols) = a.shape().last() else {
            return Err(OpError::invalid_value("expected input with >= 1 dims"));
        };
        check_eq!(a_cols, k, "input columns should match weight rows")?;

        let a_rows = a.len() / k.max(1);
        let a_mat = a.reshaped_in(pool, [a_rows, k]).auto_return(pool);

        // Quantized activations use the full `u8` range unless the GEMM kernel
        // can saturate, in which case they are restricted to 7 bits.
        let gemm = GemmExecutor::<u8, i8, i32>::default();
        let levels = if gemm.may_saturate() { 127. } else { 255. };

        let (a_quant, a_scale, a_zero) = quantize_rows(pool, a_mat.view(), levels);
        let a_quant = a_quant.auto_return(pool);
        let a_zero = NdTensorView::from_data([a_rows], a_zero.as_slice());

        let acc = matmul_integer(
            pool,
            a_quant.view().as_dyn(),
            b.as_dyn(),
            Some(a_zero.as_dyn()),
            None,
            packed_b,
        )?
        .auto_return(pool);

        // Convert accumulators to float, apply scales and add bias.
        let b_scale = b_scale.to_contiguous();
        let bias = bias.map(|b| b.to_contiguous());
        let mut out_shape: SmallVec<[usize; 4]> = a.shape().into();
        *out_shape.last_mut().unwrap() = n;
        let mut output = Tensor::<f32>::uninit_in(pool, &out_shape);
        output
            .data_mut()
            .unwrap()
            .par_chunks_mut(n.max(1))
            .zip(acc.data().unwrap().par_chunks(n.max(1)))
            .zip(a_scale.par_iter())
            .for_each(|((out_row, acc_row), &a_scale)| {
                let scaled = out_row.iter_mut().zip(acc_row).zip(b_scale.data());
                if let Some(bias) = &bias {
                    for (((y, &acc), &b_scale), &bias) in scaled.zip(bias.data()) {
                        y.write(acc as f32 * (a_scale * b_scale) + bias);
                    }
                } else {
                    for ((y, &acc), &b_scale) in scaled {
                        y.write(acc as f32 * (a_scale * b_scale));
                    }
                }
            });

        // Safety: Every row of the output was initialized.
        let output = unsafe { output.assume_init() };
        output.into_op_result()
    }

    fn prepack_inputs(&self) -> SmallVec<[usize; 1]> {
        [1].into()
    }

    fn prepack(
        &self,
        index: usize,
        input: ValueView,
        _zero_point: Option<ValueView>,
    ) -> Option<PrepackedInput> {
        if index == 1 {
            matmul_prepack_b::<u8, i8, i32>(input, None)
        } else {
            None
        }
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::Fixed(ValueType::Tensor(DataType::Float))].into())
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(&shape_ops::MatMul)
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::{NdTensor, Tensor};

    use rten_gemm::GemmExecutor;

    use super::MatMulDynamicQuant;
    use crate::operator::{InputList, OperatorExt};

    #[test]
    fn test_matmul_dynamic_quant() {
        let mut rng = XorShiftRng::new(1234);
        let (m, k, n) = (5, 32, 7);

        // Rows with different magnitudes and signs, to check that each row is
        // quantized separately.
        let a = Tensor::<f32>::from_fn(&[1, m, k], |idx| {
            let x = ((idx[1] * 7 + idx[2] * 3) % 11) as f32 - 3.;
            x * 10f32.powi(idx[1] as i32 - 2)
        });
        let b = NdTensor::<i8, 2>::rand([k, n], &mut rng).map(|&x| x.max(-127));
        let b_scale = NdTensor::from_fn([n], |[i]| 0.01 * (i + 1) as f32);
        let bias = NdTensor::from_fn([n], |[i]| i as f32 - 3.);

        for bias in [None, Some(bias.view())] {
            let inputs = [
                Some(a.view().into()),
                Some(b.view().into()),
                Some(b_scale.view().into()),
                bias.map(|b| b.into()),
            ];
            let result: Tensor<f32> = MatMulDynamicQuant {}
                .run_simple(InputList::from_optional(&inputs))
                .unwrap();
            assert_eq!(result.shape(), &[1, m, n]);

            for row in 0..m {
                // Quantize the row as the operator does, with the 7-bit range
                // used if the GEMM can saturate.
                let (lo, hi) = (0..k)
                    .map(|c| a[[0, row, c]])
                    .fold((0f32, 0f32), |(lo, hi), x| (lo.min(x), hi.max(x)));
                let levels = if GemmExecutor::<u8, i8, i32>::default().may_saturate() {
                    127.
                } else {
                    255.
                };
                let scale = (hi - lo) / levels;
                let zero_point = (-lo / scale).round_ties_even();
                let a_quant =
                    |c| ((a[[0, row, c]] / scale + zero_point).round_ties_even()).clamp(0., levels);

                for col in 0..n {
                    let bias = bias.map(|b| b[col]).unwrap_or(0.);
                    let acc: f32 = (0..k)
                        .map(|c| (a_quant(c) - zero_point) * b[[c, col]] as f32)
                        .sum();
                    let expected = acc * scale * b_scale[[col]] + bias;
                    let actual = result[[0, row, col]];
                    assert!(
                        (actual - expected).abs() <= 1e-4 * expected.abs().max(1.),
                        "row {row} col {col}: {actual} != {expected}"
                    );

                    // The result should also be close to an f32 matmul with
                    // the dequantized weights.
                    let f32_expected: f32 = (0..k)
                        .map(|c| a[[0, row, c]] * b[[c, col]] as f32 * b_scale[[col]])
                        .sum::<f32>()
                        + bias;
                    assert!((actual - f32_expected).abs() <= 0.05 * f32_expected.abs().max(1.));
                }
            }
        }
    }
}
