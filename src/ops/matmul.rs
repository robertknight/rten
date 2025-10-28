use rayon::prelude::*;
use rten_base::byte_cast::{Pod, cast_pod_vec};
use rten_gemm::{
    BiasVector, BlockQuantizedError, BlockQuantizedMatrix, GemmExecutor, GemmInT, GemmInputA,
    GemmInputB, GemmOptions, GemmOutT, GemmUninitOptions, PackedBMatrix, QuantParams,
};
use rten_tensor::prelude::*;
use rten_tensor::{CowNdTensor, Matrix, NdTensor, NdTensorView, Tensor, TensorView};
use rten_vecmath::ExtendInit;
use smallvec::SmallVec;

use crate::buffer_pool::{AutoReturn, BufferPool};
use crate::operator::{
    IntoOpResult, OpError, OpRunContext, Operator, OutputList, PrepackedInput, static_dims,
};
use crate::ops::binary_elementwise::broadcast_shapes;
use crate::ops::layout::expand_to;
use crate::value::ValueView;

/// Compute the General Matrix Multiplication (GEMM) `c = alpha * (ab) + beta * c`.
///
/// If `transpose_a` or `transpose_b` are set, the `a` and `b` inputs
/// respectively are transposed before multiplying them.
///
/// nb. This is named `gemm_op` to avoid confusion with `gemm::gemm`.
pub fn gemm<LhsT: GemmInT, RhsT: GemmInT, OutT: GemmOutT>(
    pool: &BufferPool,
    a: TensorView<LhsT>,
    b: TensorView<RhsT>,
    c: Option<TensorView<OutT>>,
    alpha: f32,
    beta: OutT,
    transpose_a: bool,
    transpose_b: bool,
) -> Result<Tensor<OutT>, OpError>
where
    GemmExecutor<LhsT, RhsT, OutT>: Default,
{
    let a = static_dims!(a, 2)?;
    let b = static_dims!(b, 2)?;

    let a = if transpose_a { a.transposed() } else { a };
    let b = if transpose_b { b.transposed() } else { b };

    let out_shape = &[a.size(0), b.size(1)][..];
    let gemm = GemmExecutor::<LhsT, RhsT, OutT>::default();

    let output = match c {
        Some(c) if beta != OutT::zero() => {
            if !c.can_broadcast_to(out_shape) {
                return Err(OpError::IncompatibleInputShapes(
                    "Cannot broadcast c to output shape",
                ));
            }
            let mut output = expand_to(pool, c, out_shape);
            gemm.gemm(
                output.data_mut().unwrap(),
                GemmInputA::Unpacked(a.nd_view()),
                GemmInputB::Unpacked(b.nd_view()),
                GemmOptions {
                    alpha,
                    beta,
                    ..Default::default()
                },
            )
            .unwrap();
            output
        }
        _ => {
            let out_len = out_shape.iter().product();
            let mut output = pool.alloc(out_len);
            output.extend_init(|uninit| {
                gemm.gemm_uninit(
                    &mut uninit[..out_len],
                    GemmInputA::Unpacked(a.nd_view()),
                    GemmInputB::Unpacked(b.nd_view()),
                    GemmUninitOptions {
                        alpha,
                        ..Default::default()
                    },
                )
                .unwrap()
            });
            Tensor::from_data(out_shape, output)
        }
    };

    Ok(output)
}

#[derive(Debug)]
pub struct Gemm {
    pub alpha: f32,
    pub beta: f32,
    pub transpose_a: bool,
    pub transpose_b: bool,
}

impl Operator for Gemm {
    fn name(&self) -> &str {
        "Gemm"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(3)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let a = inputs.require_as(0)?;
        let b = inputs.require_as(1)?;
        let c = inputs.get_as(2)?;
        gemm::<f32, f32, f32>(
            ctx.pool(),
            a,
            b,
            c,
            self.alpha,
            self.beta,
            self.transpose_a,
            self.transpose_b,
        )
        .into_op_result()
    }
}

/// Hints for how a batched MatMul should be performed. This exists to enable
/// comparisons in tests and benchmarks.
#[derive(Copy, Clone, Debug, PartialEq)]
enum MatmulStrategy {
    /// Use the best strategy for the input shapes.
    Auto,

    /// Perform separate GEMM calls for each pair of matrices to multiply in
    /// the batch.
    #[cfg(test)]
    Batch,
}

fn matmul_prepack_b<LhsT: GemmInT, RhsT: GemmInT, OutT: GemmOutT>(
    input: ValueView,
) -> Option<PrepackedInput>
where
    GemmExecutor<LhsT, RhsT, OutT>: Default,
    PrepackedInput: From<PackedBMatrix<RhsT>>,
    for<'a> TensorView<'a, RhsT>: TryFrom<ValueView<'a>>,
{
    let executor = GemmExecutor::default();
    let tensor: TensorView<RhsT> = input.try_into().ok()?;
    let matrix: Matrix<RhsT> = tensor.try_into().ok()?;
    Some(executor.prepack_b(matrix).into())
}

pub fn matmul<LhsT: GemmInT, RhsT: GemmInT, OutT: Default + GemmOutT>(
    pool: &BufferPool,
    a: TensorView<LhsT>,
    b: TensorView<RhsT>,
    packed_b: Option<&PackedBMatrix<RhsT>>,
) -> Result<Tensor<OutT>, OpError>
where
    GemmExecutor<LhsT, RhsT, OutT>: Default,
{
    matmul_impl(
        pool,
        a,
        b,
        packed_b,
        MatmulStrategy::Auto,
        None,
        None,
        None, /* a_quant */
        None, /* b_quant */
    )
}

fn matmul_impl<LhsT: GemmInT, RhsT: GemmInT, OutT: Default + GemmOutT>(
    pool: &BufferPool,
    mut a: TensorView<LhsT>,
    mut b: TensorView<RhsT>,
    packed_b: Option<&PackedBMatrix<RhsT>>,
    strategy: MatmulStrategy,
    bias: Option<BiasVector<OutT>>,
    alpha: Option<f32>,
    a_quant: Option<QuantParams<LhsT>>,
    b_quant: Option<QuantParams<RhsT>>,
) -> Result<Tensor<OutT>, OpError>
where
    GemmExecutor<LhsT, RhsT, OutT>: Default,
{
    if a.ndim() < 1 || b.ndim() < 1 {
        return Err(OpError::InvalidValue("Inputs must have >= 1 dimensions"));
    }

    // Expand vector inputs to matrices. This follows the rules of `numpy.matmul`.
    // See https://numpy.org/doc/stable/reference/generated/numpy.matmul.html.
    let a_is_vec = a.ndim() == 1;
    if a_is_vec {
        a.insert_axis(0);
    }
    let b_is_vec = b.ndim() == 1;
    if b_is_vec {
        b.insert_axis(1);
    }

    let a_rows = a.size(a.ndim() - 2);
    let a_cols = a.size(a.ndim() - 1);

    let b_rows = b.size(b.ndim() - 2);
    let b_cols = b.size(b.ndim() - 1);

    if a_cols != b_rows {
        return Err(OpError::IncompatibleInputShapes(
            "Columns of first matrix does not match rows of second matrix",
        ));
    }

    let a_prefix = &a.shape()[..a.ndim() - 2];
    let b_prefix = &b.shape()[..b.ndim() - 2];

    let num_a_matrices: usize = a_prefix.iter().product();
    let num_b_matrices: usize = b_prefix.iter().product();

    let out_prefix = broadcast_shapes(a_prefix, b_prefix)
        .ok_or(OpError::IncompatibleInputShapes("Cannot broadcast shapes"))?;
    let out_shape = &[out_prefix.as_slice(), &[a_rows, b_cols]].concat();

    // A batched matrix multiplication with `[A, M, K] x [K, N]`, where `A` can
    // consist of multiple dimensions, can be converted to a non-batched matmul
    // by reshaping the inputs as `[A * M, K]` * `[K, N]`, and then reshaping
    // the `[A * M, N]` output to `[A, M, N]`.
    //
    // The upside is that one larger matmul is likely to be more efficient than
    // `A` smaller matmuls. This is especially true if `M` is small (eg. 1).
    if strategy == MatmulStrategy::Auto && num_a_matrices > 1 && num_b_matrices == 1 {
        // nb. We assume `a` is likely already contiguous, so this will be cheap.
        let a_contig = a.to_contiguous_in(pool).auto_return(pool);
        let a_matrix = a_contig.reshaped([num_a_matrices * a_rows, a_cols].as_slice());

        // Broadcast zero point to match new row count.
        let a_quant: Option<Vec<LhsT>> = a_quant.map(|a_quant| {
            a_quant
                .zero_point
                .iter()
                .copied()
                .cycle()
                .take(a_matrix.size(0))
                .collect()
        });

        let mut output = matmul_impl(
            pool,
            a_matrix.view(),
            b.clone(),
            packed_b,
            strategy,
            bias,
            alpha,
            a_quant.as_ref().map(|zero_point| QuantParams {
                zero_point: zero_point.as_slice(),
            }),
            b_quant,
        )?;
        output.reshape(out_shape);
        return Ok(output);
    }

    // Early exit if the output is empty.
    if out_shape.iter().product::<usize>() == 0 {
        // Don't need to use the pool here since the buffer has zero size.
        return Ok(Tensor::zeros(out_shape));
    }

    let a_broadcast_shape = [out_prefix.as_slice(), &[a_rows, a_cols]].concat();
    let b_broadcast_shape = [out_prefix.as_slice(), &[b_rows, b_cols]].concat();

    let a_broadcast = a.broadcast(a_broadcast_shape.as_slice());
    let b_broadcast = b.broadcast(b_broadcast_shape.as_slice());

    let gemm = GemmExecutor::default();

    // Prepack inputs if they are re-used, to amortize packing cost, or we
    // were already called with prepacked inputs.
    //
    // We don't prepack when the "A" matrix is a vector because that uses a
    // special case vector-matrix algorithm that doesn't benefit from packing.
    let prepacked_a = (num_a_matrices == 1 && num_b_matrices > 1 && a_rows > 1).then(|| {
        let a_matrix = a.inner_iter::<2>().next().unwrap();
        gemm.prepack_a_in(pool, a_matrix).auto_return(pool)
    });
    let prepacked_a = prepacked_a.as_deref();

    let prepacked_b =
        (num_b_matrices == 1 && num_a_matrices > 1 && a_rows > 1 && packed_b.is_none()).then(
            || {
                let b_matrix = b.inner_iter::<2>().next().unwrap();
                gemm.prepack_b_in(pool, b_matrix).auto_return(pool)
            },
        );
    let prepacked_b = prepacked_b
        .as_deref()
        .or(if a_rows > 1 { packed_b } else { None });

    let a_mats: SmallVec<[_; 1]> = a_broadcast
        .inner_iter::<2>()
        .map(|mat| {
            if let Some(packed) = prepacked_a {
                GemmInputA::Packed(packed)
            } else {
                GemmInputA::Unpacked(mat)
            }
        })
        .collect();

    let b_mats: SmallVec<[_; 1]> = b_broadcast
        .inner_iter::<2>()
        .map(|mat| {
            if let Some(packed) = prepacked_b {
                GemmInputB::Packed(packed)
            } else {
                GemmInputB::Unpacked(mat)
            }
        })
        .collect();

    let out_len = out_shape.iter().product();

    let mut out_data = pool.alloc(out_len);

    out_data.extend_init(|uninit_out_data| {
        gemm.batched_gemm_uninit(
            &mut uninit_out_data[..out_len],
            &a_mats,
            &b_mats,
            GemmUninitOptions {
                alpha: alpha.unwrap_or(1.),
                bias,
                a_quant,
                b_quant,
            },
        )
        .unwrap()
    });

    let mut output = Tensor::from_data(out_shape, out_data);
    if a_is_vec {
        output.remove_axis(output.ndim() - 2);
    }
    if b_is_vec {
        output.remove_axis(output.ndim() - 1);
    }

    Ok(output)
}

#[derive(Clone, Debug)]
pub struct MatMul {}

impl Operator for MatMul {
    fn name(&self) -> &str {
        "MatMul"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(2)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let a = inputs.require_as(0)?;
        let b = inputs.require_as(1)?;
        let packed_b = match inputs.get_prepacked(1) {
            Some(PrepackedInput::FloatBMatrix(pb)) => Some(pb),
            _ => None,
        };
        matmul::<f32, f32, f32>(ctx.pool(), a, b, packed_b).into_op_result()
    }

    fn prepack_inputs(&self) -> SmallVec<[usize; 1]> {
        [1].into()
    }

    fn prepack(&self, index: usize, input: ValueView) -> Option<PrepackedInput> {
        if index == 1 {
            matmul_prepack_b::<f32, f32, f32>(input)
        } else {
            None
        }
    }
}

pub fn matmul_fused<LhsT: GemmInT, RhsT: GemmInT, OutT: Default + GemmOutT>(
    pool: &BufferPool,
    a: TensorView<LhsT>,
    b: TensorView<RhsT>,
    packed_b: Option<&PackedBMatrix<RhsT>>,
    bias: Option<BiasVector<OutT>>,
    alpha: Option<f32>,
) -> Result<Tensor<OutT>, OpError>
where
    GemmExecutor<LhsT, RhsT, OutT>: Default,
{
    matmul_impl(
        pool,
        a,
        b,
        packed_b,
        MatmulStrategy::Auto,
        bias,
        alpha,
        None, /* a_quant */
        None, /* b_quant */
    )
}

/// MatMul with fused addition of bias and scaling of result.
#[derive(Clone, Debug)]
pub struct FusedMatMul {
    /// Scaling factor to apply to result of matrix multiplication. Defaults to 1.
    pub alpha: Option<f32>,
}

impl Operator for FusedMatMul {
    fn name(&self) -> &str {
        "FusedMatMul"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(3)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let a = inputs.require_as(0)?;
        let b = inputs.require_as(1)?;
        let packed_b = match inputs.get_prepacked(1) {
            Some(PrepackedInput::FloatBMatrix(pb)) => Some(pb),
            _ => None,
        };

        let bias = inputs
            .get_as::<NdTensorView<f32, 1>>(2)?
            .map(|b| b.to_contiguous_in(ctx.pool()));
        let bias = bias.as_ref().map(|b| BiasVector::Row(b.data().unwrap()));

        matmul_fused(ctx.pool(), a, b, packed_b, bias, self.alpha).into_op_result()
    }

    fn prepack_inputs(&self) -> SmallVec<[usize; 1]> {
        [1].into()
    }

    fn prepack(&self, index: usize, input: ValueView) -> Option<PrepackedInput> {
        if index == 1 {
            matmul_prepack_b::<f32, f32, f32>(input)
        } else {
            None
        }
    }
}

/// Normalize a zero point input by converting it to a vector.
///
/// The spec for `MatMulInteger` allows for the zero point to be a scalar,
/// vector or a batch of vectors. The batch case is currently not supported.
pub fn zero_point_to_vec<T>(
    zero_point: Option<TensorView<T>>,
    expected_len: usize,
) -> Result<Option<NdTensorView<T, 1>>, OpError> {
    match zero_point {
        Some(zp) if zp.ndim() == 0 => Ok(Some(zp.broadcast([expected_len]))),
        Some(zp) if zp.ndim() == 1 => {
            if zp.size(0) != expected_len {
                Err(OpError::InvalidValue("Zero point has incorrect size"))
            } else {
                Ok(Some(zp.nd_view()))
            }
        }
        Some(_) => Err(OpError::UnsupportedValue(
            "Only scalar or vector zero points are supported",
        )),
        None => Ok(None),
    }
}

pub fn matmul_integer<LhsT, RhsT>(
    pool: &BufferPool,
    a: TensorView<LhsT>,
    b: TensorView<RhsT>,
    a_zero_point: Option<TensorView<LhsT>>,
    b_zero_point: Option<TensorView<RhsT>>,
    packed_b: Option<&PackedBMatrix<RhsT>>,
) -> Result<Tensor<i32>, OpError>
where
    LhsT: GemmInT,
    RhsT: GemmInT,
    GemmExecutor<LhsT, RhsT, i32>: Default,
{
    let a_rows = if a.ndim() > 1 {
        a.size(a.ndim() - 2)
    } else {
        1
    };
    let b_cols = if b.ndim() > 1 {
        b.size(b.ndim() - 1)
    } else {
        1
    };

    let a_zero = zero_point_to_vec(a_zero_point, a_rows)?.map(|zp| zp.to_contiguous());
    let a_quant = a_zero.as_ref().map(|zp| QuantParams {
        zero_point: zp.data().unwrap(),
    });

    let b_zero = zero_point_to_vec(b_zero_point, b_cols)?.map(|zp| zp.to_contiguous());
    let b_quant = b_zero.as_ref().map(|zp| QuantParams {
        zero_point: zp.data().unwrap(),
    });

    matmul_impl(
        pool,
        a,
        b,
        packed_b,
        MatmulStrategy::Auto,
        None,
        None,
        a_quant,
        b_quant,
    )
}

#[derive(Debug, Default)]
pub struct MatMulInteger {}

impl Operator for MatMulInteger {
    fn name(&self) -> &str {
        "MatMulInteger"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(4)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let a = inputs.require(0)?;
        let b = inputs.require(1)?;

        macro_rules! matmul_integer {
            ($a:expr, $b:expr) => {{
                let a_zero_point = inputs.get_as(2)?;
                let b_zero_point = inputs.get_as(3)?;
                let packed_b = inputs
                    .get_prepacked(1)
                    .and_then(|packed| packed.try_into().ok());
                matmul_integer(ctx.pool(), $a, $b, a_zero_point, b_zero_point, packed_b)
                    .into_op_result()
            }};
        }

        match (a, b) {
            (ValueView::UInt8Tensor(a), ValueView::Int8Tensor(b)) => matmul_integer!(a, b),

            // GEMM doesn't support other int8 signed-ness combinations yet.
            (ValueView::Int8Tensor(_), ValueView::Int8Tensor(_)) => Err(OpError::UnsupportedType),
            (ValueView::Int8Tensor(_), ValueView::UInt8Tensor(_)) => Err(OpError::UnsupportedType),
            (ValueView::UInt8Tensor(_), ValueView::UInt8Tensor(_)) => Err(OpError::UnsupportedType),

            _ => Err(OpError::UnsupportedType),
        }
    }

    fn prepack_inputs(&self) -> SmallVec<[usize; 1]> {
        [1].into()
    }

    fn prepack(&self, index: usize, input: ValueView) -> Option<PrepackedInput> {
        if index == 1 {
            matmul_prepack_b::<u8, i8, i32>(input)
        } else {
            None
        }
    }
}

/// Cast elements in `data` to f32 and scale by the per-column scales in `scale`.
fn cast_scale(
    pool: &BufferPool,
    mut data: Tensor<i32>,
    scale: NdTensorView<f32, 1>,
) -> Result<Tensor<f32>, OpError> {
    if data.size(data.ndim() - 1) != scale.size(0) {
        return Err(OpError::IncompatibleInputShapes(
            "Scale length does not match tensor columns",
        ));
    }

    let scale = scale.to_contiguous_in(pool);
    let scale_data = scale.data().unwrap();

    // Convert i32 elements to f32 in-place and multiply by column scale.
    let output_data = data.data_mut().expect("should be contiguous");
    output_data.par_chunks_mut(scale.len()).for_each(|chunk| {
        for (el, scale) in chunk.iter_mut().zip(scale_data) {
            let scaled = *el as f32 * scale;
            *el = scaled.cast_bytes();
        }
    });

    // Transmute tensor from i32 to f32.
    let shape = data.shape().to_vec();
    let data = cast_pod_vec::<i32, f32>(data.into_data()).unwrap();
    Ok(Tensor::from_data(&shape, data))
}

#[derive(Debug, Default)]
pub struct MatMulIntegerToFloat {
    matmul: MatMulInteger,
}

impl Operator for MatMulIntegerToFloat {
    fn name(&self) -> &str {
        "MatMulIntegerToFloat"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(5)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let output: Tensor<i32> = self.matmul.run(ctx)?.remove(0).try_into().unwrap();
        let scale = ctx.inputs().require_as(4)?;
        cast_scale(ctx.pool(), output, scale).into_op_result()
    }

    fn prepack_inputs(&self) -> SmallVec<[usize; 1]> {
        self.matmul.prepack_inputs()
    }

    fn prepack(&self, index: usize, input: ValueView) -> Option<PrepackedInput> {
        self.matmul.prepack(index, input)
    }
}

fn matmul_nbits(
    pool: &BufferPool,
    lhs: NdTensorView<f32, 3>,
    rhs: NdTensorView<u8, 3>,
    scales: NdTensorView<f32, 2>,
    bits: u8,
) -> Result<NdTensor<f32, 3>, OpError> {
    let [batch, rows, lhs_cols] = lhs.shape();

    let a_mats: SmallVec<[_; 1]> = lhs.inner_iter::<2>().map(GemmInputA::Unpacked).collect();
    let b_mat = BlockQuantizedMatrix::new(rhs, scales, bits).map_err(|err| {
        OpError::UnsupportedValue(match err {
            BlockQuantizedError::UnsupportedBlockSize => "Unsupported K block size",
            BlockQuantizedError::UnsupportedElementSize => "Unsupported bits-per-element",
            BlockQuantizedError::NonContiguousInput => "RHS input is not contiguous",
        })
    })?;
    let b_mats: SmallVec<[_; 1]> = std::iter::repeat(GemmInputB::BlockQuantized(b_mat))
        .take(batch)
        .collect();

    let out_shape = [batch, rows, b_mat.cols()];
    let out_len = out_shape.iter().product();
    let mut out_data = pool.alloc(out_len);

    if lhs_cols != b_mat.rows() {
        return Err(OpError::IncompatibleInputShapes(
            "Columns of first matrix does not match rows of second matrix",
        ));
    }

    let gemm = GemmExecutor::default();
    out_data.extend_init(|uninit_out_data| {
        gemm.batched_gemm_uninit(
            &mut uninit_out_data[..out_len],
            &a_mats,
            &b_mats,
            GemmUninitOptions::default(),
        )
        .unwrap()
    });

    Ok(NdTensor::from_data(out_shape, out_data))
}

/// Matrix multiplication of un-quantized LHS by block-quantized RHS.
///
/// See https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MatMulNBits.
#[derive(Debug)]
pub struct MatMulNBits {
    pub bits: u8,
    pub block_size: usize,
}

impl Operator for MatMulNBits {
    fn name(&self) -> &str {
        "MatMulNBits"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(3)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let lhs: NdTensorView<f32, 3> = ctx.inputs().require_as(0)?;
        let rhs: NdTensorView<u8, 3> = ctx.inputs().require_as(1)?;

        // Current spec requires scales to be 2D, but earlier versions used 1D
        // scales. See https://github.com/microsoft/onnxruntime/pull/24828.
        let scales: TensorView<f32> = ctx.inputs().require_as(2)?;

        let scales: CowNdTensor<f32, 2> = match scales.ndim() {
            2 => scales.to_contiguous_in(ctx.pool()).try_into().unwrap(),
            1 => {
                let k = lhs.size(2);
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

        matmul_nbits(ctx.pool(), lhs, rhs, scales.view(), self.bits).into_op_result()
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_bench::run_bench;
    use rten_gemm::{
        BiasVector, BlockQuantizedMatrix, GemmExecutor, GemmInT, GemmInputA, GemmInputB,
        GemmOptions, GemmOutT, QuantParams,
    };
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::expect_equal;
    use rten_tensor::{NdTensor, NdTensorView, Tensor, TensorView};
    use rten_testing::TestCases;

    use crate::buffer_pool::AutoReturn;
    use crate::buffer_pool::BufferPool;
    use crate::operator::{InputList, Operator, OperatorExt};
    use crate::ops::binary_elementwise::broadcast_shapes;

    use super::{
        FusedMatMul, MatMul, MatMulInteger, MatMulNBits, MatmulStrategy, OpError, OpRunContext,
        cast_scale, gemm, matmul, matmul_fused, matmul_impl, matmul_integer,
    };

    fn gemm_tensors(c: &mut Tensor, a: &Tensor, b: &Tensor, alpha: f32, beta: f32) {
        c.make_contiguous();
        GemmExecutor::default()
            .gemm(
                c.data_mut().unwrap(),
                GemmInputA::Unpacked(a.nd_view()),
                GemmInputB::Unpacked(b.nd_view()),
                GemmOptions {
                    alpha,
                    beta,
                    ..Default::default()
                },
            )
            .unwrap();
    }

    #[derive(Default)]
    struct MatMulOpts<'a, LhsT, RhsT, OutT> {
        bias: Option<BiasVector<'a, OutT>>,
        alpha: Option<f32>,
        a_zero: Option<TensorView<'a, LhsT>>,
        b_zero: Option<TensorView<'a, RhsT>>,
    }

    /// Multiply matrices in `a` by corresponding matrices in `b`.
    fn reference_matmul<LhsT: GemmInT, RhsT: GemmInT, OutT: GemmOutT + Default>(
        mut a: TensorView<LhsT>,
        mut b: TensorView<RhsT>,
        opts: MatMulOpts<LhsT, RhsT, OutT>,
    ) -> Tensor<OutT>
    where
        GemmExecutor<LhsT, RhsT, OutT>: Default,
    {
        let MatMulOpts {
            bias,
            alpha,
            a_zero,
            b_zero,
        } = opts;

        // Expand vector inputs to matrices. This follows the rules of
        // `numpy.matmul`.
        let a_is_vec = a.ndim() == 1;
        if a_is_vec {
            a.insert_axis(0);
        }
        let b_is_vec = b.ndim() == 1;
        if b_is_vec {
            b.insert_axis(1);
        }

        let a_rows = a.size(a.ndim() - 2);
        let b_cols = b.size(b.ndim() - 1);
        let a_prefix = &a.shape()[..a.ndim() - 2];
        let b_prefix = &b.shape()[..b.ndim() - 2];
        let out_prefix = broadcast_shapes(a_prefix, b_prefix).unwrap();
        let out_shape = &[out_prefix.as_slice(), &[a_rows, b_cols]].concat();
        let mut c = Tensor::zeros(out_shape);

        let a_batch_dims = a.ndim() - 2;
        let b_batch_dims = b.ndim() - 2;
        let out_prefix = &c.shape()[..c.ndim() - 2];

        let a_bcast = [out_prefix, &a.shape()[a_batch_dims..]].concat();
        let b_bcast = [out_prefix, &b.shape()[b_batch_dims..]].concat();

        let a_zero = a_zero.map(|zp| zp.broadcast([a_rows]).to_vec());
        let a_quant = a_zero.as_ref().map(|zp| QuantParams { zero_point: &zp });

        let b_zero = b_zero.map(|zp| zp.broadcast([b_cols]).to_vec());
        let b_quant = b_zero.as_ref().map(|zp| QuantParams { zero_point: &zp });

        let gemm = GemmExecutor::<LhsT, RhsT, OutT>::default();
        a.broadcast(a_bcast.as_slice())
            .inner_iter::<2>()
            .zip(b.broadcast(b_bcast.as_slice()).inner_iter::<2>())
            .zip(c.inner_iter_mut::<2>())
            .for_each(|((a, b), mut c)| {
                gemm.gemm(
                    c.data_mut().unwrap(),
                    GemmInputA::Unpacked(a),
                    GemmInputB::Unpacked(b),
                    GemmOptions {
                        alpha: alpha.unwrap_or(1.),
                        beta: OutT::default(),
                        bias,
                        a_quant,
                        b_quant,
                    },
                )
                .unwrap();
            });

        match (a_is_vec, b_is_vec) {
            (true, false) => c.remove_axis(c.ndim() - 2),
            (false, true) => c.remove_axis(c.ndim() - 1),
            (true, true) => {
                c.remove_axis(c.ndim() - 1);
                c.remove_axis(c.ndim() - 1);
            }
            (false, false) => {}
        }

        c
    }

    #[test]
    fn test_cast_scale() {
        #[derive(Debug)]
        struct Case {
            input: Tensor<i32>,
            scales: NdTensor<f32, 1>,
            expected: Result<Tensor<f32>, OpError>,
        }

        let cases = [
            Case {
                input: [[1, 2], [3, 4]].into(),
                scales: [2., 3.].into(),
                expected: Ok([[2., 6.], [6., 12.]].into()),
            },
            Case {
                input: [[1, 2], [3, 4]].into(),
                scales: [2., 3., 4.].into(),
                expected: Err(OpError::IncompatibleInputShapes(
                    "Scale length does not match tensor columns",
                )),
            },
        ];

        cases.test_each(|case| {
            let pool = BufferPool::new();
            let result = cast_scale(&pool, case.input.clone(), case.scales.view());
            assert_eq!(result, case.expected);
        });
    }

    #[test]
    fn test_gemm_op() -> Result<(), Box<dyn Error>> {
        let pool = BufferPool::new();

        let mut rng = XorShiftRng::new(1234);
        let a = Tensor::rand(&[3, 10], &mut rng);
        let b = Tensor::rand(&[10, 8], &mut rng);

        let mut expected = Tensor::zeros(&[3, 8]);
        gemm_tensors(&mut expected, &a, &b, 1., 1.);

        let result = gemm(&pool, a.view(), b.view(), None, 1.0, 1.0, false, false).unwrap();

        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_gemm_op_transposed() -> Result<(), Box<dyn Error>> {
        let pool = BufferPool::new();

        let mut rng = XorShiftRng::new(1234);
        let a = Tensor::rand(&[10, 3], &mut rng);
        let b = Tensor::rand(&[8, 10], &mut rng);

        let mut a_transposed = a.clone();
        a_transposed.permute(&[1, 0]);
        let mut b_transposed = b.clone();
        b_transposed.permute(&[1, 0]);
        let mut expected = Tensor::zeros(&[3, 8]);
        gemm_tensors(&mut expected, &a_transposed, &b_transposed, 1., 1.);

        let result = gemm(&pool, a.view(), b.view(), None, 1.0, 1.0, true, true).unwrap();

        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_gemm_op_adds_c() -> Result<(), Box<dyn Error>> {
        let pool = BufferPool::new();

        let mut rng = XorShiftRng::new(1234);
        let a = Tensor::rand(&[3, 10], &mut rng);
        let b = Tensor::rand(&[10, 8], &mut rng);
        let c = Tensor::rand(&[3, 8], &mut rng);

        let mut expected = c.clone();
        gemm_tensors(&mut expected, &a, &b, 1., 1.);

        let result = gemm(
            &pool,
            a.view(),
            b.view(),
            Some(c.view()),
            1.0,
            1.0,
            false,
            false,
        )
        .unwrap();

        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_gemm_op_invalid_inputs() {
        let pool = BufferPool::new();

        let mut rng = XorShiftRng::new(1234);
        let a = Tensor::rand(&[3, 10], &mut rng);
        let b = Tensor::rand(&[10, 8], &mut rng);
        let c = Tensor::rand(&[3, 5], &mut rng);

        let result = gemm(
            &pool,
            a.view(),
            b.view(),
            Some(c.view()),
            1.0,
            1.0,
            false,
            false,
        );

        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes(
                "Cannot broadcast c to output shape"
            ))
        );
    }

    #[test]
    fn test_matmul() {
        #[derive(Debug)]
        struct Case<'a> {
            a_shape: &'a [usize],
            b_shape: &'a [usize],
            out_shape: &'a [usize],
        }

        let cases = [
            // Simple matmul
            Case {
                a_shape: &[3, 10],
                b_shape: &[10, 8],
                out_shape: &[3, 8],
            },
            // LHS is a 1-sized batch
            Case {
                a_shape: &[1, 3, 10],
                b_shape: &[10, 8],
                out_shape: &[1, 3, 8],
            },
            // RHS is a 1-sized batch
            Case {
                a_shape: &[3, 10],
                b_shape: &[1, 10, 8],
                out_shape: &[1, 3, 8],
            },
            // LHS and RHS are 1-sized batches
            Case {
                a_shape: &[1, 3, 10],
                b_shape: &[1, 10, 8],
                out_shape: &[1, 3, 8],
            },
            // LHS input is a batch
            Case {
                a_shape: &[2, 3, 10],
                b_shape: &[10, 8],
                out_shape: &[2, 3, 8],
            },
            // RHS input is a batch
            Case {
                a_shape: &[3, 10],
                b_shape: &[2, 10, 8],
                out_shape: &[2, 3, 8],
            },
            // Both inputs are batches
            Case {
                a_shape: &[2, 3, 10],
                b_shape: &[2, 10, 8],
                out_shape: &[2, 3, 8],
            },
            // LHS is a vector
            Case {
                a_shape: &[4],
                b_shape: &[4, 8],
                out_shape: &[8],
            },
            // RHS is a vector
            Case {
                a_shape: &[4, 6],
                b_shape: &[6],
                out_shape: &[4],
            },
            // LHS and RHS are both vectors
            Case {
                a_shape: &[4],
                b_shape: &[4],
                out_shape: &[],
            },
        ];

        cases.test_each(|case| {
            let &Case {
                out_shape,
                a_shape,
                b_shape,
            } = case;

            let pool = BufferPool::new();
            let mut rng = XorShiftRng::new(1234);
            let a = Tensor::<f32>::rand(a_shape, &mut rng);
            let b = Tensor::<f32>::rand(b_shape, &mut rng);
            let expected = reference_matmul(a.view(), b.view(), MatMulOpts::default());
            let result = matmul(&pool, a.view(), b.view(), None).unwrap();
            assert_eq!(result.shape(), out_shape);
            expect_equal(&result, &expected).unwrap();
        });
    }

    #[test]
    fn test_matmul_with_prepacked_inputs() -> Result<(), Box<dyn Error>> {
        struct Case {
            op: Box<dyn Operator>,
            bias_input: bool,
        }

        let cases = [
            Case {
                op: Box::new(MatMul {}),
                bias_input: false,
            },
            Case {
                op: Box::new(FusedMatMul { alpha: None }),
                bias_input: true,
            },
        ];

        let mut rng = XorShiftRng::new(1234);

        let a = Tensor::rand(&[5, 10], &mut rng);

        // The unpacked and pre-packed versions of an input should use the
        // same data. Here we intentionally use different tensors with
        // the same shape so we can verify if the packed data was used.
        let b = Tensor::<f32>::rand(&[10, 3], &mut rng);

        // Dummy zero bias.
        let bias = Tensor::<f32>::zeros(&[3]);

        for Case { op, bias_input } in cases {
            let packed_b_input = Tensor::rand(&[10, 3], &mut rng);
            let packed_b = op.prepack(1, packed_b_input.view().into()).unwrap();

            let expected = reference_matmul(a.view(), packed_b_input.view(), MatMulOpts::default());

            let pool = BufferPool::new();
            let get_prepacked = |idx| {
                if idx == 1 { Some(&packed_b) } else { None }
            };
            let mut inputs =
                InputList::from(&[a.view().into(), b.view().into()]).with_prepacked(&get_prepacked);
            if bias_input {
                inputs.push(bias.view());
            }

            let ctx = OpRunContext::new(&pool, &inputs);
            let mut result = op.run(&ctx).unwrap();
            let result: Tensor<f32> = result.remove(0).try_into().unwrap();

            expect_equal(&result, &expected)?;
        }

        Ok(())
    }

    #[test]
    fn test_matmul_fused() {
        let mut rng = XorShiftRng::new(1234);
        let a = Tensor::rand(&[10, 15], &mut rng);
        let b = Tensor::rand(&[15, 5], &mut rng);
        let bias_data: Vec<f32> = (0..b.size(b.ndim() - 1)).map(|_| rng.next_f32()).collect();

        #[derive(Debug)]
        struct Case<'a> {
            bias: Option<BiasVector<'a, f32>>,
            alpha: Option<f32>,
        }

        let cases = [
            Case {
                bias: Some(BiasVector::Row(&bias_data)),
                alpha: None,
            },
            Case {
                bias: None,
                alpha: Some(0.5),
            },
        ];

        cases.test_each(|case| {
            let Case { bias, alpha } = case;

            let pool = BufferPool::new();
            let expected = reference_matmul(
                a.view(),
                b.view(),
                MatMulOpts {
                    bias: bias.clone(),
                    alpha: *alpha,
                    ..Default::default()
                },
            );
            let result = matmul_fused(&pool, a.view(), b.view(), None, *bias, *alpha).unwrap();
            expect_equal(&result, &expected).unwrap();
        })
    }

    #[test]
    fn test_matmul_invalid() {
        #[derive(Debug)]
        struct Case<'a> {
            a_shape: &'a [usize],
            b_shape: &'a [usize],
            error: OpError,
        }

        let cases = [
            Case {
                a_shape: &[],
                b_shape: &[10, 8],
                error: OpError::InvalidValue("Inputs must have >= 1 dimensions"),
            },
            Case {
                a_shape: &[3, 10],
                b_shape: &[],
                error: OpError::InvalidValue("Inputs must have >= 1 dimensions"),
            },
            Case {
                a_shape: &[3, 10],
                b_shape: &[11, 8],
                error: OpError::IncompatibleInputShapes(
                    "Columns of first matrix does not match rows of second matrix",
                ),
            },
            Case {
                a_shape: &[2, 3, 10],
                b_shape: &[3, 10, 8],
                error: OpError::IncompatibleInputShapes("Cannot broadcast shapes"),
            },
        ];

        cases.test_each(|case| {
            let Case {
                a_shape,
                b_shape,
                error,
            } = case;

            let pool = BufferPool::new();

            let mut rng = XorShiftRng::new(1234);
            let a = Tensor::<f32>::rand(a_shape, &mut rng);
            let b = Tensor::<f32>::rand(b_shape, &mut rng);

            let result = matmul(&pool, a.view(), b.view(), None);
            assert_eq!(result.as_ref(), Err(error));
        })
    }

    #[test]
    fn test_matmul_zero_sized_dim() {
        #[derive(Clone, Debug)]
        struct Case {
            m: usize,
            n: usize,
            k: usize,
        }

        let cases = [
            Case { m: 5, n: 0, k: 10 },
            Case { m: 0, n: 5, k: 10 },
            Case { m: 5, n: 10, k: 0 },
        ];

        cases.test_each_clone(|Case { m, n, k }| {
            let pool = BufferPool::new();
            let mut rng = XorShiftRng::new(1234);
            let a = Tensor::<f32>::rand(&[m, k], &mut rng);
            let b = Tensor::<f32>::rand(&[k, n], &mut rng);
            let result = matmul(&pool, a.view(), b.view(), None).unwrap();

            assert_eq!(result.shape(), &[m, n]);
            if k == 0 {
                assert!(result.iter().all(|x| *x == 0.));
            }
        })
    }

    #[test]
    fn test_matmul_integer() {
        #[derive(Debug)]
        struct Case {
            a: Tensor<u8>,
            b: Tensor<i8>,
            a_zero_point: Option<Tensor<u8>>,
            b_zero_point: Option<Tensor<i8>>,
            expected_err: Option<OpError>,
        }

        let cases = [
            // No zero point
            Case {
                a: Tensor::from([[1, 2], [3, 4]]),
                b: Tensor::from([[5, 6], [7, 8]]),
                a_zero_point: None,
                b_zero_point: None,
                expected_err: None,
            },
            // Scalar zero points
            Case {
                a: Tensor::from([[1, 2], [3, 4]]),
                b: Tensor::from([[5, 6], [7, 8]]),
                a_zero_point: Some(Tensor::from(127)),
                b_zero_point: Some(Tensor::from(-50)),
                expected_err: None,
            },
            // Vector zero points
            Case {
                a: Tensor::from([[1, 2], [3, 4]]),
                b: Tensor::from([[5, 6], [7, 8]]),
                a_zero_point: Some(Tensor::from([1, 2])),
                b_zero_point: Some(Tensor::from([3, 4])),
                expected_err: None,
            },
            // LHS batch input with vector zero point
            Case {
                a: Tensor::zeros(&[3, 2, 2]),
                b: Tensor::from([[5, 6], [7, 8]]),
                a_zero_point: Some(Tensor::from([1, 2])),
                b_zero_point: Some(Tensor::from([3, 4])),
                expected_err: None,
            },
            // LHS input has one row
            Case {
                a: Tensor::from([[1, 2, 3, 4]]),
                b: Tensor::from([[5, 6], [7, 8], [9, 10], [11, 12]]),
                a_zero_point: Some(Tensor::from([1])),
                b_zero_point: Some(Tensor::from([3, 4])),
                expected_err: None,
            },
            // LHS is a vector
            Case {
                a: Tensor::from([1, 2]),
                b: Tensor::from([[1, 2], [3, 4]]),
                a_zero_point: Some(Tensor::from([1])),
                b_zero_point: Some(Tensor::from([2, 3])),
                expected_err: None,
            },
            // RHS is a vector
            Case {
                a: Tensor::from([[1, 2], [3, 4]]),
                b: Tensor::from([1, 2]),
                a_zero_point: Some(Tensor::from([1, 2])),
                b_zero_point: Some(Tensor::from([3])),
                expected_err: None,
            },
            // Incorrect zero point size
            Case {
                a: Tensor::from([[1, 2], [3, 4]]),
                b: Tensor::from([[5, 6], [7, 8]]),
                a_zero_point: Some(Tensor::from([1, 2, 4])),
                b_zero_point: Some(Tensor::from([3, 4])),
                expected_err: Some(OpError::InvalidValue("Zero point has incorrect size")),
            },
            // Non-scalar zero points
            Case {
                a: Tensor::from([[2, 2], [2, 2]]),
                b: Tensor::from([[2, 2], [2, 2]]),
                a_zero_point: Some(Tensor::from([[2, 2], [2, 2]])),
                b_zero_point: None,
                expected_err: Some(OpError::UnsupportedValue(
                    "Only scalar or vector zero points are supported",
                )),
            },
            Case {
                a: Tensor::from([[2, 2], [2, 2]]),
                b: Tensor::from([[2, 2], [2, 2]]),
                a_zero_point: None,
                b_zero_point: Some(Tensor::from([[2, 2], [2, 2]])),
                expected_err: Some(OpError::UnsupportedValue(
                    "Only scalar or vector zero points are supported",
                )),
            },
            // Empty output
            Case {
                a: Tensor::zeros(&[0, 2]),
                b: Tensor::zeros(&[2, 3]),
                a_zero_point: None,
                b_zero_point: None,
                expected_err: None,
            },
            // K dim mismatch between LHS and RHS
            Case {
                a: Tensor::zeros(&[1, 2]),
                b: Tensor::zeros(&[3, 1]),
                a_zero_point: None,
                b_zero_point: None,
                expected_err: Some(OpError::IncompatibleInputShapes(
                    "Columns of first matrix does not match rows of second matrix",
                )),
            },
            // LHS is a scalar
            Case {
                a: Tensor::zeros(&[]),
                b: Tensor::zeros(&[3, 1]),
                a_zero_point: None,
                b_zero_point: None,
                expected_err: Some(OpError::InvalidValue("Inputs must have >= 1 dimensions")),
            },
            // RHS is a scalar
            Case {
                a: Tensor::zeros(&[1, 2]),
                b: Tensor::zeros(&[]),
                a_zero_point: None,
                b_zero_point: None,
                expected_err: Some(OpError::InvalidValue("Inputs must have >= 1 dimensions")),
            },
            Case {
                a: Tensor::zeros(&[2, 2, 2]),
                b: Tensor::zeros(&[3, 2, 2]),
                a_zero_point: None,
                b_zero_point: None,
                expected_err: Some(OpError::IncompatibleInputShapes("Cannot broadcast shapes")),
            },
        ];

        cases.test_each(|case| {
            let Case {
                a,
                b,
                a_zero_point,
                b_zero_point,
                expected_err,
            } = case;

            let pool = BufferPool::new();
            let result = matmul_integer(
                &pool,
                a.view(),
                b.view(),
                a_zero_point.as_ref().map(|zp| zp.view()),
                b_zero_point.as_ref().map(|zp| zp.view()),
                None,
            );

            match (result, expected_err) {
                (Ok(result), None) => {
                    let expected = reference_matmul(
                        a.view(),
                        b.view(),
                        MatMulOpts {
                            a_zero: a_zero_point.as_ref().map(|zp| zp.view()),
                            b_zero: b_zero_point.as_ref().map(|zp| zp.view()),
                            ..Default::default()
                        },
                    );
                    assert_eq!(result, expected);
                }
                (result, expected_err) => {
                    assert_eq!(result.err(), *expected_err);
                }
            }
        })
    }

    #[test]
    fn test_matmul_integer_with_prepacked_inputs() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);
        let op = MatMulInteger {};

        let a = Tensor::<u8>::rand(&[5, 10], &mut rng);

        // The unpacked and pre-packed versions of an input should use the
        // same data. Here we intentionally use different tensors with
        // the same shape so we can verify if the packed data was used.
        let b = Tensor::<i8>::rand(&[10, 3], &mut rng);
        let packed_b_input = Tensor::<i8>::rand(&[10, 3], &mut rng);
        let packed_b = op.prepack(1, packed_b_input.view().into()).unwrap();

        let expected = reference_matmul(a.view(), packed_b_input.view(), MatMulOpts::default());

        let pool = BufferPool::new();
        let get_prepacked = |idx| {
            if idx == 1 { Some(&packed_b) } else { None }
        };
        let inputs =
            InputList::from(&[a.view().into(), b.view().into()]).with_prepacked(&get_prepacked);
        let ctx = OpRunContext::new(&pool, &inputs);
        let mut result = op.run(&ctx).unwrap();
        let result: Tensor<i32> = result.remove(0).try_into().unwrap();

        expect_equal(&result, &expected)?;

        Ok(())
    }

    fn reference_matmul_nbits(
        lhs: NdTensorView<f32, 3>,
        rhs: NdTensorView<u8, 3>,
        scales: NdTensorView<f32, 2>,
        n_bits: u8,
    ) -> NdTensor<f32, 3> {
        let [batch, m, _k] = lhs.shape();
        let [n, _k_blocks, _block_size] = rhs.shape();
        let bqm = BlockQuantizedMatrix::new(rhs, scales, n_bits).unwrap();

        let mut output = NdTensor::zeros([batch, m, n]);
        let gemm = GemmExecutor::default();
        for (mut out, a) in output.inner_iter_mut::<2>().zip(lhs.inner_iter()) {
            gemm.gemm(
                out.data_mut().unwrap(),
                GemmInputA::Unpacked(a),
                GemmInputB::BlockQuantized(bqm),
                GemmOptions::default(),
            )
            .unwrap();
        }

        output
    }

    #[test]
    fn test_matmul_nbits() {
        let mut rng = XorShiftRng::new(1234);

        let batch = 2;
        let block_size = 16;
        let block_bytes = block_size / 2;
        let m = 4;
        let k = block_size * 2;
        let n = 8;
        let n_bits = 4;

        let lhs = NdTensor::<f32, 3>::rand([batch, m, k], &mut rng);
        let rhs = NdTensor::<u8, 3>::rand([n, k / block_size, block_bytes], &mut rng);
        let scales = NdTensor::<f32, 2>::rand([n, k / block_size], &mut rng);
        let expected = reference_matmul_nbits(lhs.view(), rhs.view(), scales.view(), n_bits);

        let op = MatMulNBits {
            bits: n_bits,
            block_size,
        };

        // With 2D scales.
        let result: NdTensor<f32, 3> = op
            .run_simple((lhs.view(), rhs.view(), scales.view()))
            .unwrap();
        assert_eq!(result.shape(), [batch, m, n]);
        expect_equal(&result, &expected).unwrap();

        // With 1D scales (older models)
        let result: NdTensor<f32, 3> = op
            .run_simple((
                lhs.view(),
                rhs.view(),
                scales.reshaped([scales.len()]).view(),
            ))
            .unwrap();
        assert_eq!(result.shape(), [batch, m, n]);
        expect_equal(&result, &expected).unwrap();
    }

    #[test]
    #[ignore]
    fn bench_matmul() {
        struct Case {
            a_batch: usize,
            a_rows: usize,
            a_cols: usize,
            b_cols: usize,
        }

        let mut cases = Vec::new();
        let a_cols = 512;
        let b_cols = 1536;

        for a_batch in [1, 10, 128, 256, 512, 1024] {
            for a_rows in [1, 16, 32, 64] {
                cases.push(Case {
                    a_batch,
                    a_rows,
                    a_cols,
                    b_cols,
                });
            }
        }

        for Case {
            a_batch,
            a_rows,
            a_cols,
            b_cols,
        } in cases
        {
            let mut rng = XorShiftRng::new(1234);
            let a = Tensor::<f32>::rand(&[a_batch, a_rows, a_cols], &mut rng);
            let b = Tensor::<f32>::rand(&[a_cols, b_cols], &mut rng);

            let run_trial = |strategy| {
                let trials = 10;
                let desc = format!(
                    "matmul [{a_batch},{a_rows},{a_cols}] x [{a_cols},{b_cols}], strategy={strategy:?}",
                );
                let pool = BufferPool::new();
                run_bench(trials, Some(&desc), || {
                    matmul_impl(
                        &pool,
                        a.view(),
                        b.view(),
                        None,
                        strategy,
                        None,
                        None,
                        None,
                        None,
                    )
                    .unwrap()
                    .auto_return(&pool);
                });
            };

            run_trial(MatmulStrategy::Batch);
            run_trial(MatmulStrategy::Auto);
            println!();
        }
    }
}
