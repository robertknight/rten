use rayon::prelude::*;

use rten_tensor::prelude::*;
use rten_tensor::{Matrix, NdTensorView, Tensor, TensorView};
use smallvec::SmallVec;

use crate::gemm::{
    BiasVector, GemmExecutor, GemmInT, GemmInputA, GemmInputB, GemmOutT, PackedBMatrix, QuantParams,
};
use crate::ops::binary_elementwise::broadcast_shapes;
use crate::ops::layout::expand_to;
use crate::ops::{
    static_dims, Input, IntoOpResult, OpError, OpRunContext, Operator, OutputList, PrepackedInput,
};
use crate::tensor_pool::{AutoReturn, TensorPool};

/// Compute the General Matrix Multiplication (GEMM) `c = alpha * (ab) + beta * c`.
///
/// If `transpose_a` or `transpose_b` are set, the `a` and `b` inputs
/// respectively are transposed before multiplying them.
///
/// nb. This is named `gemm_op` to avoid confusion with `gemm::gemm`.
pub fn gemm_op<LhsT: GemmInT, RhsT: GemmInT, OutT: GemmOutT>(
    pool: &TensorPool,
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
            let out_row_stride = output.stride(0);
            gemm.gemm(
                output.data_mut().unwrap(),
                out_row_stride,
                GemmInputA::Unpacked(a.nd_view()),
                GemmInputB::Unpacked(b.nd_view()),
                alpha,
                beta,
                None, // bias
                None, // a_quant
                None, // b_quant
            )
            .unwrap();
            output
        }
        _ => {
            let mut output = Tensor::uninit_in(pool, out_shape);
            let out_row_stride = output.stride(0);
            gemm.gemm_uninit(
                output.data_mut().unwrap(),
                out_row_stride,
                GemmInputA::Unpacked(a.nd_view()),
                GemmInputB::Unpacked(b.nd_view()),
                alpha,
                None, // bias
                None, // a_quant
                None, // b_quant
            )
            .unwrap();
            // Safety: `gemm_uninit` initialized all elements
            unsafe { output.assume_init() }
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

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let a = inputs.require_as(0)?;
        let b = inputs.require_as(1)?;
        let c = inputs.get_as(2)?;
        gemm_op::<f32, f32, f32>(
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
    input: Input,
) -> Option<PrepackedInput>
where
    GemmExecutor<LhsT, RhsT, OutT>: Default,
    PrepackedInput: From<PackedBMatrix<RhsT>>,
    for<'a> TensorView<'a, RhsT>: TryFrom<Input<'a>>,
{
    let executor = GemmExecutor::default();
    let tensor: TensorView<RhsT> = input.try_into().ok()?;
    let matrix: Matrix<RhsT> = tensor.try_into().ok()?;
    Some(executor.prepack_b(matrix).into())
}

pub fn matmul<LhsT: GemmInT, RhsT: GemmInT, OutT: Default + GemmOutT>(
    pool: &TensorPool,
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
    pool: &TensorPool,
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

    let mut output = Tensor::uninit_in(pool, out_shape);
    if output.is_empty() {
        // nb. We don't need to alloc from the pool here, since the buffer
        // is already empty.
        return Ok(Tensor::zeros(out_shape));
    }

    let a_broadcast_shape = [out_prefix.as_slice(), &[a_rows, a_cols]].concat();
    let b_broadcast_shape = [out_prefix.as_slice(), &[b_rows, b_cols]].concat();

    let a_broadcast = a.broadcast(a_broadcast_shape.as_slice());
    let b_broadcast = b.broadcast(b_broadcast_shape.as_slice());

    let out_row_stride = output.stride(output.ndim() - 2);
    let out_batches = output
        .data_mut()
        .unwrap()
        .chunks_mut(out_row_stride * a_rows);

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

    a_broadcast
        .inner_iter::<2>()
        .zip(b_broadcast.inner_iter::<2>())
        .zip(out_batches)
        .par_bridge()
        .for_each(|((a_mat, b_mat), out_mat)| {
            let a_input = if let Some(packed) = prepacked_a {
                GemmInputA::Packed(packed)
            } else {
                GemmInputA::Unpacked(a_mat)
            };

            let b_input = if let Some(packed) = prepacked_b {
                GemmInputB::Packed(packed)
            } else {
                GemmInputB::Unpacked(b_mat)
            };

            gemm.gemm_uninit(
                out_mat,
                out_row_stride,
                a_input,
                b_input,
                alpha.unwrap_or(1.),
                bias,
                a_quant,
                b_quant,
            )
            .unwrap();
        });

    // Safety: Loop above initialized all output elements.
    let mut output = unsafe { output.assume_init() };

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

    fn prepack(&self, index: usize, input: Input) -> Option<PrepackedInput> {
        if index == 1 {
            matmul_prepack_b::<f32, f32, f32>(input)
        } else {
            None
        }
    }
}

pub fn matmul_fused<LhsT: GemmInT, RhsT: GemmInT, OutT: Default + GemmOutT>(
    pool: &TensorPool,
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

    fn prepack(&self, index: usize, input: Input) -> Option<PrepackedInput> {
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
    pool: &TensorPool,
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

#[derive(Debug)]
pub struct MatMulInteger {}

impl Operator for MatMulInteger {
    fn name(&self) -> &str {
        "MatMulInteger"
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
            (Input::UInt8Tensor(a), Input::Int8Tensor(b)) => matmul_integer!(a, b),

            // GEMM doesn't support other int8 signed-ness combinations yet.
            (Input::Int8Tensor(_), Input::Int8Tensor(_)) => Err(OpError::UnsupportedType),
            (Input::Int8Tensor(_), Input::UInt8Tensor(_)) => Err(OpError::UnsupportedType),
            (Input::UInt8Tensor(_), Input::UInt8Tensor(_)) => Err(OpError::UnsupportedType),

            _ => Err(OpError::UnsupportedType),
        }
    }

    fn prepack_inputs(&self) -> SmallVec<[usize; 1]> {
        [1].into()
    }

    fn prepack(&self, index: usize, input: Input) -> Option<PrepackedInput> {
        if index == 1 {
            matmul_prepack_b::<u8, i8, i32>(input)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_bench::run_bench;
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::expect_equal;
    use rten_tensor::{Tensor, TensorView};
    use rten_testing::TestCases;

    use crate::gemm::{
        BiasVector, GemmExecutor, GemmInT, GemmInputA, GemmInputB, GemmOutT, QuantParams,
    };
    use crate::ops::binary_elementwise::broadcast_shapes;
    use crate::ops::tests::new_pool;
    use crate::ops::{InputList, Operator};
    use crate::tensor_pool::AutoReturn;

    use super::{
        gemm_op, matmul, matmul_fused, matmul_impl, matmul_integer, FusedMatMul, MatMul,
        MatMulInteger, MatmulStrategy, OpError, OpRunContext,
    };

    fn gemm_tensors(c: &mut Tensor, a: &Tensor, b: &Tensor, alpha: f32, beta: f32) {
        c.make_contiguous();
        let c_row_stride = c.stride(c.ndim() - 2);
        GemmExecutor::default()
            .gemm(
                c.data_mut().unwrap(),
                c_row_stride,
                GemmInputA::Unpacked(a.nd_view()),
                GemmInputB::Unpacked(b.nd_view()),
                alpha,
                beta,
                None, // bias
                None, // a_quant
                None, // b_quant
            )
            .unwrap()
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
                let c_row_stride = c.stride(0);
                gemm.gemm(
                    c.data_mut().unwrap(),
                    c_row_stride,
                    GemmInputA::Unpacked(a),
                    GemmInputB::Unpacked(b),
                    alpha.unwrap_or(1.),
                    OutT::default(), /* beta */
                    bias,
                    a_quant,
                    b_quant,
                )
                .unwrap()
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
    fn test_gemm_op() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();

        let mut rng = XorShiftRng::new(1234);
        let a = Tensor::rand(&[3, 10], &mut rng);
        let b = Tensor::rand(&[10, 8], &mut rng);

        let mut expected = Tensor::zeros(&[3, 8]);
        gemm_tensors(&mut expected, &a, &b, 1., 1.);

        let result = gemm_op(&pool, a.view(), b.view(), None, 1.0, 1.0, false, false).unwrap();

        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_gemm_op_transposed() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();

        let mut rng = XorShiftRng::new(1234);
        let a = Tensor::rand(&[10, 3], &mut rng);
        let b = Tensor::rand(&[8, 10], &mut rng);

        let mut a_transposed = a.clone();
        a_transposed.permute(&[1, 0]);
        let mut b_transposed = b.clone();
        b_transposed.permute(&[1, 0]);
        let mut expected = Tensor::zeros(&[3, 8]);
        gemm_tensors(&mut expected, &a_transposed, &b_transposed, 1., 1.);

        let result = gemm_op(&pool, a.view(), b.view(), None, 1.0, 1.0, true, true).unwrap();

        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_gemm_op_adds_c() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();

        let mut rng = XorShiftRng::new(1234);
        let a = Tensor::rand(&[3, 10], &mut rng);
        let b = Tensor::rand(&[10, 8], &mut rng);
        let c = Tensor::rand(&[3, 8], &mut rng);

        let mut expected = c.clone();
        gemm_tensors(&mut expected, &a, &b, 1., 1.);

        let result = gemm_op(
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
        let pool = new_pool();

        let mut rng = XorShiftRng::new(1234);
        let a = Tensor::rand(&[3, 10], &mut rng);
        let b = Tensor::rand(&[10, 8], &mut rng);
        let c = Tensor::rand(&[3, 5], &mut rng);

        let result = gemm_op(
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

            let pool = new_pool();
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

            let pool = new_pool();
            let get_prepacked = |idx| {
                if idx == 1 {
                    Some(&packed_b)
                } else {
                    None
                }
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

            let pool = new_pool();
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

            let pool = new_pool();

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
            let pool = new_pool();
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

            let pool = new_pool();
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

        let pool = new_pool();
        let get_prepacked = |idx| {
            if idx == 1 {
                Some(&packed_b)
            } else {
                None
            }
        };
        let inputs =
            InputList::from(&[a.view().into(), b.view().into()]).with_prepacked(&get_prepacked);
        let ctx = OpRunContext::new(&pool, &inputs);
        let mut result = op.run(&ctx).unwrap();
        let result: Tensor<i32> = result.remove(0).try_into().unwrap();

        expect_equal(&result, &expected)?;

        Ok(())
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
                let pool = new_pool();
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
