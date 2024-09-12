use rayon::prelude::*;

use rten_tensor::prelude::*;
use rten_tensor::{Tensor, TensorView};

use crate::check_dims;
use crate::gemm::{GemmExecutor, GemmInT, GemmInputA, GemmInputB, GemmOutT};
use crate::iter_util::range_chunks;
use crate::ops::binary_elementwise::broadcast_shapes;
use crate::ops::layout::expand_to;
use crate::ops::{InputList, IntoOpResult, OpError, Operator, OutputList};
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
    check_dims!(a, 2);
    check_dims!(b, 2);

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
            );
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
            );
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

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let a = inputs.require_as(0)?;
        let b = inputs.require_as(1)?;
        let c = inputs.get_as(2)?;
        gemm_op::<f32, f32, f32>(
            pool,
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

pub fn matmul<LhsT: GemmInT, RhsT: GemmInT, OutT: Default + GemmOutT>(
    pool: &TensorPool,
    a: TensorView<LhsT>,
    b: TensorView<RhsT>,
) -> Result<Tensor<OutT>, OpError>
where
    GemmExecutor<LhsT, RhsT, OutT>: Default,
{
    matmul_impl(pool, a, b, MatmulStrategy::Auto)
}

fn matmul_impl<LhsT: GemmInT, RhsT: GemmInT, OutT: Default + GemmOutT>(
    pool: &TensorPool,
    a: TensorView<LhsT>,
    b: TensorView<RhsT>,
    strategy: MatmulStrategy,
) -> Result<Tensor<OutT>, OpError>
where
    GemmExecutor<LhsT, RhsT, OutT>: Default,
{
    if a.ndim() < 2 || b.ndim() < 2 {
        return Err(OpError::InvalidValue("Inputs must have >= 2 dimensions"));
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
        let mut output = matmul(pool, a_matrix, b.clone())?;
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

    // Prepack re-used inputs to amortize packing cost.
    //
    // We don't prepack when the "A" matrix is a vector because that uses a
    // special case vector-matrix algorithm that doesn't benefit from packing.
    let prepacked_a = (num_a_matrices == 1 && num_b_matrices > 1 && a_rows > 1).then(|| {
        let a_matrix = a.inner_iter::<2>().next().unwrap();
        gemm.prepack_a_in(pool, a_matrix).auto_return(pool)
    });
    let prepacked_a = prepacked_a.as_deref();

    let prepacked_b = (num_a_matrices > 1 && num_b_matrices == 1 && a_rows > 1).then(|| {
        let b_matrix = b.inner_iter::<2>().next().unwrap();
        gemm.prepack_b_in(pool, b_matrix).auto_return(pool)
    });
    let prepacked_b = prepacked_b.as_deref();

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
                1., // alpha
            );
        });

    // Safety: Loop above initialized all output elements.
    let output = unsafe { output.assume_init() };

    Ok(output)
}

#[derive(Clone, Debug)]
pub struct MatMul {}

impl Operator for MatMul {
    fn name(&self) -> &str {
        "MatMul"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let a = inputs.require_as(0)?;
        let b = inputs.require_as(1)?;
        matmul::<f32, f32, f32>(pool, a, b).into_op_result()
    }
}

pub fn matmul_integer(
    pool: &TensorPool,
    a: TensorView<u8>,
    b: TensorView<i8>,
    a_zero_point: Option<TensorView<u8>>,
    b_zero_point: Option<TensorView<i8>>,
) -> Result<Tensor<i32>, OpError> {
    if a.ndim() < 2 || b.ndim() < 2 {
        return Err(OpError::InvalidValue("Inputs must have >= 2 dimensions"));
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

    let out_prefix = broadcast_shapes(a_prefix, b_prefix)
        .ok_or(OpError::IncompatibleInputShapes("Cannot broadcast shapes"))?;
    let out_shape = &[out_prefix.as_slice(), &[a_rows, b_cols]].concat();

    let mut output = Tensor::<i32>::uninit_in(pool, out_shape);
    if output.is_empty() {
        // nb. We don't need to alloc from the pool here, since the buffer
        // is already empty.
        return Ok(Tensor::zeros(out_shape));
    }

    let a_broadcast_shape = [out_prefix.as_slice(), &[a_rows, a_cols]].concat();
    let b_broadcast_shape = [out_prefix.as_slice(), &[b_rows, b_cols]].concat();

    let a_broadcast = a.broadcast(a_broadcast_shape.as_slice());
    let b_broadcast = b.broadcast(b_broadcast_shape.as_slice());

    fn is_scalar<T>(tensor: &Option<TensorView<T>>) -> bool {
        tensor.as_ref().map(|zp| zp.ndim() == 0).unwrap_or(true)
    }

    if !is_scalar(&a_zero_point) || !is_scalar(&b_zero_point) {
        return Err(OpError::UnsupportedValue(
            "Only scalar zero points are supported",
        ));
    }

    let a_zero = a_zero_point.and_then(|zp| zp.item()).copied().unwrap_or(0) as i32;
    let b_zero = b_zero_point.and_then(|zp| zp.item()).copied().unwrap_or(0) as i32;

    a_broadcast
        .inner_iter::<2>()
        .zip(b_broadcast.inner_iter::<2>())
        .zip(output.inner_iter_mut::<2>())
        .par_bridge()
        .for_each(|((a_mat, b_mat), mut out_mat)| {
            let [m, k] = a_mat.shape();
            let [bk, n] = b_mat.shape();
            assert_eq!(k, bk);
            assert_eq!(out_mat.shape(), [m, n]);

            // Do some extremely rudimentary cache blocking.
            for col_block in range_chunks(0..n, 32) {
                for depth_block in range_chunks(0..k, 32) {
                    for row_block in range_chunks(0..m, 32) {
                        for j in col_block.clone() {
                            for i in row_block.clone() {
                                let mut out = 0i32;
                                for k in depth_block.clone() {
                                    // Safety: `[i, k]` is in-bounds for `a_mat`.
                                    let a = unsafe { *a_mat.get_unchecked([i, k]) } as i32 - a_zero;
                                    // Safety: `[k, j]` is in-bounds for `b_mat`.
                                    let b = unsafe { *b_mat.get_unchecked([k, j]) } as i32 - b_zero;
                                    out += a * b;
                                }
                                unsafe {
                                    // Safety: `[i, j]` is in-bounds for `b_mat`.
                                    let el = out_mat.get_unchecked_mut([i, j]);
                                    if depth_block.start == 0 {
                                        el.write(out);
                                    } else {
                                        el.write(el.assume_init() + out);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

    // Safety: Loop above initialized all output elements.
    let output = unsafe { output.assume_init() };

    Ok(output)
}

#[derive(Debug)]
pub struct MatMulInteger {}

impl Operator for MatMulInteger {
    fn name(&self) -> &str {
        "MatMulInteger"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let a = inputs.require_as(0)?;
        let b = inputs.require_as(1)?;
        let a_zero_point = inputs.get_as(2)?;
        let b_zero_point = inputs.get_as(3)?;
        matmul_integer(pool, a, b, a_zero_point, b_zero_point).into_op_result()
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_bench::run_bench;
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::expect_equal;
    use rten_tensor::{Tensor, TensorView, TensorViewMut};

    use crate::gemm::gemm;
    use crate::ops::binary_elementwise::broadcast_shapes;
    use crate::ops::tests::new_pool;
    use crate::tensor_pool::AutoReturn;

    use super::{gemm_op, matmul, matmul_impl, matmul_integer, MatmulStrategy, OpError};

    fn gemm_tensors(c: &mut Tensor, a: &Tensor, b: &Tensor, alpha: f32, beta: f32) {
        c.make_contiguous();
        let c_row_stride = c.stride(c.ndim() - 2);
        gemm(
            c.data_mut().unwrap(),
            c_row_stride,
            a.nd_view(),
            b.nd_view(),
            alpha,
            beta,
        )
    }

    /// Multiply matrices in `a` by corresponding matrices in `b` and write to
    /// `c`. The shapes of `a` and `b` are broadcast so that their first N-2
    /// dims match `c`.
    fn reference_matmul(mut c: TensorViewMut, a: TensorView, b: TensorView) {
        let a_batch_dims = a.ndim() - 2;
        let b_batch_dims = b.ndim() - 2;
        let out_prefix = &c.shape()[..c.ndim() - 2];

        let a_bcast = [out_prefix, &a.shape()[a_batch_dims..]].concat();
        let b_bcast = [out_prefix, &b.shape()[b_batch_dims..]].concat();

        a.broadcast(a_bcast.as_slice())
            .inner_iter::<2>()
            .zip(b.broadcast(b_bcast.as_slice()).inner_iter::<2>())
            .zip(c.inner_iter_mut::<2>())
            .for_each(|((a, b), mut c)| {
                let c_row_stride = c.stride(0);
                gemm(
                    c.data_mut().unwrap(),
                    c_row_stride,
                    a,
                    b,
                    1., /* alpha */
                    0., /* beta */
                )
            });
    }

    fn reference_matmul_integer(
        a: TensorView<u8>,
        b: TensorView<i8>,
        a_zero_point: Option<TensorView<u8>>,
        b_zero_point: Option<TensorView<i8>>,
    ) -> Tensor<i32> {
        let a_batch_dims = a.ndim() - 2;
        let b_batch_dims = b.ndim() - 2;

        let a_prefix = &a.shape()[..a.ndim() - 2];
        let b_prefix = &b.shape()[..b.ndim() - 2];
        let out_prefix = broadcast_shapes(a_prefix, b_prefix).unwrap();
        let mut out_shape = out_prefix.to_vec();
        out_shape.push(a.size(a.ndim() - 2));
        out_shape.push(b.size(b.ndim() - 1));
        let mut out = Tensor::<i32>::zeros(&out_shape);

        let a_bcast = [out_prefix.as_slice(), &a.shape()[a_batch_dims..]].concat();
        let b_bcast = [out_prefix.as_slice(), &b.shape()[b_batch_dims..]].concat();

        let a_zero_point = a_zero_point.and_then(|zp| zp.item()).copied().unwrap_or(0) as i32;
        let b_zero_point = b_zero_point.and_then(|zp| zp.item()).copied().unwrap_or(0) as i32;

        a.broadcast(a_bcast.as_slice())
            .inner_iter::<2>()
            .zip(b.broadcast(b_bcast.as_slice()).inner_iter::<2>())
            .zip(out.inner_iter_mut::<2>())
            .for_each(|((a, b), mut c)| {
                let [n_rows, n_cols] = c.shape();
                let depth = a.size(1);

                for i in 0..n_rows {
                    for j in 0..n_cols {
                        let mut y = 0;
                        for k in 0..depth {
                            let a_el = (a[[i, k]] as i32) - a_zero_point;
                            let b_el = (b[[k, j]] as i32) - b_zero_point;
                            y += a_el * b_el;
                        }
                        c[[i, j]] = y;
                    }
                }
            });

        out
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
    fn test_matmul() -> Result<(), Box<dyn Error>> {
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
        ];

        let pool = new_pool();

        for Case {
            a_shape,
            b_shape,
            out_shape,
        } in cases
        {
            let mut rng = XorShiftRng::new(1234);
            let a = Tensor::rand(a_shape, &mut rng);
            let b = Tensor::rand(b_shape, &mut rng);
            let mut expected = Tensor::zeros(out_shape);

            reference_matmul(expected.view_mut(), a.view(), b.view());
            let result = matmul(&pool, a.view(), b.view()).unwrap();
            expect_equal(&result, &expected)?;
        }

        Ok(())
    }

    #[test]
    fn test_matmul_invalid() -> Result<(), Box<dyn Error>> {
        struct Case<'a> {
            a_shape: &'a [usize],
            b_shape: &'a [usize],
            error: OpError,
        }

        let cases = [
            Case {
                a_shape: &[3],
                b_shape: &[10, 8],
                error: OpError::InvalidValue("Inputs must have >= 2 dimensions"),
            },
            Case {
                a_shape: &[3, 10],
                b_shape: &[10],
                error: OpError::InvalidValue("Inputs must have >= 2 dimensions"),
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

        let pool = new_pool();
        for Case {
            a_shape,
            b_shape,
            error,
        } in cases
        {
            let mut rng = XorShiftRng::new(1234);
            let a = Tensor::rand(a_shape, &mut rng);
            let b = Tensor::rand(b_shape, &mut rng);

            let result = matmul(&pool, a.view(), b.view());
            assert_eq!(result, Err(error));
        }

        Ok(())
    }

    #[test]
    fn test_matmul_zero_sized_dim() {
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

        let pool = new_pool();
        for Case { m, n, k } in cases {
            let mut rng = XorShiftRng::new(1234);
            let a = Tensor::rand(&[m, k], &mut rng);
            let b = Tensor::rand(&[k, n], &mut rng);
            let result = matmul(&pool, a.view(), b.view()).unwrap();

            assert_eq!(result.shape(), &[m, n]);
            if k == 0 {
                assert!(result.iter().all(|x| *x == 0.));
            }
        }
    }

    #[test]
    fn test_matmul_integer() -> Result<(), Box<dyn Error>> {
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
            // Non-scalar zero points
            Case {
                a: Tensor::from([[2, 2], [2, 2]]),
                b: Tensor::from([[2, 2], [2, 2]]),
                a_zero_point: Some(Tensor::from([[2, 2], [2, 2]])),
                b_zero_point: None,
                expected_err: Some(OpError::UnsupportedValue(
                    "Only scalar zero points are supported",
                )),
            },
            Case {
                a: Tensor::from([[2, 2], [2, 2]]),
                b: Tensor::from([[2, 2], [2, 2]]),
                a_zero_point: None,
                b_zero_point: Some(Tensor::from([[2, 2], [2, 2]])),
                expected_err: Some(OpError::UnsupportedValue(
                    "Only scalar zero points are supported",
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
            // Mismatched shapes
            Case {
                a: Tensor::zeros(&[1, 2]),
                b: Tensor::zeros(&[3, 1]),
                a_zero_point: None,
                b_zero_point: None,
                expected_err: Some(OpError::IncompatibleInputShapes(
                    "Columns of first matrix does not match rows of second matrix",
                )),
            },
            Case {
                a: Tensor::zeros(&[1]),
                b: Tensor::zeros(&[3, 1]),
                a_zero_point: None,
                b_zero_point: None,
                expected_err: Some(OpError::InvalidValue("Inputs must have >= 2 dimensions")),
            },
            Case {
                a: Tensor::zeros(&[1, 2]),
                b: Tensor::zeros(&[1]),
                a_zero_point: None,
                b_zero_point: None,
                expected_err: Some(OpError::InvalidValue("Inputs must have >= 2 dimensions")),
            },
            Case {
                a: Tensor::zeros(&[2, 2, 2]),
                b: Tensor::zeros(&[3, 2, 2]),
                a_zero_point: None,
                b_zero_point: None,
                expected_err: Some(OpError::IncompatibleInputShapes("Cannot broadcast shapes")),
            },
        ];

        let pool = new_pool();

        for Case {
            a,
            b,
            a_zero_point,
            b_zero_point,
            expected_err,
        } in cases
        {
            let result = matmul_integer(
                &pool,
                a.view(),
                b.view(),
                a_zero_point.as_ref().map(|zp| zp.view()),
                b_zero_point.as_ref().map(|zp| zp.view()),
            );

            match (result, expected_err) {
                (Ok(result), None) => {
                    let expected = reference_matmul_integer(
                        a.view(),
                        b.view(),
                        a_zero_point.as_ref().map(|zp| zp.view()),
                        b_zero_point.as_ref().map(|zp| zp.view()),
                    );
                    assert_eq!(result, expected);
                }
                (result, expected_err) => {
                    assert_eq!(result.err(), expected_err);
                }
            }
        }

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
            let a = Tensor::rand(&[a_batch, a_rows, a_cols], &mut rng);
            let b = Tensor::rand(&[a_cols, b_cols], &mut rng);

            let run_trial = |strategy| {
                let trials = 10;
                let desc = format!(
                    "matmul [{a_batch},{a_rows},{a_cols}] x [{a_cols},{b_cols}], strategy={strategy:?}",
                );
                let pool = new_pool();
                run_bench(trials, Some(&desc), || {
                    matmul_impl(&pool, a.view(), b.view(), strategy)
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
