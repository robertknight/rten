use rayon::prelude::*;

use rten_tensor::prelude::*;
use rten_tensor::{Tensor, TensorView};

use crate::check_dims;
use crate::gemm::{gemm, GemmExecutor, GemmInputA, GemmInputB};
use crate::ops::binary_elementwise::broadcast_shapes;
use crate::ops::layout::expand_to;
use crate::ops::{InputList, IntoOpResult, OpError, Operator, Output};

#[derive(Debug)]
pub struct Gemm {
    pub alpha: f32,
    pub beta: f32,
    pub transpose_a: bool,
    pub transpose_b: bool,
}

/// Compute the General Matrix Multiplication (GEMM) `c = alpha * (ab) + beta * c`.
///
/// If `transpose_a` or `transpose_b` are set, the `a` and `b` inputs
/// respectively are transposed before multiplying them.
///
/// nb. This is named `gemm_op` to avoid confusion with `gemm::gemm`.
pub fn gemm_op(
    a: TensorView,
    b: TensorView,
    c: Option<TensorView>,
    alpha: f32,
    beta: f32,
    transpose_a: bool,
    transpose_b: bool,
) -> Result<Tensor, OpError> {
    check_dims!(a, 2);
    check_dims!(b, 2);

    let a = if transpose_a { a.transposed() } else { a };
    let b = if transpose_b { b.transposed() } else { b };

    let out_shape = &[a.size(0), b.size(1)][..];
    let mut output = match c {
        Some(c) if beta != 0. => {
            if !c.can_broadcast_to(out_shape) {
                return Err(OpError::IncompatibleInputShapes(
                    "Cannot broadcast c to output shape",
                ));
            }
            expand_to(c, out_shape)
        }
        _ => Tensor::zeros(out_shape),
    };

    let out_row_stride = output.stride(0);

    gemm(
        output.data_mut().unwrap(),
        out_row_stride,
        a.nd_view(),
        b.nd_view(),
        alpha,
        beta,
    );

    Ok(output)
}

impl Operator for Gemm {
    fn name(&self) -> &str {
        "Gemm"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let a = inputs.require_as(0)?;
        let b = inputs.require_as(1)?;
        let c = inputs.get_as(2)?;
        gemm_op(
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

pub fn matmul(a: TensorView, b: TensorView) -> Result<Tensor, OpError> {
    matmul_impl(a, b, MatmulStrategy::Auto)
}

fn matmul_impl(a: TensorView, b: TensorView, strategy: MatmulStrategy) -> Result<Tensor, OpError> {
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
    if strategy == MatmulStrategy::Auto && a.ndim() > 2 && b.ndim() == 2 {
        // nb. We assume `a` is likely already contiguous, so this will be cheap.
        let a_contig = a.to_contiguous();
        let a_matrix = a_contig.reshaped([num_a_matrices * a_rows, a_cols].as_slice());
        let mut output = matmul(a_matrix, b.clone())?;
        output.reshape(out_shape);
        return Ok(output);
    }

    let mut output = Tensor::zeros(out_shape);
    if output.is_empty() {
        return Ok(output);
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

    let gemm = GemmExecutor::new();

    // Prepack re-used inputs to amortize packing cost.
    //
    // We don't prepack when the "A" matrix is a vector because that uses a
    // special case vector-matrix algorithm that doesn't benefit from packing.
    let prepacked_a = (num_a_matrices == 1 && num_b_matrices > 1 && a_rows > 1).then(|| {
        let a_matrix = a.inner_iter::<2>().next().unwrap();
        gemm.prepack_a(a_matrix)
    });
    let prepacked_b = (num_a_matrices > 1 && num_b_matrices == 1 && a_rows > 1).then(|| {
        let b_matrix = b.inner_iter::<2>().next().unwrap();
        gemm.prepack_b(b_matrix, a_cols)
    });

    a_broadcast
        .inner_iter::<2>()
        .zip(b_broadcast.inner_iter::<2>())
        .zip(out_batches)
        .par_bridge()
        .for_each(|((a_mat, b_mat), out_mat)| {
            let a_input = if let Some(prepacked_a) = prepacked_a.as_ref() {
                GemmInputA::Packed(prepacked_a)
            } else {
                GemmInputA::Unpacked(a_mat)
            };

            let b_input = if let Some(prepacked_b) = prepacked_b.as_ref() {
                GemmInputB::Packed(prepacked_b)
            } else {
                GemmInputB::Unpacked(b_mat)
            };

            gemm.gemm(
                out_mat,
                out_row_stride,
                a_input,
                b_input,
                1., // alpha
                0., // beta
            );
        });

    Ok(output)
}

#[derive(Debug)]
pub struct MatMul {}

impl Operator for MatMul {
    fn name(&self) -> &str {
        "MatMul"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let a = inputs.require_as(0)?;
        let b = inputs.require_as(1)?;
        matmul(a, b).into_op_result()
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

    use super::{gemm_op, matmul, matmul_impl, MatmulStrategy, OpError};

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

    #[test]
    fn test_gemm_op() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);
        let a = Tensor::rand(&[3, 10], &mut rng);
        let b = Tensor::rand(&[10, 8], &mut rng);

        let mut expected = Tensor::zeros(&[3, 8]);
        gemm_tensors(&mut expected, &a, &b, 1., 1.);

        let result = gemm_op(a.view(), b.view(), None, 1.0, 1.0, false, false).unwrap();

        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_gemm_op_transposed() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);
        let a = Tensor::rand(&[10, 3], &mut rng);
        let b = Tensor::rand(&[8, 10], &mut rng);

        let mut a_transposed = a.clone();
        a_transposed.permute(&[1, 0]);
        let mut b_transposed = b.clone();
        b_transposed.permute(&[1, 0]);
        let mut expected = Tensor::zeros(&[3, 8]);
        gemm_tensors(&mut expected, &a_transposed, &b_transposed, 1., 1.);

        let result = gemm_op(a.view(), b.view(), None, 1.0, 1.0, true, true).unwrap();

        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_gemm_op_adds_c() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);
        let a = Tensor::rand(&[3, 10], &mut rng);
        let b = Tensor::rand(&[10, 8], &mut rng);
        let c = Tensor::rand(&[3, 8], &mut rng);

        let mut expected = c.clone();
        gemm_tensors(&mut expected, &a, &b, 1., 1.);

        let result = gemm_op(a.view(), b.view(), Some(c.view()), 1.0, 1.0, false, false).unwrap();

        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_gemm_op_invalid_inputs() {
        let mut rng = XorShiftRng::new(1234);
        let a = Tensor::rand(&[3, 10], &mut rng);
        let b = Tensor::rand(&[10, 8], &mut rng);
        let c = Tensor::rand(&[3, 5], &mut rng);

        let result = gemm_op(a.view(), b.view(), Some(c.view()), 1.0, 1.0, false, false);

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
            let result = matmul(a.view(), b.view()).unwrap();
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

        for Case {
            a_shape,
            b_shape,
            error,
        } in cases
        {
            let mut rng = XorShiftRng::new(1234);
            let a = Tensor::rand(a_shape, &mut rng);
            let b = Tensor::rand(b_shape, &mut rng);

            let result = matmul(a.view(), b.view());
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

        for Case { m, n, k } in cases {
            let mut rng = XorShiftRng::new(1234);
            let a = Tensor::rand(&[m, k], &mut rng);
            let b = Tensor::rand(&[k, n], &mut rng);
            let result = matmul(a.view(), b.view()).unwrap();

            assert_eq!(result.shape(), &[m, n]);
            if k == 0 {
                assert!(result.iter().all(|x| *x == 0.));
            }
        }
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
                run_bench(trials, &desc, || {
                    matmul_impl(a.view(), b.view(), strategy).unwrap();
                });
            };

            run_trial(MatmulStrategy::Batch);
            run_trial(MatmulStrategy::Auto);
            println!();
        }
    }
}
