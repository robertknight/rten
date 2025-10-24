use std::error::Error;
use std::time::Instant;

use rten_bench::run_bench;
use rten_tensor::prelude::*;
use rten_tensor::rng::XorShiftRng;
use rten_tensor::test_util::{ApproxEq, expect_equal};
use rten_tensor::{Matrix, MatrixLayout, MatrixMut, NdTensor, NdTensorView, RandomSource};
use rten_testing::TestCases;

use super::{
    BiasVector, BlockQuantizedMatrix, ColOffsets, F32KernelType, GemmError, GemmExecutor, GemmInT,
    GemmInputA, GemmInputB, GemmOutT, Im2Col, QuantParams, ReducedRangeRng, RowOffsets, WithKernel,
};

/// Scale a possibly non-float value by a float.
///
/// Used for scaling by alpha in `C = alpha * AB + beta * C`.
trait MulFloat {
    fn mul_float(self, scale: f32) -> Self;
}

impl MulFloat for f32 {
    fn mul_float(self, scale: f32) -> Self {
        self * scale
    }
}

impl MulFloat for i32 {
    fn mul_float(self, scale: f32) -> Self {
        (self as f32 * scale) as Self
    }
}

/// Type that can be used as the output for the reference GEMM
/// implementation.
trait RefGemmOutT<LhsT, RhsT>:
    Default
    + GemmOutT
    + From<LhsT>
    + From<RhsT>
    + MulFloat
    + ApproxEq
    + std::fmt::Debug
    + std::ops::Sub<Output = Self>
{
}

impl<LhsT, RhsT> RefGemmOutT<LhsT, RhsT> for f32
where
    f32: From<LhsT>,
    f32: From<RhsT>,
{
}

impl<LhsT, RhsT> RefGemmOutT<LhsT, RhsT> for i32
where
    i32: From<LhsT>,
    i32: From<RhsT>,
{
}

#[derive(Clone)]
struct GemmOpts<'a, LhsT, RhsT, OutT> {
    alpha: f32,
    beta: OutT,
    bias: Option<BiasVector<'a, OutT>>,
    a_quant: Option<QuantParams<'a, LhsT>>,
    b_quant: Option<QuantParams<'a, RhsT>>,
}

impl<LhsT, RhsT, OutT: GemmOutT> Default for GemmOpts<'_, LhsT, RhsT, OutT> {
    fn default() -> Self {
        GemmOpts {
            alpha: 1.,
            beta: OutT::zero(),
            bias: None,
            a_quant: None,
            b_quant: None,
        }
    }
}

/// Reference implementation. This should produce the same results as the
/// optimized GEMM, but small numerical differences will appear in problems
/// with a large K dimension, due to the different ordering of
/// floating-point operations.
fn reference_gemm<LhsT, RhsT, OutT>(
    mut output: MatrixMut<OutT>,
    a: Matrix<LhsT>,
    b: Matrix<RhsT>,
    opts: Option<GemmOpts<LhsT, RhsT, OutT>>,
) where
    LhsT: GemmInT,
    RhsT: GemmInT,
    OutT: RefGemmOutT<LhsT, RhsT>,
{
    let GemmOpts {
        alpha,
        beta,
        bias,
        a_quant,
        b_quant,
    } = opts.unwrap_or_default();

    for r in 0..a.rows() {
        let a_zero = a_quant
            .as_ref()
            .map(|aq| OutT::from(aq.zero_point[r]))
            .unwrap_or(OutT::zero());
        for c in 0..b.cols() {
            let b_zero = b_quant
                .as_ref()
                .map(|bq| OutT::from(bq.zero_point[c]))
                .unwrap_or(OutT::zero());

            let mut accum = OutT::zero();
            for k in 0..a.cols() {
                let a_el = OutT::from(a[[r, k]]) - a_zero;
                let b_el = OutT::from(b[[k, c]]) - b_zero;
                accum = accum + a_el * b_el;
            }
            let bias = match bias {
                Some(BiasVector::Row(b)) => b[c],
                Some(BiasVector::Column(b)) => b[r],
                None => OutT::zero(),
            };
            output[[r, c]] = accum.mul_float(alpha) + beta * output[[r, c]] + bias;
        }
    }
}

fn reference_matmul<LhsT, RhsT, OutT>(
    a: Matrix<LhsT>,
    b: Matrix<RhsT>,
    opts: Option<GemmOpts<LhsT, RhsT, OutT>>,
) -> NdTensor<OutT, 2>
where
    LhsT: GemmInT,
    RhsT: GemmInT,
    OutT: RefGemmOutT<LhsT, RhsT>,
{
    if let Some(opts) = &opts {
        assert_eq!(
            opts.beta,
            OutT::zero(),
            "beta has no effect in `reference_matmul`"
        );
    }
    let mut output = NdTensor::full([a.rows(), b.cols()], OutT::zero());
    reference_gemm(output.view_mut(), a, b, opts);
    output
}

// Maximum block sizes that the GEMM implementation uses. Choosing M, N, K
// inputs larger than this will ensure that multiple blocks are used along
// that dimension.
//
// A better approach would be to make these block sizes configurable and set
// them to small values in tests, so tests can enforce the use of multiple
// blocks without needing large inputs that are slow when tests are compiled
// in debug mode.
const ROW_BLOCK_SIZE: usize = 64;
const COL_BLOCK_SIZE: usize = 1024;
const DEPTH_BLOCK_SIZE: usize = 256;

fn run_gemm<LhsT: GemmInT, RhsT: GemmInT, OutT: GemmOutT>(
    mut output: MatrixMut<OutT>,
    a: Matrix<LhsT>,
    b: Matrix<RhsT>,
    opts: Option<GemmOpts<LhsT, RhsT, OutT>>,
    gemm: Option<&GemmExecutor<LhsT, RhsT, OutT>>,
) -> super::GemmResult
where
    GemmExecutor<LhsT, RhsT, OutT>: Default,
{
    let default_gemm = GemmExecutor::default();
    let gemm = gemm.unwrap_or(&default_gemm);
    let GemmOpts {
        alpha,
        beta,
        bias,
        a_quant,
        b_quant,
    } = opts.unwrap_or_default();

    gemm.gemm(
        output.data_mut().expect("expected contiguous input"),
        GemmInputA::Unpacked(a),
        GemmInputB::Unpacked(b),
        alpha,
        beta,
        bias,
        a_quant,
        b_quant,
    )
    .map(|_| ())
}

fn run_matmul<LhsT: GemmInT, RhsT: GemmInT, OutT: GemmOutT + Default>(
    a: Matrix<LhsT>,
    b: Matrix<RhsT>,
    opts: Option<GemmOpts<LhsT, RhsT, OutT>>,
    gemm: Option<&GemmExecutor<LhsT, RhsT, OutT>>,
) -> Result<NdTensor<OutT, 2>, GemmError>
where
    GemmExecutor<LhsT, RhsT, OutT>: Default,
{
    let mut output = NdTensor::zeros([a.rows(), b.cols()]);
    run_gemm(output.view_mut(), a, b, opts, gemm)?;
    Ok(output)
}

/// Run a matmul with the reference and real implementations and verify
/// the results match.
fn run_compare_matmul<LhsT: GemmInT, RhsT: GemmInT, OutT: RefGemmOutT<LhsT, RhsT>>(
    a: Matrix<LhsT>,
    b: Matrix<RhsT>,
    opts: Option<GemmOpts<LhsT, RhsT, OutT>>,
    gemm: Option<&GemmExecutor<LhsT, RhsT, OutT>>,
) where
    GemmExecutor<LhsT, RhsT, OutT>: Default,
{
    let result = run_matmul(a.view(), b.view(), opts.clone(), gemm).unwrap();
    let expected = reference_matmul(a.view(), b.view(), opts);
    expect_equal(&result, &expected).unwrap();
}

/// Return `GemmExecutor`s with all of the available kernels for the given
/// input and output types.
fn all_gemms<L, R, O>() -> impl Iterator<Item = GemmExecutor<L, R, O>>
where
    L: GemmInT,
    R: GemmInT,
    O: GemmOutT,
    GemmExecutor<L, R, O>: WithKernel,
{
    GemmExecutor::<L, R, O>::kernel_types()
        .into_iter()
        .filter_map(|kern_type| GemmExecutor::<L, R, O>::with_kernel(kern_type))
}

// Simplest possible test case for easy debugging.
#[test]
fn test_simple_gemm_f32() -> Result<(), Box<dyn Error>> {
    let a = NdTensor::from_data([2, 2], vec![1., 2., 3., 4.]);
    let b = NdTensor::from_data([2, 2], vec![5., 6., 7., 8.]);
    run_compare_matmul(a.view(), b.view(), None, None);
    run_compare_matmul(
        a.view(),
        b.view(),
        None,
        Some(&GemmExecutor::<f32>::with_kernel(F32KernelType::Generic).unwrap()),
    );
    Ok(())
}

#[test]
fn test_simple_gemm_u8i8_i32() -> Result<(), Box<dyn Error>> {
    let a = NdTensor::from_data([2, 2], vec![1, 2, 3, 4]);
    let b = NdTensor::from_data([2, 2], vec![5, 6, 7, 8]);
    run_compare_matmul::<u8, i8, i32>(a.view(), b.view(), None, None);
    Ok(())
}

#[test]
fn test_gemm_input_errors() {
    #[derive(Debug)]
    struct Case {
        a: NdTensor<f32, 2>,
        b: NdTensor<f32, 2>,
        output_len: usize,
        expected: GemmError,
    }

    let cases = [
        Case {
            a: NdTensor::from([[1., 2.], [3., 4.]]),
            b: NdTensor::from([[1., 2.], [3., 4.]]),
            output_len: 2,
            expected: GemmError::OutputSizeMismatch,
        },
        Case {
            a: NdTensor::from([[1.], [2.]]),
            b: NdTensor::from([[1., 2.], [3., 4.]]),
            output_len: 4,
            expected: GemmError::KSizeMismatch,
        },
    ];

    cases.test_each(
        |Case {
             a,
             b,
             output_len,
             expected,
         }| {
            let gemm = GemmExecutor::default();
            let mut output = vec![0.; *output_len];
            let result = gemm.gemm(
                &mut output,
                GemmInputA::Unpacked(a.view()),
                GemmInputB::Unpacked(b.view()),
                1.,   // alpha
                0.,   // beta
                None, // bias
                None, // a_quant
                None, // b_quant
            );
            assert_eq!(result.as_ref(), Err(expected));
        },
    )
}

/// Test a GEMM kernel using all square matrices up to a given size, plus
/// various other "interesting" size combinations.
fn test_gemm_various_input_sizes<LhsT, RhsT, OutT>(
    gemm: Option<&GemmExecutor<LhsT, RhsT, OutT>>,
    mut lhs_gen: Option<&mut dyn FnMut() -> LhsT>,
    mut rhs_gen: Option<&mut dyn FnMut() -> RhsT>,
) -> Result<(), Box<dyn Error>>
where
    LhsT: GemmInT,
    RhsT: GemmInT,
    GemmExecutor<LhsT, RhsT, OutT>: Default,
    XorShiftRng: RandomSource<LhsT>,
    XorShiftRng: RandomSource<RhsT>,
    OutT: RefGemmOutT<LhsT, RhsT>,
{
    // "Interesting" sizes for the row, column and depth dimensions of the
    // computation. These are chosen to cover cases that are less than,
    // equal to and above the tile/block sizes which the algorithm divides
    // the problem into along each dimension.
    //
    // This also covers the case where each dimension is a vector.
    let col_steps = [0, 1, 2, 4, 5, 8, 1024, 1025];
    let depth_steps = [0, 1, 2, 20, 256, 300];
    let row_steps = [0, 1, 2, 8, 10, 16, 64, 80];

    let mut cases = Vec::new();

    // Simple cases where one dimension of the problem is varied to
    // different interesting values and other dimensions are kept small.
    for cs in col_steps {
        cases.push(([2, 2], [2, cs]));
    }
    for ds in depth_steps {
        cases.push(([2, ds], [ds, 2]));
    }
    for rs in row_steps {
        cases.push(([rs, 2], [2, 2]));
    }

    // Some simple square matrix tests of different sizes. This covers all
    // cases below a threshold, and then select sizes after that. This is
    // because larger sizes are slow in debug builds.
    for n in 1..20 {
        cases.push(([n, n], [n, n]));
    }
    for n in [30, 64, 65] {
        cases.push(([n, n], [n, n]));
    }

    let mut failures = Vec::new();
    for (lhs_size, rhs_size) in cases.iter().copied() {
        let mut rng = XorShiftRng::new(1234);
        let a = if let Some(lhs_gen) = lhs_gen.as_mut() {
            NdTensor::<LhsT, 2>::from_simple_fn(lhs_size, lhs_gen)
        } else {
            NdTensor::<LhsT, 2>::rand(lhs_size, &mut rng)
        };
        let b = if let Some(rhs_gen) = rhs_gen.as_mut() {
            NdTensor::<RhsT, 2>::from_simple_fn(rhs_size, rhs_gen)
        } else {
            NdTensor::<RhsT, 2>::rand(rhs_size, &mut rng)
        };

        let result = run_matmul(a.view(), b.view(), None, gemm).unwrap();
        let expected = reference_matmul(a.view(), b.view(), None);

        if let Err(err) = expect_equal(&result, &expected) {
            failures.push((lhs_size, rhs_size, err));
        }
    }

    if !failures.is_empty() {
        // Show which sizes had failures. Print extended details for the first
        // to show a sample of discrepancies.
        for (i, (lhs_size, rhs_size, err)) in failures.iter().enumerate() {
            if i < 3 {
                println!(
                    "GEMM output for {}x{}x{} did not match reference: {}",
                    lhs_size[0], rhs_size[1], lhs_size[1], err
                );
            } else {
                println!(
                    "GEMM output for {}x{}x{} did not match reference",
                    lhs_size[0], rhs_size[1], lhs_size[1],
                );
            }
        }

        return Err(format!(
            "GEMM output did not match reference for {} of {} cases",
            failures.len(),
            cases.len()
        )
        .into());
    }

    Ok(())
}

#[test]
fn test_gemm_f32() -> Result<(), Box<dyn Error>> {
    for gemm in all_gemms::<f32, f32, f32>() {
        test_gemm_various_input_sizes(Some(&gemm), None, None)?;
    }
    Ok(())
}

#[test]
fn test_gemm_u8i8_i32() -> Result<(), Box<dyn Error>> {
    for gemm in all_gemms::<u8, i8, i32>() {
        let mut rng = ReducedRangeRng::new(gemm.may_saturate(), 1234);
        test_gemm_various_input_sizes(Some(&gemm), None, Some(&mut || rng.next()))?;
    }
    Ok(())
}

#[test]
fn test_gemm_u8i8_i32_zero_point() {
    #[derive(Copy, Clone, Debug)]
    struct Case {
        m: usize,
        n: usize,
        k: usize,
    }

    let cases = [
        // Matrix-matrix
        Case { m: 5, n: 7, k: 10 },
        // Vector-matrix product.
        Case { m: 1, n: 5, k: 10 },
        Case { m: 1, n: 8, k: 4 },
        Case { m: 1, n: 16, k: 4 },
        // Vector-matrix product, K not a multiple of 4 (tile size used by
        // int8 dot product instructions).
        Case { m: 1, n: 1, k: 2 },
        // Vector-matrix, where n is large enough that work should be
        // divided into multiple blocks.
        Case {
            m: 1,
            n: 256,
            k: 10,
        },
    ];

    cases.test_each(|&Case { m, n, k }| {
        for gemm in all_gemms::<u8, i8, i32>() {
            let mut lhs_rng = XorShiftRng::new(1234);
            let mut rhs_rng = ReducedRangeRng::new(gemm.may_saturate(), 5678);

            let a = NdTensor::<u8, 2>::rand([m, k], &mut lhs_rng);
            let b = NdTensor::<i8, 2>::rand([k, n], &mut rhs_rng);

            let a_zero_point: Vec<_> = (0..a.rows()).map(|x| x as u8).collect();
            let b_zero_point: Vec<_> = (0..b.cols()).map(|x| x as i8).collect();
            let opts = Some(GemmOpts {
                a_quant: Some(QuantParams {
                    zero_point: &a_zero_point,
                }),
                b_quant: Some(QuantParams {
                    zero_point: &b_zero_point,
                }),
                ..Default::default()
            });
            run_compare_matmul(a.view(), b.view(), opts, Some(&gemm));
        }
    })
}

#[test]
fn test_gemm_u8i8_i32_invalid_zero_point() {
    let mut rng = XorShiftRng::new(1234);
    let a = NdTensor::<u8, 2>::rand([5, 10], &mut rng);
    let b = NdTensor::<i8, 2>::rand([10, 3], &mut rng);

    fn gemm_opts<'a>(a_zero_point: &'a [u8], b_zero_point: &'a [i8]) -> GemmOpts<'a, u8, i8, i32> {
        GemmOpts {
            a_quant: Some(QuantParams {
                zero_point: a_zero_point,
            }),
            b_quant: Some(QuantParams {
                zero_point: b_zero_point,
            }),
            ..Default::default()
        }
    }
    let a_zero_point: Vec<_> = (0..a.rows()).map(|row| row as u8).collect();
    let b_zero_point: Vec<_> = (0..b.cols()).map(|col| col as i8).collect();

    // LHS zero point does not match LHS rows.
    let result = run_matmul(
        a.view(),
        b.view(),
        Some(gemm_opts(&[1, 2, 3], &b_zero_point)),
        None,
    );
    assert_eq!(result, Err(GemmError::WrongQuantParamSize));

    // RHS zero point does not match RHS columns.
    let result = run_matmul(
        a.view(),
        b.view(),
        Some(gemm_opts(&a_zero_point, &[1, 2, 3, 4])),
        None,
    );
    assert_eq!(result, Err(GemmError::WrongQuantParamSize));
}

#[test]
fn test_gemm_transposed() -> Result<(), Box<dyn Error>> {
    let mut rng = XorShiftRng::new(1234);
    let mut a = NdTensor::<f32, 2>::rand([20, 30], &mut rng);
    let mut b = NdTensor::rand([10, 20], &mut rng);

    // Transpose the input matrices. This will alter their row and column
    // strides and shapes, but not re-order the data.
    a.permute([1, 0]);
    b.permute([1, 0]);

    run_compare_matmul(a.view(), b.view(), None, None);

    Ok(())
}

#[test]
fn test_gemv_u8i8_i32_transposed() {
    #[derive(Debug)]
    struct Case {
        n: usize,
        k: usize,
    }

    let cases = [
        // K multiple of 4
        Case { k: 8, n: 5 },
        // K not a multiple of 4
        Case { k: 2, n: 5 },
    ];

    cases.test_each(|&Case { k, n }| {
        for gemm in all_gemms::<u8, i8, i32>() {
            let mut lhs_rng = XorShiftRng::new(1234);
            let mut rhs_rng = ReducedRangeRng::new(gemm.may_saturate(), 5678);

            let a = NdTensor::<u8, 2>::rand([1, k], &mut lhs_rng);
            let mut b = NdTensor::<i8, 2>::rand([n, k], &mut rhs_rng);

            // Transpose the input B matrix. This will alter the row and column
            // strides and shapes, but not re-order the data.
            b.permute([1, 0]);

            run_compare_matmul(a.view(), b.view(), None, Some(&gemm));
        }
    })
}

#[test]
fn test_gemm_alpha() -> Result<(), Box<dyn Error>> {
    let mut rng = XorShiftRng::new(1234);

    let a = NdTensor::rand([10, 5], &mut rng);
    let b = NdTensor::rand([5, 15], &mut rng);

    for gemm in all_gemms::<f32, f32, f32>() {
        for alpha in [0.0, 0.5, 1.0, 2.0] {
            let opts = Some(GemmOpts {
                alpha,
                ..Default::default()
            });
            run_compare_matmul(a.view(), b.view(), opts.clone(), Some(&gemm));
        }
    }

    Ok(())
}

#[test]
fn test_gemm_beta() {
    #[derive(Debug)]
    struct Case {
        m: usize,
        n: usize,
        k: usize,
    }

    let cases = [Case { m: 10, k: 5, n: 15 }, Case { m: 10, k: 0, n: 15 }];

    cases.test_each(|&Case { m, n, k }| {
        let mut rng = XorShiftRng::new(1234);
        let a = NdTensor::rand([m, k], &mut rng);
        let b = NdTensor::rand([k, n], &mut rng);

        for gemm in all_gemms::<f32, f32, f32>() {
            for beta in [0.5, 1.0, 2.0] {
                let mut result = NdTensor::rand([m, n], &mut rng);
                let mut expected = result.clone();
                let opts = Some(GemmOpts {
                    beta,
                    ..Default::default()
                });

                run_gemm(
                    result.view_mut(),
                    a.view(),
                    b.view(),
                    opts.clone(),
                    Some(&gemm),
                )
                .unwrap();
                reference_gemm(expected.view_mut(), a.view(), b.view(), opts);

                expect_equal(&result, &expected).unwrap();
            }
        }
    })
}

#[test]
fn test_gemm_beta_zero() {
    #[derive(Debug)]
    struct Case {
        m: usize,
        n: usize,
        k: usize,
    }

    let cases = [
        // Matrix-matrix multiplication
        Case {
            m: 20,
            n: 20,
            k: 20,
        },
        Case { m: 5, n: 5, k: 0 },
        // Vector-matrix multiplication
        Case { m: 1, n: 20, k: 20 },
    ];

    cases.test_each(|&Case { m, n, k }| {
        let mut rng = XorShiftRng::new(1234);
        let a = NdTensor::rand([m, k], &mut rng);
        let b = NdTensor::rand([k, n], &mut rng);

        // Create output buffer with NANs. This will cause incorrect
        // output if the GEMM impl incorrectly does `C = beta * C * alpha *
        // AB` instead of `C = alpha * AB` where beta is zero.
        let mut result = NdTensor::full([m, n], f32::NAN);

        // Test alpha values for which we may have special cases (0, 1) and
        // the general case.
        for alpha in [0., 0.5, 1.] {
            let opts = Some(GemmOpts {
                alpha,
                ..Default::default()
            });
            run_gemm(result.view_mut(), a.view(), b.view(), opts.clone(), None).unwrap();
            let expected = reference_matmul(a.view(), b.view(), opts);
            expect_equal(&result, &expected).unwrap();
        }
    })
}

#[test]
fn test_gemm_bias() {
    #[derive(Debug)]
    struct Case {
        m: usize,
        n: usize,
        k: usize,
    }

    let cases = [
        // Matrix-matrix
        Case { m: 10, n: 15, k: 5 },
        // Vector-matrix
        Case { m: 1, n: 15, k: 5 },
        // Vector-matrix, where n > minimum block size
        Case { m: 1, n: 129, k: 1 },
        // Case where k == 0
        Case { m: 5, n: 7, k: 0 },
    ];

    cases.test_each(|&Case { m, n, k }| {
        let mut rng = XorShiftRng::new(1234);

        let a = NdTensor::rand([m, k], &mut rng);
        let b = NdTensor::rand([k, n], &mut rng);

        // Column vector bias
        let bias: Vec<f32> = (0..a.rows()).map(|b| b as f32).collect();
        let opts = Some(GemmOpts {
            bias: Some(BiasVector::Column(&bias)),
            ..Default::default()
        });
        run_compare_matmul(a.view(), b.view(), opts, None);

        // Row vector bias
        let bias: Vec<f32> = (0..b.cols()).map(|b| b as f32).collect();
        let opts = Some(GemmOpts {
            bias: Some(BiasVector::Row(&bias)),
            ..Default::default()
        });
        run_compare_matmul(a.view(), b.view(), opts, None);
    })
}

#[test]
fn test_gemm_prepack() {
    #[derive(Clone, Debug)]
    struct Case {
        m: usize,
        n: usize,
        k: usize,
    }
    let cases = [
        // Small input that uses one block along each dimension.
        Case { m: 10, n: 15, k: 5 },
        // Inputs with one dimension large enough to require multiple blocks.
        Case {
            m: ROW_BLOCK_SIZE * 2 + ROW_BLOCK_SIZE / 2,
            n: 15,
            k: 5,
        },
        Case {
            m: 10,
            n: COL_BLOCK_SIZE * 2 + COL_BLOCK_SIZE / 2,
            k: 5,
        },
        Case {
            m: 10,
            n: 15,
            k: DEPTH_BLOCK_SIZE * 2 + DEPTH_BLOCK_SIZE / 2,
        },
    ];

    cases.test_each_clone(|case| {
        let Case { m, n, k } = case;

        let mut rng = XorShiftRng::new(1234);
        let a = NdTensor::rand([m, k], &mut rng);
        let b = NdTensor::rand([k, n], &mut rng);

        let gemm = GemmExecutor::new();

        let packed_a = gemm.prepack_a(a.view());
        let packed_b = gemm.prepack_b(b.view());

        let mut result = NdTensor::zeros([m, n]);

        gemm.gemm(
            result.data_mut().unwrap(),
            GemmInputA::Packed(&packed_a),
            GemmInputB::Packed(&packed_b),
            1.,   // alpha
            1.,   // beta
            None, // bias
            None, // a_quant
            None, // b_quant
        )
        .unwrap();

        // Compare the results of pre-packed GEMM to unpacked GEMM rather
        // than reference GEMM because a) it is faster for large inputs
        // and b) in the case where K is large, the accumulated numerical
        // differences will be smaller.
        let mut expected = NdTensor::zeros(result.shape());
        gemm.gemm(
            expected.data_mut().unwrap(),
            GemmInputA::Unpacked(a.view()),
            GemmInputB::Unpacked(b.view()),
            1.,   // alpha
            1.,   // beta
            None, // bias
            None, // a_quant
            None, // b_quant
        )
        .unwrap();

        expect_equal(&result, &expected).unwrap();
    })
}

// Simplified version of the im2col builder used by convolution code.
//
// This builds a mapping between elements of an image and a
// `[chans, height x width]` matrix where `image[c, y, x]` maps to
// `im2col_matrix[c, y / width, y % width]`.
fn build_im2col<T: Copy>(
    image: NdTensorView<T, 3>,
    col_count_step: usize,
    row_count_step: usize,
) -> Im2Col<T> {
    let [chans, img_h, img_w] = image.shape();
    let [chan_stride, h_stride, w_stride] = image.strides();

    let n_cols = img_w * img_h;
    let n_cols_padded = n_cols.next_multiple_of(col_count_step);

    let rows = chans;
    let n_rows_padded = rows.next_multiple_of(row_count_step);

    let mut row_offsets = RowOffsets {
        chan: (0..rows as i32)
            .map(|chan| chan * chan_stride as i32)
            .collect(),
        y: vec![0; rows],
        x: vec![0; rows],
    };

    for _ in rows..n_rows_padded {
        row_offsets.chan.push(i32::MAX);
        row_offsets.x.push(i32::MAX);
        row_offsets.y.push(i32::MAX);
    }

    let mut col_offsets = ColOffsets {
        y: (0..n_cols)
            .map(|i| i as i32 / img_w as i32)
            .map(|y| y * h_stride as i32)
            .collect(),
        x: (0..n_cols)
            .map(|i| i as i32 % img_w as i32)
            .map(|x| x * w_stride as i32)
            .collect(),
    };
    for _ in n_cols..n_cols_padded {
        col_offsets.y.push(i32::MAX);
        col_offsets.x.push(i32::MAX);
    }

    let max_y_offset = (img_h - 1) * h_stride;
    let max_x_offset = (img_w - 1) * w_stride;

    Im2Col {
        image,
        row_offsets,
        col_offsets,
        n_cols,
        n_rows: rows,
        max_y_offset: max_y_offset as i32,
        max_x_offset: max_x_offset as i32,
    }
}

#[test]
fn test_gemm_im2col_f32() -> Result<(), Box<dyn Error>> {
    let mut rng = XorShiftRng::new(1234);
    let gemm = GemmExecutor::default();

    // nb. If the test fails, debug by setting dimensions to 1.
    let img_h = 2;
    let img_w = 2;
    let img_chans = 2;
    let kernel_chans = 3;

    let img = NdTensor::<f32, 3>::rand([img_chans, img_h, img_w], &mut rng);
    let im2col = build_im2col(
        img.view(),
        gemm.im2col_col_count_step(),
        gemm.im2col_row_count_step(),
    );

    let kernel_mat = NdTensor::<f32, 2>::rand([kernel_chans, img_chans], &mut rng);
    let mut output_mat = NdTensor::<f32, 2>::zeros([kernel_chans, img_h * img_w]);

    gemm.gemm(
        output_mat.data_mut().unwrap(),
        GemmInputA::Unpacked(kernel_mat.view()),
        GemmInputB::Im2Col(&im2col),
        1.,   // alpha
        0.,   // beta
        None, // bias
        None, // a_quant
        None, // b_quant
    )
    .unwrap();

    let mut expected = NdTensor::<f32, 2>::zeros([kernel_chans, im2col.cols()]);
    for i in 0..expected.rows() {
        for j in 0..expected.cols() {
            let mut acc = 0.;
            for k in 0..kernel_mat.cols() {
                acc += kernel_mat[[i, k]] * img[[k, j / img_w, j % img_w]];
            }
            expected[[i, j]] = acc;
        }
    }
    expect_equal(&output_mat, &expected)?;

    Ok(())
}

#[test]
fn test_gemm_im2col_u8i8_i32() -> Result<(), Box<dyn Error>> {
    let mut rng = XorShiftRng::new(1234);

    // nb. If the test fails, debug by setting dimensions to 1.
    let img_h = 2;
    let img_w = 2;
    let img_chans = 2;
    let kernel_chans = 3;

    let img = NdTensor::<i8, 3>::rand([img_chans, img_h, img_w], &mut rng);

    for gemm in all_gemms() {
        let im2col = build_im2col(
            img.view(),
            gemm.im2col_col_count_step(),
            gemm.im2col_row_count_step(),
        );
        let kernel_mat = NdTensor::<u8, 2>::rand([kernel_chans, img_chans], &mut rng);
        let mut output_mat = NdTensor::<i32, 2>::zeros([kernel_chans, img_h * img_w]);

        gemm.gemm(
            output_mat.data_mut().unwrap(),
            GemmInputA::Unpacked(kernel_mat.view()),
            GemmInputB::Im2Col(&im2col),
            1.,   // alpha
            0,    // beta
            None, // bias
            None, // a_quant
            None, // b_quant
        )
        .unwrap();

        let mut expected = NdTensor::<i32, 2>::zeros([kernel_chans, im2col.cols()]);
        for i in 0..expected.rows() {
            for j in 0..expected.cols() {
                let mut acc = 0;
                for k in 0..kernel_mat.cols() {
                    acc += kernel_mat[[i, k]] as i32 * img[[k, j / img_w, j % img_w]] as i32;
                }
                expected[[i, j]] = acc;
            }
        }
        expect_equal(&output_mat, &expected)?;
    }

    Ok(())
}

fn reference_gemm_block_quant_4bit(
    lhs: NdTensorView<f32, 2>,
    rhs: NdTensorView<u8, 3>,
    rhs_scales: NdTensorView<f32, 2>,
) -> NdTensor<f32, 2> {
    let [m, k] = lhs.shape();
    let [n, _k_blocks, block_size] = rhs.shape();
    let elems_per_block = block_size * 2;
    let zero_point = 8;

    let mut out = NdTensor::zeros([m, n]);

    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.;
            for ki in 0..k {
                let k_block = ki / elems_per_block;
                let block_idx = ki % elems_per_block;
                let scale = rhs_scales[[col, k_block]];

                let byte = rhs[[col, k_block, block_idx / 2]];
                let elem = if ki % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                let dequantized_elem = (elem as i32 - zero_point) as f32 * scale;

                acc += lhs[[row, ki]] * dequantized_elem;
            }
            out[[row, col]] = acc;
        }
    }

    out
}

#[test]
fn test_gemm_block_quantized_f32() {
    let mut rng = XorShiftRng::new(1234);

    for gemm in all_gemms() {
        let block_size = 16;
        let m = 4;
        // nb. Not a multiple of NR for any architecture, so there will be
        // padding.
        let n = 17;
        let k = 32;
        let n_bits = 4 as u8;
        let n_elem = 8 / n_bits as usize;

        let lhs = NdTensor::<f32, 2>::rand([m, k], &mut rng);
        let rhs = NdTensor::<u8, 3>::rand([n, k / (block_size * n_elem), block_size], &mut rng);
        let rhs_scales = NdTensor::<f32, 2>::rand([n, k / block_size], &mut rng);
        let rhs_bqm = BlockQuantizedMatrix::new(rhs.view(), rhs_scales.view(), n_bits).unwrap();

        assert_eq!(lhs.cols(), rhs_bqm.rows());

        let mut output = NdTensor::<f32, 2>::zeros([m, n]);
        gemm.gemm(
            output.view_mut().data_mut().unwrap(),
            GemmInputA::Unpacked(lhs.view()),
            GemmInputB::BlockQuantized(rhs_bqm),
            1.,   // alpha
            0.,   // beta
            None, // bias
            None, // a_quant
            None, // b_quant
        )
        .unwrap();

        let expected = reference_gemm_block_quant_4bit(lhs.view(), rhs.view(), rhs_scales.view());
        expect_equal(&output, &expected).unwrap();
    }
}

#[test]
fn test_gemv() {
    #[derive(Clone, Copy, Debug)]
    enum Strides {
        Contiguous,
        Transposed,
        Other,
    }

    #[derive(Debug)]
    struct Case {
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
        bias: Option<f32>,
        b_strides: Strides,
    }

    impl Default for Case {
        fn default() -> Case {
            Case {
                n: 16,
                k: 16,
                alpha: 1.,
                beta: 0.,
                bias: None,
                b_strides: Strides::Contiguous,
            }
        }
    }

    let cases = [
        // Empty inputs
        Case {
            n: 0,
            k: 1,
            ..Default::default()
        },
        Case {
            n: 1,
            k: 0,
            ..Default::default()
        },
        // Smallest possible input
        Case {
            n: 1,
            k: 1,
            ..Default::default()
        },
        // n is a multiple of the tile size (16 for AVX 2 / FMA)
        Case {
            n: 16,
            k: 16,
            ..Default::default()
        },
        // n is not an exact multiple of the tile size
        Case {
            n: 20,
            k: 16,
            ..Default::default()
        },
        // n exceeds column block size
        Case {
            n: 300,
            k: 16,
            ..Default::default()
        },
        // k exceeds depth block size
        Case {
            n: 20,
            k: 300,
            ..Default::default()
        },
        // beta value = 0.
        Case {
            n: 20,
            k: 300,
            beta: 0.,
            ..Default::default()
        },
        // Non-standard beta value
        Case {
            n: 20,
            k: 300,
            beta: 0.5,
            ..Default::default()
        },
        // Non-standard alpha value
        Case {
            n: 20,
            k: 20,
            alpha: 0.5,
            ..Default::default()
        },
        // Test with bias
        Case {
            n: 20,
            k: 20,
            bias: Some(0.5),
            ..Default::default()
        },
        // Transposed matrix. Note both `n` and `k` are chosen to not be
        // an exact multiple of column or depth tile sizes.
        Case {
            n: 21,
            k: 21,
            b_strides: Strides::Transposed,
            ..Default::default()
        },
        // Transposed matrix with beta != 0
        Case {
            n: 21,
            k: 21,
            beta: 1.,
            b_strides: Strides::Transposed,
            ..Default::default()
        },
        // Transposed matrix with alpha != 1
        Case {
            n: 20,
            k: 20,
            alpha: 0.5,
            b_strides: Strides::Transposed,
            ..Default::default()
        },
        // Matrix with non-unit strides
        Case {
            n: 21,
            k: 21,
            b_strides: Strides::Other,
            ..Default::default()
        },
        // Matrix with non-unit strides, beta != 0
        Case {
            n: 21,
            k: 21,
            beta: 0.5,
            b_strides: Strides::Other,
            ..Default::default()
        },
        // Matrix with non-unit strides, alpha != 1
        Case {
            n: 21,
            k: 21,
            alpha: 0.5,
            b_strides: Strides::Other,
            ..Default::default()
        },
    ];

    cases.test_each(|case| {
        let &Case {
            n,
            k,
            alpha,
            beta,
            bias,
            b_strides,
        } = case;

        let mut rng = XorShiftRng::new(1234);
        let a = NdTensor::rand([1, k], &mut rng);
        let mut b = NdTensor::rand([k, n], &mut rng);
        match b_strides {
            Strides::Contiguous => {}
            Strides::Transposed => {
                b.transpose();
            }
            Strides::Other => {
                b = NdTensor::from_data_with_strides([k, n / 2], b.to_vec(), [b.stride(0), 2])
                    .unwrap();
            }
        }

        let mut result = NdTensor::zeros([1, b.size(1)]);
        let bias_array = bias.map(|b| [b]);
        let opts = Some(GemmOpts {
            alpha,
            beta,
            bias: bias_array
                .as_ref()
                .map(|b| BiasVector::Column(b.as_slice())),
            ..Default::default()
        });

        run_gemm(result.view_mut(), a.view(), b.view(), opts.clone(), None).unwrap();

        let mut expected = NdTensor::zeros([1, b.size(1)]);
        reference_gemm(expected.view_mut(), a.view(), b.view(), opts);

        expect_equal(&result, &expected).unwrap();
    })
}

struct BenchCase {
    m: usize,
    n: usize,
    k: usize,
    transpose_b: bool,
}

enum Format {
    Pretty,
    Csv,
}

fn run_gemm_bench<LhsT, RhsT, OutT>(cases: &[BenchCase], format: Format)
where
    GemmExecutor<LhsT, RhsT, OutT>: Default,
    LhsT: GemmInT,
    RhsT: GemmInT,
    OutT: GemmOutT + Default,
    XorShiftRng: RandomSource<LhsT>,
    XorShiftRng: RandomSource<RhsT>,
{
    let gemm = GemmExecutor::<LhsT, RhsT, OutT>::default();
    println!("Testing kernel {}", gemm.kernel_name());

    // Print header
    match format {
        Format::Csv => {
            println!("m,n,k,duration_ms,gflops");
        }
        Format::Pretty => {}
    }

    for &BenchCase {
        m,
        n,
        k,
        transpose_b,
    } in cases
    {
        // Adjust number of iterations based on a target amount of work,
        // so that each case takes roughly the same amount of time, assuming
        // equal efficiency.
        let target_iters = 512;
        let target_ops: u64 = 512 * 512 * 512 * target_iters;
        let iters = target_ops / (m * n * k) as u64;

        // Cap the number of iterations, for cases where the equal-efficiency
        // assumption is untrue.
        let iters = iters.min(target_iters);

        let mut rng = XorShiftRng::new(1234);
        let mut result = NdTensor::<OutT, 2>::zeros([m, n]);
        let a = NdTensor::<LhsT, 2>::rand([m, k], &mut rng);
        let b = if transpose_b {
            let mut b = NdTensor::<RhsT, 2>::rand([n, k], &mut rng);
            b.transpose();
            b
        } else {
            NdTensor::<RhsT, 2>::rand([k, n], &mut rng)
        };

        let start = Instant::now();
        for _i in 0..iters {
            run_gemm(result.view_mut(), a.view(), b.view(), None, Some(&gemm)).unwrap();
        }
        let duration = start.elapsed();

        // Calculate throughput. For comparison, the theoretical maximum
        // GFLOPS for a single core (`RAYON_NUM_THREADS=1`) can be computed
        // as:
        //
        //     frequency * simd_width * fma_throughput * fma_units
        //
        // Where:
        //  - `frequency` is the max frequency in Ghz
        //  - `simd_width` is the # of f32 values in a vector register
        //  - `fma_throughput` is the number of ops/cycle
        //  - `fma_units` is the number of execution units
        //
        // On an Intel Skylake CPU for example, `simd_width` will be
        // 8 (256-bit AVX 2 / 32-bit float), `fma_throughput` is 2,
        //   `fma_units` is 2. For a 3.4Ghz CPU this would give a max
        //   theoretical peak of 3.4 * 8 * 2 * 2 = 108.8 GFLOPS.

        let flops =
            (2 * m as u64 * n as u64 * k as u64 * iters as u64) as f32 / duration.as_secs_f32();
        let gflops = flops / (10f32).powi(9);
        let duration_ms = duration.as_secs_f64() * 1000.0;

        match format {
            Format::Pretty => {
                println!(
                    "m {} n {} k {} iters {}. Duration {:.3}ms ({:.3}ms/iter). GFLOPS {:.1}",
                    m,
                    n,
                    k,
                    iters,
                    duration_ms,
                    duration_ms / iters as f64,
                    gflops,
                );
            }
            Format::Csv => {
                println!("{},{},{},{:.3},{:.1}", m, n, k, duration_ms, gflops);
            }
        }
    }
}

// Run with `cargo test --release bench_gemm_mix -- --nocapture --ignored`
#[test]
#[ignore]
fn bench_gemm_mix() {
    type Case = BenchCase;

    let cases = [
        // Square output matrix
        Case {
            m: 512,
            n: 512,
            k: 512,
            transpose_b: false,
        },
        // Larger square output matrix
        Case {
            m: 1024,
            n: 1024,
            k: 1024,
            transpose_b: false,
        },
        // Wide output matrix
        Case {
            m: 128,
            n: 2048,
            k: 512,
            transpose_b: false,
        },
        // Tall output matrix
        Case {
            m: 2048,
            n: 128,
            k: 512,
            transpose_b: false,
        },
        // Vector-matrix. This is common in transformer decoders for example.
        Case {
            m: 1,
            n: 4096,
            k: 512,
            transpose_b: false,
        },
        Case {
            m: 1,
            n: 4096,
            k: 512,
            transpose_b: true,
        },
    ];

    println!("f32 x f32 -> f32");
    run_gemm_bench::<f32, f32, f32>(&cases, Format::Pretty);

    println!("u8 x i8 -> i32");
    run_gemm_bench::<u8, i8, i32>(&cases, Format::Pretty);
}

#[test]
#[ignore]
fn bench_gemm_size_range() {
    let cases: Vec<_> = (1..512)
        .map(|size| BenchCase {
            m: size,
            n: size,
            k: size,
            transpose_b: false,
        })
        .collect();
    run_gemm_bench::<f32, f32, f32>(&cases, Format::Csv);
}

#[test]
#[ignore]
fn bench_prepack_a() {
    let gemm = GemmExecutor::<f32>::new();
    let mut rng = XorShiftRng::new(1234);
    let m = 1024;
    let n = 1024;
    let iters = 1000;
    let a = NdTensor::rand([m, n], &mut rng);

    run_bench(
        10,
        Some(&format!("m {} n {} iters {}", m, n, iters)),
        || {
            for _i in 0..iters {
                gemm.prepack_a(a.view());
            }
        },
    );
}
