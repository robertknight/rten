use std::iter::zip;

use crate::check_dims;
use crate::linalg::gemm;
use crate::ops::binary_elementwise::broadcast_shapes;
use crate::ops::{InputList, IntoOpResult, OpError, Operator, Output};
use crate::tensor::Matrix;
use crate::tensor::{Tensor, TensorLayout, TensorView};

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
/// nb. This is named `gemm_op` to avoid confusion with `linalg::gemm`.
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

    let out_shape = &[a.shape()[0], b.shape()[1]][..];
    let mut output = match c {
        Some(c) if beta != 0. => {
            if !c.can_broadcast_to(out_shape) {
                return Err(OpError::IncompatibleInputShapes(
                    "Cannot broadcast c to output shape",
                ));
            }
            let out_data: Vec<_> = c.broadcast_iter(out_shape).collect();
            Tensor::from_data(out_shape, out_data)
        }
        _ => Tensor::zeros(out_shape),
    };

    let out_row_stride = output.stride(0);

    gemm(
        output.data_mut(),
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
            a.view(),
            b.view(),
            c.map(|c| c.view()),
            self.alpha,
            self.beta,
            self.transpose_a,
            self.transpose_b,
        )
        .into_op_result()
    }
}

pub fn matmul(a: TensorView, b: TensorView) -> Result<Tensor, OpError> {
    if a.ndim() < 2 || b.ndim() < 2 {
        return Err(OpError::InvalidValue("Inputs must have >= 2 dimensions"));
    }

    let a_rows = a.shape()[a.ndim() - 2];
    let a_cols = a.shape()[a.ndim() - 1];

    let b_rows = b.shape()[b.ndim() - 2];
    let b_cols = b.shape()[b.ndim() - 1];

    if a_cols != b_rows {
        return Err(OpError::IncompatibleInputShapes(
            "Columns of first matrix does not match rows of second matrix",
        ));
    }

    let a_prefix = &a.shape()[0..a.ndim() - 2];
    let b_prefix = &b.shape()[0..b.ndim() - 2];
    let out_prefix = broadcast_shapes(a_prefix, b_prefix)
        .ok_or(OpError::IncompatibleInputShapes("Cannot broadcast shapes"))?;

    let out_shape = &[out_prefix.as_slice(), &[a_rows, b_cols]].concat();
    let mut output = Tensor::zeros(out_shape);

    let a_broadcast_shape = [out_prefix.as_slice(), &[a_rows, a_cols]].concat();
    let b_broadcast_shape = [out_prefix.as_slice(), &[b_rows, b_cols]].concat();

    let a_offsets = a
        .broadcast_offsets(&a_broadcast_shape)
        .step_by(a_rows * a_cols);
    let b_offsets = b
        .broadcast_offsets(&b_broadcast_shape)
        .step_by(b_rows * b_cols);

    let out_row_stride = output.stride(output.ndim() - 2);
    let out_batches = output.data_mut().chunks_mut(out_row_stride * a_rows);

    for (out_batch, (a_offset, b_offset)) in zip(out_batches, zip(a_offsets, b_offsets)) {
        gemm(
            out_batch,
            out_row_stride,
            Matrix::from_slice(
                &a.data()[a_offset..],
                [a_rows, a_cols],
                Some([a.stride(a.ndim() - 2), a.stride(a.ndim() - 1)]),
            )
            .unwrap(),
            Matrix::from_slice(
                &b.data()[b_offset..],
                [b_rows, b_cols],
                Some([b.stride(b.ndim() - 2), b.stride(b.ndim() - 1)]),
            )
            .unwrap(),
            1., // alpha
            0., // beta
        );
    }

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
        matmul(a.view(), b.view()).into_op_result()
    }
}

#[cfg(test)]
mod tests {
    use crate::linalg::gemm;
    use crate::ops::matmul::{gemm_op, matmul, OpError};
    use crate::rng::XorShiftRng;
    use crate::tensor::{from_data, rand, Tensor, TensorLayout};
    use crate::test_util::expect_equal;

    fn gemm_tensors(c: &mut Tensor, a: &Tensor, b: &Tensor, alpha: f32, beta: f32) {
        let c_row_stride = c.stride(c.ndim() - 2);
        gemm(
            c.data_mut(),
            c_row_stride,
            a.nd_view(),
            b.nd_view(),
            alpha,
            beta,
        )
    }

    #[test]
    fn test_gemm_op() -> Result<(), String> {
        let mut rng = XorShiftRng::new(1234);
        let a = rand(&[3, 10], &mut rng);
        let b = rand(&[10, 8], &mut rng);

        let mut expected = Tensor::zeros(&[3, 8]);
        gemm_tensors(&mut expected, &a, &b, 1., 1.);

        let result = gemm_op(a.view(), b.view(), None, 1.0, 1.0, false, false).unwrap();

        expect_equal(&result, &expected)
    }

    #[test]
    fn test_gemm_op_transposed() -> Result<(), String> {
        let mut rng = XorShiftRng::new(1234);
        let a = rand(&[10, 3], &mut rng);
        let b = rand(&[8, 10], &mut rng);

        let mut a_transposed = a.clone();
        a_transposed.permute(&[1, 0]);
        let mut b_transposed = b.clone();
        b_transposed.permute(&[1, 0]);
        let mut expected = Tensor::zeros(&[3, 8]);
        gemm_tensors(&mut expected, &a_transposed, &b_transposed, 1., 1.);

        let result = gemm_op(a.view(), b.view(), None, 1.0, 1.0, true, true).unwrap();

        expect_equal(&result, &expected)
    }

    #[test]
    fn test_gemm_op_adds_c() -> Result<(), String> {
        let mut rng = XorShiftRng::new(1234);
        let a = rand(&[3, 10], &mut rng);
        let b = rand(&[10, 8], &mut rng);
        let c = rand(&[3, 8], &mut rng);

        let mut expected = c.clone();
        gemm_tensors(&mut expected, &a, &b, 1., 1.);

        let result = gemm_op(a.view(), b.view(), Some(c.view()), 1.0, 1.0, false, false).unwrap();

        expect_equal(&result, &expected)
    }

    #[test]
    fn test_gemm_op_invalid_inputs() {
        let mut rng = XorShiftRng::new(1234);
        let a = rand(&[3, 10], &mut rng);
        let b = rand(&[10, 8], &mut rng);
        let c = rand(&[3, 5], &mut rng);

        let result = gemm_op(a.view(), b.view(), Some(c.view()), 1.0, 1.0, false, false);

        assert_eq!(
            result.err(),
            Some(OpError::IncompatibleInputShapes(
                "Cannot broadcast c to output shape"
            ))
        );
    }

    #[test]
    fn test_matmul() -> Result<(), String> {
        let mut rng = XorShiftRng::new(1234);
        let a = rand(&[3, 10], &mut rng);
        let b = rand(&[10, 8], &mut rng);

        let mut expected = Tensor::zeros(&[3, 8]);
        gemm_tensors(&mut expected, &a, &b, 1., 1.);

        let result = matmul(a.view(), b.view()).unwrap();
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_matmul_broadcast() -> Result<(), String> {
        let mut rng = XorShiftRng::new(1234);
        let mut a = rand(&[3, 10], &mut rng);
        let mut b = rand(&[10, 8], &mut rng);

        let mut expected = Tensor::zeros(&[3, 8]);
        gemm_tensors(&mut expected, &a, &b, 1., 1.);
        expected.reshape(&[1, 1, 3, 8]);

        // LHS input has excess 1 dims
        a.reshape(&[1, 1, 3, 10]);
        let result = matmul(a.view(), b.view()).unwrap();
        expect_equal(&result, &expected)?;

        // RHS input has excess 1 dims
        a.reshape(&[3, 10]);
        b.reshape(&[1, 1, 10, 8]);
        let result = matmul(a.view(), b.view()).unwrap();
        expect_equal(&result, &expected)?;

        // RHS input requires broadcasting
        let broadcast_a_shape = &[1, 4, 3, 10][..];
        let broadcast_expected_shape = &[1, 4, 3, 8][..];
        let broadcast_a = from_data(
            broadcast_a_shape.into(),
            a.broadcast_iter(broadcast_a_shape).collect(),
        );
        let broadcast_expected = from_data(
            broadcast_expected_shape.into(),
            expected.broadcast_iter(broadcast_expected_shape).collect(),
        );
        let result = matmul(broadcast_a.view(), b.view()).unwrap();
        expect_equal(&result, &broadcast_expected)?;

        // LHS input requires broadcasting
        let broadcast_b_shape = &[1, 3, 10, 8][..];
        let broadcast_expected_shape = &[1, 3, 3, 8][..];
        let broadcast_b = from_data(
            broadcast_b_shape.into(),
            b.broadcast_iter(broadcast_b_shape).collect(),
        );
        let expected = from_data(
            broadcast_expected_shape.into(),
            expected.broadcast_iter(broadcast_expected_shape).collect(),
        );
        let result = matmul(a.view(), broadcast_b.view()).unwrap();
        expect_equal(&result, &expected)?;

        Ok(())
    }
}
