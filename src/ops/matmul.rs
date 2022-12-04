use std::iter::zip;

use crate::linalg::{gemm_slice, Matrix};
use crate::ops::{
    choose_broadcast_shape, get_input_as_float, get_optional_input_as_float, Input, IntoOpResult,
    OpError, Operator, Output,
};
use crate::tensor::{from_data, zeros, Tensor};

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
    a: &Tensor,
    b: &Tensor,
    c: Option<&Tensor>,
    alpha: f32,
    beta: f32,
    transpose_a: bool,
    transpose_b: bool,
) -> Result<Tensor, OpError> {
    if alpha != 1.0 {
        return Err(OpError::UnsupportedValue(
            "Gemm only supports `alpha` value of 1.0",
        ));
    }
    if beta != 0.0 && beta != 1.0 {
        return Err(OpError::UnsupportedValue(
            "Gemm only supports `beta` values of 0.0 and 1.0",
        ));
    }

    let (a_rows, a_cols, a_row_stride, a_col_stride) = if transpose_a {
        (a.shape()[1], a.shape()[0], a.stride(1), a.stride(0))
    } else {
        (a.shape()[0], a.shape()[1], a.stride(0), a.stride(1))
    };
    let (b_rows, b_cols, b_row_stride, b_col_stride) = if transpose_b {
        (b.shape()[1], b.shape()[0], b.stride(1), b.stride(0))
    } else {
        (b.shape()[0], b.shape()[1], b.stride(0), b.stride(1))
    };

    let out_shape = &[a_rows, b_cols][..];
    let mut output = if c.is_some() && beta == 1.0 {
        let out_data = c.unwrap().broadcast_elements(out_shape).collect();
        from_data(out_shape.into(), out_data)
    } else {
        zeros(out_shape)
    };

    let out_row_stride = output.stride(0);

    gemm_slice(
        output.data_mut(),
        out_row_stride,
        Matrix {
            data: a.data(),
            rows: a_rows,
            cols: a_cols,
            row_stride: a_row_stride,
            col_stride: a_col_stride,
        },
        Matrix {
            data: b.data(),
            rows: b_rows,
            cols: b_cols,
            row_stride: b_row_stride,
            col_stride: b_col_stride,
        },
    );

    Ok(output)
}

impl Operator for Gemm {
    fn name(&self) -> &str {
        "Gemm"
    }

    fn run(&self, inputs: &[Input]) -> Result<Vec<Output>, OpError> {
        let a = get_input_as_float(inputs, 0)?;
        let b = get_input_as_float(inputs, 1)?;
        let c = get_optional_input_as_float(inputs, 2)?;
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

pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, OpError> {
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
    let out_prefix = choose_broadcast_shape(a_prefix, b_prefix);

    let out_shape = &[out_prefix, &[a_rows, b_cols]].concat();
    let mut output = zeros(out_shape);

    let a_broadcast_shape = [out_prefix, &[a_rows, a_cols]].concat();
    let b_broadcast_shape = [out_prefix, &[b_rows, b_cols]].concat();

    // TODO: Optimize `Iterator::nth` for `Offsets` to optimize the `step_by`
    // iterator.
    let a_offsets = a
        .broadcast_offsets(&a_broadcast_shape)
        .step_by(a_rows * a_cols);
    let b_offsets = b
        .broadcast_offsets(&b_broadcast_shape)
        .step_by(b_rows * b_cols);

    let out_row_stride = output.stride(output.ndim() - 2);
    let mut out_offset = 0;

    for (a_offset, b_offset) in zip(a_offsets, b_offsets) {
        gemm_slice(
            &mut output.data_mut()[out_offset..],
            out_row_stride,
            Matrix {
                data: &a.data()[a_offset..],
                rows: a_rows,
                cols: a_cols,
                row_stride: a.stride(a.ndim() - 2),
                col_stride: a.stride(a.ndim() - 1),
            },
            Matrix {
                data: &b.data()[b_offset..],
                rows: b_rows,
                cols: b_cols,
                row_stride: b.stride(b.ndim() - 2),
                col_stride: b.stride(b.ndim() - 1),
            },
        );
        out_offset += out_row_stride * a_rows;
    }

    Ok(output)
}

#[derive(Debug)]
pub struct MatMul {}

impl Operator for MatMul {
    fn name(&self) -> &str {
        "MatMul"
    }

    fn run(&self, inputs: &[Input]) -> Result<Vec<Output>, OpError> {
        let a = get_input_as_float(inputs, 0)?;
        let b = get_input_as_float(inputs, 1)?;
        matmul(a, b).into_op_result()
    }
}

#[cfg(test)]
mod tests {
    use crate::linalg::gemm;
    use crate::ops::matmul::{gemm_op, matmul};
    use crate::rng::XorShiftRNG;
    use crate::tensor::{from_data, rand, zeros};
    use crate::test_util::expect_equal;

    #[test]
    fn test_gemm_op() -> Result<(), String> {
        let mut rng = XorShiftRNG::new(1234);
        let a = rand(&[3, 10], &mut rng);
        let b = rand(&[10, 8], &mut rng);

        let mut expected = zeros(&[3, 8]);
        gemm(&mut expected, &a, &b);

        let result = gemm_op(&a, &b, None, 1.0, 1.0, false, false).unwrap();

        expect_equal(&result, &expected)
    }

    #[test]
    fn test_gemm_op_transposed() -> Result<(), String> {
        let mut rng = XorShiftRNG::new(1234);
        let a = rand(&[10, 3], &mut rng);
        let b = rand(&[8, 10], &mut rng);

        let mut a_transposed = a.clone();
        a_transposed.permute(&[1, 0]);
        let mut b_transposed = b.clone();
        b_transposed.permute(&[1, 0]);
        let mut expected = zeros(&[3, 8]);
        gemm(&mut expected, &a_transposed, &b_transposed);

        let result = gemm_op(&a, &b, None, 1.0, 1.0, true, true).unwrap();

        expect_equal(&result, &expected)
    }

    #[test]
    fn test_gemm_op_adds_c() -> Result<(), String> {
        let mut rng = XorShiftRNG::new(1234);
        let a = rand(&[3, 10], &mut rng);
        let b = rand(&[10, 8], &mut rng);
        let c = rand(&[3, 8], &mut rng);

        let mut expected = c.clone();
        gemm(&mut expected, &a, &b);

        let result = gemm_op(&a, &b, Some(&c), 1.0, 1.0, false, false).unwrap();

        expect_equal(&result, &expected)
    }

    #[test]
    fn test_matmul() -> Result<(), String> {
        let mut rng = XorShiftRNG::new(1234);
        let a = rand(&[3, 10], &mut rng);
        let b = rand(&[10, 8], &mut rng);

        let mut expected = zeros(&[3, 8]);
        gemm(&mut expected, &a, &b);

        let result = matmul(&a, &b).unwrap();
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_matmul_broadcast() -> Result<(), String> {
        let mut rng = XorShiftRNG::new(1234);
        let mut a = rand(&[3, 10], &mut rng);
        let mut b = rand(&[10, 8], &mut rng);

        let mut expected = zeros(&[3, 8]);
        gemm(&mut expected, &a, &b);
        expected.reshape(&[1, 1, 3, 8]);

        // LHS input has excess 1 dims
        a.reshape(&[1, 1, 3, 10]);
        let result = matmul(&a, &b).unwrap();
        expect_equal(&result, &expected)?;

        // RHS input has excess 1 dims
        a.reshape(&[3, 10]);
        b.reshape(&[1, 1, 10, 8]);
        let result = matmul(&a, &b).unwrap();
        expect_equal(&result, &expected)?;

        // RHS input requires broadcasting
        let broadcast_a_shape = &[1, 4, 3, 10][..];
        let broadcast_expected_shape = &[1, 4, 3, 8][..];
        let broadcast_a = from_data(
            broadcast_a_shape.into(),
            a.broadcast_elements(broadcast_a_shape).collect(),
        );
        let broadcast_expected = from_data(
            broadcast_expected_shape.into(),
            expected
                .broadcast_elements(broadcast_expected_shape)
                .collect(),
        );
        let result = matmul(&broadcast_a, &b).unwrap();
        expect_equal(&result, &broadcast_expected)?;

        // LHS input requires broadcasting
        let broadcast_b_shape = &[1, 3, 10, 8][..];
        let broadcast_expected_shape = &[1, 3, 3, 8][..];
        let broadcast_b = from_data(
            broadcast_b_shape.into(),
            b.broadcast_elements(broadcast_b_shape).collect(),
        );
        let expected = from_data(
            broadcast_expected_shape.into(),
            expected
                .broadcast_elements(broadcast_expected_shape)
                .collect(),
        );
        let result = matmul(&a, &broadcast_b).unwrap();
        expect_equal(&result, &expected)?;

        Ok(())
    }
}
