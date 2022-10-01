use crate::tensor::Tensor;

/// Compute dot product of `depth` elements of `a` and `b`, stepping through
/// each array by `a_stride` and `b_stride` respectively. `N` specifies a
/// loop unrolling factor.
fn dot<const N: usize>(
    a: &[f32],
    b: &[f32],
    a_stride: usize,
    b_stride: usize,
    depth: usize,
) -> f32 {
    let n_blocks = depth / N;

    let mut result = 0.0;
    let mut accum = [0.0; N];

    for block in 0..n_blocks {
        let start_i = block * N;

        for i in 0..N {
            let k = start_i + i;
            unsafe {
                accum[i] = a.get_unchecked(a_stride * k) * b.get_unchecked(b_stride * k);
            }
        }
        result += accum.iter().fold(0.0, |sum, x| sum + x);
    }

    for k in (n_blocks * N)..depth {
        unsafe {
            result += a.get_unchecked(a_stride * k) * b.get_unchecked(b_stride * k);
        }
    }

    result
}

/// Perform a general matrix multiplication ("GEMM") of `a` and `b` and store
/// the result in `output`.
pub fn gemm(output: &mut Tensor, a: &Tensor, b: &Tensor) {
    let [a_rows, a_cols] = a.dims();
    let [b_rows, b_cols] = b.dims();

    if a_cols != b_rows {
        panic!(
            "Columns of first input {} must match rows {} of second input",
            a_cols, b_rows
        );
    }

    let mut out_view = output.unchecked_view_mut([0, 0]);
    let a_data = a.data();
    let b_data = b.data();

    for r in 0..a_rows {
        let a_row = &a_data[r * a_cols..];
        for c in 0..b_cols {
            let b_col = &b_data[c..];
            out_view[[r, c]] = dot::<4>(a_row, b_col, 1 /* a_stride */, b_cols, a_cols);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::gemm::gemm;
    use crate::rng::XorShiftRNG;
    use crate::tensor::{random_tensor, zero_tensor, Tensor};
    use crate::test_util::expect_equal;

    fn reference_gemm(a: &Tensor, b: &Tensor) -> Tensor {
        let [a_rows, a_cols] = a.dims();
        let [b_rows, b_cols] = b.dims();
        let mut output = zero_tensor(vec![a_rows, b_cols]);

        for r in 0..a_rows {
            for c in 0..b_cols {
                for k in 0..a_cols {
                    output[[r, c]] += a[[r, k]] * b[[k, c]];
                }
            }
        }

        output
    }

    #[test]
    fn test_gemm() -> Result<(), String> {
        let mut rng = XorShiftRNG::new(1234);

        let a = random_tensor(vec![30, 20], &mut rng);
        let b = random_tensor(vec![20, 10], &mut rng);

        let mut result = zero_tensor::<f32>(vec![30, 10]);

        gemm(&mut result, &a, &b);

        let expected = reference_gemm(&a, &b);
        expect_equal(&result, &expected)
    }
}
