use crate::tensor::Tensor;

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
        for c in 0..b_cols {
            let mut product = 0.;
            for k in 0..a_cols {
                unsafe {
                    product +=
                        a_data.get_unchecked(r * a_cols + k) * b_data.get_unchecked(k * b_cols + c);
                }
            }
            out_view[[r, c]] = product;
        }
    }
}

mod tests {
    use crate::gemm::gemm;
    use crate::rng::XorShiftRNG;
    use crate::tensor::{random_tensor, zero_tensor, Tensor};

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
    fn test_gemm() {
        let mut rng = XorShiftRNG::new(1234);

        let a = random_tensor(vec![30, 20], &mut rng);
        let b = random_tensor(vec![20, 10], &mut rng);

        let mut result = zero_tensor::<f32>(vec![30, 10]);

        gemm(&mut result, &a, &b);

        assert_eq!(result.data(), reference_gemm(&a, &b).data());
    }
}
