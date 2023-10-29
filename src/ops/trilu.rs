use wasnn_tensor::prelude::*;
use wasnn_tensor::{Tensor, TensorView};

use crate::ops::{Input, InputList, IntoOpResult, OpError, Operator, Output};

pub fn trilu<T: Copy + Default>(
    input: TensorView<T>,
    k: i32,
    upper: bool,
) -> Result<Tensor<T>, OpError> {
    if input.ndim() < 2 {
        return Err(OpError::InvalidValue("Input must have >= 2 dims"));
    }

    let mut output = Tensor::zeros(input.shape());

    for (mut out_mat, in_mat) in output.inner_iter_mut::<2>().zip(input.inner_iter::<2>()) {
        let [rows, cols] = out_mat.shape();

        for y in 0..rows {
            for x in 0..cols {
                let delta = y as i32 + k - x as i32;
                let copy = if upper { delta <= 0 } else { delta >= 0 };
                if copy {
                    out_mat[[y, x]] = in_mat[[y, x]];
                }
            }
        }
    }

    Ok(output)
}

#[derive(Debug)]
pub struct Trilu {
    pub upper: bool,
}

impl Operator for Trilu {
    fn name(&self) -> &str {
        "Trilu"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let k = inputs.get_as_scalar(1)?.unwrap_or(0);

        match input {
            Input::FloatTensor(input) => trilu(input.view(), k, self.upper).into_op_result(),
            Input::IntTensor(input) => trilu(input.view(), k, self.upper).into_op_result(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::{trilu, OpError};
    use wasnn_tensor::{tensor, Tensor};

    #[test]
    fn test_trilu() {
        struct Case {
            input: Tensor<i32>,
            expected: Tensor<i32>,
            upper: bool,
            k: i32,
        }

        let in_3x3 = Tensor::arange(1, 10, None).into_reshaped(&[3, 3]);

        let cases = [
            // k = 0, upper = true
            Case {
                input: in_3x3.clone(),
                expected: Tensor::from([
                    [1, 2, 3], //
                    [0, 5, 6], //
                    [0, 0, 9],
                ]),
                k: 0,
                upper: true,
            },
            // +ve k, upper = true
            Case {
                input: in_3x3.clone(),
                expected: Tensor::from([
                    [0, 2, 3], //
                    [0, 0, 6], //
                    [0, 0, 0],
                ]),
                k: 1,
                upper: true,
            },
            // -ve k, upper = true
            Case {
                input: in_3x3.clone(),
                expected: Tensor::from([
                    [1, 2, 3], //
                    [4, 5, 6], //
                    [0, 8, 9],
                ]),
                k: -1,
                upper: true,
            },
            // k = 0, upper = false
            Case {
                input: in_3x3.clone(),
                expected: Tensor::from([
                    [1, 0, 0], //
                    [4, 5, 0], //
                    [7, 8, 9],
                ]),
                k: 0,
                upper: false,
            },
            // +ve k, upper = false
            Case {
                input: in_3x3.clone(),
                expected: Tensor::from([
                    [1, 2, 0], //
                    [4, 5, 6], //
                    [7, 8, 9],
                ]),
                k: 1,
                upper: false,
            },
            // -ve k, upper = false
            Case {
                input: in_3x3.clone(),
                expected: Tensor::from([
                    [0, 0, 0], //
                    [4, 0, 0], //
                    [7, 8, 0],
                ]),
                k: -1,
                upper: false,
            },
            // Batch of matrices
            Case {
                input: Tensor::from([
                    [
                        [1, 2, 3], //
                        [4, 5, 6], //
                        [7, 8, 9],
                    ],
                    [
                        [9, 8, 7], //
                        [6, 5, 4], //
                        [3, 2, 1],
                    ],
                ]),
                expected: Tensor::from([
                    [
                        [1, 2, 3], //
                        [0, 5, 6], //
                        [0, 0, 9],
                    ],
                    [
                        [9, 8, 7], //
                        [0, 5, 4], //
                        [0, 0, 1],
                    ],
                ]),
                k: 0,
                upper: true,
            },
            // Non-square (wide) matrix
            Case {
                input: Tensor::arange(1, 16, None).into_reshaped(&[3, 5]),
                expected: Tensor::from([
                    [1, 2, 3, 4, 5],  //
                    [0, 7, 8, 9, 10], //
                    [0, 0, 13, 14, 15],
                ]),
                k: 0,
                upper: true,
            },
            // Non-square (tall) matrix
            Case {
                input: Tensor::arange(1, 16, None).into_reshaped(&[5, 3]),
                expected: Tensor::from([
                    [1, 2, 3], //
                    [0, 5, 6], //
                    [0, 0, 9], //
                    [0, 0, 0], //
                    [0, 0, 0],
                ]),
                k: 0,
                upper: true,
            },
        ];

        for Case {
            input,
            expected,
            upper,
            k,
        } in cases
        {
            let result = trilu(input.view(), k, upper).unwrap();
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_trilu_invalid() {
        let input = tensor!([1]);
        let result = trilu(input.view(), 0, true /* upper */);
        assert_eq!(
            result,
            Err(OpError::InvalidValue("Input must have >= 2 dims"))
        );
    }
}
