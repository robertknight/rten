use rten_tensor::prelude::*;

use crate::ops::{Input, InputList, OpError, Operator, Output};
use crate::tensor_pool::TensorPool;

/// Specifies a permutation to an operator input.
#[derive(Clone, Debug, PartialEq)]
struct PermuteSpec {
    index: usize,
    perm: Option<Vec<usize>>,
}

impl PermuteSpec {
    /// Apply the permutation to the matching operator input in `inputs`.
    fn apply(&self, inputs: &mut InputList) -> Result<(), OpError> {
        let Some(input) = inputs.get_mut(self.index) else {
            return Err(OpError::MissingInputs);
        };

        match input {
            Input::IntTensor(ref mut t) => {
                if let Some(perm) = self.perm.as_ref() {
                    t.permute(perm);
                } else {
                    t.transpose();
                }
            }
            Input::FloatTensor(ref mut t) => {
                if let Some(perm) = self.perm.as_ref() {
                    t.permute(perm);
                } else {
                    t.transpose();
                }
            }
        }

        Ok(())
    }
}

/// Operator which wraps another operator to permute one or more input views
/// before evaluating the wrapped operator.
#[derive(Debug)]
pub struct FusedTranspose {
    inner: Box<dyn Operator + Send + Sync>,
    perm: PermuteSpec,
    name: String,
}

impl FusedTranspose {
    pub fn wrap(
        op: Box<dyn Operator + Send + Sync>,
        input_index: usize,
        permutation: Option<&[usize]>,
    ) -> FusedTranspose {
        FusedTranspose {
            perm: PermuteSpec {
                index: input_index,
                perm: permutation.map(|slice| slice.to_vec()),
            },
            name: format!("FusedTranspose({})", op.name()),
            inner: op,
        }
    }
}

impl Operator for FusedTranspose {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, pool: &TensorPool, mut inputs: InputList) -> Result<Vec<Output>, OpError> {
        self.perm.apply(&mut inputs)?;
        self.inner.run(pool, inputs)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::prelude::*;
    use rten_tensor::Tensor;

    use super::FusedTranspose;
    use crate::ops::tests::new_pool;
    use crate::ops::{InputList, Operator, Sub};

    #[test]
    fn test_fused_transpose() -> Result<(), Box<dyn Error>> {
        struct Case {
            a: Tensor<i32>,
            b: Tensor<i32>,
            transpose_input: usize,
            expected: Tensor<i32>,
        }

        let cases = [
            Case {
                a: Tensor::from([[1, 2], [3, 4]]),
                b: Tensor::from([[0, 1], [2, 3]]),
                transpose_input: 1,

                // 1 2 - 0 2 = 1 0
                // 3 4   1 3   2 1
                expected: Tensor::from([[1, 0], [2, 1]]),
            },
            Case {
                a: Tensor::from([[1, 2], [3, 4]]),
                b: Tensor::from([[0, 1], [2, 3]]),
                transpose_input: 0,

                // 1 3 - 0 1 = 1 2
                // 2 4   2 3   0 1
                expected: Tensor::from([[1, 2], [0, 1]]),
            },
        ];

        for Case {
            a,
            b,
            transpose_input,
            expected,
        } in cases
        {
            // nb. `Sub` operator chosen because it is a simple non-commutative
            // binary op.
            let sub_op = Sub {};
            let fused_transpose =
                FusedTranspose::wrap(Box::new(sub_op), transpose_input, Some(&[1, 0]));

            let pool = new_pool();
            let mut outputs =
                fused_transpose.run(&pool, InputList::from(&[a.view().into(), b.view().into()]))?;

            let output: Tensor<i32> = outputs.remove(0).try_into().unwrap();

            assert_eq!(output, expected);
        }

        Ok(())
    }
}
