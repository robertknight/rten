use std::sync::Arc;

use rten_tensor::prelude::*;

use crate::ops::{map_input, Input, OpError, OpRunContext, Operator, OutputList};

trait TransformInput {
    fn transform(&self, input: &mut Input) -> Result<(), OpError>;
}

#[derive(Clone, Debug, PartialEq)]
struct PermuteInput {
    /// New order for axes, or `None` to reverse the axes.
    perm: Option<Vec<usize>>,
}

impl TransformInput for PermuteInput {
    fn transform(&self, input: &mut Input) -> Result<(), OpError> {
        map_input!(input, tensor, {
            if let Some(perm) = self.perm.as_ref() {
                tensor.permute(perm);
            } else {
                tensor.transpose();
            }
        });
        Ok(())
    }
}

#[derive(Clone, Debug)]
enum Transform {
    Permute(PermuteInput),
}

impl TransformInput for Transform {
    fn transform(&self, input: &mut Input) -> Result<(), OpError> {
        match self {
            Self::Permute(spec) => spec.transform(input),
        }
    }
}

pub struct TransformInputsBuilder {
    op: Arc<dyn Operator + Send + Sync>,
    transforms: Vec<(usize, Transform)>,
}

impl TransformInputsBuilder {
    pub fn new(op: Arc<dyn Operator + Send + Sync>) -> Self {
        Self {
            op,
            transforms: Vec::new(),
        }
    }

    pub fn permute(mut self, input_index: usize, perm: Option<Vec<usize>>) -> Self {
        self.transforms
            .push((input_index, Transform::Permute(PermuteInput { perm })));
        self
    }

    pub fn build(self) -> TransformInputs {
        TransformInputs {
            name: format!("TransformInputs({})", self.op.name()),
            inner: self.op,
            transforms: self.transforms,
        }
    }
}

/// Operator which wraps another operator to (cheaply) transform one or more
/// inputs before evaluating the wrapped operator.
#[derive(Debug)]
pub struct TransformInputs {
    name: String,

    inner: Arc<dyn Operator + Send + Sync>,

    /// (input_index, transform) list of transforms to apply.
    transforms: Vec<(usize, Transform)>,
}

impl Operator for TransformInputs {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let mut inputs = ctx.inputs().clone();
        for (index, transform) in &self.transforms {
            let Some(input) = inputs.get_mut(*index) else {
                return Err(OpError::MissingInputs);
            };
            transform.transform(input)?;
        }
        let inner_ctx = OpRunContext::new(ctx.pool(), &inputs);
        self.inner.run(&inner_ctx)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rten_tensor::prelude::*;
    use rten_tensor::Tensor;
    use rten_testing::TestCases;

    use super::TransformInputsBuilder;
    use crate::ops::{OperatorExt, Sub};

    #[test]
    fn test_fused_transpose() {
        #[derive(Debug)]
        struct Case {
            a: Tensor<i32>,
            b: Tensor<i32>,
            transpose_input: usize,
            expected: Tensor<i32>,
        }

        let cases = [
            Case {
                a: [[1, 2], [3, 4]].into(),
                b: [[0, 1], [2, 3]].into(),
                transpose_input: 1,

                // 1 2 - 0 2 = 1 0
                // 3 4   1 3   2 1
                expected: [[1, 0], [2, 1]].into(),
            },
            Case {
                a: [[1, 2], [3, 4]].into(),
                b: [[0, 1], [2, 3]].into(),
                transpose_input: 0,

                // 1 3 - 0 1 = 1 2
                // 2 4   2 3   0 1
                expected: [[1, 2], [0, 1]].into(),
            },
        ];

        cases.test_each(|case| {
            let Case {
                a,
                b,
                transpose_input,
                expected,
            } = case;

            // nb. `Sub` operator chosen because it is a simple non-commutative
            // binary op.
            let sub_op = Sub {};
            let fused_transpose = TransformInputsBuilder::new(Arc::new(sub_op))
                .permute(*transpose_input, Some([1, 0].into()))
                .build();

            let output: Tensor<i32> = fused_transpose.run_simple((a.view(), b.view())).unwrap();

            assert_eq!(output, *expected);
        })
    }
}
