use std::sync::Arc;

use rten_base::bit_set::BitSet;
use rten_tensor::prelude::*;

use crate::infer_shapes::InferShapes;
use crate::operator::{
    InPlaceInputs, OpError, OpRunContext, Operator, OutputList, OutputTypeList, OutputTypesContext,
};
use crate::ops::map_value_view;
use crate::value::ValueView;

trait TransformInput {
    fn transform(&self, input: &mut ValueView) -> Result<(), OpError>;
}

#[derive(Clone, Debug, PartialEq)]
struct PermuteInput {
    /// New order for axes, or `None` to reverse the axes.
    perm: Option<Vec<usize>>,
}

impl TransformInput for PermuteInput {
    fn transform(&self, input: &mut ValueView) -> Result<(), OpError> {
        map_value_view!(input, tensor, {
            if let Some(perm) = self.perm.as_ref() {
                tensor.permute(perm);
            } else {
                tensor.transpose();
            }
            Ok(())
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Transform {
    Permute(PermuteInput),
}

impl TransformInput for Transform {
    fn transform(&self, input: &mut ValueView) -> Result<(), OpError> {
        match self {
            Self::Permute(spec) => spec.transform(input),
        }
    }
}

#[derive(Debug)]
struct TransformIndex {
    input_index: usize,
    transform: Transform,
}

pub struct TransformInputsBuilder {
    transforms: Vec<TransformIndex>,
}

impl TransformInputsBuilder {
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    pub fn has_transforms(&self) -> bool {
        !self.transforms.is_empty()
    }

    pub fn permute(mut self, input_index: usize, perm: Option<Vec<usize>>) -> Self {
        self.transforms.push(TransformIndex {
            input_index,
            transform: Transform::Permute(PermuteInput { perm }),
        });
        self
    }

    pub fn build(self, op: Arc<dyn Operator + Send + Sync>) -> TransformInputs {
        TransformInputs {
            name: format!("TransformInputs({})", op.name()),
            inner: op,
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

    transforms: Vec<TransformIndex>,
}

impl TransformInputs {
    /// Return the wrapped operator.
    pub(crate) fn inner(&self) -> &(dyn Operator + Send + Sync) {
        self.inner.as_ref()
    }

    /// Return the input permutations as `(input_index, perm)` tuples, where
    /// `perm` is `None` if the transform reverses the axes.
    pub(crate) fn permutations(&self) -> impl Iterator<Item = (usize, Option<&[usize]>)> {
        self.transforms.iter().map(|t| match &t.transform {
            Transform::Permute(p) => (t.input_index, p.perm.as_deref()),
        })
    }

    /// Return true if two inputs have identical transforms applied.
    pub(crate) fn inputs_transformed_identically(&self, index_a: usize, index_b: usize) -> bool {
        let transforms_for = |index: usize| {
            self.transforms
                .iter()
                .filter(move |t| t.input_index == index)
                .map(|t| &t.transform)
        };
        transforms_for(index_a).eq(transforms_for(index_b))
    }
}

impl Operator for TransformInputs {
    fn name(&self) -> &str {
        &self.name
    }

    fn max_inputs(&self) -> Option<usize> {
        self.inner.max_inputs()
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let mut inputs = ctx.inputs().clone();
        for TransformIndex {
            input_index,
            transform,
        } in &self.transforms
        {
            let Some(input) = inputs.get_mut(*input_index) else {
                return Err(OpError::MissingInputs);
            };
            transform.transform(input)?;
        }
        let inner_ctx = OpRunContext::new(ctx.pool(), &inputs, ctx.outputs());
        self.inner.run(&inner_ctx)
    }

    fn in_place_inputs(&self) -> BitSet<u16> {
        // Allow in-place execution unless a transform is applied to one of the
        // in-place inputs.
        let in_place = self.inner.in_place_inputs();
        let transforms_in_place_input = self
            .transforms
            .iter()
            .any(|t| t.input_index < u16::BITS as usize && in_place.get(t.input_index as u32));
        if transforms_in_place_input {
            BitSet::new()
        } else {
            in_place
        }
    }

    fn run_in_place(
        &self,
        in_place: InPlaceInputs,
        ctx: &OpRunContext,
    ) -> Result<OutputList, OpError> {
        let mut inputs = ctx.inputs().clone();
        for TransformIndex {
            input_index,
            transform,
        } in &self.transforms
        {
            let Some(input) = inputs.get_mut(*input_index) else {
                return Err(OpError::MissingInputs);
            };
            transform.transform(input)?;
        }
        let inner_ctx = OpRunContext::new(ctx.pool(), &inputs, ctx.outputs());
        self.inner.run_in_place(in_place, &inner_ctx)
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        self.inner.output_types(_ctx)
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        // `TransformInputs` can reorder inputs, so the inner operator's shape
        // inference does not apply directly.
        None
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rten_tensor::prelude::*;
    use rten_tensor::{Tensor, TensorView};
    use rten_testing::TestCases;

    use super::TransformInputsBuilder;
    use crate::operator::{InputList, Operator, OperatorExt};
    use crate::ops::Sub;

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
            let fused_transpose = TransformInputsBuilder::new()
                .permute(*transpose_input, Some([1, 0].into()))
                .build(Arc::new(sub_op));

            let output: Tensor<i32> = fused_transpose.run_simple((a.view(), b.view())).unwrap();

            assert_eq!(output, *expected);
        })
    }

    #[test]
    fn test_fused_transpose_in_place() {
        #[derive(Clone, Debug)]
        struct Case {
            a: Tensor<i32>,
            b: Tensor<i32>,
            transpose_input: usize,
            // Set to `None` if in-place execution is not supported.
            expected: Option<Tensor<i32>>,
        }

        let cases = [
            // Transform of non-first input.
            Case {
                a: [[1, 2], [3, 4]].into(),
                b: [[0, 1], [2, 3]].into(),
                transpose_input: 1,
                expected: Some([[1, 0], [2, 1]].into()),
            },
            // Transform of first/in-place input.
            Case {
                a: [[1, 2], [3, 4]].into(),
                b: [[0, 1], [2, 3]].into(),
                transpose_input: 0,
                expected: None,
            },
        ];

        cases.test_each_clone(|case| {
            let Case {
                a,
                b,
                transpose_input,
                expected,
            } = case;

            // nb. `Sub` operator chosen because it is a simple non-commutative
            // binary op, which can run in place.
            let sub_op = Sub {};
            let fused_transpose = TransformInputsBuilder::new()
                .permute(transpose_input, Some([1, 0].into()))
                .build(Arc::new(sub_op));
            assert_eq!(
                !fused_transpose.in_place_inputs().is_empty(),
                expected.is_some()
            );

            if let Some(expected) = expected {
                // The in-place input occupies index 0, so `b` is passed at
                // index 1 with a `None` placeholder at index 0.
                let mut inputs = InputList::new();
                inputs.push_optional(None::<TensorView<i32>>);
                inputs.push(b.view());
                let output: Tensor<i32> = fused_transpose.run_simple_in_place(a, inputs).unwrap();
                assert_eq!(output, expected);
            }
        })
    }
}
