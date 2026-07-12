//! ONNX Runtime contrib elementwise operators.

use rten_tensor::{Layout, NdTensorView, TensorView};

use crate::infer_shapes::{InferShapes, UnaryOp};
use crate::operator::{
    IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
    OutputTypesContext,
};
use crate::ops::add;

use super::{Gelu, GetKernel, unary_op_in_place};

/// Bias Gelu
///
/// This is a fusion of `Add` and (non-approximate) [`Gelu`](super::Gelu). It
/// computes `Gelu(A + B)`, where `B` is a 1D bias tensor broadcast against
/// the last dimension of `A`.
///
/// See <https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.BiasGelu>.
#[derive(Debug)]
pub struct BiasGelu {}

impl Operator for BiasGelu {
    fn name(&self) -> &str {
        "BiasGelu"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(2)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let a: TensorView = inputs.require_as(0)?;
        let b: NdTensorView<_, 1> = inputs.require_as(1)?;

        if a.ndim() == 0 || a.size(a.ndim() - 1) != b.size(0) {
            return Err(OpError::IncompatibleInputShapes(
                "bias length does not match last dimension of input",
            ));
        }

        let sum = add(ctx.pool(), a, b.as_dyn())?;

        let kernel = Gelu { approximate: false }.get_kernel();
        unary_op_in_place(ctx.pool(), sum, &kernel).into_op_result()
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        // The output has the shape of `A + B`. `B` is a 1D bias broadcast
        // against the last axis of `A`, so this is just the shape of `A`.
        Some(&UnaryOp)
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::Tensor;
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::expect_equal;

    use super::super::tests::reference_gelu;
    use super::BiasGelu;
    use crate::operator::{OpError, OperatorExt};

    #[test]
    fn test_bias_gelu() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);
        let input = Tensor::<f32>::rand(&[3, 4], &mut rng);
        let bias = Tensor::<f32>::rand(&[4], &mut rng);

        // Reference is `Gelu(input + bias)` with `bias` broadcast on the last
        // axis.
        let last = input.size(input.ndim() - 1);
        let expected = Tensor::from_data(
            input.shape(),
            input
                .iter()
                .enumerate()
                .map(|(i, x)| reference_gelu(x + bias[[i % last]]))
                .collect::<Vec<_>>(),
        );

        let op = BiasGelu {};
        let result: Tensor = op.run_simple((input.view(), bias.view())).unwrap();

        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_bias_gelu_invalid() {
        let op = BiasGelu {};

        let cases: [(&[usize], &[usize]); 2] = [
            // Bias length does not match last dimension of input.
            (&[3, 4], &[3]),
            // Scalar input has no last dimension to broadcast the bias against.
            (&[], &[4]),
        ];

        for (input_shape, bias_shape) in cases {
            let input = Tensor::<f32>::zeros(input_shape);
            let bias = Tensor::<f32>::zeros(bias_shape);
            let result = op.run_simple::<_, Tensor>((input.view(), bias.view()));
            assert_eq!(
                result,
                Err(OpError::IncompatibleInputShapes(
                    "bias length does not match last dimension of input"
                ))
            );
        }
    }
}
