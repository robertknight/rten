//! ONNX Runtime contrib elementwise operators.

use rten_base::bit_set::BitSet;
use rten_tensor::{Layout, NdTensorView, TensorView};

use crate::infer_shapes::{InferShapes, UnaryOp};
use crate::operator::{
    InPlaceInputs, IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType,
    OutputTypeList, OutputTypesContext,
};
use crate::ops::add;

use super::{Gelu, GetKernel, Swish, unary_op, unary_op_in_place};

/// Compute `Gelu(A + B)`, where `A` is the operator's first input and `B` is
/// a 1D bias input broadcast against the last dimension of `A`.
fn bias_gelu(ctx: &OpRunContext, approximate: bool) -> Result<OutputList, OpError> {
    let inputs = ctx.inputs();
    let a: TensorView = inputs.require_as(0)?;
    let b: NdTensorView<_, 1> = inputs.require_as(1)?;

    if a.ndim() == 0 || a.size(a.ndim() - 1) != b.size(0) {
        return Err(OpError::incompatible_input_shapes(
            "bias length does not match last dimension of input",
        ));
    }

    let sum = add(ctx.pool(), a, b.as_dyn())?;

    let gelu = Gelu { approximate };
    let kernel = gelu.get_kernel();
    unary_op_in_place(ctx.pool(), sum, &kernel).into_op_result()
}

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
        bias_gelu(ctx, false /* approximate */)
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(&UnaryOp)
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }
}

/// Fast Gelu
///
/// This computes the tanh approximation of [`Gelu`](super::Gelu) over `X + B`,
/// where `B` is an optional 1D bias tensor broadcast against the last
/// dimension of `X`. It is the same as [`BiasGelu`] except that the
/// approximate variant of Gelu is used and the bias is optional.
///
/// See <https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.FastGelu>.
#[derive(Debug)]
pub struct FastGelu {}

impl Operator for FastGelu {
    fn name(&self) -> &str {
        "FastGelu"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(2)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let bias: Option<NdTensorView<f32, 1>> = ctx.inputs().get_as(1)?;
        if bias.is_some() {
            bias_gelu(ctx, true /* approximate */)
        } else {
            let input: TensorView = ctx.inputs().require_as(0)?;
            let kernel = Gelu { approximate: true }.get_kernel();
            unary_op(ctx.pool(), input, &kernel).into_op_result()
        }
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(&UnaryOp)
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }
}

/// Gelu operator in the `com.microsoft` domain.
///
/// This is an alias for the standard non-approximate [`Gelu`](super::Gelu).
///
/// See <https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Gelu>.
#[derive(Debug)]
pub struct GeluMicrosoft {}

impl Operator for GeluMicrosoft {
    fn name(&self) -> &str {
        "com.microsoft.Gelu"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(1)
    }

    fn in_place_inputs(&self) -> BitSet<u16> {
        BitSet::from_indices([0])
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        Gelu { approximate: false }.run(ctx)
    }

    fn run_in_place(
        &self,
        in_place: InPlaceInputs,
        ctx: &OpRunContext,
    ) -> Result<OutputList, OpError> {
        Gelu { approximate: false }.run_in_place(in_place, ctx)
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(&UnaryOp)
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }
}

/// Quick Gelu
///
/// This computes `x * sigmoid(alpha * x)`, the sigmoid approximation of Gelu
/// used by CLIP. It is the same function as [`Swish`](super::Swish), which
/// differs only in defaulting `alpha` to 1.0 rather than 1.702.
///
/// See <https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QuickGelu>.
#[derive(Debug)]
pub struct QuickGelu {
    pub alpha: f32,
}

impl Operator for QuickGelu {
    fn name(&self) -> &str {
        "QuickGelu"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(1)
    }

    fn in_place_inputs(&self) -> BitSet<u16> {
        BitSet::from_indices([0])
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        Swish { alpha: self.alpha }.run(ctx)
    }

    fn run_in_place(
        &self,
        in_place: InPlaceInputs,
        ctx: &OpRunContext,
    ) -> Result<OutputList, OpError> {
        Swish { alpha: self.alpha }.run_in_place(in_place, ctx)
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
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
    use rten_testing::TestCases;

    use super::super::tests::{reference_approx_gelu, reference_gelu};
    use super::{BiasGelu, FastGelu, GeluMicrosoft, QuickGelu};
    use crate::operator::{OpError, OperatorExt};

    fn reference_bias_gelu(
        input: &Tensor<f32>,
        bias: &Tensor<f32>,
        gelu: impl Fn(f32) -> f32,
    ) -> Tensor<f32> {
        let last = input.size(input.ndim() - 1);
        Tensor::from_data(
            input.shape(),
            input
                .iter()
                .enumerate()
                .map(|(i, x)| gelu(x + bias[[i % last]]))
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn test_bias_gelu() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);
        let input = Tensor::<f32>::rand(&[3, 4], &mut rng);
        let bias = Tensor::<f32>::rand(&[4], &mut rng);

        let expected = reference_bias_gelu(&input, &bias, reference_gelu);

        let op = BiasGelu {};
        let result: Tensor = op.run_simple((input.view(), bias.view())).unwrap();

        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_fast_gelu() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);
        let input = Tensor::<f32>::rand(&[3, 4], &mut rng);
        let bias = Tensor::<f32>::rand(&[4], &mut rng);

        let expected = reference_bias_gelu(&input, &bias, reference_approx_gelu);

        let op = FastGelu {};
        let result: Tensor = op.run_simple((input.view(), bias.view())).unwrap();

        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_fast_gelu_no_bias() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);
        let input = Tensor::<f32>::rand(&[3, 4], &mut rng);

        let expected = input.map(|&x| reference_approx_gelu(x));

        let op = FastGelu {};
        let result: Tensor = op.run_simple(input.view()).unwrap();

        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_gelu_microsoft() -> Result<(), Box<dyn Error>> {
        let mut rng = XorShiftRng::new(1234);
        let input = Tensor::<f32>::rand(&[3, 4], &mut rng);

        let expected = input.map(|&x| reference_gelu(x));

        let op = GeluMicrosoft {};
        let result: Tensor = op.run_simple(input.view()).unwrap();

        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_quick_gelu() -> Result<(), Box<dyn Error>> {
        #[derive(Debug)]
        struct Case {
            alpha: f32,
        }

        let cases = [Case { alpha: 1.702 }, Case { alpha: 1.0 }];

        let mut rng = XorShiftRng::new(1234);
        let input = Tensor::<f32>::rand(&[3, 4], &mut rng);

        cases.test_each(|Case { alpha }| {
            let alpha = *alpha;
            let expected = input.map(|&x| x / (1. + (-alpha * x).exp()));

            let op = QuickGelu { alpha };
            let result: Tensor = op.run_simple(input.view()).unwrap();

            expect_equal(&result, &expected).unwrap();
        });

        Ok(())
    }

    #[test]
    fn test_bias_gelu_invalid() {
        let cases: [(&[usize], &[usize]); 2] = [
            // Bias length does not match last dimension of input.
            (&[3, 4], &[3]),
            // Scalar input has no last dimension to broadcast the bias against.
            (&[], &[4]),
        ];

        for op in [&BiasGelu {} as &dyn crate::operator::Operator, &FastGelu {}] {
            for (input_shape, bias_shape) in cases {
                let input = Tensor::<f32>::zeros(input_shape);
                let bias = Tensor::<f32>::zeros(bias_shape);
                let result = op.run_simple::<_, Tensor>((input.view(), bias.view()));
                assert_eq!(
                    result,
                    Err(OpError::incompatible_input_shapes(
                        "bias length does not match last dimension of input"
                    ))
                );
            }
        }
    }
}
