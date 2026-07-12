//! ONNX Runtime contrib normalization operators.

use rten_shape_inference::ops as shape_ops;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensorView, Tensor, TensorView};

use crate::buffer_pool::AutoReturn;
use crate::infer_shapes::{InferShapes, UnaryOp};
use crate::operator::{
    IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
    OutputTypesContext,
};
use crate::ops::binary_elementwise::{add, add_in_place};

use super::{MeanNormalize, layer_normalization_impl, rms_normalization};

/// Fusion of layer normalization and addition.
///
/// This computes `norm(input + skip + bias) * gamma + beta`.
/// `mean_normalize` controls whether layer normalization is used or RMS
/// normalization.
fn skip_layer_normalization(
    ctx: &OpRunContext,
    input: TensorView,
    skip: TensorView,
    gamma: NdTensorView<f32, 1>,
    beta: Option<NdTensorView<f32, 1>>,
    bias: Option<NdTensorView<f32, 1>>,
    epsilon: f32,
    mean_normalize: MeanNormalize,
) -> Result<OutputList, OpError> {
    if !matches!(input.ndim(), 2 | 3) {
        return Err(OpError::InvalidValue("input must be 2 or 3 dimensioned"));
    }

    // `skip` may either match `input` exactly or broadcast over the batch
    // dimension (a batch size of 1, or no batch dimension at all). Its
    // trailing dimensions must match those of `input`. This matches ONNX
    // Runtime, which indexes `skip` modulo its own size.
    if !matches!(skip.ndim(), 2 | 3) {
        return Err(OpError::InvalidValue("skip must be 2 or 3 dimensioned"));
    }
    if skip.shape()[skip.ndim() - 2..] != input.shape()[input.ndim() - 2..]
        || !skip.can_broadcast_to(input.shape())
    {
        return Err(OpError::IncompatibleInputShapes(
            "skip must broadcast to input over the batch dimension",
        ));
    }

    // TODO: Fuse the addition of `skip` and `bias` with normalization.
    let mut x_plus_skip = add(ctx.pool(), input, skip)?.auto_return(ctx.pool());
    if let Some(bias) = bias {
        add_in_place(x_plus_skip.view_mut(), bias.as_dyn());
    }

    let output = layer_normalization_impl(
        ctx.pool(),
        x_plus_skip.view(),
        gamma.as_dyn(),
        beta.map(|b| b.as_dyn()),
        -1,
        Some(epsilon),
        mean_normalize,
    )?;

    let mut outputs: OutputList = [output.into()].into();
    if ctx.outputs().get(3) {
        // `mean` and `inv_std_var` are used for training. Here we push
        // dummy values.
        outputs.push(Tensor::from(0.).into()); // mean
        outputs.push(Tensor::from(0.).into()); // inv_std_var
        outputs.push(x_plus_skip.take().into());
    }

    Ok(outputs)
}

/// Simplified Layer Normalization
///
/// This is a non-standard ONNX operator for layer normalization which is
/// equivalent to the later stabilised RMSNormalization. See
/// [onnx/onnx#6582](https://github.com/onnx/onnx/issues/6582) for more
/// details.
#[derive(Debug)]
pub struct SimplifiedLayerNormalization {
    pub axis: isize,
    pub epsilon: Option<f32>,
}

impl Operator for SimplifiedLayerNormalization {
    fn name(&self) -> &str {
        "SimplifiedLayerNormalization"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(2)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input = inputs.require_as(0)?;
        let scale = inputs.require_as(1)?;

        rms_normalization(ctx.pool(), input, scale, self.axis, self.epsilon).into_op_result()
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(&UnaryOp)
    }
}

/// Skip Layer Normalization
///
/// This is a fusion of `Add` and `LayerNormalization`.
///
/// See <https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.SkipLayerNormalization>.
#[derive(Debug)]
pub struct SkipLayerNormalization {
    pub epsilon: f32,
}

impl Operator for SkipLayerNormalization {
    fn name(&self) -> &str {
        "SkipLayerNormalization"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(5)
    }

    fn max_outputs(&self) -> Option<usize> {
        Some(4)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input: TensorView<_> = inputs.require_as(0)?;
        let skip: TensorView<_> = inputs.require_as(1)?;

        // Scale (gamma) and bias (beta) applied after normalization.
        let gamma: NdTensorView<_, 1> = inputs.require_as(2)?;
        let beta: Option<NdTensorView<_, 1>> = inputs.get_as(3)?;

        // Bias added to `input + skip` before normalization.
        let bias: Option<NdTensorView<_, 1>> = inputs.get_as(4)?;

        skip_layer_normalization(
            ctx,
            input,
            skip,
            gamma,
            beta,
            bias,
            self.epsilon,
            MeanNormalize::Dynamic,
        )
    }

    fn output_types(&self, ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        let mut types = OutputTypeList::from([OutputType::CopyFromInput(0)]);
        if ctx.num_outputs > 1 {
            types.push(OutputType::CopyFromInput(0));
            types.push(OutputType::CopyFromInput(0));
            types.push(OutputType::CopyFromInput(0));
        }
        Some(types)
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(&shape_ops::SkipLayerNormalization)
    }
}

/// Skip Simplified Layer Normalization
///
/// This is a fusion of `Add` and `RMSNormalization` (also known as
/// SimplifiedLayerNormalization in Microsoft's contrib ops).
///
/// See https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.SkipSimplifiedLayerNormalization
#[derive(Debug)]
pub struct SkipSimplifiedLayerNormalization {
    pub epsilon: f32,
}

impl Operator for SkipSimplifiedLayerNormalization {
    fn name(&self) -> &str {
        "SkipSimplifiedLayerNormalisation"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(4)
    }

    fn max_outputs(&self) -> Option<usize> {
        Some(4)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input: TensorView<_> = inputs.require_as(0)?;
        let skip: TensorView<_> = inputs.require_as(1)?;

        // Scale factor, called gamma (γ) in the RMS normalization paper.
        let gamma: NdTensorView<_, 1> = inputs.require_as(2)?;

        let bias: Option<NdTensorView<_, 1>> = inputs.get_as(3)?;

        skip_layer_normalization(
            ctx,
            input,
            skip,
            gamma,
            None, // beta
            bias,
            self.epsilon,
            MeanNormalize::DynamicRootMeanSquare,
        )
    }

    fn output_types(&self, ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        let mut types = OutputTypeList::from([OutputType::CopyFromInput(0)]);
        if ctx.num_outputs > 1 {
            types.push(OutputType::CopyFromInput(0));
            types.push(OutputType::CopyFromInput(0));
            types.push(OutputType::CopyFromInput(0));
        }
        Some(types)
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(&shape_ops::SkipLayerNormalization)
    }
}

#[cfg(test)]
mod tests {
    use rten_base::bit_set::BitSet;
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::expect_equal;
    use rten_tensor::{Tensor, TensorView};
    use rten_testing::TestCases;

    use super::{SkipLayerNormalization, SkipSimplifiedLayerNormalization};
    use crate::buffer_pool::BufferPool;
    use crate::operator::{InputList, OpError, OpRunContext, Operator, OutputList};
    use crate::ops::tests::expect_eq_1e4;

    /// Wrapper around `SkipLayerNormalization` and
    /// `SkipSimplifiedLayerNormalization` so the same tests can be used with
    /// both.
    #[derive(Clone, Copy, Debug)]
    enum SkipNormOp {
        /// `SkipLayerNormalization`: mean-centering layer norm, supports `beta`.
        Standard,
        /// `SkipSimplifiedLayerNormalization`: RMS normalization, no `beta`.
        Simplified,
    }

    impl SkipNormOp {
        /// Whether normalization subtracts the mean (layer norm).
        fn subtracts_mean(self) -> bool {
            matches!(self, SkipNormOp::Standard)
        }

        /// Run the operator with the given logical inputs.
        fn run(
            self,
            input: TensorView,
            skip: TensorView,
            gamma: TensorView,
            beta: Option<TensorView>,
            bias: Option<TensorView>,
            epsilon: f32,
            outputs: BitSet<u64>,
        ) -> Result<OutputList, OpError> {
            let mut inputs = InputList::new();
            inputs.push(input);
            inputs.push(skip);
            inputs.push(gamma);

            let pool = BufferPool::new();
            match self {
                SkipNormOp::Standard => {
                    // Inputs: input, skip, gamma, beta?, bias?
                    inputs.push_optional(beta);
                    inputs.push_optional(bias);
                    let op = SkipLayerNormalization { epsilon };
                    let ctx = OpRunContext::new(&pool, &inputs, outputs);
                    op.run(&ctx)
                }
                SkipNormOp::Simplified => {
                    // Inputs: input, skip, gamma, bias?
                    assert!(
                        beta.is_none(),
                        "SkipSimplifiedLayerNormalization has no beta input"
                    );
                    inputs.push_optional(bias);
                    let op = SkipSimplifiedLayerNormalization { epsilon };
                    let ctx = OpRunContext::new(&pool, &inputs, outputs);
                    op.run(&ctx)
                }
            }
        }
    }

    /// Reference implementation of skip (simplified) layer normalization.
    ///
    /// Computes `norm(input + skip + bias) * gamma + beta` over the last
    /// dimension. When `subtract_mean` is true this is standard layer norm,
    /// otherwise it is RMS normalization (in which case `beta` is unused).
    fn reference_skip_layer_norm(
        input: TensorView,
        skip: TensorView,
        gamma: TensorView,
        beta: Option<TensorView>,
        bias: Option<TensorView>,
        epsilon: f32,
        subtract_mean: bool,
    ) -> Tensor {
        let skip = skip.broadcast(input.shape());
        let last = input.size(input.ndim() - 1);
        let gamma = gamma.to_vec();
        let beta = beta.map(|b| b.to_vec()).unwrap_or_else(|| vec![0.0; last]);
        let bias = bias.map(|b| b.to_vec()).unwrap_or_else(|| vec![0.0; last]);

        let sum: Vec<f32> = input
            .iter()
            .zip(skip.iter())
            .enumerate()
            .map(|(i, (x, s))| x + s + bias[i % last])
            .collect();

        let mut out = Vec::with_capacity(sum.len());
        for row in sum.chunks(last) {
            let mean = if subtract_mean {
                row.iter().sum::<f32>() / last as f32
            } else {
                0.0
            };
            let var = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / last as f32;
            let denom = (var + epsilon).sqrt();
            for ((x, g), b) in row.iter().zip(&gamma).zip(&beta) {
                out.push(((x - mean) / denom) * g + b);
            }
        }
        Tensor::from_data(input.shape(), out)
    }

    #[test]
    fn test_skip_layer_normalization() {
        #[derive(Debug)]
        struct Case {
            op: SkipNormOp,
            input: Tensor,
            skip: Tensor,
            gamma: Tensor,
            beta: Option<Tensor>,
            bias: Option<Tensor>,
        }

        // Shape configurations exercised for each operator variant, as
        // `(input shape, skip shape, has bias)`.
        let shape_cases: [(&[usize], &[usize], bool); 5] = [
            // 2D input, no bias
            (&[3, 4], &[3, 4], false),
            // 2D input, with bias
            (&[3, 4], &[3, 4], true),
            // 3D input (typical transformer shape: [batch, seq, hidden])
            (&[2, 3, 4], &[2, 3, 4], true),
            // 3D input with `skip` broadcast over the batch dimension
            (&[2, 3, 4], &[1, 3, 4], false),
            // 3D input with a 2D `skip` (no batch dimension)
            (&[2, 3, 4], &[3, 4], true),
        ];

        let epsilon = 1e-5;
        let mut rng = XorShiftRng::new(1234);
        let mut cases = Vec::new();
        for op in [SkipNormOp::Standard, SkipNormOp::Simplified] {
            for &(input_shape, skip_shape, has_bias) in &shape_cases {
                let last = *input_shape.last().unwrap();
                let input = Tensor::rand(input_shape, &mut rng);
                let skip = Tensor::rand(skip_shape, &mut rng);
                let gamma = Tensor::rand(&[last], &mut rng);
                let bias = has_bias.then(|| Tensor::rand(&[last], &mut rng));
                // Only the standard variant has a `beta` input.
                let beta = op.subtracts_mean().then(|| Tensor::rand(&[last], &mut rng));
                cases.push(Case {
                    op,
                    input,
                    skip,
                    gamma,
                    beta,
                    bias,
                });
            }
        }

        cases.test_each(|case| {
            let mut outputs = case
                .op
                .run(
                    case.input.view(),
                    case.skip.view(),
                    case.gamma.view(),
                    case.beta.as_ref().map(|b| b.view()),
                    case.bias.as_ref().map(|b| b.view()),
                    epsilon,
                    BitSet::from_indices([0]),
                )
                .unwrap();
            let result: Tensor = outputs.remove(0).try_into().unwrap();

            let expected = reference_skip_layer_norm(
                case.input.view(),
                case.skip.view(),
                case.gamma.view(),
                case.beta.as_ref().map(|b| b.view()),
                case.bias.as_ref().map(|b| b.view()),
                epsilon,
                case.op.subtracts_mean(),
            );
            expect_eq_1e4(&result, &expected).unwrap();
        });
    }

    #[test]
    fn test_skip_layer_normalization_optional_outputs() {
        #[derive(Debug)]
        struct Case {
            op: SkipNormOp,
            beta: Option<Tensor>,
        }

        let input = Tensor::from([[1., 2.], [3., 4.]]);
        let skip = Tensor::from([[10., 20.], [30., 40.]]);
        let gamma = Tensor::from([1., 1.]);
        let bias = Tensor::from([0.5, -0.5]);
        let epsilon = 1e-5;

        // `input + skip + bias` is independent of the normalization variant.
        let expected_sum = Tensor::from([[11.5, 21.5], [33.5, 43.5]]);

        let cases = [
            Case {
                op: SkipNormOp::Standard,
                beta: Some(Tensor::from([0.25, -0.25])),
            },
            Case {
                op: SkipNormOp::Simplified,
                beta: None,
            },
        ];

        cases.test_each(|case| {
            let mut outputs = case
                .op
                .run(
                    input.view(),
                    skip.view(),
                    gamma.view(),
                    case.beta.as_ref().map(|b| b.view()),
                    Some(bias.view()),
                    epsilon,
                    BitSet::from_indices([0, 3]),
                )
                .unwrap();
            assert_eq!(outputs.len(), 4);

            let output: Tensor = outputs.remove(0).try_into().unwrap();
            outputs.remove(0); // mean dummy
            outputs.remove(0); // inv_std_var dummy
            let input_skip_bias_sum: Tensor = outputs.remove(0).try_into().unwrap();

            let expected_output = reference_skip_layer_norm(
                input.view(),
                skip.view(),
                gamma.view(),
                case.beta.as_ref().map(|b| b.view()),
                Some(bias.view()),
                epsilon,
                case.op.subtracts_mean(),
            );

            expect_eq_1e4(&output, &expected_output).unwrap();
            expect_equal(&input_skip_bias_sum.view(), &expected_sum.view()).unwrap();
        });
    }

    #[test]
    fn test_skip_layer_normalization_invalid() {
        #[derive(Debug)]
        struct Case {
            op: SkipNormOp,
            input: Tensor,
            skip: Tensor,
            gamma: Tensor,
            expected: OpError,
        }

        let mut cases = Vec::new();
        for op in [SkipNormOp::Standard, SkipNormOp::Simplified] {
            cases.extend([
                // Mismatched input/skip shapes
                Case {
                    op,
                    input: Tensor::zeros(&[2, 4]),
                    skip: Tensor::zeros(&[2, 3]),
                    gamma: Tensor::zeros(&[4]),
                    expected: OpError::IncompatibleInputShapes(
                        "skip must broadcast to input over the batch dimension",
                    ),
                },
                // 1D input is unsupported
                Case {
                    op,
                    input: Tensor::zeros(&[4]),
                    skip: Tensor::zeros(&[4]),
                    gamma: Tensor::zeros(&[4]),
                    expected: OpError::InvalidValue("input must be 2 or 3 dimensioned"),
                },
                // 4D input is unsupported
                Case {
                    op,
                    input: Tensor::zeros(&[1, 1, 2, 4]),
                    skip: Tensor::zeros(&[1, 1, 2, 4]),
                    gamma: Tensor::zeros(&[4]),
                    expected: OpError::InvalidValue("input must be 2 or 3 dimensioned"),
                },
            ]);
        }

        cases.test_each(|case| {
            let result = case.op.run(
                case.input.view(),
                case.skip.view(),
                case.gamma.view(),
                None,
                None,
                1e-5,
                BitSet::from_indices([0]),
            );
            let err = result.err().expect("expected an error");
            assert_eq!(&err, &case.expected);
        })
    }
}
