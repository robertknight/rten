use rayon::prelude::*;
use rten_base::bit_set::BitSet;
use rten_simd::SimdUnaryOp;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensorView, Tensor, TensorView};
use rten_vecmath as vecmath;

use crate::buffer_pool::BufferPool;
use crate::infer_shapes::{InferShapes, UnaryOp};
use crate::operator::{
    InPlaceInputs, IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType,
    OutputTypeList, OutputTypesContext,
};

/// Apply the Snake activation `x + beta * sin(alpha * x)^2` in place, where
/// `x` has shape `[N, C, ...]` and `alpha` and `beta` have one value per
/// channel.
///
/// See <https://arxiv.org/abs/2006.08195>.
pub fn snake_in_place(
    pool: &BufferPool,
    mut input: Tensor,
    alpha: NdTensorView<f32, 1>,
    beta: NdTensorView<f32, 1>,
) -> Result<Tensor, OpError> {
    if input.ndim() < 2 {
        return Err(OpError::invalid_value("expected input with >= 2 dims"));
    }
    let chans = input.size(1);
    if alpha.size(0) != chans || beta.size(0) != chans {
        return Err(OpError::invalid_value(
            "alpha and beta length should match channel count",
        ));
    }
    if input.is_empty() {
        return Ok(input);
    }

    input.make_contiguous_in(pool);
    let row_len = input.shape()[2..].iter().product::<usize>();
    let data = input.data_mut().unwrap();

    data.par_chunks_mut(row_len)
        .enumerate()
        .for_each(|(row, row_data)| {
            let chan = row % chans;
            vecmath::Snake {
                alpha: alpha[chan],
                beta: beta[chan],
            }
            .map_mut(row_data);
        });

    Ok(input)
}

/// Snake activation with per-channel parameters.
///
/// This computes `x + beta * sin(alpha * x)^2`, where `x` has shape
/// `[N, C, ...]` and `alpha` and `beta` have shape `[C]`. This is not an
/// ONNX operator. It is created by fusing subgraphs of primitive operators.
#[derive(Debug)]
pub struct Snake {}

impl Operator for Snake {
    fn name(&self) -> &str {
        "Snake"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(3)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input: TensorView<f32> = inputs.require_as(0)?;
        let alpha = inputs.require_as(1)?;
        let beta = inputs.require_as(2)?;
        snake_in_place(ctx.pool(), input.to_tensor_in(ctx.pool()), alpha, beta).into_op_result()
    }

    fn in_place_inputs(&self) -> BitSet<u16> {
        BitSet::from_indices([0])
    }

    fn run_in_place(
        &self,
        in_place: InPlaceInputs,
        ctx: &OpRunContext,
    ) -> Result<OutputList, OpError> {
        let input: Tensor = in_place.into_single().try_into()?;
        let inputs = ctx.inputs();
        let alpha = inputs.require_as(1)?;
        let beta = inputs.require_as(2)?;
        snake_in_place(ctx.pool(), input, alpha, beta).into_op_result()
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
    use rten_tensor::Tensor;
    use rten_tensor::prelude::*;

    use super::snake_in_place;
    use crate::buffer_pool::BufferPool;
    use crate::operator::OpError;

    #[test]
    fn test_snake() {
        let pool = BufferPool::new();
        let input = Tensor::from([[[0.5f32, -1., 2.], [0.25, 3., -0.5]]]);
        let alpha = Tensor::from([1., 2.]);
        let beta = Tensor::from([1., 0.5]);

        let result = snake_in_place(&pool, input.clone(), alpha.nd_view(), beta.nd_view()).unwrap();

        let expected = Tensor::from_fn(input.shape(), |idx| {
            let x = input[idx];
            let c = idx[1];
            x + beta[[c]] * (alpha[[c]] * x).sin().powi(2)
        });
        for (y, e) in result.iter().zip(expected.iter()) {
            assert!((y - e).abs() < 1e-6, "{y} != {e}");
        }
    }

    #[test]
    fn test_snake_invalid() {
        let pool = BufferPool::new();
        let input = Tensor::from([[[0.5f32, -1.], [0.25, 3.]]]);
        let alpha = Tensor::from([1., 2., 3.]);
        let beta = Tensor::from([1., 0.5]);

        let result = snake_in_place(&pool, input, alpha.nd_view(), beta.nd_view());
        assert_eq!(
            result.err(),
            Some(OpError::invalid_value(
                "alpha and beta length should match channel count"
            ))
        );
    }
}
