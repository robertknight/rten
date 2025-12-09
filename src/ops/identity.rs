use rten_tensor::prelude::*;
use rten_tensor::{Tensor, TensorView};

use crate::buffer_pool::BufferPool;
use crate::infer_shapes::{InferShapes, UnaryOp};
use crate::operator::{
    IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
};
use crate::ops::map_value_view;
use crate::value::{Value, ValueView};

fn identity<T: Copy>(pool: &BufferPool, src: TensorView<T>) -> Tensor<T> {
    src.to_tensor_in(pool)
}

#[derive(Debug)]
pub struct Identity {}

impl Operator for Identity {
    fn name(&self) -> &str {
        "Identity"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(1)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input = ctx.inputs().require(0)?;
        map_value_view!(input, x, { identity(ctx.pool(), x).into_op_result() })
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Value, _ctx: &OpRunContext) -> Result<Value, OpError> {
        Ok(input)
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(&UnaryOp)
    }

    fn output_types(&self) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::Tensor;
    use rten_tensor::test_util::expect_equal;

    use crate::operator::OperatorExt;
    use crate::ops::Identity;

    #[test]
    fn test_identity() -> Result<(), Box<dyn Error>> {
        let id_op = Identity {};

        let int_input = Tensor::from([1, 2, 3]);
        let result: Tensor<i32> = id_op.run_simple(&int_input).unwrap();
        assert_eq!(result, int_input);

        let float_input = Tensor::from([1.0, 2.0, 3.0]);
        let result: Tensor<f32> = id_op.run_simple(&float_input).unwrap();
        expect_equal(&result, &float_input)?;

        Ok(())
    }
}
