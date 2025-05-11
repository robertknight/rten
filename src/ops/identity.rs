use rten_tensor::prelude::*;
use rten_tensor::{Tensor, TensorView};

use crate::ops::{
    map_input, Input, InputList, IntoOpResult, OpError, OpRunContext, Operator, Output, OutputList,
};
use crate::tensor_pool::TensorPool;

fn identity<T: Copy>(pool: &TensorPool, src: TensorView<T>) -> Tensor<T> {
    src.to_tensor_in(pool)
}

#[derive(Debug)]
pub struct Identity {}

impl Operator for Identity {
    fn name(&self) -> &str {
        "Identity"
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input = ctx.inputs().require(0)?;
        map_input!(input, x, { identity(ctx.pool(), x).into_op_result() })
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(
        &self,
        _pool: &TensorPool,
        input: Output,
        _: InputList,
    ) -> Result<Output, OpError> {
        Ok(input)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::test_util::expect_equal;
    use rten_tensor::Tensor;

    use crate::ops::{Identity, OperatorExt};

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
