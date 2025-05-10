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

    use crate::ops::tests::new_pool;
    use crate::ops::{Identity, OpRunContext, Operator};

    #[test]
    fn test_identity() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();
        let id_op = Identity {};

        let int_input = Tensor::from([1, 2, 3]);
        let inputs = (&int_input).into();
        let ctx = OpRunContext::new(&pool, &inputs);
        let result = id_op
            .run(&ctx)
            .unwrap()
            .remove(0)
            .into_tensor::<i32>()
            .unwrap();
        assert_eq!(result, int_input);

        let float_input = Tensor::from([1.0, 2.0, 3.0]);
        let inputs = (&float_input).into();
        let ctx = OpRunContext::new(&pool, &inputs);
        let result = id_op
            .run(&ctx)
            .unwrap()
            .remove(0)
            .into_tensor::<f32>()
            .unwrap();
        expect_equal(&result, &float_input)?;

        Ok(())
    }
}
