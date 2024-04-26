use std::any::Any;

use rten_tensor::prelude::*;
use rten_tensor::{Tensor, TensorView};

use crate::ops::{Input, InputList, IntoOpResult, OpError, Operator, Output};
use crate::tensor_pool::TensorPool;

fn identity<T: Any + Copy>(pool: &TensorPool, src: TensorView<T>) -> Tensor<T> {
    let buf = pool.alloc_vec(src.len());
    src.to_tensor_buf(buf)
}

#[derive(Debug)]
pub struct Identity {}

impl Operator for Identity {
    fn name(&self) -> &str {
        "Identity"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let result: Output = match input {
            Input::IntTensor(t) => identity(pool, t).into(),
            Input::FloatTensor(t) => identity(pool, t).into(),
        };
        result.into_op_result()
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
    use crate::ops::{Identity, Operator};

    #[test]
    fn test_identity() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();
        let id_op = Identity {};

        let int_input = Tensor::from_vec(vec![1, 2, 3]);
        let result = id_op
            .run(&pool, (&int_input).into())
            .unwrap()
            .remove(0)
            .into_int()
            .unwrap();
        assert_eq!(result, int_input);

        let float_input = Tensor::from_vec(vec![1.0, 2.0, 3.0]);
        let result = id_op
            .run(&pool, (&float_input).into())
            .unwrap()
            .remove(0)
            .into_float()
            .unwrap();
        expect_equal(&result, &float_input)?;

        Ok(())
    }
}
