use wasnn_tensor::prelude::*;

use crate::ops::{Input, InputList, IntoOpResult, OpError, Operator, Output};

#[derive(Debug)]
pub struct Identity {}

impl Operator for Identity {
    fn name(&self) -> &str {
        "Identity"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let result: Output = match input {
            Input::IntTensor(t) => t.to_tensor().into(),
            Input::FloatTensor(t) => t.to_tensor().into(),
        };
        result.into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, _: InputList) -> Result<Output, OpError> {
        Ok(input)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use wasnn_tensor::test_util::expect_equal;
    use wasnn_tensor::Tensor;

    use crate::ops::{Identity, Operator};

    #[test]
    fn test_identity() -> Result<(), Box<dyn Error>> {
        let id_op = Identity {};

        let int_input = Tensor::from_vec(vec![1, 2, 3]);
        let result = id_op
            .run((&int_input).into())
            .unwrap()
            .remove(0)
            .into_int()
            .unwrap();
        assert_eq!(result, int_input);

        let float_input = Tensor::from_vec(vec![1.0, 2.0, 3.0]);
        let result = id_op
            .run((&float_input).into())
            .unwrap()
            .remove(0)
            .into_float()
            .unwrap();
        expect_equal(&result, &float_input)?;

        Ok(())
    }
}
