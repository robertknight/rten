use rten_tensor::prelude::*;
use rten_tensor::{NdTensorView, SliceItem, Tensor, TensorView};

use crate::ops::{Input, InputList, IntoOpResult, OpError, Operator, OutputList};
use crate::static_dims;
use crate::tensor_pool::TensorPool;

pub fn pad<T: Copy>(
    pool: &TensorPool,
    input: TensorView<T>,
    padding: &NdTensorView<i32, 1>,
    const_val: T,
) -> Result<Tensor<T>, OpError> {
    if padding.size(0) != input.ndim() * 2 {
        return Err(OpError::InvalidValue(
            "padding length should be 2 * input dims",
        ));
    }
    if !padding.iter().all(|x| *x >= 0) {
        return Err(OpError::InvalidValue("Pad only supports positive pads"));
    }

    let out_shape: Vec<_> = input
        .shape()
        .iter()
        .enumerate()
        .map(|(i, size)| {
            let start_pad = padding[[i]] as usize;
            let end_pad = padding[[input.ndim() + i]] as usize;
            start_pad + size + end_pad
        })
        .collect();

    let non_pad_region: Vec<SliceItem> = input
        .shape()
        .iter()
        .enumerate()
        .map(|(i, size)| {
            let start_pad = padding[[i]] as usize;
            (start_pad..start_pad + size).into()
        })
        .collect();

    let mut output = Tensor::full_in(pool, &out_shape, const_val);
    output
        .slice_mut_dyn(non_pad_region.as_slice())
        .copy_from(&input);

    Ok(output)
}

#[derive(Debug)]
pub struct Pad {}

impl Operator for Pad {
    fn name(&self) -> &str {
        "Pad"
    }

    fn run(&self, pool: &TensorPool, inputs: InputList) -> Result<OutputList, OpError> {
        let input = inputs.require(0)?;
        let pads = inputs.require_as::<i32>(1)?;
        let pads = static_dims!(pads, 1)?;
        let axes = inputs.get_as::<i32>(3)?;

        if axes.is_some() {
            return Err(OpError::UnsupportedValue(
                "Pad operator does not yet support `axes` input",
            ));
        }

        match input {
            Input::Int8Tensor(t) => {
                let const_val = inputs.get_as_scalar::<i8>(2)?;
                pad(pool, t, &pads, const_val.unwrap_or(0)).into_op_result()
            }
            Input::Int32Tensor(t) => {
                let const_val = inputs.get_as_scalar::<i32>(2)?;
                pad(pool, t, &pads, const_val.unwrap_or(0)).into_op_result()
            }
            Input::FloatTensor(t) => {
                let const_val = inputs.get_as_scalar::<f32>(2)?;
                pad(pool, t, &pads, const_val.unwrap_or(0.0)).into_op_result()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rten_tensor::prelude::*;
    use rten_tensor::test_util::expect_equal;
    use rten_tensor::Tensor;

    use crate::ops::tests::new_pool;
    use crate::ops::{pad, OpError, Operator, Pad};

    fn from_slice<T: Clone>(data: &[T]) -> Tensor<T> {
        Tensor::from_data(&[data.len()], data.to_vec())
    }

    #[test]
    fn test_pad() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();

        // Same padding around each edge.
        let input = Tensor::from_data(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let expected = Tensor::from_data(
            &[4, 4],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        );
        let const_pads = &[1, 1, 1, 1];
        let result = pad(&pool, input.view(), &const_pads.into(), 0.0).unwrap();
        expect_equal(&result, &expected)?;

        // Zero padding (no-op)
        let zero_pads = &[0, 0, 0, 0];
        let result = pad(&pool, input.view(), &zero_pads.into(), 0.0).unwrap();
        expect_equal(&result, &input)?;

        // Un-even padding
        let input = Tensor::from_data(&[1, 2, 2], vec![1, 2, 3, 4]);
        let pads = &[0, 0, 0, 0, 1, 0];
        let result = pad(&pool, input.view(), &pads.into(), 0).unwrap();
        assert_eq!(result.shape(), &[1, 3, 2]);
        assert_eq!(result.data().unwrap(), &[1, 2, 3, 4, 0, 0]);

        Ok(())
    }

    #[test]
    fn test_pad_constant_val() -> Result<(), Box<dyn Error>> {
        let pool = new_pool();
        let input = Tensor::from_data(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let expected = Tensor::from_data(
            &[4, 4],
            vec![
                9., 9., 9., 9., 9., 1., 2., 9., 9., 3., 4., 9., 9., 9., 9., 9.,
            ],
        );
        let const_pads = &[1, 1, 1, 1];
        let result = pad(&pool, input.view(), &const_pads.into(), 9.).unwrap();
        expect_equal(&result, &expected)?;
        Ok(())
    }

    #[test]
    fn test_pad_op() -> Result<(), Box<dyn Error>> {
        let input = Tensor::from_data(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let pads = from_slice(&[1, 1, 1, 1]);
        let expected = Tensor::from_data(
            &[4, 4],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        );

        let pool = new_pool();
        let op = Pad {};
        let result = op
            .run(&pool, (&input, &pads).into())
            .unwrap()
            .remove(0)
            .into_float()
            .unwrap();
        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_pad_invalid_inputs() {
        let pool = new_pool();
        let input = Tensor::from_data(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let op = Pad {};

        // Wrong padding vector length.
        let invalid_pads = from_slice(&[1]);
        let result = op.run(&pool, (&input, &invalid_pads).into());
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue(
                "padding length should be 2 * input dims"
            ))
        );

        // Unsupported padding amounts.
        let invalid_pads = from_slice(&[1, 1, 1, -1]);
        let result = op.run(&pool, (&input, &invalid_pads).into());
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Pad only supports positive pads"))
        );

        // Wrong constant value type.
        let invalid_pads = from_slice(&[1, 1, 1, -1]);
        let const_int = Tensor::from(1);
        let result = op.run(&pool, (&input, &invalid_pads, &const_int).into());
        assert_eq!(result.err(), Some(OpError::IncorrectInputType));

        // Constant value not a scalar.
        let invalid_pads = from_slice(&[1, 1, 1, -1]);
        let int_vec = from_slice(&[1.0, 2.0]);
        let result = op.run(&pool, (&input, &invalid_pads, &int_vec).into());
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Expected scalar value"))
        );
    }
}
