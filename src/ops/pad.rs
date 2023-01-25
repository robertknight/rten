use crate::check_dims;
use crate::ops::{Input, InputList, IntoOpResult, OpError, Operator, Output};
use crate::tensor::{Tensor, TensorLayout};

pub fn pad<T: Copy>(
    input: &Tensor<T>,
    padding: &Tensor<i32>,
    const_val: T,
) -> Result<Tensor<T>, OpError> {
    check_dims!(padding, 1);

    if padding.shape()[0] != input.ndim() * 2 {
        return Err(OpError::InvalidValue(
            "padding length should be 2 * input dims",
        ));
    }
    if !padding.iter().all(|x| x >= 0) {
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
    let out_len = out_shape.iter().product();

    let mut output = Tensor::from_data(&out_shape, vec![const_val; out_len]);
    let mut in_iter = input.indices();
    let mut out_index = vec![0; output.shape().len()];

    while let Some(in_index) = in_iter.next() {
        out_index.copy_from_slice(in_index);
        for i in 0..out_index.len() {
            out_index[i] += padding[[i]] as usize;
        }
        output[&out_index[..]] = input[in_index];
    }

    Ok(output)
}

#[derive(Debug)]
pub struct Pad {}

impl Operator for Pad {
    fn name(&self) -> &str {
        "Pad"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let pads = inputs.require_as::<i32>(1)?;
        let axes = inputs.get_as::<i32>(3)?;

        if axes.is_some() {
            return Err(OpError::UnsupportedValue(
                "Pad operator does not yet support `axes` input",
            ));
        }

        match input {
            Input::IntTensor(t) => {
                let const_val = inputs.get_as_scalar::<i32>(2)?;
                pad(t, pads, const_val.unwrap_or(0)).into_op_result()
            }
            Input::FloatTensor(t) => {
                let const_val = inputs.get_as_scalar::<f32>(2)?;
                pad(t, pads, const_val.unwrap_or(0.0)).into_op_result()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::{pad, InputList, OpError, Operator, Pad};
    use crate::tensor::{from_data, from_scalar, Tensor, TensorLayout};
    use crate::test_util::expect_equal;

    fn from_slice<T: Copy>(data: &[T]) -> Tensor<T> {
        from_data(&[data.len()], data.into())
    }

    #[test]
    fn test_pad() -> Result<(), String> {
        // Same padding around each edge.
        let input = from_data(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let expected = from_data(
            &[4, 4],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        );
        let const_pads = from_slice(&[1, 1, 1, 1]);
        let result = pad(&input, &const_pads, 0.0).unwrap();
        expect_equal(&result, &expected)?;

        // Zero padding (no-op)
        let zero_pads = from_slice(&[0, 0, 0, 0]);
        let result = pad(&input, &zero_pads, 0.0).unwrap();
        expect_equal(&result, &input)?;

        // Un-even padding
        let input = from_data(&[1, 2, 2], vec![1, 2, 3, 4]);
        let pads = from_slice(&[0, 0, 0, 0, 1, 0]);
        let result = pad(&input, &pads, 0).unwrap();
        assert_eq!(result.shape(), &[1, 3, 2]);
        assert_eq!(result.data(), &[1, 2, 3, 4, 0, 0]);

        Ok(())
    }

    #[test]
    fn test_pad_constant_val() -> Result<(), String> {
        let input = from_data(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let expected = from_data(
            &[4, 4],
            vec![
                9., 9., 9., 9., 9., 1., 2., 9., 9., 3., 4., 9., 9., 9., 9., 9.,
            ],
        );
        let const_pads = from_slice(&[1, 1, 1, 1]);
        let result = pad(&input, &const_pads, 9.).unwrap();
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_pad_op() -> Result<(), String> {
        let input = from_data(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let pads = from_slice(&[1, 1, 1, 1]);
        let expected = from_data(
            &[4, 4],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        );

        let op = Pad {};
        let result = op
            .run(InputList::from(&[(&input).into(), (&pads).into()]))
            .unwrap()
            .remove(0)
            .into_float()
            .unwrap();
        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_pad_invalid_inputs() {
        let input = from_data(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let op = Pad {};

        // Wrong padding vector length.
        let invalid_pads = from_slice(&[1]);
        let result = op.run(InputList::from(&[(&input).into(), (&invalid_pads).into()]));
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue(
                "padding length should be 2 * input dims"
            ))
        );

        // Unsupported padding amounts.
        let invalid_pads = from_slice(&[1, 1, 1, -1]);
        let result = op.run(InputList::from(&[(&input).into(), (&invalid_pads).into()]));
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Pad only supports positive pads"))
        );

        // Wrong constant value type.
        let invalid_pads = from_slice(&[1, 1, 1, -1]);
        let const_int = from_scalar(1);
        let result = op.run(InputList::from(&[
            (&input).into(),
            (&invalid_pads).into(),
            (&const_int).into(),
        ]));
        assert_eq!(result.err(), Some(OpError::IncorrectInputType));

        // Constant value not a scalar.
        let invalid_pads = from_slice(&[1, 1, 1, -1]);
        let int_vec = from_slice(&[1.0, 2.0]);
        let result = op.run(InputList::from(&[
            (&input).into(),
            (&invalid_pads).into(),
            (&int_vec).into(),
        ]));
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Expected scalar value"))
        );
    }
}
