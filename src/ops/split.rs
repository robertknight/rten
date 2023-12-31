use rten_tensor::prelude::*;
use rten_tensor::{NdTensorView, SliceItem, Tensor, TensorView};

use crate::ops::{resolve_axis, InputList, OpError, Operator, Output};
use crate::static_dims;

pub fn split<T: Copy>(
    input: TensorView<T>,
    axis: isize,
    split: &NdTensorView<i32, 1>,
) -> Result<Vec<Tensor<T>>, OpError> {
    let axis = resolve_axis(input.ndim(), axis)?;

    if split.iter().any(|size| *size < 0) {
        return Err(OpError::InvalidValue("Split sizes must be >= 0"));
    }
    let split_sum = split.iter().sum::<i32>() as usize;
    if split_sum != input.size(axis) {
        return Err(OpError::InvalidValue(
            "Split sizes do not sum to dimension size",
        ));
    }

    let mut split_start = 0;
    let outputs = split
        .iter()
        .map(|&split_size| {
            let split_size = split_size as usize;
            let slice_range: Vec<SliceItem> = (0..input.ndim())
                .map(|dim| {
                    if dim == axis {
                        (split_start..split_start + split_size).into()
                    } else {
                        SliceItem::full_range()
                    }
                })
                .collect();

            split_start += split_size;

            input.view().slice(slice_range.as_slice()).to_tensor()
        })
        .collect();

    Ok(outputs)
}

#[derive(Debug)]
pub struct Split {
    pub axis: isize,
}

impl Operator for Split {
    fn name(&self) -> &str {
        "Split"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require_as::<f32>(0)?;
        let splits = inputs.require_as::<i32>(1)?;
        let splits = static_dims!(splits, 1)?;

        split(input.view(), self.axis, &splits)
            .map(|tensors| tensors.into_iter().map(|t| t.into()).collect())
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::{tensor, View};

    use crate::ops::{split, OpError};

    #[test]
    fn test_split() {
        let input = tensor!((5, 2); [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        // Split with positive axis
        let splits = &[1, 1];
        let results = split(input.view(), 1, &splits.into()).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].data().unwrap(), &[0., 2., 4., 6., 8.]);
        assert_eq!(results[1].data().unwrap(), &[1., 3., 5., 7., 9.]);

        // Split with negative axis
        let splits = &[1, 1];
        let results = split(input.view(), -1, &splits.into()).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].data().unwrap(), &[0., 2., 4., 6., 8.]);
        assert_eq!(results[1].data().unwrap(), &[1., 3., 5., 7., 9.]);
    }

    #[test]
    fn test_split_invalid_inputs() {
        let input = tensor!((5, 2); [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        let splits = &[1, 1];
        let result = split(input.view(), 2, &splits.into());
        assert_eq!(result.err(), Some(OpError::InvalidValue("Axis is invalid")));

        let result = split(input.view(), -3, &splits.into());
        assert_eq!(result.err(), Some(OpError::InvalidValue("Axis is invalid")));

        let splits = &[1, 2];
        let result = split(input.view(), 1, &splits.into());
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue(
                "Split sizes do not sum to dimension size"
            ))
        );

        let splits = &[1, -2];
        let result = split(input.view(), 1, &splits.into());
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue("Split sizes must be >= 0"))
        );
    }
}
