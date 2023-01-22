use std::iter::zip;

use crate::ops::{resolve_axis, InputList, OpError, Operator, Output};
use crate::tensor::{SliceRange, Tensor};

pub fn split<T: Copy>(
    input: &Tensor<T>,
    axis: isize,
    split: &[usize],
) -> Result<Vec<Tensor<T>>, OpError> {
    let axis = resolve_axis(input.ndim(), axis)?;
    let split_sum: usize = split.iter().sum();
    if split_sum != input.shape()[axis] {
        return Err(OpError::InvalidValue(
            "split sizes do not sum to dimension size",
        ));
    }

    let mut outputs = Vec::new();
    let mut start = 0;

    for split_size in split {
        let slice_ranges: Vec<SliceRange> = input
            .shape()
            .iter()
            .copied()
            .enumerate()
            .map(|(dim, size)| {
                if dim == axis {
                    SliceRange::new(start as isize, (start + split_size) as isize, 1)
                } else {
                    SliceRange::new(0, size as isize, 1)
                }
            })
            .collect();
        let elements = input.slice_iter(&slice_ranges).collect();
        let slice_shape = zip(input.shape().iter(), slice_ranges)
            .map(|(&dim_size, range)| range.steps(dim_size))
            .collect();
        let tensor = Tensor::from_data(slice_shape, elements);
        outputs.push(tensor);

        start += split_size;
    }

    Ok(outputs)
}

#[derive(Debug)]
pub struct Split {
    pub axis: isize,
    pub split: Vec<usize>,
}

impl Operator for Split {
    fn name(&self) -> &str {
        "Split"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require_as::<f32>(0)?;
        split(input, self.axis, &self.split[..])
            .map(|tensors| tensors.into_iter().map(|t| t.into()).collect())
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::{split, OpError};
    use crate::tensor::from_data;

    #[test]
    fn test_split() {
        let input = from_data(vec![5, 2], vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        // Split with positive axis
        let results = split(&input, 1, &[1, 1]).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].data(), &[0., 2., 4., 6., 8.]);
        assert_eq!(results[1].data(), &[1., 3., 5., 7., 9.]);

        // Split with negative axis
        let results = split(&input, -1, &[1, 1]).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].data(), &[0., 2., 4., 6., 8.]);
        assert_eq!(results[1].data(), &[1., 3., 5., 7., 9.]);
    }

    #[test]
    fn test_split_invalid_inputs() {
        let input = from_data(vec![5, 2], vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        let result = split(&input, 2, &[1, 1]);
        assert_eq!(result.err(), Some(OpError::InvalidValue("axis is invalid")));

        let result = split(&input, -3, &[1, 1]);
        assert_eq!(result.err(), Some(OpError::InvalidValue("axis is invalid")));

        let result = split(&input, 1, &[1, 2]);
        assert_eq!(
            result.err(),
            Some(OpError::InvalidValue(
                "split sizes do not sum to dimension size"
            ))
        );
    }
}
