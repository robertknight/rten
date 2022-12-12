use crate::ops::layout::squeeze_in_place;
use crate::ops::{get_input, resolve_axes, Input, IntoOpResult, OpError, Operator, Output};
use crate::tensor::{Elements, IndexIterator, SliceRange, Tensor};

fn reduce<T: Copy + Default, R: Fn(Elements<T>) -> T>(
    input: &Tensor<T>,
    axes: Option<&[i32]>,
    keep_dims: bool,
    reduce_op: R,
) -> Result<Tensor<T>, OpError> {
    let resolved_axes = if let Some(axes) = axes {
        resolve_axes(input.ndim(), axes)?
    } else {
        (0..input.ndim()).collect()
    };

    let reduced_shape: Vec<usize> = input
        .shape()
        .iter()
        .enumerate()
        .map(|(dim, &size)| {
            if resolved_axes.contains(&dim) {
                1
            } else {
                size
            }
        })
        .collect();

    let outer_range: Vec<_> = (0..input.ndim())
        .map(|dim| {
            if resolved_axes.contains(&dim) {
                0..1
            } else {
                0..input.shape()[dim]
            }
        })
        .collect();

    let mut outer_iter = IndexIterator::from_ranges(&outer_range);
    let mut inner_range = Vec::with_capacity(input.ndim());
    let mut reduced_data = Vec::with_capacity(reduced_shape.iter().product());

    while let Some(index) = outer_iter.next() {
        inner_range.clear();
        inner_range.extend(index.iter().enumerate().map(|(dim, &idx)| {
            if resolved_axes.contains(&dim) {
                SliceRange::new(0, input.shape()[dim] as isize, 1)
            } else {
                SliceRange::new(idx as isize, idx as isize + 1, 1)
            }
        }));
        reduced_data.push(reduce_op(input.slice_elements(&inner_range)));
    }

    let mut reduced = Tensor::<T>::from_data(reduced_shape, reduced_data);

    if !keep_dims {
        squeeze_in_place(&mut reduced, Some(&resolved_axes));
    }

    Ok(reduced)
}

pub fn reduce_mean(
    input: &Tensor,
    axes: Option<&[i32]>,
    keep_dims: bool,
) -> Result<Tensor, OpError> {
    reduce(input, axes, keep_dims, |elements| {
        let len = elements.len() as f32;
        elements.sum::<f32>() / len
    })
}

#[derive(Debug)]
pub struct ReduceMean {
    pub axes: Option<Vec<i32>>,
    pub keep_dims: bool,
}

impl Operator for ReduceMean {
    fn name(&self) -> &str {
        "ReduceMean"
    }

    fn run(&self, inputs: &[Input]) -> Result<Vec<Output>, OpError> {
        let input = get_input(inputs, 0)?;
        reduce_mean(
            input,
            self.axes.as_ref().map(|axis| &axis[..]),
            self.keep_dims,
        )
        .into_op_result()
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::{reduce_mean, OpError};
    use crate::tensor::{from_data, from_scalar, from_vec};
    use crate::test_util::expect_equal;

    #[test]
    fn test_reduce_mean() -> Result<(), String> {
        let input = from_data(vec![3, 3], vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        // Test with `keep_dims` off
        let result = reduce_mean(&input, Some(&[-1]), false /* keep_dims */).unwrap();
        let expected = from_vec(vec![2., 5., 8.]);
        expect_equal(&result, &expected)?;

        // Test with `keep_dims` on
        let result = reduce_mean(&input, Some(&[-1]), true /* keep_dims */).unwrap();
        let expected = from_data(vec![3, 1], vec![2., 5., 8.]);
        expect_equal(&result, &expected)?;

        // Reduce first dim
        let result = reduce_mean(&input, Some(&[0]), false /* keep_dims */).unwrap();
        let expected = from_vec(vec![4., 5., 6.]);
        expect_equal(&result, &expected)?;

        // Reduce all axes
        let result = reduce_mean(&input, None, false /* keep_dims */).unwrap();
        let expected = from_scalar(5.);
        expect_equal(&result, &expected)?;

        // Test case from ONNX spec
        let input = from_data(
            vec![3, 2, 2],
            vec![5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.],
        );
        let expected = from_data(vec![3, 2], vec![12.5, 1.5, 35., 1.5, 57.5, 1.5]);
        let result = reduce_mean(&input, Some(&[1]), false /* keep_dims */).unwrap();
        expect_equal(&result, &expected)?;

        Ok(())
    }

    #[test]
    fn test_reduce_mean_invalid_inputs() {
        let input = from_data(vec![3, 3], vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        let result = reduce_mean(&input, Some(&[3]), false /* keep_dims */);
        assert_eq!(result.err(), Some(OpError::InvalidValue("axis is invalid")));

        let result = reduce_mean(&input, Some(&[-3]), false /* keep_dims */);
        assert_eq!(result.err(), Some(OpError::InvalidValue("axis is invalid")));
    }
}
