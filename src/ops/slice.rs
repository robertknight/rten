use std::iter::zip;

use crate::ops::{resolve_axis, Input, InputList, IntoOpResult, OpError, Operator, Output};
use crate::tensor::{SliceRange, Tensor};

/// Compute the effective starts, ends and steps for each input dimension in
/// a Slice operation.
///
/// See https://onnx.ai/onnx/operators/onnx__Slice.html.
fn slice_ranges(
    input_shape: &[usize],
    starts: &Tensor<i32>,
    ends: &Tensor<i32>,
    axes: Option<&Tensor<i32>>,
    steps: Option<&Tensor<i32>>,
) -> Result<Vec<SliceRange>, OpError> {
    // FIXME: Verify that `starts`, `ends`, `axes` and `steps` are vectors with
    // compatible lengths.

    if let Some(steps) = steps {
        if steps.ndim() != 1 {
            return Err(OpError::InvalidValue("`steps` should be a vector"));
        }
        for step in steps.elements() {
            if step == 0 {
                return Err(OpError::InvalidValue("steps must be non-zero"));
            }
        }
    }

    let mut ranges: Vec<SliceRange> = input_shape
        .iter()
        .map(|dim_size| SliceRange::new(0, *dim_size as isize, 1))
        .collect();
    for (i, (start, end)) in zip(starts.elements(), ends.elements()).enumerate() {
        let axis = if let Some(axes) = axes {
            resolve_axis(input_shape.len(), axes[[i]] as isize)?
        } else {
            i
        };

        let step = steps.map(|s| s[[i]]).unwrap_or(1);
        ranges[axis] = SliceRange::new(start as isize, end as isize, step as isize);
    }
    Ok(ranges)
}

/// Return a copy of a tensor which only retains a subset of a given dimension.
pub fn slice<T: Copy>(
    input: &Tensor<T>,
    starts: &Tensor<i32>,
    ends: &Tensor<i32>,
    axes: Option<&Tensor<i32>>,
    steps: Option<&Tensor<i32>>,
) -> Result<Tensor<T>, OpError> {
    let ranges = slice_ranges(input.shape(), starts, ends, axes, steps)?;
    let sliced_data = input.slice_elements(&ranges).collect();
    let sliced_shape = ranges
        .iter()
        .enumerate()
        .map(|(dim, range)| range.steps(input.shape()[dim]))
        .collect();
    Ok(Tensor::from_data(sliced_shape, sliced_data))
}

/// Clip the dimensions of the input tensor specified by `axes` to the ranges
/// given by `starts` and `ends`.
pub fn slice_in_place<T: Copy>(
    input: &mut Tensor<T>,
    starts: &Tensor<i32>,
    ends: &Tensor<i32>,
    axes: Option<&Tensor<i32>>,
) -> Result<(), OpError> {
    let ranges = slice_ranges(input.shape(), starts, ends, axes, None)?;
    for (dim, range) in ranges.iter().enumerate() {
        let dim_size = input.shape()[dim];
        input.clip_dim(dim, range.resolve(dim_size));
    }
    Ok(())
}

#[derive(Debug)]
pub struct Slice {}

impl Operator for Slice {
    fn name(&self) -> &str {
        "Slice"
    }

    fn run(&self, inputs: InputList) -> Result<Vec<Output>, OpError> {
        let input = inputs.require(0)?;
        let starts = inputs.require_as::<i32>(1)?;
        let ends = inputs.require_as::<i32>(2)?;
        let axes = inputs.get_as::<i32>(3)?;
        let steps = inputs.get_as::<i32>(4)?;

        let result: Result<Output, OpError> = match input {
            Input::FloatTensor(input) => slice(input, starts, ends, axes, steps).map(|t| t.into()),
            Input::IntTensor(input) => slice(input, starts, ends, axes, steps).map(|t| t.into()),
        };
        result.into_op_result()
    }

    fn can_run_in_place(&self) -> bool {
        true
    }

    fn run_in_place(&self, input: Output, other: InputList) -> Result<Output, OpError> {
        let starts = other.require_as::<i32>(0)?;
        let ends = other.require_as::<i32>(1)?;
        let axes = other.get_as::<i32>(2)?;
        let steps = other.get_as::<i32>(3)?;

        // Fall back to copying if non-default steps are given.
        if let Some(steps) = steps {
            if steps.elements().any(|step| step != 1) {
                let mut inputs: Vec<_> = vec![(&input).into()];
                inputs.extend(other.iter());
                return self
                    .run(InputList::from(&inputs))
                    .map(|mut outputs| outputs.remove(0));
            }
        }

        match input {
            Output::IntTensor(mut output) => {
                slice_in_place(&mut output, starts, ends, axes)?;
                Ok(output.into())
            }
            Output::FloatTensor(mut output) => {
                slice_in_place(&mut output, starts, ends, axes)?;
                Ok(output.into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::{slice, slice_in_place};
    use crate::rng::XorShiftRNG;
    use crate::tensor::{from_data, rand, Tensor};
    use crate::test_util::expect_equal;

    fn from_slice<T: Copy>(data: &[T]) -> Tensor<T> {
        from_data(vec![data.len()], data.into())
    }

    #[test]
    fn test_slice_in_place() {
        // Slice with +ve and in-bounds endpoints.
        let mut rng = XorShiftRNG::new(5678);
        let mut input = rand(&[2, 2, 5, 3], &mut rng);

        let starts = from_slice(&[2]);
        let ends = from_slice(&[4]);
        let axes = from_slice(&[2]);

        slice_in_place(&mut input, &starts, &ends, Some(&axes)).unwrap();

        assert_eq!(
            input.shape(),
            vec![2, 2, ends[[0]] as usize - starts[[0]] as usize, 3]
        );

        // Slice with -ve endpoints.
        let mut input = Tensor::from_vec((0..10).collect());
        let starts = from_slice(&[-9]);
        let ends = from_slice(&[-6]);
        slice_in_place(&mut input, &starts, &ends, None).unwrap();
        assert_eq!(input.elements_vec(), &[1, 2, 3]);

        // Slice with out-of-bounds end.
        let mut input = Tensor::from_vec((0..10).collect());
        let starts = from_slice(&[5]);
        let ends = from_slice(&[20]);
        slice_in_place(&mut input, &starts, &ends, None).unwrap();
        assert_eq!(input.elements_vec(), &[5, 6, 7, 8, 9]);

        // Slice with out-of-bounds start.
        let mut input = Tensor::from_vec((0..10).collect());
        let starts = from_slice(&[-20]);
        let ends = from_slice(&[5]);
        slice_in_place(&mut input, &starts, &ends, None).unwrap();
        assert_eq!(input.elements_vec(), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_slice_first_dim() {
        let mut rng = XorShiftRNG::new(5678);
        let input = rand(&[5, 2, 5, 3], &mut rng);

        let starts = from_slice(&[2]);
        let ends = from_slice(&[4]);
        let axes = from_slice(&[0]);

        let sliced = slice(&input, &starts, &ends, Some(&axes), None).unwrap();
        let shape = sliced.shape();

        assert_eq!(
            shape,
            vec![ends[[0]] as usize - starts[[0]] as usize, 2, 5, 3]
        );
        assert_eq!(sliced.len(), shape.iter().fold(1, |len, x| len * x));

        for w in 0..shape[0] {
            for x in 0..shape[1] {
                for y in 0..shape[2] {
                    for z in 0..shape[3] {
                        assert_eq!(
                            sliced[[w, x, y, z]],
                            input[[w + starts[[0]] as usize, x, y, z]]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_slice_inner_dim() {
        let mut rng = XorShiftRNG::new(5678);
        let input = rand(&[2, 2, 5, 3], &mut rng);

        let starts = from_slice(&[2]);
        let ends = from_slice(&[4]);
        let axes = from_slice(&[2]);

        let sliced = slice(&input, &starts, &ends, Some(&axes), None).unwrap();
        let shape = sliced.shape();

        assert_eq!(
            sliced.shape(),
            vec![2, 2, ends[[0]] as usize - starts[[0]] as usize, 3]
        );
        assert_eq!(sliced.len(), shape.iter().fold(1, |len, x| len * x));

        for w in 0..shape[0] {
            for x in 0..shape[1] {
                for y in 0..shape[2] {
                    for z in 0..shape[3] {
                        assert_eq!(
                            sliced[[w, x, y, z]],
                            input[[w, x, y + starts[[0]] as usize, z]]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_slice_noop() {
        let mut rng = XorShiftRNG::new(5678);
        let input = rand(&[5, 2, 5, 3], &mut rng);

        for dim in 0..input.shape().len() {
            let dim_size = input.shape()[dim] as i32;

            let starts = from_slice(&[0]);
            let ends = from_slice(&[dim_size]);
            let axes = from_slice(&[dim as i32]);

            let sliced = slice(&input, &starts, &ends, Some(&axes), None).unwrap();
            assert_eq!(sliced.shape(), input.shape());
            assert_eq!(sliced.data(), input.data());
        }
    }

    #[test]
    fn test_slice_negative_axes() {
        let input = from_data(vec![3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let starts = from_slice(&[0]);
        let ends = from_slice(&[2]);

        let axes = from_slice(&[-1]);
        let sliced = slice(&input, &starts, &ends, Some(&axes), None).unwrap();
        assert_eq!(sliced.elements().collect::<Vec<_>>(), &[1, 2, 4, 5, 7, 8]);

        let axes = from_slice(&[-2]);
        let sliced = slice(&input, &starts, &ends, Some(&axes), None).unwrap();
        assert_eq!(sliced.elements().collect::<Vec<_>>(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_slice_negative_starts() {
        let input = from_data(vec![3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let axes = from_slice(&[-1]);
        let ends = from_slice(&[2]);

        let starts = from_slice(&[-3]);
        let sliced = slice(&input, &starts, &ends, Some(&axes), None).unwrap();
        assert_eq!(sliced.elements().collect::<Vec<_>>(), &[1, 2, 4, 5, 7, 8]);

        let starts = from_slice(&[-2]);
        let sliced = slice(&input, &starts, &ends, Some(&axes), None).unwrap();
        assert_eq!(sliced.elements().collect::<Vec<_>>(), &[2, 5, 8]);
    }

    #[test]
    fn test_slice_negative_ends() {
        let input = from_data(vec![3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let axes = from_slice(&[-1]);
        let starts = from_slice(&[0]);

        let ends = from_slice(&[-1]);
        let sliced = slice(&input, &starts, &ends, Some(&axes), None).unwrap();
        assert_eq!(sliced.elements().collect::<Vec<_>>(), &[1, 2, 4, 5, 7, 8]);

        let ends = from_slice(&[-2]);
        let sliced = slice(&input, &starts, &ends, Some(&axes), None).unwrap();
        assert_eq!(sliced.elements().collect::<Vec<_>>(), &[1, 4, 7]);
    }

    #[test]
    fn test_slice_clamps_starts_and_ends() -> Result<(), String> {
        let mut rng = XorShiftRNG::new(5678);
        let input = rand(&[20, 20], &mut rng);

        // Simulate how a range without a start/end may be given in a model.
        //
        // The ONNX Slice spec does not support unbounded ranges (like
        // `array[start:]` in numpy) but instead recommends the use of INT_MAX /
        // -INT_MAX together with clamping to achieve the same result.
        let starts = from_slice(&[-i32::MAX, -100]);
        let ends = from_slice(&[i32::MAX, 100]);

        let sliced = slice(&input, &starts, &ends, None, None).unwrap();

        expect_equal(&sliced, &input)
    }

    #[test]
    fn test_slice_with_step() {
        let input = from_slice(&[1, 2, 3, 4, 5]);

        struct Case<'a> {
            start: i32,
            end: i32,
            step: i32,
            expected_shape: &'a [usize],
            expected_elements: &'a [i32],
        }

        let cases = [
            // Positive step > 1
            Case {
                start: 0,
                end: 5,
                step: 2,
                expected_shape: &[3],
                expected_elements: &[1, 3, 5],
            },
            // Negative step
            Case {
                start: 5,
                end: -6,
                step: -1,
                expected_shape: &[5],
                expected_elements: &[5, 4, 3, 2, 1],
            },
            // Negative step with clamped start
            Case {
                start: 100,
                end: -6,
                step: -1,
                expected_shape: &[5],
                expected_elements: &[5, 4, 3, 2, 1],
            },
            // Negative step with clamped end
            Case {
                start: 5,
                end: -100,
                step: -1,
                expected_shape: &[5],
                expected_elements: &[5, 4, 3, 2, 1],
            },
        ];

        for case in cases {
            let starts = from_slice(&[case.start]);
            let ends = from_slice(&[case.end]);
            let axes = from_slice(&[0]);
            let steps = from_slice(&[case.step]);

            let sliced = slice(&input, &starts, &ends, Some(&axes), Some(&steps)).unwrap();

            assert_eq!(sliced.shape(), case.expected_shape);
            assert_eq!(sliced.data(), case.expected_elements);
        }
    }
}
