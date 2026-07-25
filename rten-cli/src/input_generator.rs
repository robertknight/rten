//! Generation of random values for model inputs.

use std::error::Error;

use rten::{DataType, Dimension, Value, ValueType};
use rten_tensor::Tensor;

use crate::dim_size::DimSize;
use crate::input_range::InputRange;

#[derive(Debug)]
pub enum GenerateError {
    UnsupportedDataType(ValueType),
}

impl std::fmt::Display for GenerateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedDataType(dtype) => {
                write!(f, "generation of {dtype} inputs is not supported")
            }
        }
    }
}

impl Error for GenerateError {}

pub struct RandomInputGenerator {
    rng: fastrand::Rng,
}

impl RandomInputGenerator {
    pub fn new() -> Self {
        RandomInputGenerator {
            rng: fastrand::Rng::new(),
        }
    }

    /// Generate a random value for an input using the name, shape and dtype
    /// properties from the model as well as configuration provided when
    /// running the CLI.
    ///
    /// `range` specifies the range of generated values, overriding the
    /// defaults chosen based on the input's name and dtype.
    ///
    /// `on_resolve_size` is invoked for each dynamic dimension size that
    /// is resolved, specifying the dimension name and index of the entry in
    /// `dim_sizes` that was used, if any.
    pub fn generate(
        &mut self,
        name: &str,
        value_type: Option<ValueType>,
        shape: &[Dimension],
        dim_sizes: &[DimSize],
        range: Option<&InputRange>,
        mut on_resolve_size: impl FnMut(&str, Option<usize>),
    ) -> Result<Value, GenerateError> {
        let dtype = match value_type {
            Some(ValueType::Tensor(dtype)) => Some(dtype),
            Some(vtype) => {
                return Err(GenerateError::UnsupportedDataType(vtype));
            }
            None => None,
        };

        let resolved_shape: Vec<usize> = shape
            .iter()
            .map(|dim| match dim {
                Dimension::Symbolic(dim_name) => {
                    if let Some((idx, dim_size)) = dim_sizes
                        .iter()
                        .enumerate()
                        .find(|(_i, ds)| ds.matches(name, dim_name))
                    {
                        on_resolve_size(dim_name, Some(idx));
                        dim_size.size
                    } else {
                        on_resolve_size(dim_name, None);
                        1
                    }
                }
                Dimension::Fixed(size) => *size,
            })
            .collect();

        fn random_ints<T, F: FnMut() -> T>(shape: &[usize], generate: F) -> Value
        where
            Value: From<Tensor<T>>,
        {
            Tensor::from_simple_fn(shape, generate).into()
        }

        let range = range.map(|r| (r.min, r.max));

        // Guess suitable content for the input based on its name. Name-based
        // guesses are skipped if an explicit value range was specified.
        let value = match name {
            // If this is a mask, use all ones on the assumption that we
            // don't want to mask anything out.
            name if range.is_none()
                && name.ends_with("_mask")
                && matches!(dtype, Some(DataType::Int32) | None) =>
            {
                Value::from(Tensor::full(&resolved_shape, 1i32))
            }

            // Inputs such as `token_type_ids`, `position_ids`, `input_ids`.
            // We use zero as a value that is likely to be valid for all
            // of these.
            name if range.is_none()
                && name.ends_with("_ids")
                && matches!(dtype, Some(DataType::Int32) | None) =>
            {
                Value::from(Tensor::<i32>::zeros(&resolved_shape))
            }

            // Optimum can export "merged" transformer models which have two
            // branches. One accepts KV-cache inputs and the other does not.
            // Set this to false as a "safer" value because we don't have
            // cached outputs from a previous run.
            "use_cache_branch"
                if range.is_none() && matches!(dtype, Some(DataType::Int32) | None) =>
            {
                Value::from(Tensor::<i32>::zeros(&resolved_shape))
            }

            // For anything else, random values. The default ranges are
            // intended to be suitable for many models.
            //
            // For int types the float bounds are converted with saturating `as`
            // casts, so a range which is wider than the dtype is narrowed to
            // it.
            _ => match dtype {
                Some(DataType::Float) | None => {
                    let (min, max) = range.unwrap_or((0., 1.));
                    Value::from(Tensor::from_simple_fn(&resolved_shape, || {
                        min + self.rng.f32() * (max - min)
                    }))
                }
                Some(DataType::Int32) => {
                    let (min, max) = range.map_or((0, 255), |(min, max)| (min as i32, max as i32));
                    random_ints(&resolved_shape, || self.rng.i32(min..=max))
                }
                Some(DataType::Int8) => {
                    let (min, max) = range.map_or((0, 127), |(min, max)| (min as i8, max as i8));
                    random_ints(&resolved_shape, || self.rng.i8(min..=max))
                }
                Some(DataType::UInt8) => {
                    let (min, max) = range.map_or((0, 255), |(min, max)| (min as u8, max as u8));
                    random_ints(&resolved_shape, || self.rng.u8(min..=max))
                }
                Some(dtype) => {
                    return Err(GenerateError::UnsupportedDataType(ValueType::Tensor(dtype)));
                }
            },
        };

        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_testing::TestCases;

    use super::*;

    /// Generate a random input with a given name and dtype, using the value
    /// range specified by `range`, if any.
    fn generate(name: &str, dtype: DataType, range: Option<(f32, f32)>) -> Value {
        let range = range.map(|(min, max)| InputRange {
            input_name: name.to_string(),
            min,
            max,
        });
        let mut generator = RandomInputGenerator::new();
        generator
            .generate(
                name,
                Some(ValueType::Tensor(dtype)),
                &[Dimension::Fixed(64)],
                &[],
                range.as_ref(),
                |_dim_name, _dim_size_idx| {},
            )
            .unwrap()
    }

    /// Extract the elements of a tensor value as floats, so that values of
    /// different dtypes can be compared in the same way.
    fn elements(value: &Value) -> Vec<f32> {
        match value {
            Value::FloatTensor(tensor) => tensor.to_vec(),
            Value::Int32Tensor(tensor) => tensor.map(|x| *x as f32).to_vec(),
            Value::Int8Tensor(tensor) => tensor.map(|x| *x as f32).to_vec(),
            Value::UInt8Tensor(tensor) => tensor.map(|x| *x as f32).to_vec(),
            value => panic!("unexpected value type {:?}", value.dtype()),
        }
    }

    #[test]
    fn test_generate_value_range() {
        #[derive(Debug)]
        struct Case {
            dtype: DataType,
            range: Option<(f32, f32)>,

            /// Bounds that all generated values are expected to fall within.
            expected: (f32, f32),
        }

        let cases = [
            // Default ranges, used when no range is specified.
            Case {
                dtype: DataType::Float,
                range: None,
                expected: (0., 1.),
            },
            Case {
                dtype: DataType::Int32,
                range: None,
                expected: (0., 255.),
            },
            Case {
                dtype: DataType::Int8,
                range: None,
                expected: (0., 127.),
            },
            Case {
                dtype: DataType::UInt8,
                range: None,
                expected: (0., 255.),
            },
            // Explicitly specified ranges.
            Case {
                dtype: DataType::Float,
                range: Some((-1.5, 2.5)),
                expected: (-1.5, 2.5),
            },
            Case {
                dtype: DataType::Int32,
                range: Some((5., 7.)),
                expected: (5., 7.),
            },
            Case {
                dtype: DataType::Int8,
                range: Some((-5., 5.)),
                expected: (-5., 5.),
            },
            Case {
                dtype: DataType::UInt8,
                range: Some((200., 255.)),
                expected: (200., 255.),
            },
            // Ranges which are wider than the dtype are narrowed to it.
            Case {
                dtype: DataType::Int8,
                range: Some((-1000., 1000.)),
                expected: (-128., 127.),
            },
            Case {
                dtype: DataType::UInt8,
                range: Some((-1000., 1000.)),
                expected: (0., 255.),
            },
        ];

        cases.test_each(
            |Case {
                 dtype,
                 range,
                 expected,
             }| {
                let (min, max) = *expected;
                let value = generate("x", *dtype, *range);
                let outside: Vec<f32> = elements(&value)
                    .into_iter()
                    .filter(|x| *x < min || *x > max)
                    .collect();
                assert!(
                    outside.is_empty(),
                    "values outside {min}:{max}: {outside:?}"
                );
            },
        )
    }

    #[test]
    fn test_generate_name_heuristics() {
        // Inputs whose names match certain patterns get fixed values which are
        // more likely to be valid than random ones.
        assert_eq!(
            elements(&generate("attention_mask", DataType::Int32, None)),
            [1.; 64]
        );
        assert_eq!(
            elements(&generate("input_ids", DataType::Int32, None)),
            [0.; 64]
        );
        assert_eq!(
            elements(&generate("use_cache_branch", DataType::Int32, None)),
            [0.; 64]
        );

        // An explicitly specified range overrides these heuristics.
        assert_eq!(
            elements(&generate("input_ids", DataType::Int32, Some((5., 5.)))),
            [5.; 64]
        );
    }
}
