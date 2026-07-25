use std::cmp::Ordering;

use crate::name_value::{self, ParseError, Token};

/// Specifies the size for a dynamic input dimension.
#[derive(Clone, Debug, PartialEq)]
pub struct DimSize {
    /// Name of model input. If `None`, this matches all inputs.
    pub input_name: Option<String>,

    /// Name of the dynamically-sized dimension.
    pub dim_name: String,

    /// Dimension size
    pub size: usize,
}

impl DimSize {
    /// Return true if `self` specifies the size for a given input dimension.
    pub fn matches(&self, input_name: &str, dim_name: &str) -> bool {
        match self {
            DimSize {
                input_name: Some(in_name),
                dim_name: dn,
                size: _,
            } if in_name == input_name && dn == dim_name => true,
            DimSize {
                input_name: None,
                dim_name: dn,
                size: _,
            } if dn == dim_name => true,
            _ => false,
        }
    }

    /// Parse a dimension size specifier in the form `dim_name=size` or
    /// `input_name.dim_name=size`.
    pub fn parse(spec: &str) -> Result<DimSize, ParseError> {
        let Some((name_tokens, size_str)) = name_value::split(spec) else {
            return Err(ParseError::new(
                spec,
                "expected <name>=<size> but no '=' was found",
            ));
        };

        let (input_name, dim_name) = match name_tokens.as_slice() {
            [Token::Text(dim)] => (None, dim),
            [Token::Text(input), Token::Dot, Token::Text(dim)] => (Some(input), dim),
            _ => {
                return Err(ParseError::new(spec, "invalid input or dimension name"));
            }
        };

        let size: usize = size_str.parse().map_err(|_| {
            ParseError::new(
                spec,
                "invalid dimension size. Must be a non-negative integer",
            )
        })?;

        Ok(DimSize {
            input_name: input_name.map(|s| s.to_string()),
            dim_name: dim_name.to_string(),
            size,
        })
    }

    /// Sort and de-duplicate entries in `sizes`.
    ///
    /// Entries are sorted with more specific sizes first (ie. those that
    /// specify an input name), then by name.
    pub fn sort_dedup(sizes: &mut Vec<DimSize>) {
        // Sort entries to group duplicates and prioritize those with input names
        // before those without.
        sizes.sort_by(|a, b| match (&a.input_name, &b.input_name) {
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (Some(a_name), Some(b_name)) => match a_name.cmp(b_name) {
                Ordering::Equal => a.dim_name.cmp(&b.dim_name),
                ord => ord,
            },
            (None, None) => a.dim_name.cmp(&b.dim_name),
        });

        // Remove duplicate entries, keeping only the last one.
        // `dedup_by` keeps only the first entry, hence we reverse before and after.
        sizes.reverse();
        sizes.dedup_by(|a, b| a.input_name == b.input_name && a.dim_name == b.dim_name);
        sizes.reverse();
    }
}

#[cfg(test)]
mod tests {
    use rten_testing::TestCases;

    use super::DimSize;
    use crate::name_value::ParseError;

    #[test]
    fn test_parse() {
        #[derive(Debug)]
        struct Case<'a> {
            spec: &'a str,
            expected: Result<DimSize, ParseError>,
        }

        let cases = [
            Case {
                spec: "batch_size=1",
                expected: Ok(DimSize {
                    input_name: None,
                    dim_name: "batch_size".to_string(),
                    size: 1,
                }),
            },
            Case {
                spec: "input_ids.batch_size=1",
                expected: Ok(DimSize {
                    input_name: Some("input_ids".to_string()),
                    dim_name: "batch_size".to_string(),
                    size: 1,
                }),
            },
            Case {
                spec: "foobar",
                expected: Err(ParseError::new(
                    "foobar",
                    "expected <name>=<size> but no '=' was found",
                )),
            },
            Case {
                spec: "a.b.c=1",
                expected: Err(ParseError::new(
                    "a.b.c=1",
                    "invalid input or dimension name",
                )),
            },
            Case {
                spec: "foobar=g",
                expected: Err(ParseError::new(
                    "foobar=g",
                    "invalid dimension size. Must be a non-negative integer",
                )),
            },
            Case {
                spec: "foobar=-1",
                expected: Err(ParseError::new(
                    "foobar=-1",
                    "invalid dimension size. Must be a non-negative integer",
                )),
            },
        ];

        cases.test_each(|Case { spec, expected }| {
            let dim_size = DimSize::parse(spec);
            assert_eq!(dim_size, *expected);
        })
    }

    #[test]
    fn test_matches() {
        let dim_size = DimSize::parse("batch_size=1").unwrap();
        assert!(dim_size.matches("any_input_name", "batch_size"));
        assert!(!dim_size.matches("any_input_name", "other_dim"));

        let dim_size = DimSize::parse("input_name.batch_size=1").unwrap();
        assert!(dim_size.matches("input_name", "batch_size"));
        assert!(!dim_size.matches("other_input_name", "batch_size"));
        assert!(!dim_size.matches("input_name", "other_dim"));
    }

    #[test]
    fn test_sort_dedup() {
        let mut dim_sizes: Vec<DimSize> = [
            DimSize::parse("batch_size=1").unwrap(),
            DimSize::parse("batch_size=2").unwrap(),
            DimSize::parse("specific_input.batch_size=3").unwrap(),
        ]
        .into();

        DimSize::sort_dedup(&mut dim_sizes);

        assert_eq!(
            dim_sizes,
            [
                // Sizes with input names should be listed first.
                DimSize::parse("specific_input.batch_size=3").unwrap(),
                // When there are duplicates, the last entry should be kept.
                DimSize::parse("batch_size=2").unwrap(),
            ]
        );
    }
}
