use std::cmp::Ordering;

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
        let parts: Vec<&str> = spec.split('=').collect();
        let (name_spec, size_spec) = match parts[..] {
            [name, size] => (name, size),
            _ => {
                return Err(ParseError::new(spec, ParseErrorKind::InvalidFormat));
            }
        };

        let name_parts: Vec<_> = name_spec.split('.').collect();
        let (input_name, dim_name) = match &name_parts[..] {
            [dim_name] => (None, dim_name),
            [input_name, dim_name] => (Some(input_name), dim_name),
            _ => {
                return Err(ParseError::new(spec, ParseErrorKind::InvalidName));
            }
        };

        let size: usize = size_spec
            .parse()
            .map_err(|_| ParseError::new(spec, ParseErrorKind::InvalidSize))?;

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

#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::enum_variant_names)] // Don't warn about all variants having "Invalid" prefix.
enum ParseErrorKind {
    /// Dimension size spec doesn't match "name=size"
    InvalidFormat,
    /// Dimension size spec has an invalid input or dimension name
    InvalidName,
    /// Dimension size spec has an invalid size
    InvalidSize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParseError {
    spec: String,
    kind: ParseErrorKind,
}

impl ParseError {
    fn new(spec: &str, kind: ParseErrorKind) -> ParseError {
        ParseError {
            spec: spec.to_string(),
            kind,
        }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            ParseErrorKind::InvalidFormat => write!(
                fmt,
                "invalid format for dimension size spec \"{}\". expected \"dim_name=size\" or \"input_name.dim_name=size\"",
                self.spec
            ),
            ParseErrorKind::InvalidName => {
                write!(fmt, "invalid name in dimension size spec \"{}\"", self.spec)
            }
            ParseErrorKind::InvalidSize => write!(
                fmt,
                "invalid dimension size in \"{}\". Must be a non-negative integer.",
                self.spec
            ),
        }
    }
}

impl std::error::Error for ParseError {}

#[cfg(test)]
mod tests {
    use rten_testing::TestCases;

    use super::{DimSize, ParseError, ParseErrorKind};

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
                expected: Err(ParseError::new("foobar", ParseErrorKind::InvalidFormat)),
            },
            Case {
                spec: "foobar=g",
                expected: Err(ParseError::new("foobar=g", ParseErrorKind::InvalidSize)),
            },
            Case {
                spec: "foobar=-1",
                expected: Err(ParseError::new("foobar=-1", ParseErrorKind::InvalidSize)),
            },
        ];

        cases.test_each(|Case { spec, expected }| {
            let dim_size = DimSize::parse(&spec);
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
