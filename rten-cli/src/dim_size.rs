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
        let tokens = tokenize(spec);
        let Some(eq_pos) = tokens.iter().position(|tok| matches!(tok, Token::Equals)) else {
            return Err(ParseError::new(
                spec,
                ParseErrorKind::InvalidFormat {
                    message: "expected <name>=<size> but no '=' was found".into(),
                },
            ));
        };

        let (name_spec, size_spec) = tokens.split_at(eq_pos);

        let [Token::Equals, Token::Text(size_str)] = size_spec else {
            return Err(ParseError::new(
                spec,
                ParseErrorKind::InvalidFormat {
                    message: "expected specifier to end with '=<size>'".into(),
                },
            ));
        };

        let (input_name, dim_name) = match name_spec {
            [Token::Text(dim)] => (None, dim),
            [Token::Text(input), Token::Dot, Token::Text(dim)] => (Some(input), dim),
            _ => {
                return Err(ParseError::new(spec, ParseErrorKind::InvalidName));
            }
        };

        let size: usize = size_str
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

enum Token {
    Equals,
    Dot,
    Text(String),
}

fn tokenize(spec: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut in_quote = false;

    for ch in spec.chars() {
        match ch {
            '=' if !in_quote => {
                tokens.push(Token::Equals);
            }
            '.' if !in_quote => {
                tokens.push(Token::Dot);
            }
            '"' => in_quote = !in_quote,
            ch => {
                if let Some(tok) = tokens.last_mut()
                    && let Token::Text(text) = tok
                {
                    text.push(ch);
                } else {
                    tokens.push(Token::Text(ch.into()));
                }
            }
        }
    }

    tokens
}

#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::enum_variant_names)] // Don't warn about all variants having "Invalid" prefix.
enum ParseErrorKind {
    /// Dimension size spec doesn't match "name=size"
    InvalidFormat { message: String },
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
        match &self.kind {
            ParseErrorKind::InvalidFormat { message } => write!(
                fmt,
                "invalid format for dimension size spec \"{}\": {}",
                self.spec, message
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
                spec: "x.\"dim.name\"=1",
                expected: Ok(DimSize {
                    input_name: Some("x".to_string()),
                    dim_name: "dim.name".to_string(),
                    size: 1,
                }),
            },
            Case {
                spec: "\"input.name\".dim=1",
                expected: Ok(DimSize {
                    input_name: Some("input.name".to_string()),
                    dim_name: "dim".to_string(),
                    size: 1,
                }),
            },
            Case {
                spec: "foobar",
                expected: Err(ParseError::new(
                    "foobar",
                    ParseErrorKind::InvalidFormat {
                        message: "expected <name>=<size> but no '=' was found".into(),
                    },
                )),
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
