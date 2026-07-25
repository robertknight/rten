use crate::name_value::{self, ParseError, Token};

/// Specifies the range of randomly generated values for a model input.
#[derive(Clone, Debug, PartialEq)]
pub struct InputRange {
    /// Name of model input.
    pub input_name: String,

    /// Minimum generated value.
    pub min: f32,

    /// Maximum generated value.
    pub max: f32,
}

impl InputRange {
    /// Return true if `self` specifies the value range for a given input.
    pub fn matches(&self, input_name: &str) -> bool {
        self.input_name == input_name
    }

    /// Parse a value range specifier in the form `input_name=min:max`.
    pub fn parse(spec: &str) -> Result<InputRange, ParseError> {
        let Some((name_tokens, range_str)) = name_value::split(spec) else {
            return Err(ParseError::new(
                spec,
                "expected <name>=<min>:<max> but no '=' was found",
            ));
        };

        let input_name =
            join_name(&name_tokens).ok_or_else(|| ParseError::new(spec, "invalid input name"))?;

        let Some((min_str, max_str)) = range_str.split_once(':') else {
            return Err(ParseError::new(
                spec,
                "expected <min>:<max> but no ':' was found",
            ));
        };

        let parse_value = |value: &str| {
            value
                .parse::<f32>()
                .map_err(|_| ParseError::new(spec, "invalid min or max value. Must be a number"))
        };
        let min = parse_value(min_str)?;
        let max = parse_value(max_str)?;

        if !min.is_finite() || !max.is_finite() || min > max {
            return Err(ParseError::new(
                spec,
                "invalid value range. Must be finite with min <= max",
            ));
        }

        Ok(InputRange {
            input_name,
            min,
            max,
        })
    }

    /// Sort and de-duplicate entries in `ranges`.
    ///
    /// Entries are sorted by input name.
    pub fn sort_dedup(ranges: &mut Vec<InputRange>) {
        // Sort entries to group duplicates.
        ranges.sort_by(|a, b| a.input_name.cmp(&b.input_name));

        // Remove duplicate entries, keeping only the last one.
        // `dedup_by` keeps only the first entry, hence we reverse before and after.
        ranges.reverse();
        ranges.dedup_by(|a, b| a.input_name == b.input_name);
        ranges.reverse();
    }
}

/// Join the tokens of a name into a single string, or `None` if the name is
/// empty.
fn join_name(tokens: &[Token]) -> Option<String> {
    if tokens.is_empty() {
        return None;
    }
    let name = tokens.iter().fold(String::new(), |mut name, token| {
        match token {
            Token::Dot => name.push('.'),
            Token::Text(text) => name.push_str(text),
        }
        name
    });
    Some(name)
}

#[cfg(test)]
mod tests {
    use rten_testing::TestCases;

    use super::InputRange;
    use crate::name_value::ParseError;

    #[test]
    fn test_parse() {
        #[derive(Debug)]
        struct Case<'a> {
            spec: &'a str,
            expected: Result<InputRange, ParseError>,
        }

        let cases = [
            Case {
                spec: "input_ids=0:1000",
                expected: Ok(InputRange {
                    input_name: "input_ids".to_string(),
                    min: 0.,
                    max: 1000.,
                }),
            },
            // Non-integer and negative bounds.
            Case {
                spec: "x=-1.5:1.5",
                expected: Ok(InputRange {
                    input_name: "x".to_string(),
                    min: -1.5,
                    max: 1.5,
                }),
            },
            // Input names containing periods, quoted and unquoted.
            Case {
                spec: "\"past.0.key\"=0:1",
                expected: Ok(InputRange {
                    input_name: "past.0.key".to_string(),
                    min: 0.,
                    max: 1.,
                }),
            },
            Case {
                spec: "past.0.key=0:1",
                expected: Ok(InputRange {
                    input_name: "past.0.key".to_string(),
                    min: 0.,
                    max: 1.,
                }),
            },
            Case {
                spec: "foobar",
                expected: Err(ParseError::new(
                    "foobar",
                    "expected <name>=<min>:<max> but no '=' was found",
                )),
            },
            Case {
                spec: "x=5",
                expected: Err(ParseError::new(
                    "x=5",
                    "expected <min>:<max> but no ':' was found",
                )),
            },
            Case {
                spec: "=0:1",
                expected: Err(ParseError::new("=0:1", "invalid input name")),
            },
            Case {
                spec: "x=a:b",
                expected: Err(ParseError::new(
                    "x=a:b",
                    "invalid min or max value. Must be a number",
                )),
            },
            Case {
                spec: "x=10:0",
                expected: Err(ParseError::new(
                    "x=10:0",
                    "invalid value range. Must be finite with min <= max",
                )),
            },
            Case {
                spec: "x=0:inf",
                expected: Err(ParseError::new(
                    "x=0:inf",
                    "invalid value range. Must be finite with min <= max",
                )),
            },
        ];

        cases.test_each(|Case { spec, expected }| {
            let range = InputRange::parse(spec);
            assert_eq!(range, *expected);
        })
    }

    #[test]
    fn test_matches() {
        let range = InputRange::parse("input_ids=0:1").unwrap();
        assert!(range.matches("input_ids"));
        assert!(!range.matches("other_input"));
    }

    #[test]
    fn test_sort_dedup() {
        let mut ranges: Vec<InputRange> = [
            InputRange::parse("input_ids=0:1").unwrap(),
            InputRange::parse("input_ids=0:2").unwrap(),
            InputRange::parse("attention_mask=0:3").unwrap(),
        ]
        .into();

        InputRange::sort_dedup(&mut ranges);

        assert_eq!(
            ranges,
            [
                InputRange::parse("attention_mask=0:3").unwrap(),
                // When there are duplicates, the last entry should be kept.
                InputRange::parse("input_ids=0:2").unwrap(),
            ]
        );
    }
}
