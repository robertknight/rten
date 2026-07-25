//! Parsing helpers for command line specifiers in the form `<name>=<value>`.

/// A token in the name part of a `<name>=<value>` specifier.
#[derive(Debug, PartialEq)]
pub enum Token {
    /// A `.` separator between parts of a name.
    Dot,
    /// A run of literal characters, with any quoting removed.
    Text(String),
}

/// Split a specifier in the form `<name>=<value>` into the tokenized name and
/// the raw value.
///
/// Parts of the name may be quoted with `"` in order to include `.` or `=`
/// characters (eg. `"input.name".dim=3`).
///
/// Returns `None` if the specifier contains no unquoted `=`.
pub fn split(spec: &str) -> Option<(Vec<Token>, &str)> {
    let mut tokens = Vec::new();
    let mut in_quote = false;

    for (pos, ch) in spec.char_indices() {
        match ch {
            '=' if !in_quote => {
                return Some((tokens, &spec[pos + 1..]));
            }
            '.' if !in_quote => {
                tokens.push(Token::Dot);
            }
            '"' => in_quote = !in_quote,
            ch => {
                if let Some(Token::Text(text)) = tokens.last_mut() {
                    text.push(ch);
                } else {
                    tokens.push(Token::Text(ch.into()));
                }
            }
        }
    }

    None
}

/// Error parsing a `<name>=<value>` specifier.
#[derive(Clone, Debug, PartialEq)]
pub struct ParseError {
    spec: String,
    message: String,
}

impl ParseError {
    pub fn new(spec: &str, message: impl Into<String>) -> ParseError {
        ParseError {
            spec: spec.to_string(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(fmt, "invalid specifier \"{}\": {}", self.spec, self.message)
    }
}

impl std::error::Error for ParseError {}

#[cfg(test)]
mod tests {
    use rten_testing::TestCases;

    use super::{Token, split};

    fn text(text: &str) -> Token {
        Token::Text(text.to_string())
    }

    #[test]
    fn test_split() {
        #[derive(Debug)]
        struct Case<'a> {
            spec: &'a str,
            expected: Option<(Vec<Token>, &'a str)>,
        }

        let cases = [
            Case {
                spec: "name=value",
                expected: Some((vec![text("name")], "value")),
            },
            // Periods separate parts of the name.
            Case {
                spec: "a.b=1",
                expected: Some((vec![text("a"), Token::Dot, text("b")], "1")),
            },
            // Quoting allows periods and `=` to be used in a name.
            Case {
                spec: "\"a.b\"=1",
                expected: Some((vec![text("a.b")], "1")),
            },
            Case {
                spec: "\"a=b\".c=1",
                expected: Some((vec![text("a=b"), Token::Dot, text("c")], "1")),
            },
            // Only the first unquoted `=` separates the name and value. The
            // value is returned unparsed.
            Case {
                spec: "name=1.5=2",
                expected: Some((vec![text("name")], "1.5=2")),
            },
            // Empty names and values.
            Case {
                spec: "=1",
                expected: Some((Vec::new(), "1")),
            },
            Case {
                spec: "name=",
                expected: Some((vec![text("name")], "")),
            },
            // Specifiers with no unquoted `=`.
            Case {
                spec: "name",
                expected: None,
            },
            Case {
                spec: "\"a=b\"",
                expected: None,
            },
        ];

        cases.test_each(|Case { spec, expected }| {
            assert_eq!(split(spec), *expected);
        })
    }
}
