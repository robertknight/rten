//! Parser for Einsum equations.

use std::error::Error;
use std::fmt;

/// Error produced when parsing an invalid Einsum equation with
/// [`EinsumExpr::parse`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParseError {
    /// An input term contains characters other than lowercase ASCII letters
    /// or more than one ellipsis (`...`).
    InvalidInputTerm,
    /// The output term contains characters other than lowercase ASCII letters
    /// or more than one ellipsis (`...`).
    InvalidOutputTerm,
    /// The output term contains a repeated label.
    RepeatedOutputLabels,
    /// The output term contains a label which does not appear in any input
    /// term.
    UnknownOutputLabel,
}

impl ParseError {
    /// Return a human-readable description of the error.
    pub fn as_str(&self) -> &'static str {
        match self {
            ParseError::InvalidInputTerm => "Input term is invalid",
            ParseError::InvalidOutputTerm => "Output term is invalid",
            ParseError::RepeatedOutputLabels => "Einsum output term contains repeated labels",
            ParseError::UnknownOutputLabel => {
                "Einsum output term contains a label not present in any input term"
            }
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Error for ParseError {}

/// A parsed equation for an Einsum operator.
///
/// Einsum expressions have the form `abc,def,...->xyz` where the `->xyz` part
/// is optional. If omitted, it is inferred as the alphabetically ordered set of
/// letters from the left hand side that do not repeat.
///
/// See <https://onnx.ai/onnx/operators/onnx__Einsum.html>.
#[derive(Clone, Debug, PartialEq)]
pub struct EinsumExpr {
    /// Terms describing the dimension labels of each input.
    pub inputs: Vec<String>,
    /// Term describing the dimension labels of the output.
    pub output: String,
}

impl EinsumExpr {
    /// Parse an Einsum expression.
    ///
    /// An empty left-hand side (eg. in "" or "->") is treated as a single
    /// empty term, corresponding to a scalar input.
    pub fn parse(expr: &str) -> Result<EinsumExpr, ParseError> {
        let mut parts = expr.trim().splitn(2, "->").map(|part| part.trim());

        let lhs = parts.next().unwrap_or_default();

        let inputs: Vec<String> = lhs
            .split(',')
            .map(|term| non_whitespace_chars(term).collect())
            .collect();
        if inputs.iter().any(|term| !is_valid_term(term)) {
            return Err(ParseError::InvalidInputTerm);
        }

        let output: String = match parts.next() {
            Some(rhs) => non_whitespace_chars(rhs).collect(),
            None => default_output(&inputs),
        };

        if !is_valid_term(&output) {
            return Err(ParseError::InvalidOutputTerm);
        }
        if contains_repeated_chars(&output) {
            return Err(ParseError::RepeatedOutputLabels);
        }
        if output
            .chars()
            .any(|c| c.is_ascii_lowercase() && !inputs.iter().any(|term| term.contains(c)))
        {
            return Err(ParseError::UnknownOutputLabel);
        }

        Ok(EinsumExpr { inputs, output })
    }

    /// Check operator inputs are compatible with the parsed equation and
    /// calculate the number of dimensions represented by `...` in input terms.
    ///
    /// Returns the number of dimensions represented by `...` in input terms.
    pub fn validate_inputs(
        &self,
        input_ndims: impl ExactSizeIterator<Item = Option<usize>>,
    ) -> Result<usize, ValidateError> {
        if input_ndims.len() != self.inputs.len() {
            return Err(ValidateError::IncorrectInputCount);
        }

        let mut broadcast_ndim = None;
        for (term, ndim) in self.inputs.iter().zip(input_ndims) {
            let has_ellipsis = term.contains("...");
            let Some(ndim) = ndim else {
                // Without the rank we can neither rank-check a non-ellipsis
                // term nor count an ellipsis term's broadcast dimensions.
                if has_ellipsis {
                    return Err(ValidateError::UnknownRank);
                }
                continue;
            };

            // `term` contains at most one "..." occurrence (validated during
            // parsing). For an ellipsis term the remaining labels are the
            // non-broadcast dimensions, and the input may have additional dims
            // which the ellipsis stands for. A term without an ellipsis must
            // have exactly one label per dimension.
            let non_broadcast = if has_ellipsis {
                term.len() - "...".len()
            } else {
                term.len()
            };
            let rank_ok = if has_ellipsis {
                ndim >= non_broadcast
            } else {
                ndim == non_broadcast
            };
            if !rank_ok {
                return Err(ValidateError::RankMismatch);
            }

            if ndim > MAX_DIMS {
                return Err(ValidateError::TooManyDims);
            }

            if has_ellipsis {
                let this_broadcast = ndim - non_broadcast;
                match broadcast_ndim {
                    None => broadcast_ndim = Some(this_broadcast),
                    Some(b) if b == this_broadcast => {}
                    Some(_) => return Err(ValidateError::BroadcastMismatch),
                }
            }
        }
        Ok(broadcast_ndim.unwrap_or(0))
    }
}

/// Error produced by [`EinsumExpr::validate_inputs`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValidateError {
    /// The number of inputs does not match the number of input terms in the
    /// equation.
    IncorrectInputCount,
    /// An input term contains an ellipsis but the corresponding input's rank is
    /// unknown, so the broadcast count cannot be determined.
    UnknownRank,
    /// An input's rank is incompatible with its term.
    RankMismatch,
    /// An input has more than [`MAX_DIMS`] dimensions.
    TooManyDims,
    /// Inputs whose terms contain an ellipsis disagree on the number of
    /// dimensions the ellipsis represents.
    BroadcastMismatch,
}

/// Compute the default output term for an equation with no explicit `->` part.
///
/// The default consists of (1) the broadcast (ellipsis) dimensions if any input
/// has an ellipsis, followed by (2) the lowercase-letter labels which appear
/// exactly once across all input terms, in alphabetical order.
fn default_output(inputs: &[String]) -> String {
    const N_LETTERS: usize = 26;

    // Count occurrences of each lowercase ASCII letter.
    let mut char_count = [0; N_LETTERS];
    for ch in inputs
        .iter()
        .flat_map(|term| term.chars().filter(|c| c.is_ascii_lowercase()))
    {
        char_count[(ch as u8 - b'a') as usize] += 1;
    }

    // Generate output as the ellipsis (if any) followed by alphabetically
    // ordered letters which appear only once in the input.
    let mut output = String::with_capacity(N_LETTERS);
    if inputs.iter().any(|term| term.contains("...")) {
        output.push_str("...");
    }
    for i in 0..N_LETTERS as u8 {
        if char_count[i as usize] == 1 {
            output.push((b'a' + i) as char);
        }
    }
    output
}

/// Return true if `term` is a valid sequence of dimension labels: lowercase
/// ASCII letters with at most one ellipsis (`...`).
fn is_valid_term(term: &str) -> bool {
    if let Some((lhs, rhs)) = term.split_once("...") {
        is_valid_term(lhs) && !rhs.contains("...") && is_valid_term(rhs)
    } else {
        term.chars().all(|c| c.is_ascii_lowercase())
    }
}

/// Maximum number of dimensions an ellipsis (`...`) can represent.
///
/// [`expand_ellipsis`] replaces an ellipsis with single ASCII digits
/// (`'0'..='9'`) used as placeholder labels, which limits the count to 10.
pub const MAX_DIMS: usize = 10;

/// Replace an ellipsis representing a fixed number of dimensions with a sequence
/// of digit labels.
///
/// Digits are used because they are not allowed as dimension labels in input
/// Einsum equations.
///
/// eg. `expand_ellipsis("i...j", 3)` returns `"i012j"`.
pub fn expand_ellipsis(term: &str, broadcast_ndim: usize) -> String {
    assert!(broadcast_ndim <= MAX_DIMS);
    if let Some((lhs, rhs)) = term.split_once("...") {
        lhs.chars()
            .chain((0..broadcast_ndim as u8).map(|i| (b'0' + i) as char))
            .chain(rhs.chars())
            .collect()
    } else {
        term.to_string()
    }
}

fn non_whitespace_chars(s: &str) -> impl Iterator<Item = char> + '_ {
    s.chars().filter(|c| !c.is_ascii_whitespace())
}

fn contains_repeated_chars(term: &str) -> bool {
    term.chars()
        .filter(|c| *c != '.')
        .any(|c1| term.chars().filter(|c2| c1 == *c2).count() > 1)
}

#[cfg(test)]
mod tests {
    use rten_testing::TestCases;

    use super::{EinsumExpr, ParseError, ValidateError, expand_ellipsis};

    fn expr(inputs: &[&str], output: &str) -> EinsumExpr {
        EinsumExpr {
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            output: output.to_string(),
        }
    }

    #[test]
    fn test_parse() {
        #[derive(Debug)]
        struct Case<'a> {
            equation: &'a str,
            expected: Result<EinsumExpr, ParseError>,
        }

        let cases = [
            // Explicit output.
            Case {
                equation: "ij->ij",
                expected: Ok(expr(&["ij"], "ij")),
            },
            // Transpose.
            Case {
                equation: "ij->ji",
                expected: Ok(expr(&["ij"], "ji")),
            },
            // Multiple input terms.
            Case {
                equation: "ij,jk->ik",
                expected: Ok(expr(&["ij", "jk"], "ik")),
            },
            // Whitespace is stripped from terms.
            Case {
                equation: " i j , j k -> i k ",
                expected: Ok(expr(&["ij", "jk"], "ik")),
            },
            // Implicit output: unique labels in alphabetical order, repeated
            // labels (j) dropped.
            Case {
                equation: "ij,jk",
                expected: Ok(expr(&["ij", "jk"], "ik")),
            },
            // Implicit output with an ellipsis: "..." is prepended.
            Case {
                equation: "...ij",
                expected: Ok(expr(&["...ij"], "...ij")),
            },
            // Implicit output that reduces every label to a scalar.
            Case {
                equation: "i,i",
                expected: Ok(expr(&["i", "i"], "")),
            },
            // Ellipsis in input and output terms.
            Case {
                equation: "...ij->...ji",
                expected: Ok(expr(&["...ij"], "...ji")),
            },
            // Empty equation. The left-hand side is a single empty term,
            // corresponding to a scalar input.
            Case {
                equation: "",
                expected: Ok(expr(&[""], "")),
            },
            // Whitespace-only equation.
            Case {
                equation: "  ",
                expected: Ok(expr(&[""], "")),
            },
            // Explicit form with a single empty (scalar) term.
            Case {
                equation: "->",
                expected: Ok(expr(&[""], "")),
            },
            // Upper-case letters in an input term.
            Case {
                equation: "IJ,JK",
                expected: Err(ParseError::InvalidInputTerm),
            },
            // A period that is not part of an ellipsis.
            Case {
                equation: "i.j",
                expected: Err(ParseError::InvalidInputTerm),
            },
            // More than one ellipsis in a term.
            Case {
                equation: "i...j...",
                expected: Err(ParseError::InvalidInputTerm),
            },
            // Invalid output term.
            Case {
                equation: "ij,jk->IK",
                expected: Err(ParseError::InvalidOutputTerm),
            },
            // Repeated labels in the output term.
            Case {
                equation: "ij->ii",
                expected: Err(ParseError::RepeatedOutputLabels),
            },
            // Output label which does not appear in any input term.
            Case {
                equation: "ij->ik",
                expected: Err(ParseError::UnknownOutputLabel),
            },
        ];

        cases.test_each(|case| {
            assert_eq!(EinsumExpr::parse(case.equation), case.expected);
        });
    }

    #[test]
    fn test_validate_inputs() {
        #[derive(Debug)]
        struct Case<'a> {
            equation: &'a str,
            input_ndims: &'a [Option<usize>],
            expected: Result<usize, ValidateError>,
        }

        let cases = [
            // No ellipsis: broadcast count is zero when ranks match the terms.
            Case {
                equation: "ij,jk->ik",
                input_ndims: &[Some(2), Some(2)],
                expected: Ok(0),
            },
            // A non-ellipsis term must have one label per input dimension.
            Case {
                equation: "ij,jk->ik",
                input_ndims: &[Some(2), Some(3)],
                expected: Err(ValidateError::RankMismatch),
            },
            // A single ellipsis input determines the count.
            Case {
                equation: "...ij->...ji",
                input_ndims: &[Some(4)],
                expected: Ok(2),
            },
            // Multiple ellipsis inputs which agree.
            Case {
                equation: "...ij,...jk->...ik",
                input_ndims: &[Some(4), Some(4)],
                expected: Ok(2),
            },
            // Ellipsis standing for zero dimensions.
            Case {
                equation: "...ij->...ij",
                input_ndims: &[Some(2)],
                expected: Ok(0),
            },
            // Unknown rank for an ellipsis input.
            Case {
                equation: "...ij->...ji",
                input_ndims: &[None],
                expected: Err(ValidateError::UnknownRank),
            },
            // Unknown rank for a non-ellipsis input doesn't matter.
            Case {
                equation: "...ij,jk->...ik",
                input_ndims: &[Some(3), None],
                expected: Ok(1),
            },
            // Input has fewer dims than its ellipsis term's non-ellipsis labels.
            Case {
                equation: "...ij->...ji",
                input_ndims: &[Some(1)],
                expected: Err(ValidateError::RankMismatch),
            },
            // Ellipsis inputs disagree on the broadcast count.
            Case {
                equation: "...,...->...",
                input_ndims: &[Some(1), Some(2)],
                expected: Err(ValidateError::BroadcastMismatch),
            },
            // Number of inputs doesn't match the number of terms.
            Case {
                equation: "ij,jk->ik",
                input_ndims: &[Some(2)],
                expected: Err(ValidateError::IncorrectInputCount),
            },
            // An input has more than `MAX_DIMS` dimensions (non-ellipsis term).
            Case {
                equation: "abcdefghijk",
                input_ndims: &[Some(11)],
                expected: Err(ValidateError::TooManyDims),
            },
            // An input has more than `MAX_DIMS` dimensions (ellipsis term).
            Case {
                equation: "...",
                input_ndims: &[Some(11)],
                expected: Err(ValidateError::TooManyDims),
            },
        ];

        cases.test_each(|case| {
            let expr = EinsumExpr::parse(case.equation).unwrap();
            assert_eq!(
                expr.validate_inputs(case.input_ndims.iter().copied()),
                case.expected
            );
        });
    }

    #[test]
    fn test_expand_ellipsis() {
        #[derive(Debug)]
        struct Case<'a> {
            term: &'a str,
            broadcast_ndim: usize,
            expected: &'a str,
        }

        let cases = [
            // Ellipsis between labels expands to digit placeholders.
            Case {
                term: "i...j",
                broadcast_ndim: 3,
                expected: "i012j",
            },
            // An ellipsis standing for zero dimensions is removed.
            Case {
                term: "i...j",
                broadcast_ndim: 0,
                expected: "ij",
            },
            // A term with no ellipsis is unchanged.
            Case {
                term: "ij",
                broadcast_ndim: 2,
                expected: "ij",
            },
            // Leading ellipsis.
            Case {
                term: "...ij",
                broadcast_ndim: 2,
                expected: "01ij",
            },
        ];

        cases.test_each(|case| {
            assert_eq!(
                expand_ellipsis(case.term, case.broadcast_ndim),
                case.expected
            );
        });
    }
}
