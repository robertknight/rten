//! Helpers for formatting and printing information about model inputs and
//! outputs.

use std::fmt::{Display, Formatter};
use std::ops::Range;
use std::str::FromStr;

use rten::{Dimension, Model, NodeId, Value, ValueOrView, ValueType};
use rten_tensor::Layout;

/// Format an input or output shape as a `[dim0, dim1, ...]` string, where each
/// dimension is represented by its fixed size or symbolic name.
fn format_shape(shape: &[Dimension]) -> String {
    let dims = shape
        .iter()
        .map(|dim| match dim {
            Dimension::Fixed(value) => value.to_string(),
            Dimension::Symbolic(name) => name.clone(),
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{}]", dims)
}

/// Format a list of numbers into comma-separated ranges.
/// e.g. [1, 2, 3, 10, 12, 13, 14] formats as "1-3, 10, 12-14".
fn format_ranges(numbers: &[i32]) -> String {
    let mut numbers = numbers.to_vec();
    numbers.sort();

    let ranges: Vec<Range<i32>> =
        numbers
            .into_iter()
            .fold(Vec::<Range<i32>>::new(), |mut ranges, num| {
                if let Some(prev_range) = ranges.last_mut()
                    && num == prev_range.end + 1
                {
                    prev_range.end += 1;
                } else {
                    ranges.push(num..num);
                }
                ranges
            });

    ranges
        .iter()
        .map(|r| {
            if r.start == r.end {
                r.start.to_string()
            } else {
                format!("{}-{}", r.start, r.end)
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// Names for a group of inputs which consist of a prefix, optional number and
/// suffix.
///
/// For example "present.0.key" and "present.1.key" would be grouped together,
/// but "present.0.value" would be a separate group.
#[derive(Debug, PartialEq)]
struct GroupName {
    prefix: String,
    suffix: String,
    numbers: Vec<i32>,
}

impl GroupName {
    /// Parse an input name in the form "{prefix}{number}{suffix}" where all
    /// parts are optional.
    fn parse(name: &str) -> Self {
        let Some(num_start) = name.find(|ch: char| ch.is_ascii_digit()) else {
            return Self {
                prefix: name.to_string(),
                suffix: String::new(),
                numbers: Vec::new(),
            };
        };
        let num_end = name[num_start..]
            .find(|ch: char| !ch.is_ascii_digit())
            .map(|offset| num_start + offset)
            .unwrap_or(name.len());

        if let Ok(number) = i32::from_str(&name[num_start..num_end]) {
            Self {
                prefix: name[..num_start].to_string(),
                suffix: name[num_end..].to_string(),
                numbers: [number].into(),
            }
        } else {
            Self {
                prefix: name.to_string(),
                suffix: String::new(),
                numbers: Vec::new(),
            }
        }
    }

    fn matches(&self, other: &GroupName) -> bool {
        self.prefix == other.prefix && self.suffix == other.suffix
    }

    fn merge(&mut self, other: &GroupName) {
        self.numbers.extend(&other.numbers)
    }
}

impl Display for GroupName {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.numbers.as_slice() {
            [] => write!(f, "{}{}", self.prefix, self.suffix),
            [num] => write!(f, "{}{}{}", self.prefix, num, self.suffix),
            numbers => {
                let ranges = format_ranges(numbers);
                // "<prefix>{<range>}<suffix>"
                write!(f, "{}{{{}}}{}", self.prefix, ranges, self.suffix)
            }
        }
    }
}

/// Print a summary of the names and shapes of a list of input or output node IDs.
pub fn print_input_output_list(model: &Model, node_ids: &[NodeId]) {
    struct Group {
        name: GroupName,
        dtype: Option<ValueType>,
        shape: Option<Vec<Dimension>>,
    }

    let mut groups: Vec<Group> = Vec::new();

    for &node_id in node_ids {
        let Some(info) = model.node_info(node_id) else {
            continue;
        };
        let name = info.name().unwrap_or("(unknown)");
        let dtype = info.dtype();
        let shape = info.shape();

        let name = GroupName::parse(name);

        if let Some(group) = groups
            .iter_mut()
            .find(|g| g.name.matches(&name) && g.dtype == dtype && g.shape == shape)
        {
            group.name.merge(&name);
        } else {
            groups.push(Group { name, dtype, shape });
        }
    }

    // Print information about each group of inputs or outputs.
    for g in groups {
        println!(
            "  {}: {} {}",
            g.name,
            g.dtype
                .map(|dt| dt.to_string())
                .unwrap_or("(unknown dtype)".to_string()),
            g.shape
                .map(|dims| format_shape(&dims))
                .unwrap_or("(unknown shape)".to_string())
        );
    }
}

/// Display information about the actual shapes used for model inference.
pub fn print_input_shapes(model: &Model, inputs: &[(NodeId, ValueOrView<'_>)]) {
    struct Group {
        name: GroupName,
        shape: Vec<usize>,
    }

    let mut groups: Vec<Group> = Vec::new();

    for (id, input) in inputs.iter() {
        let info = model.node_info(*id);
        let name = info
            .as_ref()
            .and_then(|ni| ni.name())
            .unwrap_or("(unnamed)");

        let name = GroupName::parse(name);

        if let Some(group) = groups
            .iter_mut()
            .find(|g| g.name.matches(&name) && g.shape == input.shape().as_slice())
        {
            group.name.merge(&name);
        } else {
            groups.push(Group {
                name,
                shape: input.shape().to_vec(),
            });
        }
    }

    for Group { name, shape } in groups {
        println!("  Input \"{name}\" shape {shape:?}");
    }
}

/// Display information about the actual shapes of model outputs.
pub fn print_output_shapes(model: &Model, outputs: &[Value]) {
    struct Group {
        name: GroupName,
        shape: Vec<usize>,
    }

    let mut groups: Vec<Group> = Vec::new();

    for (id, output) in model.output_ids().iter().zip(outputs) {
        let info = model.node_info(*id);
        let name = info
            .as_ref()
            .and_then(|ni| ni.name())
            .unwrap_or("(unnamed)");
        let name = GroupName::parse(name);

        if let Some(group) = groups
            .iter_mut()
            .find(|g| g.name.matches(&name) && g.shape == output.shape().as_slice())
        {
            group.name.merge(&name);
        } else {
            groups.push(Group {
                name,
                shape: output.shape().to_vec(),
            });
        }
    }

    for Group { name, shape } in groups {
        println!("  Output \"{name}\" shape {shape:?}");
    }
}

#[cfg(test)]
mod tests {
    use super::GroupName;

    #[test]
    fn test_group_name_parse() {
        // Single number
        assert_eq!(
            GroupName::parse("present.12.key"),
            GroupName {
                prefix: "present.".to_string(),
                suffix: ".key".to_string(),
                numbers: vec![12],
            }
        );

        // Multiple numbers
        assert_eq!(
            GroupName::parse("present.12.5.key"),
            GroupName {
                prefix: "present.".to_string(),
                suffix: ".5.key".to_string(),
                numbers: vec![12],
            }
        );

        // No numbers
        assert_eq!(
            GroupName::parse("input_ids"),
            GroupName {
                prefix: "input_ids".to_string(),
                suffix: String::new(),
                numbers: Vec::new(),
            }
        );
    }

    #[test]
    fn test_group_name_to_string() {
        // No numbers
        let g = GroupName {
            prefix: "present".to_string(),
            suffix: ".key".to_string(),
            numbers: [].into(),
        };
        assert_eq!(g.to_string(), "present.key");

        // Single number
        let g = GroupName {
            prefix: "present.".to_string(),
            suffix: ".key".to_string(),
            numbers: [1].into(),
        };
        assert_eq!(g.to_string(), "present.1.key");

        // Multiple numbers
        let g = GroupName {
            prefix: "present.".to_string(),
            suffix: ".key".to_string(),
            numbers: [14, 1, 2, 3, 10, 12, 13].into(),
        };
        assert_eq!(g.to_string(), "present.{1-3,10,12-14}.key");
    }
}
