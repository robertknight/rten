use std::collections::HashMap;
use std::fmt;

use smallvec::SmallVec;

/// Trait for text data table sources.
///
/// Tables can be formatted using [Table::display] to get a wrapper that
/// implements [Display].
trait Table {
    /// Return the number of rows in this table.
    fn rows(&self) -> usize;

    /// Return the column headings for this table. This also determines the
    /// number of columns.
    fn headings(&self) -> &[&str];

    /// Return the text for a given table cell.
    fn cell(&self, row: usize, col: usize) -> String;

    /// Return the maximum number of characters used by any entry in column
    /// `col`.
    fn max_width(&self, col: usize) -> usize {
        (0..self.rows())
            .map(|row| self.cell(row, col).len())
            .max()
            .unwrap_or(0)
    }

    /// Return a wrapper around this table which implements [Display].
    fn display(&self, indent: usize) -> DisplayTable<Self>
    where
        Self: Sized,
    {
        DisplayTable {
            table: self,
            indent,
        }
    }
}

struct DisplayTable<'a, T: Table> {
    table: &'a T,
    indent: usize,
}

impl<'a, T: Table> fmt::Display for DisplayTable<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        let col_widths: Vec<_> = self
            .table
            .headings()
            .iter()
            .enumerate()
            .map(|(col, heading)| heading.len().max(self.table.max_width(col)))
            .collect();
        let col_padding = 2;

        let write_indent =
            |f: &mut fmt::Formatter<'_>| write!(f, "{:<indent$}", "", indent = self.indent);

        // Write table head
        write_indent(f)?;
        for (col_heading, width) in self.table.headings().iter().zip(col_widths.iter()) {
            write!(f, "{:<width$}", col_heading, width = width + col_padding)?;
        }
        writeln!(f)?;
        write_indent(f)?;
        for width in &col_widths {
            write!(
                f,
                "{:-<width$}{:<padding$}",
                "",
                "",
                width = width,
                padding = col_padding
            )?;
        }
        writeln!(f)?;

        // Write table body
        for row in 0..self.table.rows() {
            write_indent(f)?;
            for (col, width) in col_widths.iter().enumerate() {
                write!(
                    f,
                    "{:<width$}",
                    self.table.cell(row, col),
                    width = width + col_padding
                )?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

/// Timing statistics gathered from a graph run.
pub struct RunTiming<'a> {
    /// Records of graph step execution times
    pub records: &'a [TimingRecord<'a>],

    /// Total time spent in memory allocations or de-allocations that were
    /// tracked.
    pub alloc_time: f32,

    /// Total time for the graph run.
    pub total_time: f32,
}

impl<'a> RunTiming<'a> {
    fn total_op_time(&self) -> f32 {
        self.records.iter().map(|tr| tr.elapsed_micros).sum::<f32>() / 1000.0
    }

    /// Return a struct that formats output with the given options.
    pub fn display(&self, sort: TimingSort, include_shapes: bool) -> impl fmt::Display + '_ {
        FormattedRunTiming {
            timing: self,
            sort,
            include_shapes,
        }
    }
}

impl<'a> fmt::Display for RunTiming<'a> {
    /// Format timings with the default sort order (see [TimingSort]).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        self.display(TimingSort::ByTime, false /* include_shapes */)
            .fmt(f)
    }
}

/// Record of execution time for a single (operator, input_shape) combination.
struct TimingByShapeRecord {
    /// Formatted representation of input shapes
    shape: String,

    /// Total elapsed time for this operator with a given shape
    total_ms: f32,

    /// Number of times the operator was run with this shape
    count: usize,

    /// Total number of elements in all inputs, or the sum of the product
    /// of each input's shape.
    ///
    /// Note that some operator inputs are parameters rather than data
    /// (eg. permutation for a transpose operator), which can distort figures.
    /// It would be better if operators could provide a metric for the amount
    /// of work required for a given set of inputs.
    ///
    /// This count is for each run of the operator.
    input_elements: usize,

    /// Name of an example operator node in the graph that contributed to this
    /// record.
    node_name: String,
}

/// [Display]-able table containing a breakdown of operator execution time
/// by input shape.
struct TimingByShapeTable {
    rows: Vec<TimingByShapeRecord>,
}

impl Table for TimingByShapeTable {
    fn rows(&self) -> usize {
        self.rows.len()
    }

    fn headings(&self) -> &[&str] {
        &[
            "Shape",
            "Count",
            "Mean (ms)",
            "Total (ms)",
            "ns/input elem",
            "Example node",
        ]
    }

    fn cell(&self, row: usize, col: usize) -> String {
        let row = self.rows.get(row).expect("invalid row");
        match col {
            0 => row.shape.clone(),
            1 => row.count.to_string(),
            2 => format!("{:.3}", row.total_ms / row.count as f32),
            3 => format!("{:.3}", row.total_ms),
            4 => format!(
                "{:.3}",
                (row.total_ms * 1_000_000.0) / (row.input_elements * row.count) as f32
            ),
            5 => row.node_name.clone(),
            _ => panic!("invalid column"),
        }
    }
}

/// Format a tensor shape as a "[dim_0, dim_1, ...]" string.
fn shape_to_string(shape: &[usize]) -> String {
    let mut shape_str = String::new();
    shape_str.push('[');
    for (i, size) in shape.iter().enumerate() {
        if i > 0 {
            shape_str.push_str(", ");
        }
        shape_str.push_str(&format!("{}", size));
    }
    shape_str.push(']');
    shape_str
}

/// Format a list of operator input shapes as a string.
fn shapes_to_string(shapes: &[InputShape]) -> String {
    let formatted_shapes: Vec<_> = shapes
        .iter()
        .map(|shape| {
            shape
                .as_ref()
                .map(|s| shape_to_string(s))
                .unwrap_or("_".to_string())
        })
        .collect();
    formatted_shapes.join(", ")
}

/// Wrapper around a `RunTiming` that includes formatting configuration for the
/// `Display` implementation.
struct FormattedRunTiming<'a> {
    timing: &'a RunTiming<'a>,
    sort: TimingSort,
    include_shapes: bool,
}

impl<'a> FormattedRunTiming<'a> {
    /// Create a table that breaks down execution times for all runs of `op_name`
    /// by input shape.
    fn timing_by_shape_table(&self, op_name: &str) -> impl Table {
        let mut time_by_shape_rows: Vec<TimingByShapeRecord> = self
            .timing
            .records
            .iter()
            .filter(|record| record.name == op_name)
            .fold(HashMap::new(), |mut timings, record| {
                let formatted_shapes = shapes_to_string(&record.input_shapes);
                let input_elements = record
                    .input_shapes
                    .iter()
                    .map(|shape| {
                        shape
                            .as_ref()
                            .map(|s| s.iter().product::<usize>())
                            .unwrap_or(0)
                    })
                    .sum::<usize>();
                let (cum_time, count, _, _) = timings.entry(formatted_shapes).or_insert((
                    0.,
                    0,
                    input_elements,
                    record.node_name,
                ));
                *cum_time += record.elapsed_micros / 1000.0;
                *count += 1;
                timings
            })
            .into_iter()
            .map(
                |(shape, (total_ms, count, input_elements, node_name))| TimingByShapeRecord {
                    shape,
                    total_ms,
                    count,
                    input_elements,
                    node_name: node_name.to_string(),
                },
            )
            .collect();

        time_by_shape_rows.sort_by(|a, b| a.total_ms.total_cmp(&b.total_ms).reverse());

        TimingByShapeTable {
            rows: time_by_shape_rows,
        }
    }
}

impl<'a> fmt::Display for FormattedRunTiming<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> std::fmt::Result {
        let mut op_timings: Vec<_> = self
            .timing
            .records
            .iter()
            .fold(HashMap::new(), |mut timings, record| {
                let total_op_time = timings.entry(record.name).or_insert(0.);
                *total_op_time += record.elapsed_micros / 1000.0;
                timings
            })
            .into_iter()
            .collect();

        // Add `[Other]` for all unaccounted time.
        let total_op_time = self.timing.total_op_time();
        op_timings.push((
            "[Other]",
            self.timing.total_time - total_op_time - self.timing.alloc_time,
        ));
        op_timings.push(("[Mem alloc/free]", self.timing.alloc_time));

        op_timings.sort_by(|(a_name, a_time), (b_name, b_time)| match self.sort {
            TimingSort::ByName => a_name.cmp(b_name),
            TimingSort::ByTime => a_time.total_cmp(b_time).reverse(),
        });

        let rows: Vec<_> = op_timings
            .iter()
            .map(|(op_name, op_total_time)| {
                let run_percent = (*op_total_time / self.timing.total_time) * 100.;
                [
                    op_name.to_string(),
                    format!("{:.2}ms", op_total_time),
                    format!("({:.2}%)", run_percent),
                ]
            })
            .collect();
        let col_widths: Vec<usize> = (0..3)
            .map(|col| rows.iter().fold(0, |width, row| row[col].len().max(width)))
            .collect();

        for row in rows {
            writeln!(
                f,
                "{0:1$} {2:3$} {4:5$}",
                row[0], col_widths[0], row[1], col_widths[1], row[2], col_widths[2]
            )?;

            let op_name = &row[0];
            if self.include_shapes && !op_name.starts_with('[') {
                writeln!(f)?;
                self.timing_by_shape_table(op_name)
                    .display(4 /* indent */)
                    .fmt(f)?;
                write!(f, "\n\n")?;
            }
        }

        Ok(())
    }
}

/// Shape of a single input to an operator. May be `None` if this is a
/// positional input that was not provided.
pub type InputShape = Option<SmallVec<[usize; 4]>>;

/// Timing record for a single graph computation step.
pub struct TimingRecord<'a> {
    /// Operator name (eg. `MatMul`)
    pub name: &'a str,

    /// Name of the graph node
    pub node_name: &'a str,

    /// Shapes of the operator's inputs
    pub input_shapes: Vec<InputShape>,

    /// Execution time of this step in microseconds
    pub elapsed_micros: f32,
}

/// Specifies sort order for graph run timings.
#[derive(Clone, Default, PartialEq)]
pub enum TimingSort {
    /// Sort timings by operator name
    ByName,

    /// Sort timings by time, descending
    #[default]
    ByTime,
}
