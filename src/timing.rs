use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::ops::Sub;
use std::time::Duration;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use std::time;

use crate::NodeId;
use crate::value::ValueMeta;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
type Seconds = f64;

/// A wrapper around [`std::time::Instant`] that provides a fallback on
/// platforms (WebAssembly) where `Instant::now` is unsupported.
#[derive(Copy, Clone, PartialEq)]
pub struct Instant {
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    inner: time::Instant,
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    inner: Seconds,
}

impl Instant {
    pub fn now() -> Self {
        Instant {
            #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
            inner: time::Instant::now(),

            // This could use `performance.now()` when available.
            #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
            inner: 0.0,
        }
    }
}

impl Sub<Instant> for Instant {
    type Output = Duration;

    fn sub(self, rhs: Instant) -> Duration {
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            self.inner - rhs.inner
        }

        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            Duration::from_secs_f64(self.inner - rhs.inner)
        }
    }
}

/// Trait for text data table sources.
///
/// Tables can be formatted using [`Table::display`] to get a wrapper that
/// implements [`Display`].
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

    /// Return a wrapper around this table which implements [`Display`].
    fn display(&self, indent: usize) -> DisplayTable<'_, Self>
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

impl<T: Table> fmt::Display for DisplayTable<'_, T> {
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

    /// Total time for the graph run
    pub total_time: Duration,
}

impl RunTiming<'_> {
    /// Return a struct that formats output with the given options.
    pub fn display(&self, sort: TimingSort, include_shapes: bool) -> impl fmt::Display + '_ {
        FormattedRunTiming {
            timing: self,
            sort,
            include_shapes,
        }
    }
}

impl fmt::Display for RunTiming<'_> {
    /// Format timings with the default sort order (see [`TimingSort`]).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        self.display(TimingSort::ByTime, false /* include_shapes */)
            .fmt(f)
    }
}

/// Record of execution time for a single (operator, input_shape) combination.
struct TimingByShapeRecord {
    /// Formatted representation of input shapes
    shape: String,

    /// Total elapsed time for this operator
    total_duration: Duration,

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

/// [`Display`]-able table containing a breakdown of operator execution time
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
        let total_ms = row.total_duration.as_secs_f64() * 1000.0;
        let total_ns = total_ms * 1_000_000.0;
        match col {
            0 => row.shape.clone(),
            1 => row.count.to_string(),
            2 => format!("{:.3}", total_ms / row.count as f64),
            3 => format!("{:.3}", total_ms),
            4 => format!("{:.3}", total_ns / (row.input_elements * row.count) as f64),
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
fn shapes_to_string(meta: &[(Option<NodeId>, Option<ValueMeta>)]) -> String {
    let formatted_shapes: Vec<_> = meta
        .iter()
        .map(|meta| {
            meta.1
                .as_ref()
                .map(|m| shape_to_string(&m.shape))
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

impl FormattedRunTiming<'_> {
    /// Create a table that breaks down execution times for all runs of `op_name`
    /// by input shape.
    fn timing_by_shape_table(&self, op_name: &str) -> impl Table {
        let mut time_by_shape_rows: Vec<TimingByShapeRecord> = self
            .timing
            .records
            .iter()
            .filter(|record| record.name == op_name)
            .fold(HashMap::new(), |mut timings, record| {
                let formatted_shapes = shapes_to_string(&record.input_meta);
                let input_elements = record
                    .input_meta
                    .iter()
                    .map(|meta| {
                        meta.1
                            .as_ref()
                            .map(|meta| meta.shape.iter().product::<usize>())
                            .unwrap_or(0)
                    })
                    .sum::<usize>();
                let (cum_time, count, _, _) = timings.entry(formatted_shapes).or_insert((
                    Duration::ZERO,
                    0,
                    input_elements,
                    record.node_name,
                ));
                *cum_time += record.elapsed;
                *count += 1;
                timings
            })
            .into_iter()
            .map(
                |(shape, (total, count, input_elements, node_name))| TimingByShapeRecord {
                    shape,
                    total_duration: total,
                    count,
                    input_elements,
                    node_name: node_name.to_string(),
                },
            )
            .collect();

        time_by_shape_rows.sort_by(|a, b| a.total_duration.cmp(&b.total_duration).reverse());

        TimingByShapeTable {
            rows: time_by_shape_rows,
        }
    }
}

impl fmt::Display for FormattedRunTiming<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> std::fmt::Result {
        let mut op_timings: Vec<_> = self
            .timing
            .records
            .iter()
            .fold(HashMap::new(), |mut timings, record| {
                let total_op_time = timings.entry(record.name).or_insert(Duration::ZERO);
                *total_op_time += record.elapsed;
                timings
            })
            .into_iter()
            .collect();

        op_timings.sort_by(|(a_name, a_time), (b_name, b_time)| match self.sort {
            TimingSort::ByName => a_name.cmp(b_name),
            TimingSort::ByTime => a_time.cmp(b_time).reverse(),
        });

        let rows: Vec<_> = op_timings
            .into_iter()
            .map(|(op_name, op_total_time)| {
                let op_total_time_ms = op_total_time.as_secs_f64() * 1000.0;
                let total_time_ms = self.timing.total_time.as_secs_f64() * 1000.0;
                let run_percent = (op_total_time_ms / total_time_ms) * 100.;
                [
                    op_name.to_string(),
                    format!("{:.2}ms", op_total_time_ms),
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

/// Timing record for a single graph computation step.
#[derive(Clone)]
pub struct TimingRecord<'a> {
    /// Operator name (eg. `MatMul`)
    pub name: &'a str,

    /// Name of the graph node
    pub node_name: &'a str,

    /// IDs and shapes of the operator's inputs
    pub input_meta: Vec<(Option<NodeId>, Option<ValueMeta>)>,

    /// Execution time of this step
    pub elapsed: Duration,
}

/// Specifies sort order for graph run timings.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum TimingSort {
    /// Sort timings by operator name
    ByName,

    /// Sort timings by time, descending
    #[default]
    ByTime,
}

/// Filter which records (ie. which operator nodes) are included in run timings.
#[derive(Clone, Debug, PartialEq)]
pub enum TimingFilter {
    /// Include only certain operators (eg. MatMul, Add).
    Operator(String),
}

impl TimingFilter {
    fn matches(&self, record: &TimingRecord) -> bool {
        match self {
            Self::Operator(op) => record.name == op,
        }
    }
}

/// Formatting options for use with [`Profiler::print`].
pub struct ProfileFormat {
    pub sort: TimingSort,

    pub filter: Vec<TimingFilter>,

    /// Whether to break down operator timings by the shape of inputs.
    pub timing_by_shape: bool,
}

/// Profiler that collects operator timing and memory allocation metrics
/// during a graph run.
pub struct Profiler<'a> {
    records: Vec<TimingRecord<'a>>,

    /// Total number of memory allocations from a shared pool.
    pool_allocs: usize,

    /// Number of allocations from the shared pool that were fulfilled from
    /// the pool rather than the OS.
    pool_hits: usize,
}

impl<'a> Profiler<'a> {
    /// Create a profiler with initial capacity for `num_records` records for
    /// individual operator timings.
    pub fn with_capacity(num_records: usize) -> Self {
        Profiler {
            records: Vec::with_capacity(num_records),
            pool_allocs: 0,
            pool_hits: 0,
        }
    }

    /// Add a timing record for a single operator execution.
    pub fn add_record(&mut self, record: TimingRecord<'a>) {
        self.records.push(record);
    }

    /// Update memory allocation metrics.
    pub fn add_pool_metrics(&mut self, allocs: usize, hits: usize) {
        self.pool_allocs += allocs;
        self.pool_hits += hits;
    }

    /// Print a summary of the profile to stdout.
    pub fn print(&self, opts: ProfileFormat) {
        // Apply filters to timing records.
        let mut records = Cow::Borrowed(&self.records);
        if !opts.filter.is_empty() {
            records
                .to_mut()
                .retain(|entry| opts.filter.iter().any(|f| f.matches(entry)));
        }

        // Print overall stats for all operators.
        let run_duration: Duration = records.iter().map(|r| r.elapsed).sum();
        let run_duration_ms = run_duration.as_secs_f64() * 1000.0;
        println!(
            "{} ops evaluated in {:.3}ms",
            records.len(),
            run_duration_ms,
        );

        // Print memory-related stats.
        println!("Pool allocs {} hits {}", self.pool_allocs, self.pool_hits,);

        // Print detailed breakdown by operator.
        let timing = RunTiming {
            records: &records,
            total_time: run_duration,
        };

        print!(
            "\n{}",
            timing.display(opts.sort.clone(), opts.timing_by_shape)
        );
    }
}
