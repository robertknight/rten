use std::fmt::{Debug, Error, Formatter};

use crate::layout::{Layout, MatrixLayout, SizeArray};
use crate::{AsView, NdTensorView, Storage, TensorBase};

/// Configuration for debug formatting of a tensor.
#[derive(Clone, Debug)]
struct FormatOptions {
    /// Maximum number of columns to print before eliding.
    pub max_columns: usize,

    /// Maximum number of rows to print before eliding.
    pub max_rows: usize,

    /// Maximum number of sub-matrices to print before eliding.
    pub max_matrices: usize,
}

impl Default for FormatOptions {
    fn default() -> Self {
        FormatOptions {
            max_columns: 10,
            max_rows: 10,
            max_matrices: 10,
        }
    }
}

/// Format a single vector of a tensor as a list (`[0, 1, 2, ... n]`).
fn write_vector<'a, T: Debug + 'a>(
    opts: &FormatOptions,
    f: &mut Formatter<'_>,
    row: NdTensorView<T, 1>,
) -> Result<(), Error> {
    let mut entries = row.iter();
    let head_len = opts.max_columns / 2;
    let tail_len = opts.max_columns.div_ceil(2);
    let body_len = row.len().saturating_sub(head_len + tail_len);

    let mut data_fmt = f.debug_list();
    data_fmt.entries(entries.by_ref().take(head_len));
    if body_len > 0 {
        data_fmt.entry(&format_args!("..."));
        entries.nth(body_len - 1);
    }
    data_fmt.entries(entries);
    data_fmt.finish()
}

/// Format a single sub-matrix from a tensor.
///
/// `extra_indent` specifies the amount of additional indentation to
/// apply to rows after the first one. The first row is assumed not to
/// require any indentation.
fn write_matrix<T: Debug>(
    opts: &FormatOptions,
    f: &mut Formatter<'_>,
    mat: NdTensorView<T, 2>,
    extra_indent: usize,
) -> Result<(), Error> {
    write!(f, "[")?;
    for row in 0..mat.rows().min(opts.max_rows) {
        write_vector(opts, f, mat.slice(row))?;

        if row < mat.rows().min(opts.max_rows) - 1 {
            write!(f, ",\n{:>width$}", ' ', width = extra_indent + 1)?;
        } else if mat.rows() > opts.max_rows {
            write!(f, ",\n{}...", " ".repeat(extra_indent + 1))?;
        }
    }
    write!(f, "]")?;
    Ok(())
}

/// A [`Debug`]-implementing wrapper around a tensor reference with custom
/// formatting options.
struct FormatTensor<'a, S: Storage, L: Layout> {
    tensor: &'a TensorBase<S, L>,
    opts: FormatOptions,
}

impl<'a, S: Storage, L: Layout> FormatTensor<'a, S, L> {
    fn new(tensor: &'a TensorBase<S, L>, opts: FormatOptions) -> Self {
        Self { tensor, opts }
    }
}

impl<S: Storage, L: Layout + Clone> Debug for FormatTensor<'_, S, L>
where
    S::Elem: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        let tensor = self.tensor;

        match tensor.ndim() {
            0 => write!(f, "({:?})", tensor.item().unwrap())?,
            1 => write_vector(&self.opts, f, tensor.nd_view())?,
            n => {
                // Format tensors with >= 2 dims as a sequence of matrices.
                let outer_dims = n - 2;
                write!(f, "{}", "[".repeat(outer_dims))?;

                let n_matrices: usize = tensor.shape().iter().take(outer_dims).product();

                for (i, mat) in tensor
                    .inner_iter::<2>()
                    .enumerate()
                    .take(self.opts.max_matrices)
                {
                    if i > 0 {
                        write!(f, "{}", " ".repeat(outer_dims))?;
                    }

                    write_matrix(&self.opts, f, mat, outer_dims)?;

                    if i < n_matrices.min(self.opts.max_matrices) - 1 {
                        write!(f, ",\n\n")?;
                    } else if n_matrices > self.opts.max_matrices {
                        write!(f, "\n\n{}...\n\n", " ".repeat(outer_dims))?;
                    }
                }

                write!(f, "{}", "]".repeat(outer_dims))?;
            }
        }

        write!(
            f,
            ", shape={:?}, strides={:?}",
            tensor.shape(),
            tensor.strides()
        )
    }
}

impl<S: Storage, L: Layout + Clone> Debug for TensorBase<S, L>
where
    S::Elem: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{:?}", FormatTensor::new(self, FormatOptions::default()))
    }
}

#[cfg(test)]
mod tests {
    use rten_testing::TestCases;

    use super::{FormatOptions, FormatTensor};
    use crate::Tensor;

    #[test]
    fn test_debug() {
        #[derive(Clone, Debug)]
        struct Case<'a> {
            tensor: Tensor,
            opts: FormatOptions,
            expected: &'a str,
        }

        let cases = [
            // Scalar
            Case {
                tensor: Tensor::from(2.),
                opts: FormatOptions::default(),
                expected: "(2.0), shape=[], strides=[]",
            },
            // Small and large values
            Case {
                tensor: Tensor::from([1e-8, 1e-7]),
                opts: FormatOptions::default(),
                expected: "[1e-8, 1e-7], shape=[2], strides=[1]",
            },
            // Empty vector
            Case {
                tensor: Tensor::from([0.; 0]),
                opts: FormatOptions::default(),
                expected: "[], shape=[0], strides=[1]",
            },
            // Vector with length less than `max_columns`.
            Case {
                tensor: Tensor::from([1., 2., 3., 4.]),
                opts: FormatOptions {
                    max_columns: 5,
                    ..Default::default()
                },
                expected: "[1.0, 2.0, 3.0, 4.0], shape=[4], strides=[1]",
            },
            // Vector with length `max_columns`
            Case {
                tensor: Tensor::arange(1., 6., None),
                opts: FormatOptions {
                    max_columns: 5,
                    ..Default::default()
                },
                expected: "[1.0, 2.0, 3.0, 4.0, 5.0], shape=[5], strides=[1]",
            },
            // Vector with length `max_columns + 1`
            Case {
                tensor: Tensor::arange(1., 7., None),
                opts: FormatOptions {
                    // nb. max_columns is odd so the tail is longer
                    max_columns: 5,
                    ..Default::default()
                },
                expected: "[1.0, 2.0, ..., 4.0, 5.0, 6.0], shape=[6], strides=[1]",
            },
            // Vector with length `max_columns + N`
            Case {
                tensor: Tensor::arange(1., 22., None),
                opts: FormatOptions {
                    max_columns: 10,
                    ..Default::default()
                },
                expected: "[1.0, 2.0, 3.0, 4.0, 5.0, ..., 17.0, 18.0, 19.0, 20.0, 21.0], shape=[21], strides=[1]",
            },
            // Matrix
            Case {
                tensor: Tensor::from([[1., 2.], [3., 4.]]),
                opts: FormatOptions::default(),
                expected: "
[[1.0, 2.0],
 [3.0, 4.0]], shape=[2, 2], strides=[2, 1]"
                    .trim(),
            },
            // Matrix with elided rows
            Case {
                tensor: Tensor::from([[1., 2.], [3., 4.], [5., 6.]]),
                opts: FormatOptions {
                    max_rows: 2,
                    ..Default::default()
                },
                expected: "
[[1.0, 2.0],
 [3.0, 4.0],
 ...], shape=[3, 2], strides=[2, 1]"
                    .trim(),
            },
            // 3D
            Case {
                tensor: Tensor::from([[[1., 2.], [3., 4.]]]),
                opts: FormatOptions::default(),
                expected: "
[[[1.0, 2.0],
  [3.0, 4.0]]], shape=[1, 2, 2], strides=[4, 2, 1]"
                    .trim(),
            },
            // 3D
            Case {
                tensor: Tensor::from([
                    [[1., 2.], [3., 4.]],
                    [[1., 2.], [3., 4.]],
                    [[1., 2.], [3., 4.]],
                ]),
                opts: FormatOptions {
                    max_matrices: 2,
                    ..Default::default()
                },
                expected: "
[[[1.0, 2.0],
  [3.0, 4.0]],

 [[1.0, 2.0],
  [3.0, 4.0]]

 ...

], shape=[3, 2, 2], strides=[4, 2, 1]"
                    .trim(),
            },
            // 4D
            Case {
                tensor: Tensor::from([[[1., 2.], [3., 4.]]]).into_shape([1, 1, 2, 2].as_slice()),
                opts: FormatOptions::default(),
                expected: "
[[[[1.0, 2.0],
   [3.0, 4.0]]]], shape=[1, 1, 2, 2], strides=[4, 4, 2, 1]"
                    .trim(),
            },
        ];

        cases.test_each_clone(|case| {
            let debug_str = format!("{:?}", FormatTensor::new(&case.tensor, case.opts));
            assert_eq!(debug_str, case.expected);
        })
    }
}
