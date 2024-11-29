use std::fmt::{Debug, Error, Formatter};

use crate::{AsView, Layout, MatrixLayout, MutLayout, NdTensorView, Storage, TensorBase};

/// Entry in the formatted representation of a tensor's data.
enum Entry<T: Debug> {
    Value(T),

    /// "..." used to elide long dimensions.
    Ellipsis,
}

impl<T: Debug> Debug for Entry<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        match self {
            Entry::Value(val) => write!(f, "{:?}", val),
            Entry::Ellipsis => write!(f, "..."),
        }
    }
}

/// Configuration for debug formatting of a tensor.
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

/// A [`Debug`]-implementing wrapper around a tensor reference with custom
/// formatting options.
struct FormatTensor<'a, S: Storage, L: MutLayout> {
    tensor: &'a TensorBase<S, L>,
    opts: FormatOptions,
}

impl<'a, S: Storage, L: MutLayout> FormatTensor<'a, S, L> {
    fn new(tensor: &'a TensorBase<S, L>, opts: FormatOptions) -> Self {
        Self { tensor, opts }
    }

    /// Format a single vector of a tensor as a list (`[0, 1, 2, ... n]`).
    fn write_vector<T: Debug>(
        &self,
        f: &mut Formatter<'_>,
        row: impl ExactSizeIterator<Item = T> + Clone,
    ) -> Result<(), Error> {
        let len = row.len();

        let head = row.clone().take(self.opts.max_columns / 2);
        let tail = row
            .clone()
            .skip(self.opts.max_columns / 2)
            .skip(len.saturating_sub(self.opts.max_columns));

        let mut data_fmt = f.debug_list();
        data_fmt.entries(head.map(Entry::Value));
        if len > self.opts.max_columns {
            data_fmt.entry(&Entry::<T>::Ellipsis);
        }
        data_fmt.entries(tail.map(Entry::Value));
        data_fmt.finish()
    }

    /// Format a single sub-matrix from a tensor.
    ///
    /// `extra_indent` specifies the amount of additional indentation to
    /// apply to rows after the first one. The first row is assumed not to
    /// require any indentation.
    fn write_matrix<T: Debug>(
        &self,
        f: &mut Formatter<'_>,
        mat: NdTensorView<T, 2>,
        extra_indent: usize,
    ) -> Result<(), Error> {
        write!(f, "[")?;
        for row in 0..mat.rows().min(self.opts.max_rows) {
            self.write_vector(f, mat.slice(row).iter())?;

            if row < mat.rows().min(self.opts.max_rows) - 1 {
                write!(f, ",\n{:>width$}", ' ', width = extra_indent + 1)?;
            } else if mat.rows() > self.opts.max_rows {
                write!(f, ",\n{}...", " ".repeat(extra_indent + 1))?;
            }
        }
        write!(f, "]")?;
        Ok(())
    }
}

impl<S: Storage, L: MutLayout> Debug for FormatTensor<'_, S, L>
where
    S::Elem: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        let tensor = self.tensor;

        match tensor.ndim() {
            0 => write!(f, "({:?})", tensor.item().unwrap())?,
            1 => self.write_vector(f, tensor.iter())?,
            n => {
                // Format tensors with >= 2 dims as a sequence of matrices.
                let outer_dims = n - 2;
                write!(f, "{}", "[".repeat(outer_dims))?;

                let n_matrices: usize = tensor.shape().as_ref().iter().take(outer_dims).product();

                for (i, mat) in tensor
                    .inner_iter::<2>()
                    .enumerate()
                    .take(self.opts.max_matrices)
                {
                    if i > 0 {
                        write!(f, "{}", " ".repeat(outer_dims))?;
                    }

                    self.write_matrix(f, mat, outer_dims)?;

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

impl<S: Storage, L: MutLayout> Debug for TensorBase<S, L>
where
    S::Elem: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{:?}", FormatTensor::new(self, FormatOptions::default()))
    }
}

#[cfg(test)]
mod tests {
    use super::{FormatOptions, FormatTensor};
    use crate::Tensor;

    #[test]
    fn test_debug() {
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
            // Empty vector
            Case {
                tensor: Tensor::from([0.; 0]),
                opts: FormatOptions::default(),
                expected: "[], shape=[0], strides=[1]",
            },
            // Short vector
            Case {
                tensor: Tensor::from([1., 2., 3., 4.]),
                opts: FormatOptions::default(),
                expected: "[1.0, 2.0, 3.0, 4.0], shape=[4], strides=[1]",
            },
            // Small and large values
            Case {
                tensor: Tensor::from([1e-8, 1e-7]),
                opts: FormatOptions::default(),
                expected: "[1e-8, 1e-7], shape=[2], strides=[1]",
            },
            // Long vector
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
 [3.0, 4.0]], shape=[2, 2], strides=[2, 1]".trim(),
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
 ...], shape=[3, 2], strides=[2, 1]".trim(),
            },
            // 3D
            Case {
                tensor: Tensor::from([[[1., 2.], [3., 4.]]]),
                opts: FormatOptions::default(),
                expected: "
[[[1.0, 2.0],
  [3.0, 4.0]]], shape=[1, 2, 2], strides=[4, 2, 1]".trim(),
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

], shape=[3, 2, 2], strides=[4, 2, 1]".trim(),
            },
            // 4D
            Case {
                tensor: Tensor::from([[[1., 2.], [3., 4.]]]).into_shape([1, 1, 2, 2].as_slice()),
                opts: FormatOptions::default(),
                expected: "
[[[[1.0, 2.0],
   [3.0, 4.0]]]], shape=[1, 1, 2, 2], strides=[4, 4, 2, 1]".trim(),
            },
        ];

        for Case {
            tensor,
            opts,
            expected,
        } in cases
        {
            let debug_str = format!("{:?}", FormatTensor::new(&tensor, opts));
            assert_eq!(debug_str, expected);
        }
    }
}
