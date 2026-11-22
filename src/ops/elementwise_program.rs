//! Fused elementwise operator which evaluates a sequence of primitive
//! operations on one small tile of its inputs at a time.
//!
//! Evaluating an expression such as `x + beta * sin(alpha * x)^2` as separate
//! graph operators makes one pass over the whole tensor per operator. Here
//! the tensor is instead split into tiles which are small enough that a tile
//! of each input and intermediate value fits in the L1 cache, and all of the
//! operations are applied to one tile before moving on to the next.

use std::ops::Range;

use rayon::prelude::*;
use rten_base::bit_set::BitSet;
use rten_simd::SimdUnaryOp;
use rten_tensor::prelude::*;
use rten_tensor::{Tensor, TensorView};
use rten_vecmath as vecmath;

use crate::buffer_pool::AutoReturn;
use crate::infer_shapes::{InferShapes, UnaryOp};
use crate::operator::{
    InPlaceInputs, IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType,
    OutputTypeList, OutputTypesContext,
};

/// Number of elements in a tile.
///
/// A tile of the output, of one other input and of two registers together use
/// 16 KB, which fits in a 32 KB L1 cache.
const TILE_LEN: usize = 1024;

/// Number of elements processed by one task when running in parallel.
const TASK_LEN: usize = 32 * TILE_LEN;

/// Source of a tile read by a [`TileOp`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Src {
    /// A tile of one of the operator's tensor inputs.
    Input(usize),
    /// A register holding the result of an earlier operation.
    Reg(usize),
}

/// Destination of a tile written by a [`TileOp`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Dst {
    Reg(usize),
    /// The operator's output. This shares a buffer with input 0.
    Output,
}

/// An operation on one tile.
///
/// The destination may be the same register as a source.
#[derive(Clone, Debug, PartialEq)]
pub enum TileOp {
    /// Compute `a + b`.
    Add { dst: Dst, a: Src, b: Src },
    /// Multiply `src` by the scalar input with index `scalar`.
    MulScalar { dst: Dst, src: Src, scalar: usize },
    /// Compute `addend + src * scalar`, where `scalar` is the index of a
    /// scalar input.
    MulScalarAdd {
        dst: Dst,
        src: Src,
        scalar: usize,
        addend: Src,
    },
    /// Compute `sin(src)`.
    Sin { dst: Dst, src: Src },
    /// Compute `sin(src)^2`.
    SinSquared { dst: Dst, src: Src },
    /// Compute `src * src`.
    Square { dst: Dst, src: Src },
}

impl TileOp {
    fn dst(&self) -> Dst {
        match *self {
            TileOp::Add { dst, .. }
            | TileOp::MulScalar { dst, .. }
            | TileOp::MulScalarAdd { dst, .. }
            | TileOp::Sin { dst, .. }
            | TileOp::SinSquared { dst, .. }
            | TileOp::Square { dst, .. } => dst,
        }
    }
}

/// A sequence of operations which compute one tile of an operator's output
/// from the corresponding tiles of its inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct TileProgram {
    ops: Vec<TileOp>,
    n_inputs: usize,
    n_scalars: usize,
    n_regs: usize,
}

impl TileProgram {
    /// Create a program which reads `n_inputs` tensor inputs and `n_scalars`
    /// scalar inputs.
    ///
    /// Only the last operation may write the output. Since the output shares
    /// a buffer with input 0, this means input 0 can be read by any operation.
    pub fn new(ops: Vec<TileOp>, n_inputs: usize, n_scalars: usize) -> TileProgram {
        let (last, rest) = ops.split_last().expect("program should not be empty");
        assert_eq!(last.dst(), Dst::Output);
        assert!(rest.iter().all(|op| op.dst() != Dst::Output));
        assert!(n_inputs >= 1);

        let n_regs = ops
            .iter()
            .filter_map(|op| match op.dst() {
                Dst::Reg(reg) => Some(reg + 1),
                Dst::Output => None,
            })
            .max()
            .unwrap_or(0);

        TileProgram {
            ops,
            n_inputs,
            n_scalars,
            n_regs,
        }
    }
}

/// Buffers used while evaluating a [`TileProgram`] on one tile.
struct TileState<'a, 'r> {
    /// Tensor inputs after the first.
    inputs: &'a [&'a [f32]],
    /// Range of each entry of `inputs` which this tile covers.
    range: Range<usize>,
    regs: &'a mut [&'r mut [f32]],
    /// Tile of the output. Until the final operation, this holds input 0.
    output: &'a mut [f32],
}

impl TileState<'_, '_> {
    /// Return the buffers for `srcs`, or `None` for sources which are the
    /// same buffer as `dst`.
    fn sources<const N: usize>(&self, dst: Dst, srcs: [Src; N]) -> [Option<&[f32]>; N] {
        srcs.map(|src| match (src, dst) {
            (Src::Input(0), Dst::Output) => None,
            (Src::Input(0), _) => Some(&self.output[..]),
            (Src::Input(index), _) => Some(&self.inputs[index - 1][self.range.clone()]),
            (Src::Reg(src_reg), Dst::Reg(dst_reg)) if src_reg == dst_reg => None,
            (Src::Reg(reg), _) => Some(&self.regs[reg][..self.range.len()]),
        })
    }

    /// Call `kernel` with the buffers for `dst` and `srcs`.
    ///
    /// Sources which are the same buffer as the destination are passed as
    /// `None`.
    fn apply<const N: usize>(
        &mut self,
        dst: Dst,
        srcs: [Src; N],
        kernel: impl FnOnce(&mut [f32], [Option<&[f32]>; N]),
    ) {
        let len = self.range.len();

        // Take the destination out of `self`, so that the remaining buffers
        // can be borrowed as sources.
        match dst {
            Dst::Reg(reg) => {
                let dst_buf = std::mem::take(&mut self.regs[reg]);
                kernel(&mut dst_buf[..len], self.sources(dst, srcs));
                self.regs[reg] = dst_buf;
            }
            Dst::Output => {
                let dst_buf = std::mem::take(&mut self.output);
                kernel(dst_buf, self.sources(dst, srcs));
                self.output = dst_buf;
            }
        }
    }

    fn run(&mut self, ops: &[TileOp], scalars: &[f32]) {
        for op in ops {
            match *op {
                TileOp::Add { dst, a, b } => self.apply(dst, [a, b], |dst, srcs| match srcs {
                    [Some(a), Some(b)] => {
                        for ((y, a), b) in dst.iter_mut().zip(a).zip(b) {
                            *y = a + b;
                        }
                    }
                    [Some(x), None] | [None, Some(x)] => {
                        for (y, x) in dst.iter_mut().zip(x) {
                            *y += x;
                        }
                    }
                    [None, None] => {
                        for y in dst {
                            *y += *y;
                        }
                    }
                }),
                TileOp::MulScalar { dst, src, scalar } => {
                    let scale = scalars[scalar];
                    self.apply(dst, [src], |dst, [src]| match src {
                        Some(src) => {
                            for (y, x) in dst.iter_mut().zip(src) {
                                *y = x * scale;
                            }
                        }
                        None => {
                            for y in dst {
                                *y *= scale;
                            }
                        }
                    })
                }
                TileOp::MulScalarAdd {
                    dst,
                    src,
                    scalar,
                    addend,
                } => {
                    let scale = scalars[scalar];
                    self.apply(dst, [src, addend], |dst, srcs| match srcs {
                        [Some(src), Some(addend)] => {
                            for ((y, x), a) in dst.iter_mut().zip(src).zip(addend) {
                                *y = a + x * scale;
                            }
                        }
                        [Some(src), None] => {
                            for (y, x) in dst.iter_mut().zip(src) {
                                *y += x * scale;
                            }
                        }
                        [None, Some(addend)] => {
                            for (y, a) in dst.iter_mut().zip(addend) {
                                *y = a + *y * scale;
                            }
                        }
                        [None, None] => {
                            for y in dst {
                                *y += *y * scale;
                            }
                        }
                    })
                }
                TileOp::Sin { dst, src } => self.apply(dst, [src], |dst, [src]| {
                    if let Some(src) = src {
                        dst.copy_from_slice(src);
                    }
                    vecmath::Sin::new().map_mut(dst);
                }),
                TileOp::SinSquared { dst, src } => self.apply(dst, [src], |dst, [src]| {
                    if let Some(src) = src {
                        dst.copy_from_slice(src);
                    }
                    vecmath::SinSquared::new().map_mut(dst);
                }),
                TileOp::Square { dst, src } => self.apply(dst, [src], |dst, [src]| match src {
                    Some(src) => {
                        for (y, x) in dst.iter_mut().zip(src) {
                            *y = x * x;
                        }
                    }
                    None => {
                        for y in dst {
                            *y *= *y;
                        }
                    }
                }),
            }
        }
    }
}

/// Operator which evaluates a [`TileProgram`].
///
/// The operator's inputs are the program's tensor inputs followed by its
/// scalar inputs. Tensor inputs all have the shape of the output. Scalar
/// inputs are broadcast against the output and have a size of one in the last
/// `inner_dims` dimensions, so that their value is constant within each _row_
/// formed by flattening those dimensions.
///
/// This is not an ONNX operator. It is created by fusing subgraphs of
/// elementwise operators.
#[derive(Debug)]
pub struct ElementwiseProgram {
    pub program: TileProgram,

    /// Number of trailing dimensions in which scalar inputs have a size of
    /// one.
    pub inner_dims: usize,
}

impl ElementwiseProgram {
    /// Evaluate the program, where `output` initially holds input 0.
    fn run_in_place_impl(
        &self,
        ctx: &OpRunContext,
        mut output: Tensor<f32>,
    ) -> Result<Tensor<f32>, OpError> {
        let TileProgram {
            ops,
            n_inputs,
            n_scalars,
            n_regs,
        } = &self.program;
        let (n_inputs, n_scalars, n_regs) = (*n_inputs, *n_scalars, *n_regs);
        let pool = ctx.pool();
        let inputs = ctx.inputs();

        let Some(outer_dims) = output.ndim().checked_sub(self.inner_dims) else {
            return Err(OpError::invalid_value("input has too few dims"));
        };
        let row_len: usize = output.shape()[outer_dims..].iter().product();

        let mut other_inputs = Vec::with_capacity(n_inputs - 1);
        for index in 1..n_inputs {
            let input: TensorView<f32> = inputs.require_as(index)?;
            if input.shape() != output.shape() {
                return Err(OpError::incompatible_input_shapes(
                    "tensor inputs should have the same shape",
                ));
            }
            other_inputs.push(input.to_contiguous_in(pool).auto_return(pool));
        }
        let other_inputs: Vec<&[f32]> = other_inputs.iter().map(|input| input.data()).collect();

        // Get the value of each scalar input for each row, with the values for
        // one row stored together.
        let mut row_shape = output.shape().to_vec();
        row_shape[outer_dims..].fill(1);
        let n_rows: usize = row_shape.iter().product();
        let mut row_scalars = vec![0.; n_rows * n_scalars];
        for index in 0..n_scalars {
            let scalar: TensorView<f32> = inputs.require_as(n_inputs + index)?;
            let Some(scalar) = scalar.try_broadcast(row_shape.as_slice()).ok() else {
                return Err(OpError::incompatible_input_shapes(
                    "scalar input should be constant within each row",
                ));
            };
            for (row, value) in scalar.iter().enumerate() {
                row_scalars[row * n_scalars + index] = *value;
            }
        }

        if output.is_empty() {
            return Ok(output);
        }

        output.make_contiguous_in(pool);
        output
            .data_mut()
            .unwrap()
            .par_chunks_mut(TASK_LEN)
            .enumerate()
            .for_each_init(
                || vec![0.; n_regs * TILE_LEN],
                |reg_buf, (task, chunk)| {
                    let mut regs: Vec<&mut [f32]> = reg_buf.chunks_mut(TILE_LEN).collect();
                    let task_start = task * TASK_LEN;

                    // Split the chunk into tiles which each lie within one row.
                    let mut pos = 0;
                    while pos < chunk.len() {
                        let start = task_start + pos;
                        let row = start / row_len;
                        let row_end = (row + 1) * row_len;
                        let len = TILE_LEN.min(chunk.len() - pos).min(row_end - start);

                        let mut tile = TileState {
                            inputs: &other_inputs,
                            range: start..start + len,
                            regs: &mut regs,
                            output: &mut chunk[pos..pos + len],
                        };
                        tile.run(ops, &row_scalars[row * n_scalars..][..n_scalars]);

                        pos += len;
                    }
                },
            );

        Ok(output)
    }
}

impl Operator for ElementwiseProgram {
    fn name(&self) -> &str {
        "ElementwiseProgram"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(self.program.n_inputs + self.program.n_scalars)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input: TensorView<f32> = ctx.inputs().require_as(0)?;
        self.run_in_place_impl(ctx, input.to_tensor_in(ctx.pool()))
            .into_op_result()
    }

    fn in_place_inputs(&self) -> BitSet<u16> {
        BitSet::from_indices([0])
    }

    fn run_in_place(
        &self,
        in_place: InPlaceInputs,
        ctx: &OpRunContext,
    ) -> Result<OutputList, OpError> {
        let input: Tensor<f32> = in_place.into_single().try_into()?;
        self.run_in_place_impl(ctx, input).into_op_result()
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(&UnaryOp)
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::Tensor;
    use rten_tensor::prelude::*;
    use rten_testing::TestCases;

    use super::{Dst, ElementwiseProgram, Src, TASK_LEN, TILE_LEN, TileOp, TileProgram};
    use crate::operator::{OpError, OperatorExt};

    /// Return a program which computes `x + beta * sin(alpha * x)^2`, where
    /// `alpha` and `beta` are the scalar inputs.
    fn snake_program() -> TileProgram {
        let reg = Dst::Reg(0);
        let reg_src = Src::Reg(0);
        TileProgram::new(
            [
                TileOp::MulScalar {
                    dst: reg,
                    src: Src::Input(0),
                    scalar: 0,
                },
                TileOp::Sin {
                    dst: reg,
                    src: reg_src,
                },
                TileOp::Square {
                    dst: reg,
                    src: reg_src,
                },
                TileOp::MulScalar {
                    dst: reg,
                    src: reg_src,
                    scalar: 1,
                },
                TileOp::Add {
                    dst: Dst::Output,
                    a: Src::Input(0),
                    b: reg_src,
                },
            ]
            .into(),
            1,
            2,
        )
    }

    fn test_input(shape: &[usize]) -> Tensor<f32> {
        let len = shape.iter().product();
        Tensor::from_data(
            shape,
            (0..len)
                .map(|i| ((i * 7919) % 2000) as f32 / 500. - 2.)
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn test_elementwise_program() {
        #[derive(Debug)]
        struct Case {
            shape: Vec<usize>,
            scalar_shape: Vec<usize>,
            inner_dims: usize,
        }

        let cases = [
            // Rows shorter than a tile.
            Case {
                shape: [2, 3, 50].into(),
                scalar_shape: [1, 3, 1].into(),
                inner_dims: 1,
            },
            // Rows which span several tiles and end with a partial tile.
            Case {
                shape: [1, 2, 2 * TILE_LEN + 7].into(),
                scalar_shape: [1, 2, 1].into(),
                inner_dims: 1,
            },
            // Rows which span several tasks.
            Case {
                shape: [1, 2, TASK_LEN + TILE_LEN + 3].into(),
                scalar_shape: [1, 2, 1].into(),
                inner_dims: 1,
            },
            // Tasks which contain several rows.
            Case {
                shape: [4, 50, 7, 11].into(),
                scalar_shape: [4, 1, 1, 1].into(),
                inner_dims: 2,
            },
            // Single scalar for the whole tensor.
            Case {
                shape: [3, 100].into(),
                scalar_shape: [].into(),
                inner_dims: 2,
            },
            // Empty input.
            Case {
                shape: [1, 2, 0].into(),
                scalar_shape: [1, 2, 1].into(),
                inner_dims: 1,
            },
        ];

        cases.test_each(|case| {
            let x = test_input(&case.shape);
            let n_scalars: usize = case.scalar_shape.iter().product();
            let alpha = Tensor::from_data(
                &case.scalar_shape,
                (0..n_scalars)
                    .map(|i| 0.5 + i as f32 * 0.25)
                    .collect::<Vec<_>>(),
            );
            let beta = alpha.map(|a| 1. / a);

            let op = ElementwiseProgram {
                program: snake_program(),
                inner_dims: case.inner_dims,
            };
            let result: Tensor<f32> = op
                .run_simple((x.view(), alpha.view(), beta.view()))
                .unwrap();

            // Evaluate the expression using the same steps as the separate
            // operators would.
            let alpha = alpha.broadcast(x.shape());
            let beta = beta.broadcast(x.shape());
            let expected = Tensor::from_fn(x.shape(), |idx| {
                let x = x[idx];
                let mut sin = [alpha[idx] * x];
                rten_simd::SimdUnaryOp::map_mut(&rten_vecmath::Sin::new(), &mut sin);
                x + beta[idx] * (sin[0] * sin[0])
            });
            assert_eq!(result, expected);
        })
    }

    #[test]
    fn test_elementwise_program_multiple_inputs() {
        // Compute `sin(a) + b`.
        let program = TileProgram::new(
            [
                TileOp::Sin {
                    dst: Dst::Reg(0),
                    src: Src::Input(1),
                },
                TileOp::Add {
                    dst: Dst::Output,
                    a: Src::Reg(0),
                    b: Src::Input(0),
                },
            ]
            .into(),
            2,
            0,
        );
        let op = ElementwiseProgram {
            program,
            inner_dims: 2,
        };
        let a = test_input(&[3, TILE_LEN + 5]);
        let b = a.map(|x| x * 0.5);

        let result: Tensor<f32> = op.run_simple((b.view(), a.view())).unwrap();

        let expected = Tensor::from_fn(a.shape(), |idx| {
            let mut sin = [a[idx]];
            rten_simd::SimdUnaryOp::map_mut(&rten_vecmath::Sin::new(), &mut sin);
            sin[0] + b[idx]
        });
        assert_eq!(result, expected);
    }

    #[test]
    fn test_elementwise_program_mul_scalar_add() {
        #[derive(Debug)]
        struct Case {
            src: Src,
            addend: Src,
        }

        // Test each combination of the output's buffer and another buffer
        // as sources.
        let cases = [
            Case {
                src: Src::Input(1),
                addend: Src::Input(0),
            },
            Case {
                src: Src::Input(0),
                addend: Src::Input(1),
            },
            Case {
                src: Src::Input(0),
                addend: Src::Input(0),
            },
            Case {
                src: Src::Input(1),
                addend: Src::Input(1),
            },
        ];

        cases.test_each(|case| {
            let program = TileProgram::new(
                [TileOp::MulScalarAdd {
                    dst: Dst::Output,
                    src: case.src,
                    scalar: 0,
                    addend: case.addend,
                }]
                .into(),
                2,
                1,
            );
            let op = ElementwiseProgram {
                program,
                inner_dims: 2,
            };
            let inputs = [
                test_input(&[3, TILE_LEN + 5]),
                test_input(&[3, TILE_LEN + 5]),
            ];
            let inputs = [inputs[0].clone(), inputs[1].map(|x| x * 0.5 + 1.)];
            let scale = Tensor::from(1.5);

            let result: Tensor<f32> = op
                .run_simple((inputs[0].view(), inputs[1].view(), scale.view()))
                .unwrap();

            let input = |src: Src| match src {
                Src::Input(index) => &inputs[index],
                Src::Reg(_) => unreachable!(),
            };
            let (src, addend) = (input(case.src), input(case.addend));
            let expected = Tensor::from_fn(src.shape(), |idx| addend[idx] + src[idx] * 1.5);
            assert_eq!(result, expected);
        })
    }

    #[test]
    fn test_elementwise_program_invalid() {
        #[derive(Debug)]
        struct Case {
            scalar_shape: Vec<usize>,
        }

        let cases = [
            // Scalar input which varies within a row.
            Case {
                scalar_shape: [1, 1, 5].into(),
            },
            // Scalar input which can't be broadcast to the tensor inputs.
            Case {
                scalar_shape: [1, 3, 1].into(),
            },
        ];

        cases.test_each(|case| {
            let x = test_input(&[1, 2, 5]);
            let alpha = Tensor::<f32>::zeros(&case.scalar_shape);
            let op = ElementwiseProgram {
                program: snake_program(),
                inner_dims: 1,
            };
            let result: Result<Tensor<f32>, _> =
                op.run_simple((x.view(), alpha.view(), alpha.view()));
            assert_eq!(
                result.err(),
                Some(OpError::incompatible_input_shapes(
                    "scalar input should be constant within each row"
                ))
            );
        })
    }
}
