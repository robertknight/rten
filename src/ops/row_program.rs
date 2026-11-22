//! Fused operator which evaluates a sequence of elementwise operations, and
//! reductions over rows, on one small tile of its inputs at a time.
//!
//! Evaluating an expression such as `x + beta * sin(alpha * x)^2` as separate
//! graph operators makes one pass over the whole tensor per operator. Here
//! the tensor is instead split into tiles which are small enough that a tile
//! of each input and intermediate value fits in the L1 cache, and all of the
//! operations are applied to one tile before moving on to the next.
//!
//! Programs can also reduce the values in each _row_ of the tensor to a
//! scalar, and use it in later operations on the same row. This requires
//! several passes over the row. For example normalizing each row to have zero
//! mean and unit variance takes one pass to compute the mean, a second to
//! compute the variance and a third to compute the output.

use std::ops::Range;

use rayon::prelude::*;
use rten_base::bit_set::BitSet;
use rten_simd::ops::{BitOps, NumOps};
use rten_simd::{Isa, Simd, SimdOp, SimdUnaryOp};
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
/// The destination may be the same register as a source. Scalar operands
/// are indices of scalars, which are the operator's scalar inputs followed by
/// the scalars computed by the program.
#[derive(Clone, Debug, PartialEq)]
pub enum TileOp {
    /// Compute `a + b`.
    Add { dst: Dst, a: Src, b: Src },
    /// Compute `src + scalar`.
    AddScalar { dst: Dst, src: Src, scalar: usize },
    /// Compute `src * scalar`.
    MulScalar { dst: Dst, src: Src, scalar: usize },
    /// Compute `addend + src * scalar`. This uses a fused multiply-add if
    /// available.
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
    /// Compute `src - scalar`.
    SubScalar { dst: Dst, src: Src, scalar: usize },
    /// Add the sum of the elements of `src` to a scalar.
    Sum { src: Src, scalar: usize },
}

impl TileOp {
    /// Return the tile written by this operation, if any.
    fn dst(&self) -> Option<Dst> {
        match *self {
            TileOp::Add { dst, .. }
            | TileOp::AddScalar { dst, .. }
            | TileOp::MulScalar { dst, .. }
            | TileOp::MulScalarAdd { dst, .. }
            | TileOp::Sin { dst, .. }
            | TileOp::SinSquared { dst, .. }
            | TileOp::Square { dst, .. }
            | TileOp::SubScalar { dst, .. } => Some(dst),
            TileOp::Sum { .. } => None,
        }
    }

    /// Return the scalar read or updated by this operation, if any.
    fn scalar(&self) -> Option<usize> {
        match *self {
            TileOp::AddScalar { scalar, .. }
            | TileOp::MulScalar { scalar, .. }
            | TileOp::MulScalarAdd { scalar, .. }
            | TileOp::SubScalar { scalar, .. }
            | TileOp::Sum { scalar, .. } => Some(scalar),
            TileOp::Add { .. }
            | TileOp::Sin { .. }
            | TileOp::SinSquared { .. }
            | TileOp::Square { .. } => None,
        }
    }
}

/// An operation on the scalars for one row.
#[derive(Clone, Debug, PartialEq)]
pub enum ScalarOp {
    /// Compute `a + b`.
    Add { dst: usize, a: usize, b: usize },
    /// Divide a scalar by the number of elements in the row.
    DivRowLen { scalar: usize },
    /// Compute `1 / src`.
    Reciprocal { dst: usize, src: usize },
    /// Compute `sqrt(src)`.
    Sqrt { dst: usize, src: usize },
}

impl ScalarOp {
    /// Return the scalar written by this operation.
    fn dst(&self) -> usize {
        match *self {
            ScalarOp::Add { dst, .. }
            | ScalarOp::Reciprocal { dst, .. }
            | ScalarOp::Sqrt { dst, .. } => dst,
            ScalarOp::DivRowLen { scalar } => scalar,
        }
    }

    fn run(&self, scalars: &mut [f32], row_len: usize) {
        match *self {
            ScalarOp::Add { dst, a, b } => scalars[dst] = scalars[a] + scalars[b],
            ScalarOp::DivRowLen { scalar } => scalars[scalar] /= row_len as f32,
            ScalarOp::Reciprocal { dst, src } => scalars[dst] = 1. / scalars[src],
            ScalarOp::Sqrt { dst, src } => scalars[dst] = scalars[src].sqrt(),
        }
    }
}

/// One pass over a row.
///
/// The scalar operations are run first, followed by the tile operations for
/// each tile of the row.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pass {
    pub scalar_ops: Vec<ScalarOp>,
    pub tile_ops: Vec<TileOp>,
}

impl From<Vec<TileOp>> for Pass {
    fn from(tile_ops: Vec<TileOp>) -> Pass {
        Pass {
            scalar_ops: Vec::new(),
            tile_ops,
        }
    }
}

/// A sequence of passes which compute one row of an operator's output from
/// the corresponding rows of its inputs.
///
/// Scalars computed by the program are zero at the start of each row.
#[derive(Clone, Debug, PartialEq)]
pub struct RowProgramCode {
    passes: Vec<Pass>,
    n_inputs: usize,
    n_scalar_inputs: usize,
    /// Number of scalars, including scalar inputs.
    n_scalars: usize,
    n_regs: usize,
}

impl RowProgramCode {
    /// Create a program which reads `n_inputs` tensor inputs and
    /// `n_scalar_inputs` scalar inputs.
    ///
    /// Only the last operation of the last pass may write the output. Since
    /// the output shares a buffer with input 0, this means input 0 can be
    /// read by any operation.
    pub fn new(passes: Vec<Pass>, n_inputs: usize, n_scalar_inputs: usize) -> RowProgramCode {
        let tile_ops = || passes.iter().flat_map(|pass| &pass.tile_ops);
        let n_tile_ops = tile_ops().count();
        assert!(tile_ops().enumerate().all(|(index, op)| {
            let is_last = index == n_tile_ops - 1;
            (op.dst() == Some(Dst::Output)) == is_last
        }));
        assert!(n_inputs >= 1);

        let n_regs = tile_ops()
            .filter_map(|op| match op.dst() {
                Some(Dst::Reg(reg)) => Some(reg + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let n_scalars = tile_ops()
            .filter_map(|op| op.scalar())
            .chain(
                passes
                    .iter()
                    .flat_map(|pass| &pass.scalar_ops)
                    .map(|op| op.dst()),
            )
            .map(|scalar| scalar + 1)
            .max()
            .unwrap_or(0)
            .max(n_scalar_inputs);

        RowProgramCode {
            passes,
            n_inputs,
            n_scalar_inputs,
            n_scalars,
            n_regs,
        }
    }
}

/// Computes `addend + src * scale` and writes the result to `dst`.
///
/// `src` and `addend` are `None` if they are the same buffer as `dst`.
struct MulScalarAdd<'a> {
    dst: &'a mut [f32],
    src: Option<&'a [f32]>,
    addend: Option<&'a [f32]>,
    scale: f32,
}

impl SimdOp for MulScalarAdd<'_> {
    type Output = ();

    #[inline(always)]
    fn eval<I: Isa>(self, isa: I) {
        let MulScalarAdd {
            dst,
            src,
            addend,
            scale,
        } = self;
        let ops = isa.f32();
        let scale = ops.splat(scale);
        let v_len = ops.len();

        // Process the whole vectors. There is a separate loop for each
        // combination of sources, so that the chunks of each buffer can be
        // iterated over together.
        let fma = |x: &[f32], a: &[f32]| ops.mul_add(ops.load(x), scale, ops.load(a));
        let pos = dst.len() - dst.len() % v_len;
        let dst_chunks = dst[..pos].chunks_exact_mut(v_len);
        match (src, addend) {
            (Some(src), Some(addend)) => {
                let srcs = src.chunks_exact(v_len).zip(addend.chunks_exact(v_len));
                for (y, (x, a)) in dst_chunks.zip(srcs) {
                    ops.store(fma(x, a), y);
                }
            }
            (Some(src), None) => {
                for (y, x) in dst_chunks.zip(src.chunks_exact(v_len)) {
                    ops.store(fma(x, y), y);
                }
            }
            (None, Some(addend)) => {
                for (y, a) in dst_chunks.zip(addend.chunks_exact(v_len)) {
                    ops.store(fma(y, a), y);
                }
            }
            (None, None) => {
                for y in dst_chunks {
                    ops.store(fma(y, y), y);
                }
            }
        }

        if pos < dst.len() {
            let (x, _) = ops.load_pad(&src.unwrap_or(dst)[pos..]);
            let (a, _) = ops.load_pad(&addend.unwrap_or(dst)[pos..]);
            let y = ops.mul_add(x, scale, a).to_array();
            let tail = &mut dst[pos..];
            tail.copy_from_slice(&y.as_ref()[..tail.len()]);
        }
    }
}

/// Buffers used while evaluating a [`RowProgramCode`] on one tile.
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

    /// Return the buffer for `src`.
    fn source(&self, src: Src) -> &[f32] {
        match src {
            Src::Input(0) => self.output,
            Src::Input(index) => &self.inputs[index - 1][self.range.clone()],
            Src::Reg(reg) => &self.regs[reg][..self.range.len()],
        }
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

    fn run(&mut self, ops: &[TileOp], scalars: &mut [f32]) {
        for op in ops {
            match *op {
                TileOp::AddScalar { dst, src, scalar } => {
                    let scalar = scalars[scalar];
                    self.apply(dst, [src], |dst, [src]| match src {
                        Some(src) => {
                            for (y, x) in dst.iter_mut().zip(src) {
                                *y = x + scalar;
                            }
                        }
                        None => {
                            for y in dst {
                                *y += scalar;
                            }
                        }
                    })
                }
                TileOp::SubScalar { dst, src, scalar } => {
                    let scalar = scalars[scalar];
                    self.apply(dst, [src], |dst, [src]| match src {
                        Some(src) => {
                            for (y, x) in dst.iter_mut().zip(src) {
                                *y = x - scalar;
                            }
                        }
                        None => {
                            for y in dst {
                                *y -= scalar;
                            }
                        }
                    })
                }
                TileOp::Sum { src, scalar } => {
                    scalars[scalar] += vecmath::Sum::new(self.source(src)).dispatch();
                }
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
                    self.apply(dst, [src, addend], |dst, [src, addend]| {
                        MulScalarAdd {
                            dst,
                            src,
                            addend,
                            scale,
                        }
                        .dispatch()
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

/// Operator which evaluates a [`RowProgramCode`].
///
/// The operator's inputs are the program's tensor inputs followed by its
/// scalar inputs. Tensor inputs all have the shape of the output. Scalar
/// inputs are broadcast against the output and have a size of one in the last
/// `inner_dims` dimensions, so that their value is constant within each _row_
/// formed by flattening those dimensions.
///
/// This is not an ONNX operator. It is created by fusing subgraphs of
/// elementwise operators and reductions over rows.
#[derive(Debug)]
pub struct RowProgram {
    pub program: RowProgramCode,

    /// Number of trailing dimensions which are flattened to form a row.
    pub inner_dims: usize,
}

impl RowProgram {
    /// Evaluate the program, where `output` initially holds input 0.
    fn run_in_place_impl(
        &self,
        ctx: &OpRunContext,
        mut output: Tensor<f32>,
    ) -> Result<Tensor<f32>, OpError> {
        let RowProgramCode {
            passes,
            n_inputs,
            n_scalar_inputs,
            n_scalars,
            n_regs,
        } = &self.program;
        let (n_inputs, n_scalar_inputs, n_scalars, n_regs) =
            (*n_inputs, *n_scalar_inputs, *n_scalars, *n_regs);
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
        let mut row_scalars = vec![0.; n_rows * n_scalar_inputs];
        for index in 0..n_scalar_inputs {
            let scalar: TensorView<f32> = inputs.require_as(n_inputs + index)?;
            let Some(scalar) = scalar.try_broadcast(row_shape.as_slice()).ok() else {
                return Err(OpError::incompatible_input_shapes(
                    "scalar input should be constant within each row",
                ));
            };
            for (row, value) in scalar.iter().enumerate() {
                row_scalars[row * n_scalar_inputs + index] = *value;
            }
        }

        if output.is_empty() {
            return Ok(output);
        }

        // Run all of the passes over `segment`, which is all or part of a
        // row and starts at offset `start` in the output.
        let run_segment = |regs: &mut [&mut [f32]],
                           scalars: &mut [f32],
                           row: usize,
                           start: usize,
                           segment: &mut [f32]| {
            let (scalar_inputs, computed_scalars) = scalars.split_at_mut(n_scalar_inputs);
            scalar_inputs.copy_from_slice(&row_scalars[row * n_scalar_inputs..][..n_scalar_inputs]);
            computed_scalars.fill(0.);

            for pass in passes {
                for op in &pass.scalar_ops {
                    op.run(scalars, segment.len());
                }
                let mut tile_start = start;
                for tile in segment.chunks_mut(TILE_LEN) {
                    let range = tile_start..tile_start + tile.len();
                    tile_start = range.end;
                    let mut tile = TileState {
                        inputs: &other_inputs,
                        range,
                        regs,
                        output: tile,
                    };
                    tile.run(&pass.tile_ops, scalars);
                }
            }
        };
        let init = || (vec![0.; n_regs * TILE_LEN], vec![0.; n_scalars]);

        output.make_contiguous_in(pool);
        let data = output.data_mut().unwrap();
        if let [_] = passes.as_slice() {
            // With a single pass, each part of a row can be computed
            // separately, so divide long rows between tasks.
            data.par_chunks_mut(TASK_LEN).enumerate().for_each_init(
                init,
                |(reg_buf, scalars), (task, chunk)| {
                    let mut regs: Vec<&mut [f32]> = reg_buf.chunks_mut(TILE_LEN).collect();
                    let task_start = task * TASK_LEN;

                    // Split the chunk into segments which each lie within
                    // one row.
                    let mut pos = 0;
                    while pos < chunk.len() {
                        let start = task_start + pos;
                        let row = start / row_len;
                        let row_end = (row + 1) * row_len;
                        let len = (chunk.len() - pos).min(row_end - start);
                        run_segment(&mut regs, scalars, row, start, &mut chunk[pos..pos + len]);
                        pos += len;
                    }
                },
            );
        } else {
            data.par_chunks_mut(row_len)
                .enumerate()
                .with_min_len((TASK_LEN / row_len).max(1))
                .for_each_init(init, |(reg_buf, scalars), (row, row_data)| {
                    let mut regs: Vec<&mut [f32]> = reg_buf.chunks_mut(TILE_LEN).collect();
                    run_segment(&mut regs, scalars, row, row * row_len, row_data);
                });
        }

        Ok(output)
    }
}

impl Operator for RowProgram {
    fn name(&self) -> &str {
        "RowProgram"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(self.program.n_inputs + self.program.n_scalar_inputs)
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
    use rten_tensor::test_util::{expect_equal, expect_equal_with_tolerance};
    use rten_testing::TestCases;

    use super::{Dst, Pass, RowProgram, RowProgramCode, ScalarOp, Src, TASK_LEN, TILE_LEN, TileOp};
    use crate::operator::{OpError, OperatorExt};

    /// Return a program which computes `x + beta * sin(alpha * x)^2`, where
    /// `alpha` and `beta` are the scalar inputs.
    fn snake_program() -> RowProgramCode {
        let reg = Dst::Reg(0);
        let reg_src = Src::Reg(0);
        RowProgramCode::new(
            vec![Pass::from(vec![
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
            ])],
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
    fn test_row_program() {
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

            let op = RowProgram {
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
    fn test_row_program_multiple_inputs() {
        // Compute `sin(a) + b`.
        let program = RowProgramCode::new(
            vec![Pass::from(vec![
                TileOp::Sin {
                    dst: Dst::Reg(0),
                    src: Src::Input(1),
                },
                TileOp::Add {
                    dst: Dst::Output,
                    a: Src::Reg(0),
                    b: Src::Input(0),
                },
            ])],
            2,
            0,
        );
        let op = RowProgram {
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
    fn test_row_program_mul_scalar_add() {
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
            let program = RowProgramCode::new(
                vec![Pass::from(vec![TileOp::MulScalarAdd {
                    dst: Dst::Output,
                    src: case.src,
                    scalar: 0,
                    addend: case.addend,
                }])],
                2,
                1,
            );
            let op = RowProgram {
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
            expect_equal(&result, &expected).unwrap();
        })
    }

    #[test]
    fn test_row_program_multiple_passes() {
        // Normalize each row to have zero mean and unit variance, then
        // compute `normalized * gamma + beta`. Scalars are the inputs
        // (epsilon, gamma, beta) followed by the mean, variance, variance +
        // epsilon, standard deviation and its reciprocal.
        let reg = Dst::Reg(0);
        let reg_src = Src::Reg(0);
        let x = Src::Input(0);
        let program = RowProgramCode::new(
            vec![
                vec![TileOp::Sum { src: x, scalar: 3 }].into(),
                Pass {
                    scalar_ops: vec![ScalarOp::DivRowLen { scalar: 3 }],
                    tile_ops: vec![
                        TileOp::SubScalar {
                            dst: reg,
                            src: x,
                            scalar: 3,
                        },
                        TileOp::Square {
                            dst: reg,
                            src: reg_src,
                        },
                        TileOp::Sum {
                            src: reg_src,
                            scalar: 4,
                        },
                    ],
                },
                Pass {
                    scalar_ops: vec![
                        ScalarOp::DivRowLen { scalar: 4 },
                        ScalarOp::Add { dst: 5, a: 4, b: 0 },
                        ScalarOp::Sqrt { dst: 6, src: 5 },
                        ScalarOp::Reciprocal { dst: 7, src: 6 },
                    ],
                    tile_ops: vec![
                        TileOp::SubScalar {
                            dst: reg,
                            src: x,
                            scalar: 3,
                        },
                        TileOp::MulScalar {
                            dst: reg,
                            src: reg_src,
                            scalar: 7,
                        },
                        TileOp::MulScalar {
                            dst: reg,
                            src: reg_src,
                            scalar: 1,
                        },
                        TileOp::AddScalar {
                            dst: Dst::Output,
                            src: reg_src,
                            scalar: 2,
                        },
                    ],
                },
            ],
            1,
            3,
        );
        let op = RowProgram {
            program,
            inner_dims: 1,
        };

        #[derive(Debug)]
        struct Case {
            shape: Vec<usize>,
        }

        let cases = [
            // Rows shorter than a tile.
            Case {
                shape: [2, 3, 50].into(),
            },
            // Rows which span several tiles.
            Case {
                shape: [2, 3, 2 * TILE_LEN + 7].into(),
            },
            // Many short rows, which are grouped into tasks.
            Case {
                shape: [1, 4 * TASK_LEN / 300, 300].into(),
            },
        ];

        cases.test_each(|case| {
            let x = test_input(&case.shape);
            let chans = case.shape[1];
            let epsilon = Tensor::from(1e-5);
            let gamma = Tensor::from_data(
                &[1, chans, 1],
                (0..chans)
                    .map(|i| 0.5 + i as f32 * 0.25)
                    .collect::<Vec<_>>(),
            );
            let beta = gamma.map(|x| x - 1.);

            let result: Tensor<f32> = op
                .run_simple((x.view(), epsilon.view(), gamma.view(), beta.view()))
                .unwrap();

            let mut expected = x.clone();
            for (row, mut lane) in expected.lanes_mut(2).enumerate() {
                let chan = row % chans;
                let lane = lane.as_slice_mut().unwrap();
                let len = lane.len() as f32;
                let mean = lane.iter().sum::<f32>() / len;
                let var = lane.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / len;
                let std = (var + 1e-5).sqrt();
                for x in lane.iter_mut() {
                    *x = (*x - mean) / std * gamma[[0, chan, 0]] + beta[[0, chan, 0]];
                }
            }
            expect_equal_with_tolerance(&result, &expected, 1e-4, 1e-4).unwrap();
        })
    }

    #[test]
    fn test_row_program_invalid() {
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
            let op = RowProgram {
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
