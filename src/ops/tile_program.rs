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

use std::borrow::Cow;
use std::ops::Range;

use rayon::prelude::*;
use rten_base::bit_set::BitSet;
use rten_simd::ops::{BitOps, NumOps};
use rten_simd::{Isa, Simd, SimdOp, SimdUnaryOp};
use rten_tensor::prelude::*;
use rten_tensor::{Tensor, TensorView};
use rten_vecmath as vecmath;
use smallvec::SmallVec;

use crate::buffer_pool::AutoReturn;
use crate::infer_shapes::{
    InferShapes, InferShapesContext, InferShapesError, SymTensor, SymbolGen,
};
use crate::operator::{
    InPlaceInputs, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
    OutputTypesContext,
};

/// Number of elements in a tile.
///
/// A tile of the output, of one other input and of two registers together use
/// 16 KB, which fits in a 32 KB L1 cache.
const TILE_LEN: usize = 1024;

/// Number of elements processed by one task when running in parallel.
const TASK_LEN: usize = 32 * TILE_LEN;

/// Identifies a register, which holds one tile of an intermediate value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RegId(pub u8);

impl RegId {
    /// Return the ID of the register with a given index, or `None` if there
    /// are too many registers.
    pub fn new(index: usize) -> Option<RegId> {
        index.try_into().ok().map(RegId)
    }

    const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Source of a tile read by a [`TileOp`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Src {
    /// A tile of one of the operator's tensor inputs.
    Input(usize),
    /// A register holding the result of an earlier operation.
    Reg(RegId),
    /// A tile of one of the operator's outputs, which was written by an
    /// earlier operation. Output 0 cannot be read.
    Output(usize),
}

/// Destination of a tile written by a [`TileOp`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Dst {
    Reg(RegId),
    /// A tile of one of the operator's outputs. Output 0 shares a buffer with
    /// input 0.
    Output(usize),
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
    /// Compute `cos(src)`.
    Cos { dst: Dst, src: Src },
    /// Compute `a * b`.
    Mul { dst: Dst, a: Src, b: Src },
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
    /// Compute `ln(1 + exp(src))`.
    Softplus { dst: Dst, src: Src },
    /// Compute `src * src`.
    Square { dst: Dst, src: Src },
    /// Compute `src - scalar`.
    SubScalar { dst: Dst, src: Src, scalar: usize },
    /// Compute `tanh(src)`.
    Tanh { dst: Dst, src: Src },
    /// Add the sum of the elements of `src` to a scalar.
    Sum { src: Src, scalar: usize },
}

impl TileOp {
    /// Return the tile written by this operation, if any.
    const fn dst(&self) -> Option<Dst> {
        match *self {
            TileOp::Add { dst, .. }
            | TileOp::AddScalar { dst, .. }
            | TileOp::Cos { dst, .. }
            | TileOp::Mul { dst, .. }
            | TileOp::MulScalar { dst, .. }
            | TileOp::MulScalarAdd { dst, .. }
            | TileOp::Sin { dst, .. }
            | TileOp::SinSquared { dst, .. }
            | TileOp::Softplus { dst, .. }
            | TileOp::Square { dst, .. }
            | TileOp::SubScalar { dst, .. }
            | TileOp::Tanh { dst, .. } => Some(dst),
            TileOp::Sum { .. } => None,
        }
    }

    /// Return the scalar read or updated by this operation, if any.
    const fn scalar(&self) -> Option<usize> {
        match *self {
            TileOp::AddScalar { scalar, .. }
            | TileOp::MulScalar { scalar, .. }
            | TileOp::MulScalarAdd { scalar, .. }
            | TileOp::SubScalar { scalar, .. }
            | TileOp::Sum { scalar, .. } => Some(scalar),
            TileOp::Add { .. }
            | TileOp::Cos { .. }
            | TileOp::Mul { .. }
            | TileOp::Sin { .. }
            | TileOp::SinSquared { .. }
            | TileOp::Softplus { .. }
            | TileOp::Square { .. }
            | TileOp::Tanh { .. } => None,
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
    const fn dst(&self) -> usize {
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
///
/// The operations can be borrowed from static data, for programs which are
/// defined at compile time.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pass {
    pub scalar_ops: Cow<'static, [ScalarOp]>,
    pub tile_ops: Cow<'static, [TileOp]>,
}

impl From<Vec<TileOp>> for Pass {
    fn from(tile_ops: Vec<TileOp>) -> Pass {
        Pass {
            scalar_ops: Cow::Borrowed(&[]),
            tile_ops: tile_ops.into(),
        }
    }
}

/// Return the contents of `cow` as a slice.
const fn cow_slice<'a, T: Clone>(cow: &'a Cow<'static, [T]>) -> &'a [T] {
    match cow {
        Cow::Borrowed(items) => items,
        Cow::Owned(items) => items.as_slice(),
    }
}

/// Specifies which dimensions of a [`TileProgram`]'s output form a row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RowDims {
    /// All dimensions, so the whole tensor is one row. This can be used by
    /// programs which have no reductions and whose scalar inputs each have a
    /// single element.
    All,
    /// The given number of trailing dimensions.
    Trailing(usize),
}

impl From<usize> for RowDims {
    fn from(inner_dims: usize) -> RowDims {
        RowDims::Trailing(inner_dims)
    }
}

/// Operator which evaluates a sequence of passes over each row of its inputs.
///
/// The operator's inputs are the program's tensor inputs followed by its
/// scalar inputs. Tensor inputs and outputs all have the same shape. Scalar
/// inputs are broadcast against the output and have a size of one in the
/// dimensions which form a row, so that their value is constant within each
/// row. Scalars computed by the program are zero at the start of each row.
///
/// Programs are created by fusing subgraphs of elementwise operators and
/// reductions over rows. They are also used to implement some ONNX operators,
/// such as [`MISH_PROGRAM`], without dedicated code for each.
#[derive(Clone, Debug, PartialEq)]
pub struct TileProgram {
    passes: Cow<'static, [Pass]>,
    n_inputs: usize,
    n_scalar_inputs: usize,
    /// Number of scalars, including scalar inputs.
    n_scalars: usize,
    n_regs: usize,
    n_outputs: usize,
    row_dims: RowDims,
    /// Name of the operator which this program implements.
    name: Option<&'static str>,
}

impl TileProgram {
    /// Create a program which reads `n_inputs` tensor inputs and
    /// `n_scalar_inputs` scalar inputs.
    ///
    /// `row_dims` specifies the dimensions which are flattened to form a row.
    /// A number is the count of trailing dimensions.
    ///
    /// See [`from_parts`](Self::from_parts) for the requirements on `passes`.
    pub fn new(
        passes: Vec<Pass>,
        n_inputs: usize,
        n_scalar_inputs: usize,
        row_dims: impl Into<RowDims>,
    ) -> TileProgram {
        Self::from_parts(
            Cow::Owned(passes),
            n_inputs,
            n_scalar_inputs,
            row_dims.into(),
        )
    }

    /// Create a program from passes which may be borrowed from static data.
    ///
    /// This can be used to define a program in a `static`, in which case the
    /// program is checked at compile time.
    ///
    /// Each output is written once, in the last pass. The last operation of
    /// that pass writes output 0. Since this shares a buffer with input 0, this
    /// means input 0 can be read by any operation.
    pub const fn from_parts(
        passes: Cow<'static, [Pass]>,
        n_inputs: usize,
        n_scalar_inputs: usize,
        row_dims: RowDims,
    ) -> TileProgram {
        let pass_list = cow_slice(&passes);
        assert!(!pass_list.is_empty(), "program has no passes");
        assert!(n_inputs >= 1);

        // Count the registers, scalars and outputs.
        let mut n_regs = 0;
        let mut n_scalars = n_scalar_inputs;
        let mut n_outputs = 0;
        let mut pass_index = 0;
        while pass_index < pass_list.len() {
            let pass = &pass_list[pass_index];
            let tile_ops = cow_slice(&pass.tile_ops);
            let mut i = 0;
            while i < tile_ops.len() {
                match tile_ops[i].dst() {
                    Some(Dst::Reg(reg)) if reg.index() >= n_regs => n_regs = reg.index() + 1,
                    Some(Dst::Output(_)) => {
                        assert!(
                            pass_index == pass_list.len() - 1,
                            "outputs must be written in the last pass"
                        );
                        n_outputs += 1;
                    }
                    _ => {}
                }
                if let Some(scalar) = tile_ops[i].scalar()
                    && scalar >= n_scalars
                {
                    n_scalars = scalar + 1;
                }
                i += 1;
            }

            let scalar_ops = cow_slice(&pass.scalar_ops);
            let mut i = 0;
            while i < scalar_ops.len() {
                if scalar_ops[i].dst() >= n_scalars {
                    n_scalars = scalar_ops[i].dst() + 1;
                }
                i += 1;
            }
            pass_index += 1;
        }

        // Check that each output is written once, and output 0 is written by
        // the last operation.
        let last_ops = cow_slice(&pass_list[pass_list.len() - 1].tile_ops);
        let mut output = 0;
        while output < n_outputs {
            let mut writes = 0;
            let mut i = 0;
            while i < last_ops.len() {
                if let Some(Dst::Output(index)) = last_ops[i].dst()
                    && index == output
                {
                    writes += 1;
                }
                i += 1;
            }
            assert!(writes == 1, "each output must be written once");
            output += 1;
        }
        assert!(
            !last_ops.is_empty()
                && matches!(last_ops[last_ops.len() - 1].dst(), Some(Dst::Output(0))),
            "the last operation must write output 0"
        );

        TileProgram {
            passes,
            n_inputs,
            n_scalar_inputs,
            n_scalars,
            n_regs,
            n_outputs,
            row_dims,
            name: None,
        }
    }

    /// Set the name which this operator reports, for programs which implement
    /// another operator.
    pub const fn with_name(mut self, name: &'static str) -> TileProgram {
        self.name = Some(name);
        self
    }
}

/// Program which implements the ONNX `Mish` operator.
///
/// This computes `x * tanh(softplus(x))`.
pub static MISH_PROGRAM: TileProgram = {
    const X: Src = Src::Input(0);
    const REG: RegId = RegId(0);
    static TILE_OPS: [TileOp; 3] = [
        TileOp::Softplus {
            dst: Dst::Reg(REG),
            src: X,
        },
        TileOp::Tanh {
            dst: Dst::Reg(REG),
            src: Src::Reg(REG),
        },
        TileOp::Mul {
            dst: Dst::Output(0),
            a: X,
            b: Src::Reg(REG),
        },
    ];
    static PASSES: [Pass; 1] = [Pass {
        scalar_ops: Cow::Borrowed(&[]),
        tile_ops: Cow::Borrowed(&TILE_OPS),
    }];
    TileProgram::from_parts(Cow::Borrowed(&PASSES), 1, 0, RowDims::All).with_name("Mish")
};

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

/// Buffers used while evaluating a [`TileProgram`] on one tile.
struct TileState<'a, 'r, 'o> {
    /// Tensor inputs after the first.
    inputs: &'a [&'a [f32]],
    /// Range of each entry of `inputs` which this tile covers.
    range: Range<usize>,
    regs: &'a mut [&'r mut [f32]],
    /// Tile of each output. Until the final operation, the first holds
    /// input 0.
    outputs: &'a mut [&'o mut [f32]],
}

impl TileState<'_, '_, '_> {
    /// Return the buffers for `srcs`, or `None` for sources which are the
    /// same buffer as `dst`.
    fn sources<const N: usize>(&self, dst: Dst, srcs: [Src; N]) -> [Option<&[f32]>; N] {
        srcs.map(|src| match (src, dst) {
            (Src::Input(0), Dst::Output(0)) => None,
            (Src::Input(0), _) => Some(&self.outputs[0][..]),
            (Src::Input(index), _) => Some(&self.inputs[index - 1][self.range.clone()]),
            (Src::Reg(src_reg), Dst::Reg(dst_reg)) if src_reg == dst_reg => None,
            (Src::Reg(reg), _) => Some(&self.regs[reg.index()][..self.range.len()]),
            // An output is written once, so it is never both a source and
            // the destination.
            (Src::Output(index), _) => Some(&self.outputs[index][..]),
        })
    }

    /// Return the buffer for `src`.
    fn source(&self, src: Src) -> &[f32] {
        match src {
            Src::Input(0) => self.outputs[0],
            Src::Input(index) => &self.inputs[index - 1][self.range.clone()],
            Src::Reg(reg) => &self.regs[reg.index()][..self.range.len()],
            Src::Output(index) => self.outputs[index],
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
                let dst_buf = std::mem::take(&mut self.regs[reg.index()]);
                kernel(&mut dst_buf[..len], self.sources(dst, srcs));
                self.regs[reg.index()] = dst_buf;
            }
            Dst::Output(index) => {
                let dst_buf = std::mem::take(&mut self.outputs[index]);
                kernel(dst_buf, self.sources(dst, srcs));
                self.outputs[index] = dst_buf;
            }
        }
    }

    /// Apply a unary kernel which operates in place.
    fn apply_unary(&mut self, dst: Dst, src: Src, kernel: impl FnOnce(&mut [f32])) {
        self.apply(dst, [src], |dst, [src]| {
            if let Some(src) = src {
                dst.copy_from_slice(src);
            }
            kernel(dst);
        })
    }

    fn run(&mut self, ops: &[TileOp], scalars: &mut [f32]) {
        for op in ops {
            match *op {
                TileOp::Cos { dst, src } => self.apply_unary(dst, src, |dst| {
                    vecmath::Cos::new().map_mut(dst);
                }),
                TileOp::Softplus { dst, src } => self.apply_unary(dst, src, |dst| {
                    vecmath::Softplus {}.map_mut(dst);
                }),
                TileOp::Tanh { dst, src } => self.apply_unary(dst, src, |dst| {
                    vecmath::Tanh {}.map_mut(dst);
                }),
                TileOp::Mul { dst, a, b } => self.apply(dst, [a, b], |dst, srcs| match srcs {
                    [Some(a), Some(b)] => {
                        for ((y, a), b) in dst.iter_mut().zip(a).zip(b) {
                            *y = a * b;
                        }
                    }
                    [Some(x), None] | [None, Some(x)] => {
                        for (y, x) in dst.iter_mut().zip(x) {
                            *y *= x;
                        }
                    }
                    [None, None] => {
                        for y in dst {
                            *y *= *y;
                        }
                    }
                }),
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
                TileOp::Sin { dst, src } => self.apply_unary(dst, src, |dst| {
                    vecmath::Sin::new().map_mut(dst);
                }),
                TileOp::SinSquared { dst, src } => self.apply_unary(dst, src, |dst| {
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

impl TileProgram {
    /// Evaluate the program, where `output` initially holds input 0.
    fn run_in_place_impl(
        &self,
        ctx: &OpRunContext,
        mut output: Tensor<f32>,
    ) -> Result<OutputList, OpError> {
        let TileProgram {
            ref passes,
            n_inputs,
            n_scalar_inputs,
            n_scalars,
            n_regs,
            n_outputs,
            row_dims,
            ..
        } = *self;
        let pool = ctx.pool();
        let inputs = ctx.inputs();

        let outer_dims = match row_dims {
            RowDims::All => 0,
            RowDims::Trailing(inner_dims) => output
                .ndim()
                .checked_sub(inner_dims)
                .ok_or(OpError::invalid_value("input has too few dims"))?,
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

        // The first output reuses the buffer of the first input. Allocate the
        // other outputs.
        let mut other_outputs: Vec<Tensor<f32>> = (1..n_outputs)
            .map(|_| Tensor::zeros_in(pool, output.shape()))
            .collect();

        // Run all of the passes over a segment, which is all or part of a
        // row and starts at offset `start` in the outputs. `segment` holds
        // the part of each output which the segment covers.
        let run_segment = |regs: &mut [&mut [f32]],
                           scalars: &mut [f32],
                           row: usize,
                           start: usize,
                           segment: &mut [&mut [f32]]| {
            let (scalar_inputs, computed_scalars) = scalars.split_at_mut(n_scalar_inputs);
            scalar_inputs.copy_from_slice(&row_scalars[row * n_scalar_inputs..][..n_scalar_inputs]);
            computed_scalars.fill(0.);

            let len = segment[0].len();
            for pass in passes.iter() {
                for op in pass.scalar_ops.iter() {
                    op.run(scalars, len);
                }
                let mut tile_start = 0;
                while tile_start < len {
                    let tile_len = TILE_LEN.min(len - tile_start);
                    let mut outputs: SmallVec<[&mut [f32]; 2]> = segment
                        .iter_mut()
                        .map(|output| &mut output[tile_start..tile_start + tile_len])
                        .collect();
                    let mut tile = TileState {
                        inputs: &other_inputs,
                        range: start + tile_start..start + tile_start + tile_len,
                        regs: &mut *regs,
                        outputs: &mut outputs,
                    };
                    tile.run(&pass.tile_ops, scalars);
                    tile_start += tile_len;
                }
            }
        };

        // With a single pass, each part of a row can be computed separately,
        // so divide long rows between tasks. Otherwise each row is computed
        // by one task, and short rows are grouped together.
        let (chunk_len, min_chunks) = match &passes[..] {
            [_] => (TASK_LEN, 1),
            _ => (row_len, (TASK_LEN / row_len.max(1)).max(1)),
        };

        // Divide the outputs into chunks, where each item holds the
        // corresponding chunk of every output.
        output.make_contiguous_in(pool);
        if !output.is_empty() {
            let mut other_chunks: Vec<_> = other_outputs
                .iter_mut()
                .map(|output| output.data_mut().unwrap().chunks_mut(chunk_len))
                .collect();
            let chunks: Vec<SmallVec<[&mut [f32]; 2]>> = output
                .data_mut()
                .unwrap()
                .chunks_mut(chunk_len)
                .map(|chunk| {
                    let other_chunks = other_chunks.iter_mut().map(|chunks| chunks.next().unwrap());
                    std::iter::once(chunk).chain(other_chunks).collect()
                })
                .collect();

            chunks
                .into_par_iter()
                .enumerate()
                .with_min_len(min_chunks)
                .for_each_init(
                    || (vec![0.; n_regs * TILE_LEN], vec![0.; n_scalars]),
                    |(reg_buf, scalars), (index, mut chunk)| {
                        let mut regs: Vec<&mut [f32]> = reg_buf.chunks_mut(TILE_LEN).collect();
                        let chunk_start = index * chunk_len;
                        let chunk_len = chunk[0].len();

                        // Split the chunk into segments which each lie
                        // within one row.
                        let mut pos = 0;
                        while pos < chunk_len {
                            let start = chunk_start + pos;
                            let row = start / row_len;
                            let row_end = (row + 1) * row_len;
                            let len = (chunk_len - pos).min(row_end - start);
                            let mut segment: SmallVec<[&mut [f32]; 2]> = chunk
                                .iter_mut()
                                .map(|output| &mut output[pos..pos + len])
                                .collect();
                            run_segment(&mut regs, scalars, row, start, &mut segment);
                            pos += len;
                        }
                    },
                );
        }

        Ok(std::iter::once(output)
            .chain(other_outputs)
            .map(|output| output.into())
            .collect())
    }
}

impl InferShapes for TileProgram {
    fn infer_shapes(
        &self,
        inputs: InferShapesContext,
        _sym_gen: &mut SymbolGen,
    ) -> Result<Vec<SymTensor>, InferShapesError> {
        let shape = match inputs.require(0)?.shape() {
            Some(shape) => SymTensor::from_shape(shape.collect()),
            None => SymTensor::unknown("unknown input shape"),
        };
        Ok(vec![shape; self.n_outputs])
    }
}

impl Operator for TileProgram {
    fn name(&self) -> &str {
        self.name.unwrap_or("TileProgram")
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(self.n_inputs + self.n_scalar_inputs)
    }

    fn max_outputs(&self) -> Option<usize> {
        Some(self.n_outputs)
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let input: TensorView<f32> = ctx.inputs().require_as(0)?;
        self.run_in_place_impl(ctx, input.to_tensor_in(ctx.pool()))
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
        self.run_in_place_impl(ctx, input)
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(self)
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some(
            (0..self.n_outputs)
                .map(|_| OutputType::CopyFromInput(0))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::Tensor;
    use rten_tensor::prelude::*;
    use rten_tensor::test_util::{expect_equal, expect_equal_with_tolerance};
    use rten_testing::TestCases;

    use super::{
        Dst, MISH_PROGRAM, Pass, RegId, ScalarOp, Src, TASK_LEN, TILE_LEN, TileOp, TileProgram,
    };
    use crate::buffer_pool::BufferPool;
    use crate::operator::{InputList, OpError, OpRunContext, Operator, OperatorExt, OutputMask};

    /// Return a program which computes `x + beta * sin(alpha * x)^2`, where
    /// `alpha` and `beta` are the scalar inputs.
    fn snake_program(inner_dims: usize) -> TileProgram {
        let reg = Dst::Reg(RegId(0));
        let reg_src = Src::Reg(RegId(0));
        TileProgram::new(
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
                    dst: Dst::Output(0),
                    a: Src::Input(0),
                    b: reg_src,
                },
            ])],
            1,
            2,
            inner_dims,
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
    fn test_tile_program() {
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

            let op = snake_program(case.inner_dims);
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
    fn test_tile_program_multiple_inputs() {
        // Compute `sin(a) + b`.
        let op = TileProgram::new(
            vec![Pass::from(vec![
                TileOp::Sin {
                    dst: Dst::Reg(RegId(0)),
                    src: Src::Input(1),
                },
                TileOp::Add {
                    dst: Dst::Output(0),
                    a: Src::Reg(RegId(0)),
                    b: Src::Input(0),
                },
            ])],
            2,
            0,
            2,
        );
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
    fn test_tile_program_mul_scalar_add() {
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
            let op = TileProgram::new(
                vec![Pass::from(vec![TileOp::MulScalarAdd {
                    dst: Dst::Output(0),
                    src: case.src,
                    scalar: 0,
                    addend: case.addend,
                }])],
                2,
                1,
                2,
            );
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
                Src::Reg(_) | Src::Output(_) => unreachable!(),
            };
            let (src, addend) = (input(case.src), input(case.addend));
            let expected = Tensor::from_fn(src.shape(), |idx| addend[idx] + src[idx] * 1.5);
            expect_equal(&result, &expected).unwrap();
        })
    }

    #[test]
    fn test_tile_program_multiple_passes() {
        // Normalize each row to have zero mean and unit variance, then
        // compute `normalized * gamma + beta`. Scalars are the inputs
        // (epsilon, gamma, beta) followed by the mean, variance, variance +
        // epsilon, standard deviation and its reciprocal.
        let reg = Dst::Reg(RegId(0));
        let reg_src = Src::Reg(RegId(0));
        let x = Src::Input(0);
        let op = TileProgram::new(
            vec![
                vec![TileOp::Sum { src: x, scalar: 3 }].into(),
                Pass {
                    scalar_ops: vec![ScalarOp::DivRowLen { scalar: 3 }].into(),
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
                    ]
                    .into(),
                },
                Pass {
                    scalar_ops: vec![
                        ScalarOp::DivRowLen { scalar: 4 },
                        ScalarOp::Add { dst: 5, a: 4, b: 0 },
                        ScalarOp::Sqrt { dst: 6, src: 5 },
                        ScalarOp::Reciprocal { dst: 7, src: 6 },
                    ]
                    .into(),
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
                            dst: Dst::Output(0),
                            src: reg_src,
                            scalar: 2,
                        },
                    ]
                    .into(),
                },
            ],
            1,
            3,
            1,
        );

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
    fn test_tile_program_multiple_outputs() {
        // Convert magnitudes (input 1) and phases (input 0) to imaginary
        // (output 0) and real (output 1) parts.
        let program = TileProgram::new(
            vec![Pass::from(vec![
                TileOp::Sin {
                    dst: Dst::Reg(RegId(0)),
                    src: Src::Input(0),
                },
                TileOp::Cos {
                    dst: Dst::Reg(RegId(1)),
                    src: Src::Reg(RegId(0)),
                },
                TileOp::Softplus {
                    dst: Dst::Reg(RegId(2)),
                    src: Src::Input(1),
                },
                TileOp::Mul {
                    dst: Dst::Output(1),
                    a: Src::Reg(RegId(2)),
                    b: Src::Reg(RegId(1)),
                },
                TileOp::Sin {
                    dst: Dst::Reg(RegId(0)),
                    src: Src::Reg(RegId(0)),
                },
                TileOp::Mul {
                    dst: Dst::Output(0),
                    a: Src::Reg(RegId(2)),
                    b: Src::Reg(RegId(0)),
                },
            ])],
            2,
            0,
            3,
        );

        #[derive(Debug)]
        struct Case {
            shape: Vec<usize>,
        }

        let cases = [
            // Input which spans several tiles.
            Case {
                shape: [1, 3, TILE_LEN + 5].into(),
            },
            // Input which spans several tasks.
            Case {
                shape: [1, 3, TASK_LEN].into(),
            },
            // Empty input.
            Case {
                shape: [1, 3, 0].into(),
            },
        ];

        cases.test_each(|case| {
            let phase = test_input(&case.shape);
            let mag = phase.map(|x| x * 0.5 + 0.25);

            let pool = BufferPool::new();
            let inputs: InputList = (phase.view(), mag.view()).into();
            let ctx = OpRunContext::new(&pool, &inputs, OutputMask::all_used(2));
            let outputs: Vec<Tensor<f32>> = program
                .run(&ctx)
                .unwrap()
                .into_iter()
                .map(|output| output.try_into().unwrap())
                .collect();

            let softplus = |x: f32| x.exp().ln_1p();
            let expected_imag = Tensor::from_fn(phase.shape(), |idx| {
                softplus(mag[idx]) * phase[idx].sin().sin()
            });
            let expected_real = Tensor::from_fn(phase.shape(), |idx| {
                softplus(mag[idx]) * phase[idx].sin().cos()
            });
            expect_equal_with_tolerance(&outputs[0], &expected_imag, 1e-5, 1e-5).unwrap();
            expect_equal_with_tolerance(&outputs[1], &expected_real, 1e-5, 1e-5).unwrap();
        })
    }

    #[test]
    fn test_mish() {
        #[derive(Debug)]
        struct Case {
            shape: Vec<usize>,
        }

        let cases = [
            Case { shape: [].into() },
            Case { shape: [5].into() },
            Case {
                shape: [2, 3, TILE_LEN + 5].into(),
            },
            Case {
                shape: [1, 2, 3, 0].into(),
            },
        ];

        cases.test_each(|case| {
            let op = MISH_PROGRAM.clone();
            assert_eq!(op.name(), "Mish");

            // Scale the input to cover large positive and negative values.
            let x = test_input(&case.shape).map(|x| x * 20.);
            let expected = x.map(|x: &f32| x * x.exp().ln_1p().tanh());

            let result: Tensor<f32> = op.run_simple(x.view()).unwrap();
            expect_equal_with_tolerance(&result, &expected, 1e-5, 1e-5).unwrap();

            let result: Tensor<f32> = op.run_simple_in_place(x.clone(), ()).unwrap();
            expect_equal_with_tolerance(&result, &expected, 1e-5, 1e-5).unwrap();
        })
    }

    #[test]
    fn test_tile_program_invalid() {
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
            let op = snake_program(1);
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
