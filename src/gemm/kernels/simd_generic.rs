use std::mem::MaybeUninit;

use rten_simd::{SimdFloat, SimdInt};
use rten_tensor::{Matrix, MatrixLayout, Storage};

use super::Lhs;
use crate::iter_util::{range_chunks_exact, unroll_loop, unroll_loop_x4};

/// Compute an output block of a vector-matrix product ("gemv" in BLAS APIs).
///
/// Multiple output columns are computed at a time, using `NR_REGS` SIMD
/// registers of type `S`. See [`Kernel::gemv_kernel`].
///
/// If `beta` is zero the output may be uninitialized. The output will always
/// be initialized after the kernel has run.
///
/// Safety: The `SimdFloat` type must be supported on the current system.
#[inline(always)]
pub unsafe fn simd_gemv<S: SimdFloat, const NR_REGS: usize>(
    out: &mut [MaybeUninit<f32>],
    a: &[f32],
    b: Matrix,
    alpha: f32,
    beta: f32,
) {
    // Handle cases where `b` does not have unit stride.
    if b.row_stride() == 1 {
        return simd_gemv_transposed::<S>(out, a, b, alpha, beta);
    } else if b.col_stride() != 1 {
        return simd_gemv_fallback(out, a, b, alpha, beta);
    }

    assert!(b.col_stride() == 1);
    assert!(a.len() == b.rows());
    assert!(out.len() == b.cols());

    let out_ptr = out.as_mut_ptr();
    let a_ptr = a.as_ptr();
    let b_ptr = b.storage().as_ptr();
    let b_row_stride = b.row_stride();

    let mut b_tiles = range_chunks_exact(0..b.cols(), NR_REGS * S::LEN);
    for b_tile in b_tiles.by_ref() {
        let mut acc = [S::zero(); NR_REGS];
        unroll_loop!(0..a.len(), k, 4, {
            let a_elt = *a_ptr.add(k);
            let a_elts = S::splat(a_elt);

            // Pre-fetch the current row for the next column tile.
            S::prefetch(b_ptr.add(k * b_row_stride + b_tile.start + NR_REGS + S::LEN));

            for i in 0..NR_REGS {
                let b_elts = S::load(b_ptr.add(k * b_row_stride + b_tile.start + i * S::LEN));
                acc[i] = a_elts.mul_add(b_elts, acc[i]);
            }
        });

        if alpha != 1. {
            let alpha_vec = S::splat(alpha);
            for i in 0..NR_REGS {
                acc[i] = acc[i].mul(alpha_vec);
            }
        }

        let get_out_tile_ptr = |i| out_ptr.add(b_tile.start + i * S::LEN);

        if beta == 0. {
            for i in 0..NR_REGS {
                acc[i].store(get_out_tile_ptr(i) as *mut f32);
            }
        } else if beta == 1. {
            for i in 0..NR_REGS {
                let out_tile_ptr = get_out_tile_ptr(i);
                let out_tile = S::load(out_tile_ptr as *mut f32).add(acc[i]);
                out_tile.store(out_tile_ptr as *mut f32);
            }
        } else {
            let beta_vec = S::splat(beta);
            for i in 0..NR_REGS {
                let out_tile_ptr = get_out_tile_ptr(i);
                let out_tile = S::load(out_tile_ptr as *mut f32).mul_add(beta_vec, acc[i]);
                out_tile.store(out_tile_ptr as *mut f32);
            }
        }
    }

    for c in b_tiles.remainder() {
        let mut acc = 0.;
        for k in 0..a.len() {
            acc += *a_ptr.add(k) * *b_ptr.add(k * b_row_stride + c);
        }
        let out_el = out_ptr.add(c);
        let tmp = if beta == 0. {
            0.
        } else {
            (*out_el).assume_init()
        };
        *out_el = MaybeUninit::new(beta * tmp + acc * alpha);
    }
}

/// Variant of [`simd_gemv`] which handles the case where `b` has unit row stride.
#[inline(always)]
unsafe fn simd_gemv_transposed<S: SimdFloat>(
    out: &mut [MaybeUninit<f32>],
    a: &[f32],
    b: Matrix,
    alpha: f32,
    beta: f32,
) {
    assert!(b.row_stride() == 1);
    assert!(a.len() == b.rows());
    assert!(out.len() == b.cols());

    let b_ptr = b.storage().as_ptr();
    let b_col_stride = b.col_stride();

    const COL_TILE: usize = 8;

    let mut col_tiles = range_chunks_exact(0..b.cols(), COL_TILE);
    for col_tile in col_tiles.by_ref() {
        let mut acc = [S::zero(); COL_TILE];

        let mut depth_tiles = range_chunks_exact(0..a.len(), S::LEN);
        for depth_tile in depth_tiles.by_ref() {
            let a_tile = S::load(a.as_ptr().add(depth_tile.start));
            for i in 0..COL_TILE {
                let b_col_ptr = b_ptr.add((col_tile.start + i) * b_col_stride);
                let b_tile = S::load(b_col_ptr.add(depth_tile.start));
                acc[i] = a_tile.mul_add(b_tile, acc[i]);
            }
        }

        let mut acc: [f32; COL_TILE] = std::array::from_fn(|i| acc[i].sum());
        for k in depth_tiles.remainder() {
            let ak = *a.get_unchecked(k);
            for i in 0..COL_TILE {
                let b_col_ptr = b_ptr.add((col_tile.start + i) * b_col_stride);
                let bk = *b_col_ptr.add(k);
                acc[i] = ak.mul_add(bk, acc[i]);
            }
        }

        if beta == 0. {
            for i in 0..COL_TILE {
                out[col_tile.start + i].write(alpha * acc[i]);
            }
        } else {
            for i in 0..COL_TILE {
                // Safety: Output is initialized when `beta` is non-zero.
                let out_val =
                    alpha * acc[i] + beta * unsafe { out[col_tile.start + i].assume_init() };
                out[col_tile.start + i].write(out_val);
            }
        }
    }

    let last_col_tile = col_tiles.remainder();
    if !last_col_tile.is_empty() {
        simd_gemv_fallback(
            &mut out[last_col_tile.clone()],
            a,
            b.slice((.., last_col_tile)),
            alpha,
            beta,
        );
    }
}

/// Variant of [`simd_gemv`] which handles the case where `b` has non-unit strides
/// for rows and columns.
///
/// This doesn't benefit from SIMD operations. It is at least inlined so it
/// can benefit from the kernel's instruction set (eg. for FMA operations).
#[inline(always)]
fn simd_gemv_fallback(out: &mut [MaybeUninit<f32>], a: &[f32], b: Matrix, alpha: f32, beta: f32) {
    assert!(a.len() == b.rows());
    assert!(out.len() == b.cols());

    for (col, out) in out.iter_mut().enumerate() {
        let mut acc = 0.;
        for (k, ak) in (0..a.len()).zip(a.iter()) {
            let bk = unsafe { *b.get_unchecked([k, col]) };
            acc = ak.mul_add(bk, acc);
        }
        acc *= alpha;
        if beta == 0. {
            out.write(acc);
        } else {
            // Safety: Output is initialized when `beta` is non-zero.
            out.write(acc + beta * unsafe { out.assume_init() });
        }
    }
}

/// A helper to instantiate calls to the SIMD gemm kernel with different values
/// for const generic parameters.
pub struct GemmDispatch<'a, S: SimdFloat, const MR: usize, const NR_REGS: usize> {
    tile_ptr: *mut f32,
    tile_row_stride: usize,
    a: Lhs<'a, f32>,
    b: &'a [f32],
    depth: usize,
    alpha: f32,
    beta: f32,

    _marker: std::marker::PhantomData<S>,
}

impl<'a, S: SimdFloat, const MR: usize, const NR_REGS: usize> GemmDispatch<'a, S, MR, NR_REGS> {
    pub unsafe fn new(
        tile_ptr: *mut f32,
        tile_row_stride: usize,
        a: Lhs<'a, f32>,
        b: &'a [f32],
        depth: usize,
        alpha: f32,
        beta: f32,
    ) -> Self {
        GemmDispatch {
            tile_ptr,
            tile_row_stride,
            a,
            b,
            depth,
            alpha,
            beta,
            _marker: std::marker::PhantomData,
        }
    }

    /// Run the kernel to update an output tile with `ROWS` rows.
    #[inline(always)]
    pub unsafe fn dispatch<const ROWS: usize>(&self) {
        simd_gemm::<S, MR, NR_REGS, ROWS>(
            self.tile_ptr,
            self.tile_row_stride,
            self.a,
            self.b,
            self.depth,
            self.alpha,
            self.beta,
        )
    }
}

/// Compute a tile of matrix-multiplication output.
///
/// - `S` specifies the SIMD vector type
/// - `MR` is the number of rows in a full tile
/// - `NR_REGS` is the width of a full tile as a multiple of `S::LEN`
/// - `ROWS` is the number of rows that are actually used.
///
/// See [`Kernel::kernel`].
///
/// Safety: The `SimdFloat` type must be supported on the current system.
#[inline(always)]
pub unsafe fn simd_gemm<S: SimdFloat, const MR: usize, const NR_REGS: usize, const ROWS: usize>(
    tile_ptr: *mut f32,
    tile_row_stride: usize,
    a: Lhs<f32>,
    b: &[f32],
    depth: usize,
    alpha: f32,
    beta: f32,
) {
    assert!(b.len() >= depth * NR_REGS * S::LEN);
    assert!(depth > 0);
    let (a_ptr, a_row_stride) = match a {
        Lhs::Packed(data) => {
            let min_len = depth * MR * size_of::<f32>();
            assert!(
                data.len() >= min_len,
                "packed data len {} smaller than required {}",
                data.len(),
                min_len
            );
            (data.as_ptr() as *const f32, depth)
        }
        Lhs::Unpacked {
            data,
            len,
            row_stride,
        } => {
            // Offset 1 past last element we'll access.
            let end_offset = (ROWS - 1) * row_stride + depth;
            assert!(len >= end_offset);
            (data, row_stride)
        }
    };
    let b_ptr = b.as_ptr();

    let mut tmp = [[S::zero(); NR_REGS]; ROWS];
    let mut b_rows = [S::zero(); NR_REGS];

    unroll_loop_x4!(0..depth - 1, k, {
        let b_off = k * NR_REGS * S::LEN;

        // Prefetch B for the next iteration
        S::prefetch(b_ptr.add((k + 1) * NR_REGS * S::LEN));

        for i in 0..NR_REGS {
            b_rows[i] = S::load(b_ptr.add(b_off + i * S::LEN));
        }

        for i in 0..ROWS {
            let a_val = *a_ptr.add(i * a_row_stride + k);
            let a_broadcast = S::splat(a_val);

            for j in 0..NR_REGS {
                tmp[i][j] = a_broadcast.mul_add(b_rows[j], tmp[i][j]);
            }
        }
    });

    // Prefetch output before the final computation loop
    for i in 0..ROWS {
        S::prefetch_write(tile_ptr.add(tile_row_stride * i));
    }

    // Perform final outer product update.
    let k = depth - 1;
    let b_off = k * NR_REGS * S::LEN;

    for i in 0..NR_REGS {
        b_rows[i] = S::load(b_ptr.add(b_off + i * S::LEN));
    }

    for i in 0..ROWS {
        let a_val = *a_ptr.add(i * a_row_stride + k);
        let a_broadcast = S::splat(a_val);

        for j in 0..NR_REGS {
            tmp[i][j] = a_broadcast.mul_add(b_rows[j], tmp[i][j]);
        }
    }

    let get_out_ptr = |i, j| tile_ptr.add(tile_row_stride * i + j * S::LEN);

    // Write to output tile.
    //
    // We have special cases for zero/one values of alpha and beta, both for
    // performance in the common cases where (alpha, beta) are (0, 1) or (1, 1)
    // and because when beta is zero, the destination may be uninitialized and
    // must not be read.
    if beta == 0. && alpha == 1. {
        for i in 0..ROWS {
            for j in 0..NR_REGS {
                let out_ptr = get_out_ptr(i, j);
                tmp[i][j].store(out_ptr);
            }
        }
    } else if beta == 1. && alpha == 1. {
        for i in 0..ROWS {
            for j in 0..NR_REGS {
                let out_ptr = get_out_ptr(i, j);
                let out_val = S::load(out_ptr).add(tmp[i][j]);
                out_val.store(out_ptr);
            }
        }
    } else if beta == 0. {
        let alpha_broadcast = S::splat(alpha);

        for i in 0..ROWS {
            for j in 0..NR_REGS {
                let out_ptr = get_out_ptr(i, j);
                let out_val = tmp[i][j].mul(alpha_broadcast);
                out_val.store(out_ptr);
            }
        }
    } else {
        let alpha_broadcast = S::splat(alpha);
        let beta_broadcast = S::splat(beta);

        for i in 0..ROWS {
            for j in 0..NR_REGS {
                let out_ptr = get_out_ptr(i, j);
                let out_val = S::load(out_ptr).mul(beta_broadcast);
                let out_val = tmp[i][j].mul_add(alpha_broadcast, out_val);
                out_val.store(out_ptr);
            }
        }
    }
}

/// Compute an i32 matrix multiplication tile with maximum size `MR x NR` using
/// packed blocks of A and B int8 inputs. `NR` must equal `S::LEN`.
///
/// Whether int8 values in `a` and `b` are treated as signed depends on the
/// `dot_product` function.
///
/// `dot_product(a, b, c)` is a function that computes `c + dot(a, b)` where
/// `c` contains packed i32 values and `a` and `b` contain groups of 4 packed
/// 8-bit integers.
///
/// If `accumulate` is true, the output referenced by `tile_ptr` must be
/// initialized and the result will be added to it. If false, the `tile_ptr`
/// may be uninitialized and will be initialized with the result.
#[inline(always)]
pub unsafe fn simd_int8_gemm<S: SimdInt, const MR: usize, const NR: usize>(
    tile_ptr: *mut i32,
    tile_row_stride: usize,
    a: &[u8],
    b: &[u8],
    used_rows: usize,
    used_cols: usize,
    depth: usize,
    accumulate: bool,
    a_zero_points: [i32; MR],
    b_zero_points: [i32; NR],
    a_row_sums: &[i32; MR],
    b_col_sums: &[i32; NR],
    dot_product: unsafe fn(S, S, S) -> S,
) {
    assert_eq!(S::LEN, NR);

    // The value for each element in the output tile is computed as:
    //
    // c = (a[0] - a_zero_point) * (b[0] - b_zero_point) + ...
    //
    // (or `c += ...` when beta=1)
    //
    // Where `a_zero_point` is the zero point for the row of A and
    // `b_zero_point` is the zero point for the column of B.
    //
    // This can be expanded and re-arranged into:
    //
    // c = a[0]b[0] - a[0] * b_zero_point - b[0] * a_zero_point + a_zero_point * b_zero_point + ...
    // c = dot(a, b) - sum(a) * b_zero_point - sum(b) * a_zero_point + k * a_zero_point * b_zero_point
    // c = k * a_zero_point * b_zero_point + dot(a, b) - sum(a) * b_zero_point - sum(b) * a_zero_point
    //
    // The `k * a_zero_point * b_zero_point` term is computed first as the
    // initial value of the accumulator tile, then we loop over K and add
    // the dot product of each row and column. Finally the scaled row
    // and column sums are subtracted.

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let n_depth_tiles = depth.div_ceil(4);
    let b_zero = S::load(b_zero_points.as_ptr());

    // Initialize output tile with `k * a_zero_point[row] * b_zero_point[col]`
    let k_mul_b_zero = S::splat(depth as i32).mul(b_zero);
    let mut tmp = [k_mul_b_zero; MR];
    for row in 0..MR {
        let a_zero = S::splat(a_zero_points[row]);
        tmp[row] = tmp[row].mul(a_zero);
    }

    // Loop over K dimension and compute dot product of panel of A with panel of
    // B.
    for k_block in 0..n_depth_tiles {
        // Load `[4, NR]` microtile from B
        let b_vec = S::load(b_ptr.add(k_block * NR * 4) as *const i32);

        // Each iteration broadcasts 4x int 8 values from A, computes NR
        // dot products and accumulates into the output tile.
        for row in 0..MR {
            let a_val = *(a_ptr.add(k_block * MR * 4 + row * 4) as *const i32);
            let a_vec = S::splat(a_val);
            tmp[row] = dot_product(a_vec, b_vec, tmp[row]);
        }
    }

    // Scale zero points by row and column sums and subtract from output tile.
    let b_col_sums = S::load(b_col_sums.as_ptr());
    for row in 0..MR {
        let a_zero = S::splat(a_zero_points[row]);
        let a_sum = S::splat(a_row_sums[row]);

        let a_sum_mul_b_zero = a_sum.mul(b_zero);
        let b_sum_mul_a_zero = b_col_sums.mul(a_zero);
        tmp[row] = tmp[row].sub(a_sum_mul_b_zero);
        tmp[row] = tmp[row].sub(b_sum_mul_a_zero);
    }

    // Write from accumulator in registers back to output.
    let output_tile_ptr = |row| tile_ptr.add(row * tile_row_stride);

    #[allow(clippy::collapsible_else_if)]
    if !accumulate {
        if used_rows == MR && used_cols == NR {
            // Full output tile
            for row in 0..MR {
                let tile_ptr = output_tile_ptr(row);
                tmp[row].store(tile_ptr);
            }
        } else {
            // Partial output tile
            for r in 0..used_rows {
                let tile_ptr = output_tile_ptr(r);
                let tmp = tmp[r].to_array();
                for c in 0..used_cols {
                    tile_ptr.add(c).write(tmp[c]);
                }
            }
        }
    } else {
        if used_rows == MR && used_cols == NR {
            // Full output tile
            for row in 0..MR {
                let tile_ptr = output_tile_ptr(row);
                let out = S::load(tile_ptr).add(tmp[row]);
                out.store(tile_ptr);
            }
        } else {
            // Partial output tile
            for r in 0..used_rows {
                let tile_ptr = output_tile_ptr(r);
                let tmp = tmp[r].to_array();
                for c in 0..used_cols {
                    *tile_ptr.add(c) += tmp[c];
                }
            }
        }
    }
}

// Mask that when XOR-ed with packed i8 values shifts them to u8 by adding 128,
// or the opposite if XOR-ed with packed u8 values.
const I8_U8_SHIFT_MASK: i32 = 0x80808080u32 as i32;

/// Keep the input values unchanged.
pub const CAST_SAME: u8 = 0;

/// Convert i8 input values to u8 by adding 128.
pub const CAST_I8_TO_U8: u8 = 1;

/// Convert u8 input values to i8 by subtracting 128.
pub const CAST_U8_TO_I8: u8 = 2;

/// Return adjustment to apply to the zero point when changing the sign of
/// int8 inputs.
const fn zero_shift(cast_mode: u8) -> i32 {
    match cast_mode {
        CAST_I8_TO_U8 => 128,
        CAST_U8_TO_I8 => -128,
        _ => 0,
    }
}

/// Trait for loading and sign or zero extending int8 values to int32 into a
/// SIMD vector.
pub trait LoadExtend<T> {
    unsafe fn load_extend(src: *const T) -> Self;
}

impl<S: SimdInt> LoadExtend<u8> for S {
    unsafe fn load_extend(src: *const u8) -> Self {
        S::load_extend_u8(src)
    }
}

impl<S: SimdInt> LoadExtend<i8> for S {
    unsafe fn load_extend(src: *const i8) -> Self {
        S::load_extend_i8(src)
    }
}

/// Compute a vector-matrix product between an int8 vector and int8 matrix,
/// producing an i32 vector.
///
/// This is a specialization of [`simd_int8_gemm`] for the case where the LHS
/// input is a vector. In this case the kernel inputs are not packed.
///
/// The LHS and RHS inputs may be signed or unsigned. The dot product function
/// may expect inputs with a different sign than the LHS and RHS type. To handle
/// this `CAST_A` and `CAST_B` specify how the input elements should be
/// transformed before being passed to the dot product function. A matching
/// adjustment is applied to the corresponding zero point. The values must be
/// one of [`CAST_SAME`], [`CAST_I8_TO_U8`] or [`CAST_U8_TO_I8`].
///
/// # Safety
///
/// - Instructions used by SIMD type `S` and `dot_product` must be supported.
#[inline(always)]
pub unsafe fn simd_int8_gemv<
    S,
    LhsT: Copy + Into<i32> + std::fmt::Debug,
    RhsT: Copy + Into<i32> + std::fmt::Debug,
    const CAST_A: u8,
    const CAST_B: u8,
>(
    out: &mut [MaybeUninit<i32>],
    a: &[LhsT],
    b: Matrix<RhsT>,
    accumulate: bool,
    a_zero_point: LhsT,
    b_zero_points: Option<&[RhsT]>,
    dot_product: unsafe fn(S, S, S) -> S,
) where
    S: SimdInt + LoadExtend<RhsT>,
{
    assert_eq!(out.len(), b.cols());
    assert_eq!(b.rows(), a.len());
    assert_eq!(a.as_ptr() as usize % align_of::<i32>(), 0);

    if b.row_stride() == 1 {
        return simd_int8_gemv_transposed::<S, LhsT, RhsT, CAST_A, CAST_B>(
            out,
            a,
            b,
            accumulate,
            a_zero_point,
            b_zero_points,
            dot_product,
        );
    } else if b.col_stride() != 1 {
        return simd_int8_gemv_fallback::<LhsT, RhsT, CAST_A, CAST_B>(
            out,
            a,
            b,
            accumulate,
            a_zero_point,
            b_zero_points,
        );
    }

    let a_zero_shift = zero_shift(CAST_A);
    let b_zero_shift = zero_shift(CAST_B);
    let bit_flip_mask = S::splat(I8_U8_SHIFT_MASK);

    let a_ptr = a.as_ptr();
    let depth = a.len();
    let b_ptr = b.storage().as_ptr();
    let b_row_stride = b.row_stride();

    let row_sum: i32 = a.iter().map(|&x| x.into() + a_zero_shift).sum();

    let mut col_tiles = range_chunks_exact(0..b.cols(), S::LEN);
    for col_tile in col_tiles.by_ref() {
        let b_ptr = b_ptr.add(col_tile.start);
        let mut acc = S::zero();
        let mut col_sums = S::zero();
        let one_u8 = S::splat(i32::from_le_bytes([1; 4]));

        // Loop over K tiles of size 4.
        let mut k_tiles = range_chunks_exact(0..depth, 4);
        for k_tile in k_tiles.by_ref() {
            // Broadcast 4 values from A.
            let a = S::splat(*(a_ptr.add(k_tile.start) as *const i32));
            let a = if CAST_A != CAST_SAME {
                a.xor(bit_flip_mask)
            } else {
                a
            };

            // Load `S::LEN` groups of 4 values from B.
            let b = S::load_interleave_i8(
                b_ptr.add(k_tile.start * b_row_stride) as *const i8,
                b_ptr.add((k_tile.start + 1) * b_row_stride) as *const i8,
                b_ptr.add((k_tile.start + 2) * b_row_stride) as *const i8,
                b_ptr.add((k_tile.start + 3) * b_row_stride) as *const i8,
            );
            let b = if CAST_B != CAST_SAME {
                b.xor(bit_flip_mask)
            } else {
                b
            };

            // Compute `C += dot(A, B)` for each of the `S::LEN` columns.
            acc = dot_product(a, b, acc);
            col_sums = dot_product(one_u8, b, col_sums);
        }

        for k in k_tiles.remainder() {
            let a = S::splat((*a_ptr.add(k)).into());
            let a = if CAST_A != CAST_SAME {
                a.add(S::splat(a_zero_shift))
            } else {
                a
            };

            let b = S::load_extend(b_ptr.add(k * b_row_stride));
            let b = if CAST_B != CAST_SAME {
                b.add(S::splat(b_zero_shift))
            } else {
                b
            };

            acc = a.mul(b).add(acc);
            col_sums = col_sums.add(b);
        }

        // Subtract zero points. This is equivalent to doing
        // `acc += (a - a_zero) * (b - b_zero)` in the loop over K, but more
        // efficient.
        let row_sum_vec = S::splat(row_sum);
        let depth_vec = S::splat(depth as i32);
        let a_zero_vec = S::splat(a_zero_point.into()).add(S::splat(a_zero_shift));
        let b_zero_vec = if let Some(b_zero) = b_zero_points {
            S::load_extend(b_zero.as_ptr().add(col_tile.start))
        } else {
            S::zero()
        };
        let b_zero_vec = b_zero_vec.add(S::splat(b_zero_shift));

        acc = depth_vec
            .mul(a_zero_vec)
            .mul(b_zero_vec)
            .add(acc)
            .sub(row_sum_vec.mul(b_zero_vec))
            .sub(col_sums.mul(a_zero_vec));

        let out_ptr = out.as_ptr().add(col_tile.start) as *mut i32;
        if !accumulate {
            acc.store(out_ptr);
        } else {
            S::load(out_ptr).add(acc).store(out_ptr);
        }
    }

    for col in col_tiles.remainder() {
        let mut acc = 0;
        let mut col_sum = 0;
        for (k, &a) in a.iter().enumerate() {
            let mut b_val: i32 = (*b.get_unchecked([k, col])).into();
            if CAST_B != CAST_SAME {
                b_val += b_zero_shift;
            }
            let a_val: i32 = if CAST_A != CAST_SAME {
                a.into() + a_zero_shift
            } else {
                a.into()
            };
            acc += a_val * b_val;
            col_sum += b_val;
        }

        // Subtract zero points. This is equivalent to doing
        // `acc += (a - a_zero) * (b - b_zero)` in the loop over K, but more
        // efficient.
        let a_zero: i32 = a_zero_point.into() + a_zero_shift;
        let b_zero = b_zero_points.map(|bq| bq[col].into()).unwrap_or(0) + b_zero_shift;
        acc = depth as i32 * a_zero * b_zero + acc - row_sum * b_zero - col_sum * a_zero;

        let out = out.as_ptr().add(col) as *mut i32;
        if !accumulate {
            out.write(acc);
        } else {
            *out += acc;
        }
    }
}

/// Variant of [`simd_int8_gemv`] for the case where the RHS has unit row stride.
#[inline(always)]
unsafe fn simd_int8_gemv_transposed<
    S: SimdInt,
    LhsT: Copy + std::fmt::Debug + Into<i32>,
    RhsT: Copy + std::fmt::Debug + Into<i32>,
    const CAST_A: u8,
    const CAST_B: u8,
>(
    out: &mut [MaybeUninit<i32>],
    a: &[LhsT],
    b: Matrix<RhsT>,
    accumulate: bool,
    a_zero_point: LhsT,
    b_zero_points: Option<&[RhsT]>,
    dot_product: unsafe fn(S, S, S) -> S,
) {
    let bit_flip_mask = S::splat(I8_U8_SHIFT_MASK);
    let a_zero_shift = zero_shift(CAST_A);
    let b_zero_shift = zero_shift(CAST_B);
    let depth = a.len();

    let row_sum: i32 = a.iter().map(|&x| x.into() + a_zero_shift).sum();
    let a_ptr = a.as_ptr();
    let b_ptr = b.storage().as_ptr();
    let one_u8 = S::splat(i32::from_le_bytes([1; 4]));

    for col in 0..b.cols() {
        let b_ptr = b_ptr.add(col * b.col_stride());
        let mut acc = S::zero();
        let mut col_sum = S::zero();

        // nb. `S::LEN` refers to `i32` values, but `depth` is a count of u8/i8
        // values.
        let mut k_tiles = range_chunks_exact(0..depth, S::LEN * 4);

        for k_tile in k_tiles.by_ref() {
            let a = S::load(a_ptr.add(k_tile.start) as *const i32);
            let a = if CAST_A != CAST_SAME {
                a.xor(bit_flip_mask)
            } else {
                a
            };
            let b = S::load(b_ptr.add(k_tile.start) as *const i32);
            let b = if CAST_B != CAST_SAME {
                b.xor(bit_flip_mask)
            } else {
                b
            };

            acc = dot_product(a, b, acc);
            col_sum = dot_product(one_u8, b, col_sum);
        }

        let mut acc = acc.sum();
        let mut col_sum = col_sum.sum();

        for k in k_tiles.remainder() {
            let a: i32 = (*a_ptr.add(k)).into() + a_zero_shift;
            let b: i32 = (*b_ptr.add(k)).into() + b_zero_shift;
            acc += a * b;
            col_sum += b;
        }

        let a_zero = a_zero_point.into() + a_zero_shift;
        let b_zero = b_zero_points
            .map(|bz| bz[col].into() + b_zero_shift)
            .unwrap_or(b_zero_shift);
        let acc = (depth as i32 * a_zero * b_zero) + acc - row_sum * b_zero - col_sum * a_zero;

        let out_ptr = out.get_unchecked_mut(col);
        if !accumulate {
            out_ptr.write(acc);
        } else {
            out_ptr.write(out_ptr.assume_init() + acc);
        }
    }
}

/// Fallback for [`simd_int8_gemv`] when RHS has neither unit column stride nor
/// unit row stride.
#[inline(always)]
fn simd_int8_gemv_fallback<
    LhsT: Copy + Into<i32>,
    RhsT: Copy + Into<i32>,
    const CAST_A: u8,
    const CAST_B: u8,
>(
    out: &mut [MaybeUninit<i32>],
    a: &[LhsT],
    b: Matrix<RhsT>,
    accumulate: bool,
    a_zero_point: LhsT,
    b_zero_points: Option<&[RhsT]>,
) {
    let a_zero_shift = zero_shift(CAST_A);
    let b_zero_shift = zero_shift(CAST_B);
    let depth = a.len();
    for (out, col) in out.iter_mut().zip(0..b.cols()) {
        let b_zero = b_zero_points
            .map(|bz| bz[col].into() + b_zero_shift)
            .unwrap_or(0);
        let mut acc = 0;
        let mut row_sum = 0;
        let mut col_sum = 0;

        for k in 0..depth {
            let a_el = unsafe { (*a.get_unchecked(k)).into() };
            let a_el = if CAST_A != CAST_SAME {
                a_el + a_zero_shift
            } else {
                a_el
            };

            let b_el = unsafe { (*b.get_unchecked([k, col])).into() };
            let b_el = if CAST_B != CAST_SAME {
                b_el + b_zero_shift
            } else {
                b_el
            };
            acc += a_el * b_el;
            row_sum += a_el;
            col_sum += b_el;
        }

        // Subtract zero points. This is equivalent to doing
        // `acc += (a - a_zero) * (b - b_zero)` in the loop over K, but more
        // efficient.
        let a_zero = a_zero_point.into() + a_zero_shift;
        acc = depth as i32 * a_zero * b_zero + acc - row_sum * b_zero - col_sum * a_zero;

        if !accumulate {
            out.write(acc);
        } else {
            // Safety: Output is initialized when `accumulate` is true
            unsafe {
                out.write(out.assume_init() + acc);
            }
        }
    }
}
