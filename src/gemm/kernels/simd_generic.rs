use rten_simd::ops::{Extend, Interleave, NumOps};
use rten_simd::{Isa, Simd};
use rten_tensor::{Matrix, MatrixLayout, Storage};

use super::{Int8DotProduct, Lhs, MatVecOutput};
use crate::iter_util::{range_chunks_exact, unroll_loop, unroll_loop_x4};

/// Compute an output block of a vector-matrix product ("gemv" in BLAS APIs).
///
/// Multiple output columns are computed at a time, using `NR_REGS` SIMD
/// registers of type `I::F32`. See [`Kernel::gemv_kernel`].
#[inline(always)]
pub fn simd_gemv<I: Isa, const NR_REGS: usize>(
    isa: I,
    out: MatVecOutput<f32, f32>,
    a: &[f32],
    b: Matrix,
    alpha: f32,
) {
    // Handle cases where `b` does not have unit stride.
    if b.row_stride() == 1 {
        return simd_gemv_transposed(isa, out, a, b, alpha);
    } else if b.col_stride() != 1 {
        return simd_gemv_fallback(out, a, b, alpha);
    }

    assert_eq!(a.len(), b.rows());
    assert_eq!(out.data.len(), b.cols());
    assert_eq!(b.col_stride(), 1);

    let ops = isa.f32();
    let out_ptr = out.data.as_mut_ptr();
    let a_ptr = a.as_ptr();
    let b_ptr = b.storage().as_ptr();
    let b_row_stride = b.row_stride();
    let v_len = ops.len();

    let mut b_tiles = range_chunks_exact(0..b.cols(), NR_REGS * v_len);
    for b_tile in b_tiles.by_ref() {
        let mut acc = [ops.zero(); NR_REGS];
        unroll_loop!(0..a.len(), k, 4, {
            let a_elt = unsafe { *a_ptr.add(k) };
            let a_elts = ops.splat(a_elt);

            // Pre-fetch the current row for the next column tile.
            ops.prefetch(unsafe { b_ptr.add(k * b_row_stride + b_tile.start + NR_REGS + v_len) });

            for i in 0..NR_REGS {
                let b_elts = unsafe {
                    ops.load_ptr(b_ptr.add(k * b_row_stride + b_tile.start + i * ops.len()))
                };
                acc[i] = ops.mul_add(a_elts, b_elts, acc[i]);
            }
        });

        if alpha != 1. {
            let alpha_vec = ops.splat(alpha);
            for i in 0..NR_REGS {
                acc[i] = ops.mul(acc[i], alpha_vec);
            }
        }

        let get_out_tile_ptr = |i| unsafe { out_ptr.add(b_tile.start + i * v_len) };

        if out.beta == 0. {
            for i in 0..NR_REGS {
                unsafe {
                    ops.store_ptr(acc[i], get_out_tile_ptr(i) as *mut f32);
                }
            }
        } else if out.beta == 1. {
            for i in 0..NR_REGS {
                let out_tile_ptr = get_out_tile_ptr(i);
                let out_tile = unsafe { ops.load_ptr(out_tile_ptr as *mut f32) };
                let out_tile = ops.add(out_tile, acc[i]);
                unsafe { ops.store_ptr(out_tile, out_tile_ptr as *mut f32) };
            }
        } else {
            let beta_vec = ops.splat(out.beta);
            for i in 0..NR_REGS {
                let out_tile_ptr = get_out_tile_ptr(i);
                let out_tile = unsafe { ops.load_ptr(out_tile_ptr as *mut f32) };
                let out_tile = ops.mul_add(out_tile, beta_vec, acc[i]);
                unsafe { ops.store_ptr(out_tile, out_tile_ptr as *mut f32) };
            }
        }
    }

    for c in b_tiles.remainder() {
        let mut acc = 0.;
        for (k, ax) in a.iter().enumerate() {
            acc += ax * unsafe { *b_ptr.add(k * b_row_stride + c) };
        }
        let out_el = unsafe { out.data.get_unchecked_mut(c) };
        let tmp = if out.beta == 0. {
            0.
        } else {
            unsafe { out_el.assume_init() }
        };
        out_el.write(out.beta * tmp + acc * alpha);
    }
}

/// Variant of [`simd_gemv`] which handles the case where `b` has unit row stride.
#[inline(always)]
fn simd_gemv_transposed<I: Isa>(
    isa: I,
    mut out: MatVecOutput<f32>,
    a: &[f32],
    b: Matrix,
    alpha: f32,
) {
    assert_eq!(b.row_stride(), 1);
    assert_eq!(a.len(), b.rows());
    assert_eq!(out.data.len(), b.cols());

    let ops = isa.f32();
    let b_ptr = b.storage().as_ptr();
    let b_col_stride = b.col_stride();

    const COL_TILE: usize = 8;

    let mut col_tiles = range_chunks_exact(0..b.cols(), COL_TILE);
    for col_tile in col_tiles.by_ref() {
        let mut acc = [ops.zero(); COL_TILE];

        let mut depth_tiles = range_chunks_exact(0..a.len(), ops.len());
        for depth_tile in depth_tiles.by_ref() {
            let a_tile = unsafe { ops.load_ptr(a.as_ptr().add(depth_tile.start)) };
            for i in 0..COL_TILE {
                let b_col_ptr = unsafe { b_ptr.add((col_tile.start + i) * b_col_stride) };
                let b_tile = unsafe { ops.load_ptr(b_col_ptr.add(depth_tile.start)) };
                acc[i] = ops.mul_add(a_tile, b_tile, acc[i]);
            }
        }

        let mut acc: [f32; COL_TILE] = std::array::from_fn(|i| ops.sum(acc[i]));
        for k in depth_tiles.remainder() {
            let ak = unsafe { *a.get_unchecked(k) };
            for i in 0..COL_TILE {
                let b_col_ptr = unsafe { b_ptr.add((col_tile.start + i) * b_col_stride) };
                let bk = unsafe { *b_col_ptr.add(k) };
                acc[i] = ak.mul_add(bk, acc[i]);
            }
        }

        if out.beta == 0. {
            for i in 0..COL_TILE {
                out.data[col_tile.start + i].write(alpha * acc[i]);
            }
        } else {
            for i in 0..COL_TILE {
                // Safety: Output is initialized when `beta` is non-zero.
                let out_val = alpha * acc[i]
                    + out.beta * unsafe { out.data[col_tile.start + i].assume_init() };
                out.data[col_tile.start + i].write(out_val);
            }
        }
    }

    let last_col_tile = col_tiles.remainder();
    if !last_col_tile.is_empty() {
        simd_gemv_fallback(
            out.slice_mut(last_col_tile.clone()),
            a,
            b.slice((.., last_col_tile)),
            alpha,
        );
    }
}

/// Variant of [`simd_gemv`] which handles the case where `b` has non-unit strides
/// for rows and columns.
///
/// This doesn't benefit from SIMD operations. It is at least inlined so it
/// can benefit from the kernel's instruction set (eg. for FMA operations).
#[inline(always)]
fn simd_gemv_fallback(out: MatVecOutput<f32>, a: &[f32], b: Matrix, alpha: f32) {
    assert_eq!(a.len(), b.rows());
    assert_eq!(out.data.len(), b.cols());

    for (col, out_el) in out.data.iter_mut().enumerate() {
        let mut acc = 0.;
        for (k, ak) in (0..a.len()).zip(a.iter()) {
            let bk = unsafe { *b.get_unchecked([k, col]) };
            acc = ak.mul_add(bk, acc);
        }
        acc *= alpha;
        if out.beta == 0. {
            out_el.write(acc);
        } else {
            // Safety: Output is initialized when `beta` is non-zero.
            out_el.write(acc + out.beta * unsafe { out_el.assume_init() });
        }
    }
}

/// A helper to instantiate calls to the SIMD gemm kernel with different values
/// for const generic parameters.
pub struct GemmDispatch<'a, I: Isa, const MR: usize, const NR_REGS: usize> {
    isa: I,
    tile_ptr: *mut f32,
    tile_row_stride: usize,
    a: Lhs<'a, f32>,
    b: &'a [f32],
    depth: usize,
    alpha: f32,
    beta: f32,
}

impl<'a, I: Isa, const MR: usize, const NR_REGS: usize> GemmDispatch<'a, I, MR, NR_REGS> {
    pub unsafe fn new(
        isa: I,
        tile_ptr: *mut f32,
        tile_row_stride: usize,
        a: Lhs<'a, f32>,
        b: &'a [f32],
        depth: usize,
        alpha: f32,
        beta: f32,
    ) -> Self {
        GemmDispatch {
            isa,
            tile_ptr,
            tile_row_stride,
            a,
            b,
            depth,
            alpha,
            beta,
        }
    }

    /// Run the kernel to update an output tile with `ROWS` rows.
    #[inline(always)]
    pub unsafe fn dispatch<const ROWS: usize>(&self) {
        simd_gemm::<I, MR, NR_REGS, ROWS>(
            self.isa,
            self.tile_ptr,
            self.tile_row_stride,
            self.a,
            self.b,
            self.depth,
            self.alpha,
            self.beta,
        )
    }

    /// Run the kernel to update an output tile with `ROWS` rows.
    ///
    /// This is a variant of `dispatch` for architectures (Arm) which can
    /// efficiently broadcast a lane from one vector into a new vector used
    /// as an FMA operand.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub unsafe fn dispatch_broadcast_lane<const ROWS: usize>(&self) {
        simd_gemm_broadcast_lane::<I, MR, NR_REGS, ROWS>(
            self.isa,
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
/// - `MR` is the number of rows in a full tile
/// - `NR_REGS` is the width of a full tile as a multiple of `isa.i32().len()`
/// - `ROWS` is the number of rows that are actually used.
///
/// See [`Kernel::kernel`].
///
/// # Safety
///
/// - `tile_ptr.add(tile_row_stride * row + col)` must be a valid pointer for
///   `row ∈ [0, MR)` and `col ∈ [0, NR)`.
/// - Values pointed to by `tile_ptr` must be initialized if `beta` is non-zero
#[inline(always)]
pub unsafe fn simd_gemm<I: Isa, const MR: usize, const NR_REGS: usize, const ROWS: usize>(
    isa: I,
    tile_ptr: *mut f32,
    tile_row_stride: usize,
    a: Lhs<f32>,
    b: &[f32],
    depth: usize,
    alpha: f32,
    beta: f32,
) {
    let ops = isa.f32();

    assert!(b.len() >= depth * NR_REGS * ops.len());
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

    let mut tmp = [[ops.zero(); NR_REGS]; ROWS];
    let mut b_rows = [ops.zero(); NR_REGS];

    unroll_loop_x4!(0..depth - 1, k, {
        let b_off = k * NR_REGS * ops.len();

        // Prefetch B for the next iteration
        ops.prefetch(b_ptr.add((k + 1) * NR_REGS * ops.len()));

        for i in 0..NR_REGS {
            b_rows[i] = ops.load_ptr(b_ptr.add(b_off + i * ops.len()));
        }

        for i in 0..ROWS {
            let a_val = *a_ptr.add(i * a_row_stride + k);
            let a_broadcast = ops.splat(a_val);

            for j in 0..NR_REGS {
                tmp[i][j] = ops.mul_add(a_broadcast, b_rows[j], tmp[i][j]);
            }
        }
    });

    // Prefetch output before the final computation loop
    for i in 0..ROWS {
        ops.prefetch_write(tile_ptr.add(tile_row_stride * i));
    }

    // Perform final outer product update.
    let k = depth - 1;
    let b_off = k * NR_REGS * ops.len();

    for i in 0..NR_REGS {
        b_rows[i] = ops.load_ptr(b_ptr.add(b_off + i * ops.len()));
    }

    for i in 0..ROWS {
        let a_val = *a_ptr.add(i * a_row_stride + k);
        let a_broadcast = ops.splat(a_val);

        for j in 0..NR_REGS {
            tmp[i][j] = ops.mul_add(a_broadcast, b_rows[j], tmp[i][j]);
        }
    }

    let get_out_ptr = |i, j| tile_ptr.add(tile_row_stride * i + j * ops.len());

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
                ops.store_ptr(tmp[i][j], out_ptr);
            }
        }
    } else if beta == 1. && alpha == 1. {
        for i in 0..ROWS {
            for j in 0..NR_REGS {
                let out_ptr = get_out_ptr(i, j);
                let out_val = ops.add(ops.load_ptr(out_ptr), tmp[i][j]);
                ops.store_ptr(out_val, out_ptr);
            }
        }
    } else if beta == 0. {
        let alpha_broadcast = ops.splat(alpha);

        for i in 0..ROWS {
            for j in 0..NR_REGS {
                let out_ptr = get_out_ptr(i, j);
                let out_val = ops.mul(tmp[i][j], alpha_broadcast);
                ops.store_ptr(out_val, out_ptr);
            }
        }
    } else {
        let alpha_broadcast = ops.splat(alpha);
        let beta_broadcast = ops.splat(beta);

        for i in 0..ROWS {
            for j in 0..NR_REGS {
                let out_ptr = get_out_ptr(i, j);
                let out_val = ops.mul(ops.load_ptr(out_ptr), beta_broadcast);
                let out_val = ops.mul_add(tmp[i][j], alpha_broadcast, out_val);
                ops.store_ptr(out_val, out_ptr);
            }
        }
    }
}

/// Variant of [`simd_gemm`] for architectures (Arm) which can efficiently
/// broadcast a lane from a vector to a new vector used as an operand for
/// FMA.
///
/// On Arm, the combination of broadcast lane + FMA will be fused into
/// FMLA by element [^1].
///
/// [^1]: https://developer.arm.com/documentation/dui0801/g/A64-SIMD-Vector-Instructions/FMLA--vector--by-element-?lang=en
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn simd_gemm_broadcast_lane<
    I: Isa,
    const MR: usize,
    const NR_REGS: usize,
    const ROWS: usize,
>(
    isa: I,
    tile_ptr: *mut f32,
    tile_row_stride: usize,
    a: Lhs<f32>,
    b: &[f32],
    depth: usize,
    alpha: f32,
    beta: f32,
) {
    let ops = isa.f32();

    assert!(b.len() >= depth * NR_REGS * ops.len());
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

    let mut tmp = [[ops.zero(); NR_REGS]; ROWS];
    let mut b_rows = [ops.zero(); NR_REGS];
    let mut a_tiles = [ops.zero(); ROWS];

    let v_len = ops.len();

    macro_rules! k_step {
        ($k_base:ident, $k_offset:literal) => {
            let b_off = ($k_base + $k_offset) * NR_REGS * v_len;
            for i in 0..NR_REGS {
                b_rows[i] = ops.load_ptr(b_ptr.add(b_off + i * v_len));
            }
            for i in 0..ROWS {
                // On Arm, the `broadcast_lane` and `mul_add` operations can be
                // fused into a single FMLA (by element) operation.
                let a_broadcast = ops.broadcast_lane::<$k_offset>(a_tiles[i]);
                for j in 0..NR_REGS {
                    tmp[i][j] = ops.mul_add(a_broadcast, b_rows[j], tmp[i][j]);
                }
            }
        };
    }

    // Columns of A we can load into a register at once.
    const VEC_LEN: usize = 4;
    let mut k_base = 0;
    while depth - k_base >= VEC_LEN {
        for i in 0..ROWS {
            a_tiles[i] = ops.load_ptr(a_ptr.add(i * a_row_stride + k_base));
        }
        k_step!(k_base, 0);
        k_step!(k_base, 1);
        k_step!(k_base, 2);
        k_step!(k_base, 3);
        k_base += VEC_LEN;
    }
    while k_base < depth {
        for i in 0..ROWS {
            let a_val = *a_ptr.add(i * a_row_stride + k_base);
            a_tiles[i] = ops.splat(a_val);
        }
        k_step!(k_base, 0);
        k_base += 1;
    }

    let get_out_ptr = |i, j| tile_ptr.add(tile_row_stride * i + j * ops.len());

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
                ops.store_ptr(tmp[i][j], out_ptr);
            }
        }
    } else if beta == 1. && alpha == 1. {
        for i in 0..ROWS {
            for j in 0..NR_REGS {
                let out_ptr = get_out_ptr(i, j);
                let out_val = ops.add(ops.load_ptr(out_ptr), tmp[i][j]);
                ops.store_ptr(out_val, out_ptr);
            }
        }
    } else if beta == 0. {
        let alpha_broadcast = ops.splat(alpha);

        for i in 0..ROWS {
            for j in 0..NR_REGS {
                let out_ptr = get_out_ptr(i, j);
                let out_val = ops.mul(tmp[i][j], alpha_broadcast);
                ops.store_ptr(out_val, out_ptr);
            }
        }
    } else {
        let alpha_broadcast = ops.splat(alpha);
        let beta_broadcast = ops.splat(beta);

        for i in 0..ROWS {
            for j in 0..NR_REGS {
                let out_ptr = get_out_ptr(i, j);
                let out_val = ops.mul(ops.load_ptr(out_ptr), beta_broadcast);
                let out_val = ops.mul_add(tmp[i][j], alpha_broadcast, out_val);
                ops.store_ptr(out_val, out_ptr);
            }
        }
    }
}

/// Compute an i32 matrix multiplication tile with maximum size `MR x NR` using
/// packed blocks of A and B int8 inputs. `NR` must equal `NR_REGS * isa.i32().len()`.
///
/// Whether int8 values in `a` and `b` are treated as signed depends on the
/// `dot` implementation.
///
/// # Safety
///
/// - `tile_ptr.add(tile_row_stride * row + col)` must be a valid pointer where
///   `row < used_rows` and `col < used_cols`.
/// - If `accumulate` is true, the output referenced by `tile_ptr` must be
///   initialized and the result will be added to it. If false, `tile_ptr` may
///   point to uninitialized data and will be initialized with the result.
#[inline(always)]
pub unsafe fn simd_int8_gemm<I: Isa, D, const MR: usize, const NR: usize, const NR_REGS: usize>(
    isa: I,
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
    dot: D,
) where
    D: Int8DotProduct<X8 = I::I8, I32 = I::I32> + Copy,
{
    let ops = isa.i32();
    let i8_ops = isa.i8();

    assert_eq!(ops.len() * NR_REGS, NR);

    // Packed buffers contain `[MR, 4]` microtiles of A and transposed `[4, NR]`
    // microtiles of B.
    assert_eq!(a.len(), MR * depth.next_multiple_of(4));
    assert_eq!(b.len(), NR * depth.next_multiple_of(4));

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
    let b_zero = ops.load_many::<NR_REGS>(&b_zero_points);

    // Initialize output tile with `k * a_zero_point[row] * b_zero_point[col]`
    let k_mul_b_zero: [I::I32; NR_REGS] =
        std::array::from_fn(|i| ops.mul(ops.splat(depth as i32), b_zero[i]));
    let mut tmp = [k_mul_b_zero; MR];
    for row in 0..MR {
        let a_zero = ops.splat(a_zero_points[row]);
        for i in 0..NR_REGS {
            tmp[row][i] = ops.mul(tmp[row][i], a_zero);
        }
    }

    // Loop over K dimension and compute dot product of panel of A with panel of
    // B.
    for k_block in 0..n_depth_tiles {
        // Load `[4, NR]` microtile from B
        let b_vec: [I::I8; NR_REGS] = std::array::from_fn(|i| {
            i8_ops.load_ptr(b_ptr.add((k_block * NR + i * ops.len()) * 4) as *const i8)
        });

        // Multiply a MRx4 tile of A with a 4xNR tile of B.
        //
        // On Arm we can load the A tile with one instruction and then use an
        // indexed dot product for each row to multiply one row of A from that
        // tile by all columns of B. On other architectures we use a separate
        // scalar load for each row of A and broadcast to a 4x4 tile which is
        // then multiplied by the columns of B.

        if D::supports_indexed_dot_product() {
            let a_tile = unsafe { ops.load_ptr(a_ptr.add(k_block * MR * 4) as *const i32) }
                .reinterpret_cast::<I::I8>();

            macro_rules! k_step {
                ($row:literal) => {
                    for i in 0..NR_REGS {
                        tmp[$row][i] =
                            dot.indexed_dot_product::<$row>(b_vec[i], a_tile, tmp[$row][i]);
                    }
                };
            }

            // This code path is currently only used on Arm Neon, where MR=4.
            debug_assert_eq!(MR, 4);
            k_step!(0);
            k_step!(1);
            k_step!(2);
            k_step!(3);
        } else {
            for row in 0..MR {
                let a_val = unsafe { *(a_ptr.add(k_block * MR * 4 + row * 4) as *const i32) };
                let a_vec = ops.splat(a_val).reinterpret_cast::<I::I8>();

                for i in 0..NR_REGS {
                    tmp[row][i] = dot.dot_product(a_vec, b_vec[i], tmp[row][i]);
                }
            }
        }
    }

    // Scale zero points by row and column sums and subtract from output tile.
    let b_col_sums: [I::I32; NR_REGS] =
        std::array::from_fn(|i| ops.load_ptr(b_col_sums.as_ptr().add(i * ops.len())));
    for row in 0..MR {
        let a_zero = ops.splat(a_zero_points[row]);
        let a_sum = ops.splat(a_row_sums[row]);

        for i in 0..NR_REGS {
            let a_sum_mul_b_zero = ops.mul(a_sum, b_zero[i]);
            let b_sum_mul_a_zero = ops.mul(b_col_sums[i], a_zero);
            let sum = ops.add(a_sum_mul_b_zero, b_sum_mul_a_zero);
            tmp[row][i] = ops.sub(tmp[row][i], sum);
        }
    }

    // Write from accumulator in registers back to output.
    let output_tile_ptr =
        |row, col_block| tile_ptr.add(row * tile_row_stride + col_block * ops.len());

    #[allow(clippy::collapsible_else_if)]
    if !accumulate {
        if used_rows == MR && used_cols == NR {
            // Full output tile
            for row in 0..MR {
                for c_block in 0..NR_REGS {
                    let tile_ptr = output_tile_ptr(row, c_block);
                    ops.store_ptr(tmp[row][c_block], tile_ptr);
                }
            }
        } else {
            // Partial output tile
            for r in 0..used_rows {
                for c_block in 0..NR_REGS {
                    let tile_ptr = output_tile_ptr(r, c_block);
                    let used_cols = used_cols.saturating_sub(c_block * ops.len()).min(ops.len());
                    let tmp = tmp[r][c_block].to_array();

                    for c in 0..used_cols {
                        tile_ptr.add(c).write(tmp[c]);
                    }
                }
            }
        }
    } else {
        if used_rows == MR && used_cols == NR {
            // Full output tile
            for row in 0..MR {
                for c_block in 0..NR_REGS {
                    let tile_ptr = output_tile_ptr(row, c_block);
                    let out = ops.add(ops.load_ptr(tile_ptr), tmp[row][c_block]);
                    ops.store_ptr(out, tile_ptr);
                }
            }
        } else {
            // Partial output tile
            for r in 0..used_rows {
                for c_block in 0..NR_REGS {
                    let tile_ptr = output_tile_ptr(r, c_block);
                    let used_cols = used_cols.saturating_sub(c_block * ops.len()).min(ops.len());
                    let tmp = tmp[r][c_block].to_array();

                    for c in 0..used_cols {
                        *tile_ptr.add(c) += tmp[c];
                    }
                }
            }
        }
    }
}

// Mask that when XOR-ed with packed i8 values shifts them to u8 by adding 128.
const I8_U8_SHIFT_MASK: i8 = 0x80u8 as i8;

/// Compute a vector-matrix product between a u8 vector and i8 matrix, producing
/// an i32 vector.
///
/// This is a specialization of [`simd_int8_gemm`] for the case where the LHS
/// input is a vector. In this case the kernel inputs are not packed.
///
/// `CAST_B_U8` specifies that the dot product implementation expects its second
/// argument to contain `u8` rather than `i8` values. If true, the values of
/// B are shifted by 128 and the same adjustment is applied to zero points.
#[inline(always)]
pub fn simd_int8_gemv<I: Isa, const CAST_B_U8: bool>(
    isa: I,
    out: MatVecOutput<i32, bool>,
    a: &[u8],
    b: Matrix<i8>,
    a_zero_point: u8,
    b_zero_points: Option<&[i8]>,
    dot: impl Int8DotProduct<X8 = I::I8, I32 = I::I32> + Copy,
) {
    // Verify that input and output dimensions are compatible.
    assert_eq!(out.data.len(), b.cols());
    assert_eq!(b.rows(), a.len());
    assert_eq!(
        b_zero_points.map(|zp| zp.len()).unwrap_or(b.cols()),
        b.cols()
    );

    // Inner loop loads 4x u8 values at a time as an i32.
    assert_eq!(a.as_ptr() as usize % align_of::<i32>(), 0);

    if b.row_stride() == 1 {
        // Safety: Input and output dimensions are compatible.
        unsafe {
            return simd_int8_gemv_transposed::<_, CAST_B_U8>(
                isa,
                out,
                a,
                b,
                a_zero_point,
                b_zero_points,
                dot,
            );
        }
    } else if b.col_stride() != 1 {
        // Safety: Input and output dimensions are compatible.
        unsafe {
            return simd_int8_gemv_fallback::<CAST_B_U8>(out, a, b, a_zero_point, b_zero_points);
        }
    }

    let ops = isa.i32();
    let i8_ops = isa.i8();
    let i16_ops = isa.i16();

    let b_zero_shift = if CAST_B_U8 { 128 } else { 0 };
    let bit_flip_mask = i8_ops.splat(I8_U8_SHIFT_MASK);

    let a_ptr = a.as_ptr();
    let depth = a.len();
    let b_ptr = b.storage().as_ptr();
    let b_row_stride = b.row_stride();

    let row_sum: i32 = a.iter().map(|x| *x as i32).sum();

    // Iterate over one SIMD vec of int8 input columns at a time, or 4x output
    // i32 vecs.
    let mut col_tiles = range_chunks_exact(0..b.cols(), i8_ops.len());
    for col_tile in col_tiles.by_ref() {
        let b_ptr = unsafe { b_ptr.add(col_tile.start) };
        let mut acc = [ops.zero(); 4];
        let mut col_sums = [ops.zero(); 4];
        let one_u8 = i8_ops.splat(1);

        // Loop over K tiles of size 4.
        let mut k = 0;
        while k + 4 <= depth {
            // Broadcast 4 values from A.
            let a_block = unsafe { *(a_ptr.add(k) as *const i32) };
            let a = ops.splat(a_block).reinterpret_cast::<I::I8>();

            // Load 4 rows of int8 elements from B and interleave to give 4
            // transposed `[4, MR]` tiles. eg. Given 4 rows A, B, C, D if `MR` =
            // 4, the first tile is stored in column-major order and contains:
            //
            // A0 A1 A2 A3
            // B0 B1 B2 B3
            // C0 C1 C2 C3
            // D0 D1 D2 D3
            //
            // The second tile contains A4..A7 and so on.
            let b_tile_ptr: [*const i8; 4] =
                std::array::from_fn(|i| unsafe { b_ptr.add((k + i) * b_row_stride) });
            let b0 = unsafe { i8_ops.load_ptr(b_tile_ptr[0]) };
            let b1 = unsafe { i8_ops.load_ptr(b_tile_ptr[1]) };
            let b2 = unsafe { i8_ops.load_ptr(b_tile_ptr[2]) };
            let b3 = unsafe { i8_ops.load_ptr(b_tile_ptr[3]) };

            let b01_lo = i8_ops.interleave_low(b0, b1).reinterpret_cast::<I::I16>();
            let b01_hi = i8_ops.interleave_high(b0, b1).reinterpret_cast::<I::I16>();
            let b23_lo = i8_ops.interleave_low(b2, b3).reinterpret_cast::<I::I16>();
            let b23_hi = i8_ops.interleave_high(b2, b3).reinterpret_cast::<I::I16>();

            let b_tiles = [
                i16_ops.interleave_low(b01_lo, b23_lo),
                i16_ops.interleave_high(b01_lo, b23_lo),
                i16_ops.interleave_low(b01_hi, b23_hi),
                i16_ops.interleave_high(b01_hi, b23_hi),
            ]
            .map(|t| t.reinterpret_cast::<I::I8>());

            // Pre-fetch the current block of 4 rows for the next column tile.
            for i in 0..4 {
                i8_ops.prefetch(unsafe { b_tile_ptr[i].add(i8_ops.len()) });
            }

            for i in 0..4 {
                let b_tile = if CAST_B_U8 {
                    i8_ops.xor(b_tiles[i], bit_flip_mask)
                } else {
                    b_tiles[i]
                };
                acc[i] = dot.dot_product(a, b_tile, acc[i]);
                col_sums[i] = dot.dot_product(one_u8, b_tile, col_sums[i]);
            }
            k += 4;
        }

        while k < depth {
            let a_block = unsafe { (*a_ptr.add(k)).into() };
            let a = ops.splat(a_block);

            // Load one `i8` vec, sign-extend each quarter to give 4 `i32` vecs.
            let b = unsafe { i8_ops.load_ptr(b_ptr.add(k * b_row_stride)) };
            let (b01, b23) = i8_ops.extend(b);
            let (b0, b1) = i16_ops.extend(b01);
            let (b2, b3) = i16_ops.extend(b23);
            let b_rows = [b0, b1, b2, b3];

            for i in 0..4 {
                let b = b_rows[i];
                let b = if CAST_B_U8 {
                    ops.add(b, ops.splat(b_zero_shift))
                } else {
                    b
                };

                acc[i] = ops.mul_add(a, b, acc[i]);
                col_sums[i] = ops.add(col_sums[i], b);
            }
            k += 1;
        }

        // Subtract zero points. This is equivalent to doing
        // `acc += (a - a_zero) * (b - b_zero)` in the loop over K, but more
        // efficient.
        let row_sum_vec = ops.splat(row_sum);
        let depth_vec = ops.splat(depth as i32);
        let a_zero_vec = ops.splat(a_zero_point.into());

        let b_zero_vec = if let Some(b_zero) = b_zero_points {
            // Load one `i8` vec, sign-extend each quarter to give 4 `i32` vecs.
            let b = unsafe { i8_ops.load_ptr(b_zero.as_ptr().add(col_tile.start)) };
            let (b01, b23) = i8_ops.extend(b);
            let (b0, b1) = i16_ops.extend(b01);
            let (b2, b3) = i16_ops.extend(b23);
            [b0, b1, b2, b3]
        } else {
            [ops.zero(); 4]
        };

        for i in 0..4 {
            let b_zero_vec = ops.add(b_zero_vec[i], ops.splat(b_zero_shift));

            let tmp = ops.mul(depth_vec, a_zero_vec);
            let tmp = ops.mul(tmp, b_zero_vec);
            let tmp = ops.add(tmp, acc[i]);
            let tmp = ops.sub(tmp, ops.mul(row_sum_vec, b_zero_vec));
            acc[i] = ops.sub(tmp, ops.mul(col_sums[i], a_zero_vec));

            let out_ptr =
                unsafe { out.data.as_ptr().add(col_tile.start + i * ops.len()) as *mut i32 };
            if !out.beta {
                unsafe {
                    ops.store_ptr(acc[i], out_ptr);
                }
            } else {
                let tmp = unsafe { ops.load_ptr(out_ptr) };
                let tmp = ops.add(tmp, acc[i]);
                unsafe {
                    ops.store_ptr(tmp, out_ptr);
                }
            }
        }
    }

    for col in col_tiles.remainder() {
        let mut acc = 0;
        let mut col_sum = 0;
        for (k, &a) in a.iter().enumerate() {
            let mut b_val = unsafe { *b.get_unchecked([k, col]) as i32 };
            if CAST_B_U8 {
                b_val += b_zero_shift;
            }
            acc += a as i32 * b_val;
            col_sum += b_val;
        }

        // Subtract zero points. This is equivalent to doing
        // `acc += (a - a_zero) * (b - b_zero)` in the loop over K, but more
        // efficient.
        let a_zero = a_zero_point as i32;
        let b_zero = b_zero_points.map(|bq| bq[col] as i32).unwrap_or(0) + b_zero_shift;
        acc = depth as i32 * a_zero * b_zero + acc - row_sum * b_zero - col_sum * a_zero;

        let out_el = unsafe { out.data.as_ptr().add(col) as *mut i32 };
        if !out.beta {
            unsafe { out_el.write(acc) };
        } else {
            unsafe { *out_el += acc };
        }
    }
}

/// Variant of [`simd_int8_gemv`] for the case where the RHS has unit row stride.
///
/// This is unsafe as it assumes compatibility of input and output dimensions
/// has been checked by `simd_int8_gemv`.
#[inline(always)]
unsafe fn simd_int8_gemv_transposed<I: Isa, const CAST_B_U8: bool>(
    isa: I,
    out: MatVecOutput<i32, bool>,
    a: &[u8],
    b: Matrix<i8>,
    a_zero_point: u8,
    b_zero_points: Option<&[i8]>,
    dot: impl Int8DotProduct<X8 = I::I8, I32 = I::I32> + Copy,
) {
    let ops = isa.i32();
    let i8_ops = isa.i8();

    let bit_flip_mask = i8_ops.splat(I8_U8_SHIFT_MASK);
    let b_zero_shift = if CAST_B_U8 { 128 } else { 0 };
    let depth = a.len();

    let row_sum: i32 = a.iter().map(|x| *x as i32).sum();
    let a_ptr = a.as_ptr();
    let b_ptr = b.storage().as_ptr();
    let one_u8 = i8_ops.splat(1);

    for col in 0..b.cols() {
        let b_ptr = b_ptr.add(col * b.col_stride());
        let mut acc = ops.zero();
        let mut col_sum = ops.zero();
        let mut k_tiles = range_chunks_exact(0..depth, i8_ops.len());

        for k_tile in k_tiles.by_ref() {
            let a = i8_ops.load_ptr(a_ptr.add(k_tile.start) as *const i8);
            let b = i8_ops.load_ptr(b_ptr.add(k_tile.start));
            let b = if CAST_B_U8 {
                i8_ops.xor(b, bit_flip_mask)
            } else {
                b
            };

            acc = dot.dot_product(a, b, acc);
            col_sum = dot.dot_product(one_u8, b, col_sum);
        }

        let mut acc = ops.sum(acc);
        let mut col_sum = ops.sum(col_sum);

        for k in k_tiles.remainder() {
            let a = *a_ptr.add(k) as i32;
            let b = *b_ptr.add(k) as i32;
            let b = if CAST_B_U8 { b + b_zero_shift } else { b };
            acc += a * b;
            col_sum += b;
        }

        let a_zero = a_zero_point as i32;
        let b_zero = b_zero_points
            .map(|bz| bz[col] as i32 + b_zero_shift)
            .unwrap_or(b_zero_shift);
        let acc = (depth as i32 * a_zero * b_zero) + acc - row_sum * b_zero - col_sum * a_zero;

        let out_ptr = out.data.get_unchecked_mut(col);
        if !out.beta {
            out_ptr.write(acc);
        } else {
            out_ptr.write(out_ptr.assume_init() + acc);
        }
    }
}

/// Fallback for [`simd_int8_gemv`] when RHS has neither unit column stride nor
/// unit row stride.
///
/// This is unsafe as it assumes compatibility of input and output dimensions
/// has been checked by `simd_int8_gemv`.
#[inline(always)]
unsafe fn simd_int8_gemv_fallback<const CAST_B_U8: bool>(
    out: MatVecOutput<i32, bool>,
    a: &[u8],
    b: Matrix<i8>,
    a_zero_point: u8,
    b_zero_points: Option<&[i8]>,
) {
    let b_zero_shift = if CAST_B_U8 { 128 } else { 0 };
    let depth = a.len();
    for (out_el, col) in out.data.iter_mut().zip(0..b.cols()) {
        let b_zero = b_zero_points
            .map(|bz| bz[col] as i32 + b_zero_shift)
            .unwrap_or(0);
        let mut acc = 0;
        let mut row_sum = 0;
        let mut col_sum = 0;

        for k in 0..depth {
            let a_el = unsafe { *a.get_unchecked(k) } as i32;
            let b_el = unsafe { *b.get_unchecked([k, col]) } as i32;
            let b_el = if CAST_B_U8 { b_el + b_zero_shift } else { b_el };
            acc += a_el * b_el;
            row_sum += a_el;
            col_sum += b_el;
        }

        // Subtract zero points. This is equivalent to doing
        // `acc += (a - a_zero) * (b - b_zero)` in the loop over K, but more
        // efficient.
        let a_zero = a_zero_point as i32;
        acc = depth as i32 * a_zero * b_zero + acc - row_sum * b_zero - col_sum * a_zero;

        if !out.beta {
            out_el.write(acc);
        } else {
            // Safety: Output is initialized when `beta` is true
            unsafe {
                out_el.write(out_el.assume_init() + acc);
            }
        }
    }
}
