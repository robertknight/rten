use std::mem::MaybeUninit;

use rten_simd::SimdFloat;
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
