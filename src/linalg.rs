///! Optimized linear algebra functions.
///!
///! This module provides a subset of BLAS-like functions that are used by
///! neural network operators. This includes general matrix multiplication ("gemm"),
///! and vector-scalar products.
use std::ops::Range;

use crate::tensor::Tensor;

pub fn div_ceil(a: usize, b: usize) -> usize {
    if b == 1 {
        // Fast path
        return a;
    }
    let rounding = usize::from(a % b != 0);
    a / b + rounding
}

/// Compute `dest += src * scale`, also known as a vector-scalar product or
/// "axpy" operation.
///
/// `dest_stride` and `src_stride` specifies the strides to use when iterating
/// over `dest` and `src` respectively. The lengths of `dest` and `src` must
/// match after accounting for their respective strides.
pub fn add_scaled_vector(
    dest: &mut [f32],
    src: &[f32],
    dest_stride: usize,
    src_stride: usize,
    scale: f32,
) {
    // Fast path for non-strided case. We write a trivial loop and leave the
    // compiler to optimize it.
    if src_stride == 1 && dest_stride == 1 {
        if src.len() != dest.len() {
            panic!("src and dest vector sizes do not match");
        }
        for i in 0..dest.len() {
            dest[i] += src[i] * scale;
        }
        return;
    }

    let src_els = div_ceil(src.len(), src_stride);
    let dest_els = div_ceil(dest.len(), dest_stride);

    if src_els != dest_els {
        panic!("src and dest vector sizes do not match");
    }

    const N: usize = 4;
    let n_blocks = src_els / N;
    let mut val = [0.0; N];

    for b in 0..n_blocks {
        for i in 0..N {
            unsafe {
                val[i] = src.get_unchecked((b * N + i) * src_stride) * scale;
            }
        }

        for i in 0..N {
            unsafe {
                *dest.get_unchecked_mut((b * N + i) * dest_stride) += val[i];
            }
        }
    }

    for i in n_blocks * N..src_els {
        unsafe {
            *dest.get_unchecked_mut(i * dest_stride) += src.get_unchecked(i * src_stride) * scale;
        }
    }
}

struct BlockIter {
    start: usize,
    end: usize,
    step: usize,
}

impl Iterator for BlockIter {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        if self.start < self.end {
            let start = self.start;
            let end = (start + self.step).min(self.end);
            self.start += self.step;
            Some((start, end))
        } else {
            None
        }
    }
}

/// Return an iterator over (block_start, block_end) tuples of `step`-sized
/// blocks between `start` and `end`. If `end - start` is not a multiple of
/// `step` then the final block will be smaller.
fn blocks(start: usize, end: usize, step: usize) -> BlockIter {
    BlockIter { start, end, step }
}

/// Kernel that computes a small tile of a matrix multiplication output.
///
/// The tile size depends upon the kernel and is specified by the `MR` and `NR`
/// associated constants. The MR and NR values are chosen such that an `MR * NR`
/// tile can fit in registers. NR is generally determined by the width of the
/// registers used (eg. for SSE, 128 bits = 4x32 floats, so NR is 4) and MR by
/// the number available.
///
/// The kernel corresponds to Loop 6 of the algorithm in Page 4 of
/// https://dl.acm.org/doi/pdf/10.1145/2925987.
trait Kernel {
    /// Height of output tiles computed by the kernel.
    const MR: usize;

    /// Width of output tiles computed by the kernel.
    const NR: usize;

    /// Return true if this kernel is usable on the current system.
    ///
    /// It is unsafe to call `kernel` if this is false.
    fn supported() -> bool;

    /// Compute an `MR * NR`-sized tile of the output matrix.
    fn kernel(
        out: &mut [f32],
        out_row_stride: usize,
        a: &[f32],
        b: &[f32],
        depth: usize,
        alpha: f32,
        beta: f32,
    );
}

/// Optimized kernel for x64 CPUs that support AVX + FMA instructions.
#[cfg(target_arch = "x86_64")]
struct FMAKernel {}

#[cfg(target_arch = "x86_64")]
impl Kernel for FMAKernel {
    const MR: usize = 8;
    const NR: usize = 8; // AVX registers are 256 bits wide = 8 x f32

    fn supported() -> bool {
        is_x86_feature_detected!("fma")
    }

    fn kernel(
        out: &mut [f32],
        out_row_stride: usize,
        a: &[f32],
        b: &[f32],
        depth: usize,
        alpha: f32,
        beta: f32,
    ) {
        unsafe { Self::kernel_fma(out, out_row_stride, a, b, depth, alpha, beta) }
    }
}

#[cfg(target_arch = "x86_64")]
impl FMAKernel {
    #[target_feature(enable = "fma")]
    unsafe fn kernel_fma(
        out: &mut [f32],
        out_row_stride: usize,
        a: &[f32],
        b: &[f32],
        depth: usize,
        alpha: f32,
        beta: f32,
    ) {
        const MR: usize = FMAKernel::MR;
        const NR: usize = FMAKernel::NR;

        use core::arch::x86_64::{
            _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps,
            _mm256_setzero_ps, _mm256_storeu_ps,
        };

        // Check that buffer accesses below are going to be valid.
        assert!(a.len() >= depth * MR);
        assert!(b.len() >= depth * NR);
        assert!(out.len() >= (MR - 1) * out_row_stride + NR);

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        // Accumulate into a fixed-sized array to allow the compiler to generate
        // more efficient code for the loop over `depth`.
        let mut tmp = [_mm256_setzero_ps(); MR];
        for k in 0..depth {
            let a_off = k * MR;
            let b_off = k * NR;

            let b_row = _mm256_loadu_ps(b_ptr.add(b_off));
            for i in 0..MR {
                let a_val = *a_ptr.add(a_off + i);
                let a_broadcast = _mm256_set1_ps(a_val);
                tmp[i] = _mm256_fmadd_ps(a_broadcast, b_row, tmp[i]);
            }
        }

        if beta == 0. && alpha == 1. {
            for i in 0..MR {
                let out_ptr = out.as_mut_ptr().add(out_row_stride * i);
                _mm256_storeu_ps(out_ptr, tmp[i]);
            }
        } else if beta == 1. && alpha == 1. {
            for i in 0..MR {
                let out_ptr = out.as_mut_ptr().add(out_row_stride * i);
                let out_val = _mm256_add_ps(_mm256_loadu_ps(out_ptr), tmp[i]);
                _mm256_storeu_ps(out_ptr, out_val);
            }
        } else {
            let alpha_broadcast = _mm256_set1_ps(alpha);
            let beta_broadcast = _mm256_set1_ps(beta);
            for i in 0..MR {
                let out_ptr = out.as_mut_ptr().add(out_row_stride * i);
                let out_val = _mm256_mul_ps(_mm256_loadu_ps(out_ptr), beta_broadcast);
                let out_val = _mm256_fmadd_ps(tmp[i], alpha_broadcast, out_val);
                _mm256_storeu_ps(out_ptr, out_val);
            }
        }
    }
}

/// This is the base kernel that does not use architecture-specific intrinsics
/// but is autovectorization-friendly. It is expected to perform the same as
/// a kernel using SSE intrinsics (or equivalent).
struct BaseKernel {}

impl Kernel for BaseKernel {
    const MR: usize = 8;

    // The base kernel will most likely be compiled to SSE or equivalent. SSE
    // registers are 128 bits wide = 4 x f32.
    const NR: usize = 4;

    fn supported() -> bool {
        true
    }

    fn kernel(
        out: &mut [f32],
        out_row_stride: usize,
        a: &[f32],
        b: &[f32],
        depth: usize,
        alpha: f32,
        beta: f32,
    ) {
        const MR: usize = BaseKernel::MR;
        const NR: usize = BaseKernel::NR;

        assert!(a.len() >= depth * MR);
        assert!(b.len() >= depth * NR);
        assert!(out.len() >= (MR - 1) * out_row_stride + NR);

        // Accumulate into a fixed-sized array to allow the compiler to generate
        // more efficient code for the loop over `depth`.
        let mut tmp = [[0.0; NR]; MR];
        for k in 0..depth {
            let a_off = k * MR;
            let b_off = k * NR;

            for i in 0..MR {
                for j in 0..NR {
                    // Safety: Indexes are less than lengths asserted above.
                    unsafe {
                        tmp[i][j] += a.get_unchecked(a_off + i) * b.get_unchecked(b_off + j);
                    }
                }
            }
        }

        if beta == 0. && alpha == 1. {
            for i in 0..MR {
                for j in 0..NR {
                    // Safety: Index is less than length asserted above.
                    unsafe {
                        let out_el = out.get_unchecked_mut(out_row_stride * i + j);
                        *out_el = tmp[i][j];
                    }
                }
            }
        } else if beta == 1. && alpha == 1. {
            for i in 0..MR {
                for j in 0..NR {
                    // Safety: Index is less than length asserted above.
                    unsafe {
                        let out_el = out.get_unchecked_mut(out_row_stride * i + j);
                        *out_el += tmp[i][j];
                    }
                }
            }
        } else {
            for i in 0..MR {
                for j in 0..NR {
                    // Safety: Index is less than length asserted above.
                    unsafe {
                        let out_el = out.get_unchecked_mut(out_row_stride * i + j);
                        *out_el = beta * *out_el + alpha * tmp[i][j];
                    }
                }
            }
        }
    }
}

/// Pack a block of the "A" matrix for use by kernel K.
///
/// The packed buffer is laid out as a sequence of `ceil(rows.len() / K::MR)`
/// row panels. Each row panel has size `K::MR * cols.len()` and uses
/// column-major order. If `rows.len()` is not a multiple of `K::MR`, the
/// final panel is zero-padded.
fn pack_a_block<K: Kernel>(out: &mut [f32], a: Matrix, rows: Range<usize>, cols: Range<usize>) {
    let a_rows = rows.len();
    let a_cols = cols.len();

    let n_panels = round_up(a_rows, K::MR) / K::MR;
    for panel in 0..n_panels {
        let panel_offset = panel * a_cols * K::MR;
        let panel_start_row = panel * K::MR;

        if a_rows - panel_start_row >= K::MR {
            // Optimized loop for panels that don't need any padding
            let a_offset =
                (rows.start + panel_start_row) * a.row_stride + cols.start * a.col_stride;

            assert!(out.len() > panel_offset + (a_cols - 1) * K::MR + K::MR - 1);
            assert!(
                a.data.len() > a_offset + (K::MR - 1) * a.row_stride + (a_cols - 1) * a.col_stride
            );

            for col in 0..a_cols {
                for row in 0..K::MR {
                    // Safety: Indexes are less than lengths asserted above.
                    unsafe {
                        *out.get_unchecked_mut(panel_offset + col * K::MR + row) = *a
                            .data
                            .get_unchecked(a_offset + row * a.row_stride + col * a.col_stride);
                    }
                }
            }
        } else {
            // Fallback for final panel if padding is required
            for col in 0..a_cols {
                let out_col_offset = panel_offset + col * K::MR;
                for row in 0..K::MR {
                    let a_row = rows.start + panel_start_row + row;
                    out[out_col_offset + row] = if a_row < rows.end {
                        a.data[a_row * a.row_stride + (cols.start + col) * a.col_stride]
                    } else {
                        0.0
                    };
                }
            }
        }
    }
}

/// Pack a block of the "B" matrix for use by kernel K.
///
/// The packed buffer is laid out as a sequence of `ceil(cols.len() /
/// K::NR)` column panels. Each column panel has size `rows.len() *
/// K::NR` and uses row-major order. If `cols.len()` is not a multiple of
/// `K::NR`, the final panel is zero-padded.
fn pack_b_block<K: Kernel>(out: &mut [f32], b: Matrix, rows: Range<usize>, cols: Range<usize>) {
    let b_cols = cols.len();
    let b_rows = rows.len();

    let n_panels = round_up(b_cols, K::NR) / K::NR;
    for panel in 0..n_panels {
        let panel_offset = panel * b_rows * K::NR;
        let panel_start_col = panel * K::NR;

        if b_cols - panel_start_col >= K::NR {
            // Optimized loop for panels that don't need any padding
            let b_offset =
                rows.start * b.row_stride + (cols.start + panel_start_col) * b.col_stride;

            assert!(out.len() >= panel_offset + (b_rows - 1) * K::NR + K::NR);
            assert!(
                b.data.len() > b_offset + (b_rows - 1) * b.row_stride + (K::NR - 1) * b.col_stride
            );

            for row in 0..b_rows {
                for col in 0..K::NR {
                    // Safety: Indexes are less than lengths asserted above.
                    unsafe {
                        *out.get_unchecked_mut(panel_offset + row * K::NR + col) = *b
                            .data
                            .get_unchecked(b_offset + row * b.row_stride + col * b.col_stride);
                    }
                }
            }
        } else {
            // Fallback for final panel if padding is required
            for row in 0..b_rows {
                let out_row_offset = panel_offset + row * K::NR;
                let b_row_offset = (rows.start + row) * b.row_stride;

                for col in 0..K::NR {
                    let out_col = panel_start_col + col;
                    let b_offset =
                        b_row_offset + (cols.start + panel_start_col + col) * b.col_stride;

                    out[out_row_offset + col] = if out_col < b_cols {
                        b.data[b_offset]
                    } else {
                        0.0
                    };
                }
            }
        }
    }
}

/// Return the smallest multiple of `factor` that is >= `val`.
fn round_up(val: usize, factor: usize) -> usize {
    let rem = val % factor;
    if rem == 0 {
        val
    } else {
        (val + factor) - rem
    }
}

/// Perform a General Matrix Multiplication ("gemm").
///
/// This computes `output = alpha * (a @ b) + beta * output` where `@` is
/// matrix multiplication.
///
/// This is a high-level API that operates on `Tensor`s. See `gemm` for
/// a low-level API that operates on slices.
#[allow(dead_code)] // Currently only used in tests
pub fn gemm_tensors(output: &mut Tensor, a: &Tensor, b: &Tensor, alpha: f32, beta: f32) {
    let [a_rows, a_cols] = a.dims();
    let [b_rows, b_cols] = b.dims();
    let out_row_stride = output.stride(0);

    gemm(
        output.data_mut(),
        out_row_stride,
        Matrix {
            data: a.data(),
            rows: a_rows,
            cols: a_cols,
            row_stride: a.stride(0),
            col_stride: a.stride(1),
        },
        Matrix {
            data: b.data(),
            rows: b_rows,
            cols: b_cols,
            row_stride: b.stride(0),
            col_stride: b.stride(1),
        },
        alpha,
        beta,
    );
}

/// Struct specifying details of an input matrix for use in GEMM operation.
///
/// Unlike a `Tensor`, this doesn't own the data.
#[derive(Copy, Clone)]
pub struct Matrix<'a> {
    pub data: &'a [f32],
    pub rows: usize,
    pub cols: usize,
    pub row_stride: usize,
    pub col_stride: usize,
}

/// Perform a General Matrix Multiplication ("gemm").
///
/// This is a low-level API that operates directly on slices. Use `gemm` for
/// a more convenient way to multiply two 2D tensors.
///
/// This computes `output = alpha * (a @ b) + beta * output` where `@` is
/// matrix multiplication.
///
/// The implementation uses the general approach of BLIS
/// (https://github.com/flame/blis), and was informed by the matrixmultiply
/// crate (https://github.com/bluss/matrixmultiply). See Pages 3-5 of
/// https://dl.acm.org/doi/pdf/10.1145/2925987 for an outline of the algorithm.
pub fn gemm(
    out_data: &mut [f32],
    out_row_stride: usize,
    a: Matrix,
    b: Matrix,
    alpha: f32,
    beta: f32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if FMAKernel::supported() {
            return gemm_impl::<FMAKernel, { FMAKernel::MR * FMAKernel::NR }>(
                out_data,
                out_row_stride,
                a,
                b,
                alpha,
                beta,
            );
        }
    }
    gemm_impl::<BaseKernel, { BaseKernel::MR * BaseKernel::NR }>(
        out_data,
        out_row_stride,
        a,
        b,
        alpha,
        beta,
    )
}

#[cfg(test)]
pub fn gemm_base_kernel(
    out_data: &mut [f32],
    out_row_stride: usize,
    a: Matrix,
    b: Matrix,
    alpha: f32,
    beta: f32,
) {
    gemm_impl::<BaseKernel, { BaseKernel::MR * BaseKernel::NR }>(
        out_data,
        out_row_stride,
        a,
        b,
        alpha,
        beta,
    )
}

/// Perform matrix multiplication with a given kernel.
///
/// `MR_NR` should be computed as `K::MR * K::NR`. This function can't compute
/// that itself due to Rust limitations on using generic parameters in const
/// expressions.
fn gemm_impl<K: Kernel, const MR_NR: usize>(
    out_data: &mut [f32],
    out_row_stride: usize,
    a: Matrix,
    b: Matrix,
    alpha: f32,
    beta: f32,
) {
    assert!(K::supported());

    if a.cols != b.rows {
        panic!("Columns of matrix `a` must match rows of matrix `b`");
    }

    // The constant values below were taken from the matrixmultiply crate. The
    // MR and NR sizes correspond to its fallback (non-SIMD) and SSE kernels.
    // See https://dl.acm.org/doi/pdf/10.1145/2925987 for an explanation of how
    // suitable values are determined. Since we don't know exactly which CPU
    // this code will be run on, we try to pick something that will work well on
    // most systems.

    // Sizes of blocks that the width (nc), depth (kc) and height (mc)
    // dimensions are partitioned into in the outer loops. These are chosen
    // so that blocks can fit in specific cache levels.
    let nc = round_up(1024.min(b.cols), K::NR);
    let mc = round_up(64.min(a.rows), K::MR);
    let kc = 256.min(a.cols);

    // Buffer for packed blocks of the matrix. Conceptually there are two
    // buffers, but we coalesce them into one allocation.
    //
    // These currently have no alignment specified. The paper mentioned above
    // suggests that aligning to cache-line (ie. 64-byte) boundaries may help
    // performance.
    let packed_b_size = kc * nc;
    let packed_a_size = mc * kc;
    let mut packed = vec![0.; packed_b_size + packed_a_size];

    for (col_start, col_end) in blocks(0, b.cols, nc) {
        for (depth_start, depth_end) in blocks(0, a.cols, kc) {
            let panel_length = depth_end - depth_start;
            pack_b_block::<K>(
                &mut packed[..packed_b_size],
                b,
                depth_start..depth_end,
                col_start..col_end,
            );

            for (row_start, row_end) in blocks(0, a.rows, mc) {
                pack_a_block::<K>(
                    &mut packed[packed_b_size..],
                    a,
                    row_start..row_end,
                    depth_start..depth_end,
                );

                let packed_b = &packed[..packed_b_size];
                let packed_a = &packed[packed_b_size..];

                let b_panel_size = panel_length * K::NR;
                let a_panel_size = K::MR * panel_length;

                for (tile_col_start, tile_col_end) in blocks(col_start, col_end, K::NR) {
                    let b_panel_idx = (tile_col_start - col_start) / K::NR;
                    let b_panel_offset = b_panel_idx * b_panel_size;
                    let b_panel = &packed_b[b_panel_offset..b_panel_offset + b_panel_size];

                    for (tile_row_start, tile_row_end) in blocks(row_start, row_end, K::MR) {
                        let a_panel_idx = (tile_row_start - row_start) / K::MR;
                        let a_panel_offset = a_panel_idx * a_panel_size;
                        let a_panel = &packed_a[a_panel_offset..a_panel_offset + a_panel_size];

                        let out_offset = tile_row_start * out_row_stride + tile_col_start;
                        let out_tile = &mut out_data[out_offset..];

                        let used_rows = tile_row_end - tile_row_start;
                        let used_cols = tile_col_end - tile_col_start;

                        // Only use provided `beta` on the first write to this output tile. For
                        // subsequent updates accumulate.
                        let effective_beta = if depth_start == 0 { beta } else { 1.0 };

                        if used_rows == K::MR && used_cols == K::NR {
                            K::kernel(
                                out_tile,
                                out_row_stride,
                                a_panel,
                                b_panel,
                                panel_length,
                                alpha,
                                effective_beta,
                            );
                        } else {
                            // If this is not a full size tile, run the kernel on a temporary
                            // buffer that is the size of a full tile, then copy the results back
                            // to the output. This allows the same kernel implementation to be used
                            // whether the tile is full-sized or not.
                            let mut tmp_out_tile = [0.; MR_NR];

                            K::kernel(
                                &mut tmp_out_tile,
                                K::NR,
                                a_panel,
                                b_panel,
                                panel_length,
                                alpha,
                                0., // Multiplication with `effective_beta` is handled below.
                            );

                            assert!(out_tile.len() >= (used_rows - 1) * out_row_stride + used_cols);
                            assert!(tmp_out_tile.len() >= (used_rows - 1) * K::NR + used_cols);
                            for i in 0..used_rows {
                                for j in 0..used_cols {
                                    // Safety: Index is less than length asserted above.
                                    unsafe {
                                        let out_el =
                                            out_tile.get_unchecked_mut(out_row_stride * i + j);
                                        *out_el =
                                            effective_beta * *out_el + tmp_out_tile[i * K::NR + j];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::linalg::{add_scaled_vector, gemm, gemm_base_kernel, Matrix};
    use crate::rng::XorShiftRNG;
    use crate::tensor::{rand, zeros, Tensor};
    use crate::test_util::expect_equal;

    fn reference_matmul(a: &Tensor, b: &Tensor) -> Tensor {
        let [a_rows, _a_cols] = a.dims();
        let [_b_rows, b_cols] = b.dims();
        let mut output = zeros(&[a_rows, b_cols]);

        reference_gemm(&mut output, a, b, 1.0, 0.0);

        output
    }

    #[derive(Clone, Copy, Debug)]
    enum Kernel {
        /// Use the preferred kernel for the current platform
        Auto,
        /// Use the fallback/base kernel
        Base,
    }

    fn run_gemm(
        output: &mut Tensor,
        a: &Tensor,
        b: &Tensor,
        alpha: f32,
        beta: f32,
        kernel: Kernel,
    ) {
        let [a_rows, a_cols] = a.dims();
        let [b_rows, b_cols] = b.dims();
        let out_row_stride = output.stride(0);

        let gemm_fn = match kernel {
            Kernel::Auto => gemm,
            Kernel::Base => gemm_base_kernel,
        };

        gemm_fn(
            output.data_mut(),
            out_row_stride,
            Matrix {
                data: a.data(),
                rows: a_rows,
                cols: a_cols,
                row_stride: a.stride(0),
                col_stride: a.stride(1),
            },
            Matrix {
                data: b.data(),
                rows: b_rows,
                cols: b_cols,
                row_stride: b.stride(0),
                col_stride: b.stride(1),
            },
            alpha,
            beta,
        );
    }

    fn reference_gemm(output: &mut Tensor, a: &Tensor, b: &Tensor, alpha: f32, beta: f32) {
        let [a_rows, a_cols] = a.dims();
        let [_b_rows, b_cols] = b.dims();

        for r in 0..a_rows {
            for c in 0..b_cols {
                let mut accum = 0.0;
                for k in 0..a_cols {
                    accum += a[[r, k]] * b[[k, c]];
                }
                output[[r, c]] = alpha * accum + beta * output[[r, c]];
            }
        }
    }

    #[test]
    fn test_add_scaled_vector() {
        let mut dest = vec![1.0, 2.0, 3.0, 4.0];
        let src = vec![10.0, 20.0, 30.0, 40.0];

        add_scaled_vector(&mut dest, &src, 1, 1, 2.0);

        assert_eq!(&dest, &[21.0, 42.0, 63.0, 84.0]);
    }

    #[test]
    fn test_add_scaled_vector_src_stride() {
        let mut dest = vec![1.0, 2.0];
        let src = vec![10.0, 20.0, 30.0];

        add_scaled_vector(&mut dest, &src, 1, 2, 1.0);

        assert_eq!(&dest, &[11.0, 32.0]);
    }

    #[test]
    fn test_add_scaled_vector_dest_stride() {
        let mut dest = vec![1.0, 2.0, 3.0];
        let src = vec![10.0, 20.0];

        add_scaled_vector(&mut dest, &src, 2, 1, 1.0);

        assert_eq!(&dest, &[11.0, 2.0, 23.0]);
    }

    #[test]
    #[should_panic(expected = "src and dest vector sizes do not match")]
    fn test_add_scaled_vector_size_mismatch() {
        let mut dest = vec![1.0, 2.0, 3.0];
        let src = vec![10.0, 20.0];
        add_scaled_vector(&mut dest, &src, 1, 1, 1.0);
    }

    #[test]
    #[should_panic(expected = "src and dest vector sizes do not match")]
    fn test_add_scaled_vector_strided_size_mismatch() {
        let mut dest = vec![1.0, 2.0];
        let src = vec![10.0, 20.0];
        add_scaled_vector(&mut dest, &src, 2, 1, 1.0);
    }

    // Simplest possible test case for easy debugging.
    #[test]
    fn test_simple_gemm() -> Result<(), String> {
        let a = Tensor::from_data(vec![2, 2], vec![1., 2., 3., 4.]);
        let b = Tensor::from_data(vec![2, 2], vec![5., 6., 7., 8.]);
        let expected = reference_matmul(&a, &b);

        let mut result = zeros::<f32>(&[a.shape()[0], b.shape()[1]]);
        run_gemm(&mut result, &a, &b, 1., 1., Kernel::Auto);
        expect_equal(&result, &expected)?;

        let mut result = zeros::<f32>(&[a.shape()[0], b.shape()[1]]);
        run_gemm(&mut result, &a, &b, 1., 1., Kernel::Base);
        expect_equal(&result, &expected)?;

        Ok(())
    }

    fn test_gemm_with_kernel(kernel: Kernel) -> Result<(), String> {
        // "Interesting" sizes for the row, column and depth dimensions of the
        // computation. These are chosen to cover cases that are less than,
        // equal to and above the tile/block sizes which the algorithm divides
        // the problem into along each dimension.
        let col_steps = [0, 2, 4, 5, 8, 1024, 1025];
        let depth_steps = [0, 2, 20, 256, 300];
        let row_steps = [0, 2, 8, 10, 16, 64, 80];

        let mut cases = Vec::new();

        // Simple cases where one dimension of the problem is varied to
        // different interesting values and other dimensions are kept small.
        for cs in col_steps {
            cases.push(([2, 2], [2, cs]));
        }
        for ds in depth_steps {
            cases.push(([2, ds], [ds, 2]));
        }
        for rs in row_steps {
            cases.push(([rs, 2], [2, 2]));
        }

        // Some simple square matrix tests of different sizes. This covers all
        // cases below a threshold, and then select sizes after that. This is
        // because larger sizes are slow in debug builds.
        for n in 1..20 {
            cases.push(([n, n], [n, n]));
        }
        for n in [30, 64, 65] {
            cases.push(([n, n], [n, n]));
        }

        for (lhs_size, rhs_size) in cases {
            let mut rng = XorShiftRNG::new(1234);
            let a = rand(&lhs_size, &mut rng);
            let b = rand(&rhs_size, &mut rng);
            let mut result = zeros::<f32>(&[lhs_size[0], rhs_size[1]]);

            run_gemm(&mut result, &a, &b, 1., 0., kernel);

            let expected = reference_matmul(&a, &b);

            if let Err(err) = expect_equal(&result, &expected) {
                println!(
                    "GEMM output for {}x{}x{} did not match reference",
                    lhs_size[0], rhs_size[1], lhs_size[1]
                );
                return Err(err);
            }
        }

        Ok(())
    }

    #[test]
    fn test_gemm_with_fastest_kernel() -> Result<(), String> {
        test_gemm_with_kernel(Kernel::Auto)
    }

    #[test]
    fn test_gemm_with_base_kernel() -> Result<(), String> {
        test_gemm_with_kernel(Kernel::Base)
    }

    #[test]
    fn test_gemm_transposed() -> Result<(), String> {
        let mut rng = XorShiftRNG::new(1234);
        let mut a = rand(&[20, 30], &mut rng);
        let mut b = rand(&[10, 20], &mut rng);

        // Transpose the input matrices. This will alter their row and column
        // strides and shapes, but not re-order the data.
        a.permute(&[1, 0]);
        b.permute(&[1, 0]);

        let [a_rows, _] = a.dims();
        let [_, b_cols] = b.dims();

        let mut result = zeros(&[a_rows, b_cols]);
        run_gemm(&mut result, &a, &b, 1., 1., Kernel::Auto);

        let expected = reference_matmul(&a, &b);
        expect_equal(&result, &expected)
    }

    #[test]
    fn test_gemm_alpha() -> Result<(), String> {
        let mut rng = XorShiftRNG::new(1234);

        let a = rand(&[10, 5], &mut rng);
        let b = rand(&[5, 15], &mut rng);

        for kernel in [Kernel::Auto, Kernel::Base] {
            for alpha in [0.0, 0.5, 1.0, 2.0] {
                let mut result = rand(&[10, 15], &mut rng);
                let mut expected = result.clone();

                run_gemm(&mut result, &a, &b, alpha, 0.0, kernel);
                reference_gemm(&mut expected, &a, &b, alpha, 0.0);

                expect_equal(&result, &expected)?;
            }
        }

        Ok(())
    }

    #[test]
    fn test_gemm_beta() -> Result<(), String> {
        let mut rng = XorShiftRNG::new(1234);

        let a = rand(&[10, 5], &mut rng);
        let b = rand(&[5, 15], &mut rng);

        for kernel in [Kernel::Auto, Kernel::Base] {
            for beta in [0.0, 0.5, 1.0, 2.0] {
                let mut result = rand(&[10, 15], &mut rng);
                let mut expected = result.clone();

                run_gemm(&mut result, &a, &b, 1., beta, kernel);
                reference_gemm(&mut expected, &a, &b, 1., beta);

                expect_equal(&result, &expected)?;
            }
        }

        Ok(())
    }
}
