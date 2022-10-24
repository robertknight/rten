///! Optimized linear algebra functions.
///!
///! This module provides a subset of BLAS-like functions that are used by
///! neural network operators. This includes general matrix multiplication ("gemm"),
///! and vector-scalar products.
use crate::tensor::Tensor;

pub fn div_ceil(a: usize, b: usize) -> usize {
    let rounding = if a % b == 0 { 0 } else { 1 };
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

/// Kernel that computes a small tile of the output matrix of size HxW.
///
/// This corresponds to Loop 6 of the algorithm in Page 4 of
/// https://dl.acm.org/doi/pdf/10.1145/2925987.
fn kernel<const H: usize, const W: usize>(
    out: &mut [f32],
    out_row_stride: usize,
    used_rows: usize,
    used_cols: usize,
    a: &[f32],
    b: &[f32],
    depth: usize,
) {
    // Accumulate into a fixed-sized array to allow the compiler to generate
    // more efficient code for the loop over `depth`.
    let mut tmp = [[0.0; W]; H];
    for k in 0..depth {
        let a_off = k * H;
        let b_off = k * W;

        for i in 0..H {
            for j in 0..W {
                tmp[i][j] += a[a_off + i] * b[b_off + j];
            }
        }
    }

    // The `min` operations here are strictly redundant as `used_rows` <= H and
    // `used_cols` <= W. Curiously the kernel ran faster under WASM when they
    // are left in.
    for i in 0..H.min(used_rows) {
        for j in 0..W.min(used_cols) {
            out[out_row_stride * i + j] += tmp[i][j];
        }
    }
}

/// Pack a block of the "A" matrix.
///
/// The packed buffer is laid out as a sequence of `ceil(a_rows / PANEL_HEIGHT)`
/// row panels. Each row panel has size `PANEL_HEIGHT * panel_width` and uses
/// column-major order. If `a_rows` is not a multiple of `PANEL_HEIGHT`, the
/// final panel is zero-padded.
///
/// The `a` slice is assumed to start at the first element to be packed and
/// extend to include at least the last element to be packed.
fn pack_a_block<const PANEL_HEIGHT: usize>(
    out: &mut [f32],
    a: &[f32],
    a_row_stride: usize,
    a_cols: usize,
    a_rows: usize,
) {
    let n_panels = round_up(a_rows, PANEL_HEIGHT) / PANEL_HEIGHT;
    for panel in 0..n_panels {
        let panel_offset = panel * a_cols * PANEL_HEIGHT;
        let panel_start_row = panel * PANEL_HEIGHT;

        for col in 0..a_cols {
            let out_col_offset = panel_offset + col * PANEL_HEIGHT;
            for row in 0..PANEL_HEIGHT {
                let a_row = panel_start_row + row;
                out[out_col_offset + row] = if a_row < a_rows {
                    a[a_row * a_row_stride + col]
                } else {
                    0.0
                };
            }
        }
    }
}

/// Pack block of the "B" matrix.
///
/// The packed buffer is laid out as a sequence of `ceil(b_cols / PANEL_WIDTH)`
/// column panels. Each column panel has size `b_rows * PANEL_WIDTH` and
/// uses row-major order. If `b_cols` is not a multiple of `PANEL_WIDTH`, the
/// final panel is zero-padded.
///
/// The `b` slice is assumed to start at the first element to be packed and
/// extend to include at least the last element to be packed.
fn pack_b_block<const PANEL_WIDTH: usize>(
    out: &mut [f32],
    b: &[f32],
    b_row_stride: usize,
    b_cols: usize,
    b_rows: usize,
) {
    let n_panels = round_up(b_cols, PANEL_WIDTH) / PANEL_WIDTH;
    for panel in 0..n_panels {
        let panel_offset = panel * b_rows * PANEL_WIDTH;
        let panel_start_col = panel * PANEL_WIDTH;

        if b_cols - panel_start_col >= PANEL_WIDTH {
            // Optimized loop for panels that don't need any padding
            for row in 0..b_rows {
                let out_row_offset = panel_offset + row * PANEL_WIDTH;
                let b_row_offset = row * b_row_stride + panel_start_col;

                let out_row = &mut out[out_row_offset..out_row_offset + PANEL_WIDTH];
                let b_row = &b[b_row_offset..b_row_offset + PANEL_WIDTH];

                for col in 0..PANEL_WIDTH {
                    out_row[col] = b_row[col];
                }
            }
        } else {
            // Fallback for final panel if padding is required
            for row in 0..b_rows {
                let out_row_offset = panel_offset + row * PANEL_WIDTH;
                let b_row_offset = row * b_row_stride;

                for col in 0..PANEL_WIDTH {
                    let out_col = panel_start_col + col;
                    let b_offset = b_row_offset + panel_start_col + col;

                    out[out_row_offset + col] = if out_col < b_cols { b[b_offset] } else { 0.0 };
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

/// Multiply two matrices and add the results to `output`.
///
/// This is a high-level API that operates on tensors.
pub fn gemm(output: &mut Tensor, a: &Tensor, b: &Tensor) {
    let [a_rows, a_cols] = a.dims();
    let [b_rows, b_cols] = b.dims();
    let out_row_stride = output.stride(0);

    gemm_slice(
        output.data_mut(),
        out_row_stride,
        a.data(),
        a_rows,
        a_cols,
        a.stride(0),
        b.data(),
        b_rows,
        b_cols,
        b.stride(0),
    );
}

/// Multiply two matrices and add the results to `out_data`.
///
/// This is a low-level API that operates directly on slices. Use `gemm` for
/// a more convenient way to multiply two 2D tensors.
///
/// The implementation uses the general approach of BLIS
/// (https://github.com/flame/blis), and was informed by the matrixmultiply
/// crate (https://github.com/bluss/matrixmultiply). See Pages 3-5 of
/// https://dl.acm.org/doi/pdf/10.1145/2925987 for an outline of the algorithm.
pub fn gemm_slice(
    out_data: &mut [f32],
    out_row_stride: usize,
    a_data: &[f32],
    a_rows: usize,
    a_cols: usize,
    a_row_stride: usize,
    b_data: &[f32],
    b_rows: usize,
    b_cols: usize,
    b_row_stride: usize,
) {
    if a_cols != b_rows {
        panic!("Columns of matrix `a` must match rows of matrix `b`");
    }

    // The constant values below were taken from the matrixmultiply crate. The
    // MR and NR sizes correspond to its fallback kernel (ie. not using SSE,
    // AVX or other intrinsics). See https://dl.acm.org/doi/pdf/10.1145/2925987
    // for an explanation of how suitable values are determined. Since we don't
    // know exactly which CPU this code will be run on, we try to pick
    // something that will work well on most systems.

    // Sizes of blocks that the width (nc), depth (kc) and height (mc)
    // dimensions are partitioned into in the outer loops. These are chosen
    // so that blocks can fit in specific cache levels.
    let nc = round_up(1024.min(b_cols), NR);
    let mc = round_up(64.min(a_rows), MR);
    let kc = 256.min(a_cols);

    // Size of output tiles in rows (MR) and columns (NR) computed by innermost
    // loops. These are chosen so that an MRxNR tile can fit in registers.
    const MR: usize = 8;
    const NR: usize = 4;

    // Buffers for packed blocks of the matrix. These currently have no
    // alignment specified. The paper mentioned above suggests that aligning to
    // cache-line (ie. 64-byte) boundaries may help performance.
    let mut packed_b = vec![0.0; kc * nc];
    let mut packed_a = vec![0.0; mc * kc];

    for (col_start, col_end) in blocks(0, b_cols, nc) {
        for (depth_start, depth_end) in blocks(0, a_cols, kc) {
            let panel_length = depth_end - depth_start;

            pack_b_block::<NR>(
                &mut packed_b,
                &b_data[depth_start * b_row_stride + col_start..depth_end * b_row_stride],
                b_row_stride,
                col_end - col_start,
                panel_length,
            );

            for (row_start, row_end) in blocks(0, a_rows, mc) {
                pack_a_block::<MR>(
                    &mut packed_a,
                    &a_data[row_start * a_row_stride + depth_start..row_end * a_row_stride],
                    a_row_stride,
                    panel_length,
                    row_end - row_start,
                );

                let b_panel_size = panel_length * NR;
                let a_panel_size = MR * panel_length;

                for (tile_col_start, tile_col_end) in blocks(col_start, col_end, NR) {
                    let b_panel_idx = (tile_col_start - col_start) / NR;
                    let b_panel_offset = b_panel_idx * b_panel_size;
                    let b_panel = &packed_b[b_panel_offset..b_panel_offset + b_panel_size];

                    for (tile_row_start, tile_row_end) in blocks(row_start, row_end, MR) {
                        let a_panel_idx = (tile_row_start - row_start) / MR;
                        let a_panel_offset = a_panel_idx * a_panel_size;
                        let a_panel = &packed_a[a_panel_offset..a_panel_offset + a_panel_size];

                        let out_offset = tile_row_start * out_row_stride + tile_col_start;
                        let out_tile = &mut out_data[out_offset..];

                        kernel::<MR, NR>(
                            out_tile,
                            out_row_stride,
                            tile_row_end - tile_row_start,
                            tile_col_end - tile_col_start,
                            a_panel,
                            b_panel,
                            panel_length,
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::linalg::{add_scaled_vector, gemm};
    use crate::rng::XorShiftRNG;
    use crate::tensor::{random_tensor, zero_tensor, Tensor};
    use crate::test_util::expect_equal;

    fn reference_gemm(a: &Tensor, b: &Tensor) -> Tensor {
        let [a_rows, a_cols] = a.dims();
        let [_b_rows, b_cols] = b.dims();
        let mut output = zero_tensor(&[a_rows, b_cols]);

        for r in 0..a_rows {
            for c in 0..b_cols {
                for k in 0..a_cols {
                    output[[r, c]] += a[[r, k]] * b[[k, c]];
                }
            }
        }

        output
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

    #[test]
    fn test_gemm() -> Result<(), String> {
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
            let a = random_tensor(&lhs_size, &mut rng);
            let b = random_tensor(&rhs_size, &mut rng);
            let mut result = zero_tensor::<f32>(&[lhs_size[0], rhs_size[1]]);

            gemm(&mut result, &a, &b);

            let expected = reference_gemm(&a, &b);

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
}
