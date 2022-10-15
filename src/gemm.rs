use crate::tensor::Tensor;

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
    for i in 0..H.min(used_rows) {
        for j in 0..W.min(used_cols) {
            out[out_row_stride * i + j] += tmp[i][j];
        }
    }
}

/// Pack a block of the "A" matrix.
///
/// The packed buffer is laid out as a sequence of row panels. Each row panel
/// has size `PANEL_HEIGHT * panel_width` and uses column-major order.
///
/// `panel_rows` is currently required to be a multiple of `PANEL_HEIGHT`.
fn pack_a_block<const PANEL_HEIGHT: usize>(
    out: &mut [f32],
    a: &[f32],
    a_row_stride: usize,
    a_cols: usize,
    a_rows: usize,
    panel_rows: usize,
    panel_width: usize,
) {
    let n_panels = panel_rows / PANEL_HEIGHT;
    let panel_elements = panel_width * PANEL_HEIGHT;
    for panel in 0..n_panels {
        let panel_offset = panel * panel_elements;
        for col in 0..panel_width {
            let out_col_offset = panel_offset + col * PANEL_HEIGHT;
            for panel_row in 0..PANEL_HEIGHT {
                let row = (panel * PANEL_HEIGHT) + panel_row;
                let a_offset = row * a_row_stride + col;

                out[out_col_offset + panel_row] = if col < a_cols && row < a_rows {
                    a[a_offset]
                } else {
                    0.0
                };
            }
        }
    }
}

/// Pack block of the "B" matrix.
///
/// The packed buffer is laid out as a sequence of column panels. Each column
/// panel as size `panel_height * PANEL_WIDTH` and uses row-major order.
///
/// `cols` is currently required to be a multiple of `PANEL_WIDTH`.
fn pack_b_block<const PANEL_WIDTH: usize>(
    out: &mut [f32],
    b: &[f32],
    b_row_stride: usize,
    b_cols: usize,
    b_rows: usize,
    cols: usize,
    panel_height: usize,
) {
    let n_panels = cols / PANEL_WIDTH;
    let panel_elements = panel_height * PANEL_WIDTH;
    for panel in 0..n_panels {
        let panel_offset = panel * panel_elements;
        let panel_start_col = panel * PANEL_WIDTH;

        for row in 0..panel_height {
            let out_row_offset = panel_offset + row * PANEL_WIDTH;
            let b_row_offset = row * b_row_stride;

            for col in 0..PANEL_WIDTH {
                let out_col = panel_start_col + col;
                let b_offset = b_row_offset + panel_start_col + col;

                out[out_row_offset + col] = if out_col < b_cols && row < b_rows {
                    b[b_offset]
                } else {
                    0.0
                };
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
/// The implementation uses the general approach of BLIS
/// (https://github.com/flame/blis), and was informed by the matrixmultiply
/// crate (https://github.com/bluss/matrixmultiply). See Pages 3-5 of
/// https://dl.acm.org/doi/pdf/10.1145/2925987 for an outline of the algorithm.
///
pub fn gemm(output: &mut Tensor, a: &Tensor, b: &Tensor) {
    let [a_rows, a_cols] = a.dims();
    let [b_rows, b_cols] = b.dims();

    if a_cols != b_rows {
        panic!("Columns of matrix `a` must match rows of matrix `b`");
    }

    // The constant values below were taken from the matrixmultiply crate. The
    // microblock sizes correspond to its fallback kernel (ie. not using SSE,
    // AVX or other intrinsics). See https://dl.acm.org/doi/pdf/10.1145/2925987
    // for an explanation of how suitable values are determined. Since we don't
    // know exactly which CPU this code will be run on, we have to pick
    // something that will work on most systems.

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

    let out_data = output.data_mut();
    let out_row_stride = b_cols;

    // Buffers for packed blocks the matrix. These currently have no alignment
    // specified. The paper mentioned above suggests that aligning to cache-line
    // (ie. 64-byte) boundaries may help performance.
    let mut packed_b = vec![0.0; kc * nc];
    let mut packed_a = vec![0.0; mc * kc];

    let a_data = a.data();
    let b_data = b.data();
    let b_row_stride = b_cols;
    let a_row_stride = a_cols;

    for (col_start, col_end) in blocks(0, b_cols, nc) {
        for (depth_start, depth_end) in blocks(0, a_cols, kc) {
            let block_width = col_end - col_start;
            let block_depth = depth_end - depth_start;

            pack_b_block::<NR>(
                &mut packed_b,
                &b_data[depth_start * b_row_stride + col_start..depth_end * b_row_stride],
                b_row_stride,
                block_width,
                block_depth,
                round_up(block_width, NR),
                block_depth,
            );
            for (row_start, row_end) in blocks(0, a_rows, mc) {
                let block_height = row_end - row_start;

                pack_a_block::<MR>(
                    &mut packed_a,
                    &a_data[row_start * a_row_stride + depth_start..row_end * a_row_stride],
                    a_row_stride,
                    block_depth,
                    block_height,
                    round_up(block_height, MR),
                    block_depth,
                );

                for (ub_col_start, ub_col_end) in blocks(col_start, col_end, NR) {
                    for (ub_row_start, ub_row_end) in blocks(row_start, row_end, MR) {
                        let out_offset = ub_row_start * out_row_stride + ub_col_start;
                        let out_tile = &mut out_data[out_offset..];

                        let a_panel = (ub_row_start - row_start) / MR;
                        let a_panel_size = MR * kc;
                        let a_panel_offset = a_panel * a_panel_size;

                        let b_panel = (ub_col_start - col_start) / NR;
                        let b_panel_size = kc * NR;
                        let b_panel_offset = b_panel * b_panel_size;

                        let a_block = &packed_a[a_panel_offset..a_panel_offset + a_panel_size];
                        let b_block = &packed_b[b_panel_offset..b_panel_offset + b_panel_size];

                        kernel::<MR, NR>(
                            out_tile,
                            out_row_stride,
                            ub_row_end - ub_row_start,
                            ub_col_end - ub_col_start,
                            a_block,
                            b_block,
                            block_depth,
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::gemm::gemm;
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
    fn test_gemm() -> Result<(), String> {
        // "Interesting" sizes for the row, column and depth dimensions of the
        // computation. These are chosen to cover cases that are less than,
        // equal to and above the tile/block sizes which the algorithm divides
        // the problem into along each dimension.
        let row_steps = [0, 2, 8, 10, 16, 64, 80];
        let col_steps = [0, 2, 4, 5, 8, 1024, 1025];
        let depth_steps = [0, 2, 20, 256, 300];

        let mut cases = Vec::new();

        // Simple cases where one dimension of the problem is varied to
        // different interesting values and other dimensions are kept small.
        for rs in row_steps {
            cases.push(([rs, 2], [2, 2]));
        }
        for cs in col_steps {
            cases.push(([2, 2], [2, cs]));
        }
        for ds in depth_steps {
            cases.push(([2, ds], [ds, 2]));
        }

        // Some simple square matrix tests of different sizes.
        for n in [1, 2, 4, 5, 8, 30, 64, 65] {
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
