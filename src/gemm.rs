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

/// Compute dot product of `depth` elements of `a` and `b`, stepping through
/// each array by `a_stride` and `b_stride` respectively. `N` specifies a
/// loop unrolling factor.
fn dot<const N: usize>(a: &[f32], b: &[f32], depth: usize) -> f32 {
    let n_blocks = depth / N;

    let mut result = 0.0;
    let mut accum = [0.0; N];

    for block in 0..n_blocks {
        let start_i = block * N;

        for i in 0..N {
            let k = start_i + i;
            unsafe {
                accum[i] = a.get_unchecked(k) * b.get_unchecked(k);
            }
        }
        result += accum.iter().fold(0.0, |sum, x| sum + x);
    }

    for k in (n_blocks * N)..depth {
        unsafe {
            result += a.get_unchecked(k) * b.get_unchecked(k);
        }
    }

    result
}

/// Pack a block of a row-major matrix into a smaller buffer in column-major order.
fn pack(
    dest: &mut [f32],
    dest_col_stride: usize,
    src: &[f32],
    src_row_stride: usize,
    start_col: usize,
    end_col: usize,
    start_row: usize,
    end_row: usize,
) {
    for c in start_col..end_col {
        let dest_col_offset = dest_col_stride * (c - start_col);
        let dest_col = &mut dest[dest_col_offset..dest_col_offset + end_row - start_row];

        for r in 0..dest_col.len() {
            dest_col[r] = src[(src_row_stride * (r + start_row)) + c];
        }
    }
}

/// Perform a general matrix multiplication ("GEMM") of `a` and `b` and store
/// the result in `output`.
pub fn gemm(output: &mut Tensor, a: &Tensor, b: &Tensor) {
    let [a_rows, a_cols] = a.dims();
    let [b_rows, b_cols] = b.dims();

    if a_cols != b_rows {
        panic!(
            "Columns of first input {} must match rows {} of second input",
            a_cols, b_rows
        );
    }

    let out_data = output.data_mut();
    let a_data = a.data();
    let b_data = b.data();

    let row_block_size = 16;
    let col_block_size = 64;

    let mut packed_b = Vec::with_capacity(col_block_size * b_rows);
    packed_b.resize(col_block_size * b_rows, 0.0);

    for (col_start, col_end) in blocks(0, b_cols, col_block_size) {
        pack(
            &mut packed_b,
            b_rows, /* dest col stride */
            b_data,
            b_cols, /* src row stride */
            col_start,
            col_end,
            0,      /* start row */
            b_rows, /* end row */
        );

        for (row_start, row_end) in blocks(0, a_rows, row_block_size) {
            for r in row_start..row_end {
                let a_row = &a_data[r * a_cols..];
                let out_row_offset = r * b_cols;
                let out_row = &mut out_data[out_row_offset + col_start..out_row_offset + col_end];

                for c in 0..out_row.len() {
                    let b_col = &packed_b[c * b_rows..];
                    out_row[c] = dot::<4>(a_row, b_col, a_cols);
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
        let mut rng = XorShiftRNG::new(1234);

        let a = random_tensor(&[30, 20], &mut rng);
        let b = random_tensor(&[20, 10], &mut rng);

        let mut result = zero_tensor::<f32>(&[30, 10]);

        gemm(&mut result, &a, &b);

        let expected = reference_gemm(&a, &b);
        expect_equal(&result, &expected)
    }
}
