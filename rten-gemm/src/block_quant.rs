//! Matrix multiplication with block-quantized inputs.

use rten_tensor::{Contiguous, Layout, NdTensorView};

use std::mem::MaybeUninit;
use std::ops::Range;

use rayon::prelude::*;
use rten_base::iter::range_chunks;
use rten_simd::ops::{Extend, IntOps, Interleave, NumOps, ToFloat};
use rten_simd::{Isa, Simd, SimdOp};
use rten_tensor::{AsView, AssumeInit};

use crate::GemmResult;
use crate::errors::{BlockQuantizedError, GemmError};

/// Performs matrix-multiplication between an un-quantized LHS / "A" matrix
/// and a block-quantized RHS / "B" matrix.
pub struct BlockQuantizedGemm {}

impl Default for BlockQuantizedGemm {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockQuantizedGemm {
    pub fn new() -> Self {
        BlockQuantizedGemm {}
    }

    /// Multiply `lhs` by the dequantized `rhs` matrix.
    pub fn batched_gemm_uninit<'a>(
        &self,
        out: &'a mut [MaybeUninit<f32>],
        lhs: NdTensorView<f32, 3>,
        rhs: BlockQuantizedMatrix<f32>,
    ) -> GemmResult<&'a mut [f32]> {
        let [batch, m, lhs_k] = lhs.shape();
        let n = rhs.cols();

        if out.len() != n * m * batch {
            return Err(GemmError::OutputSizeMismatch);
        }
        if lhs_k != rhs.rows() {
            return Err(GemmError::KSizeMismatch);
        }

        let lhs = lhs.to_contiguous();

        let col_block = 16;
        for (b, out_mat) in out.chunks_mut(n * m).enumerate() {
            // The handling of multiple rows here is inefficient. This is
            // because the initial focus is on efficient vector-matrix products.
            for (row, out_row) in out_mat.chunks_mut(n).enumerate() {
                let lhs = lhs.slice((b, row)).data().unwrap();
                range_chunks(0..n, col_block)
                    .into_par_iter()
                    .zip(out_row.par_chunks_mut(col_block))
                    .for_each(|(col_range, out_row_chunk)| {
                        let op = VecDotMatrix {
                            lhs,
                            rhs: rhs.slice(col_range),
                            out: out_row_chunk,
                        };
                        op.dispatch();
                    });
            }
        }

        Ok(unsafe { out.assume_init() })
    }
}

/// SIMD operation which computes the product between an f32 vector and a 4-bit
/// quantized matrix.
struct VecDotMatrix<'a> {
    lhs: &'a [f32],
    rhs: BlockQuantizedMatrix<'a, f32>,
    out: &'a mut [MaybeUninit<f32>],
}

impl<'a> SimdOp for VecDotMatrix<'a> {
    type Output = &'a mut [f32];

    #[inline(always)]
    fn eval<I: Isa>(self, isa: I) -> Self::Output {
        let VecDotMatrix { lhs, rhs, out } = self;

        let ops = isa.f32();
        let i16_ops = isa.i16();
        let i32_ops = isa.i32();
        let i8_ops = isa.i8();
        let u8_ops = isa.u8();

        // Columns are processed in "vblocks" whose size is the number of
        // 4-bit elements that can be loaded into a SIMD vector. This can be
        // larger or smaller than the block size of the RHS.
        let elements_per_vec = u8_ops.len() * 2;
        let vecs_per_block = rhs.elements_per_block() / elements_per_vec;

        // Max supported vector width is 512 bits. This is because the smallest
        // supported block size is 16 and we require f32 vector width >= block
        // size.
        assert!(ops.len() <= 16);

        // Number of scale values and zero points we will use for each vblock.
        let scales_per_vblock = (elements_per_vec / rhs.elements_per_block()).max(1);
        let n_tail_scales = rhs.blocks_per_column() % scales_per_vblock;
        assert!(matches!(scales_per_vblock, 1 | 2 | 4 | 8));

        let rhs_data = rhs.quant.data();
        let rhs_cols = rhs_data.chunks_exact(rhs.blocks_per_column() * rhs.bytes_per_block());
        let scale_blocks = rhs.scales.data().chunks_exact(rhs.blocks_per_column());

        for ((col, col_scales), out) in rhs_cols.zip(scale_blocks).zip(out.iter_mut()) {
            let mut acc = [ops.zero(); 4];

            let mut row_vblocks = lhs.chunks_exact(elements_per_vec);
            let mut col_vblocks = col.chunks_exact(u8_ops.len());

            let zero_point = i8_ops.splat(8);
            let lo_mask = u8_ops.splat(0x0F);

            // Vectorized loop over "vblock" elements
            for (vblock_idx, (row_vblock, col_vblock)) in
                row_vblocks.by_ref().zip(col_vblocks.by_ref()).enumerate()
            {
                // Load packed u4 values.
                let rhs_vblock = u8_ops.load(col_vblock);

                // Unpack to u8.
                let lo = u8_ops.and(rhs_vblock, lo_mask);
                let hi = u8_ops.shift_right::<4>(rhs_vblock);
                let (lo, hi) = (
                    u8_ops.interleave_low(lo, hi),
                    u8_ops.interleave_high(lo, hi),
                );

                // Re-interpret as i8
                let lo = i8_ops.from_bits(lo.to_bits());
                let hi = i8_ops.from_bits(hi.to_bits());

                // Subtract zero point
                let lo = i8_ops.sub(lo, zero_point);
                let hi = i8_ops.sub(hi, zero_point);

                // Widen to i32
                let (a_i16, b_i16) = i8_ops.extend(lo);
                let (c_i16, d_i16) = i8_ops.extend(hi);
                let (a_i32, b_i32) = i16_ops.extend(a_i16);
                let (c_i32, d_i32) = i16_ops.extend(b_i16);
                let (e_i32, f_i32) = i16_ops.extend(c_i16);
                let (g_i32, h_i32) = i16_ops.extend(d_i16);
                let rhs_i32 = [a_i32, b_i32, c_i32, d_i32, e_i32, f_i32, g_i32, h_i32];

                // Convert to f32, apply scale and multiply with LHS.
                let vlen = ops.len();
                match scales_per_vblock {
                    1 => {
                        let scale = ops.splat(col_scales[vblock_idx / vecs_per_block]);
                        for i in 0..8 {
                            let rhs_f32 = i32_ops.to_float(rhs_i32[i]);
                            let rhs_scaled = ops.mul(rhs_f32, scale);
                            let lhs = ops.load(&row_vblock[i * vlen..][..vlen]);
                            acc[i % acc.len()] = ops.mul_add(lhs, rhs_scaled, acc[i % acc.len()]);
                        }
                    }
                    2 => {
                        let block_idx = vblock_idx * 2;
                        let scales = [
                            ops.splat(col_scales[block_idx]),
                            ops.splat(col_scales[block_idx + 1]),
                        ];
                        for i in 0..8 {
                            let rhs_f32 = i32_ops.to_float(rhs_i32[i]);
                            let rhs_scaled = ops.mul(rhs_f32, scales[i / 4]);
                            let lhs = ops.load(&row_vblock[i * vlen..][..vlen]);
                            acc[i % acc.len()] = ops.mul_add(lhs, rhs_scaled, acc[i % acc.len()]);
                        }
                    }
                    4 => {
                        let block_idx = vblock_idx * 4;
                        let scales = [
                            ops.splat(col_scales[block_idx]),
                            ops.splat(col_scales[block_idx + 1]),
                            ops.splat(col_scales[block_idx + 2]),
                            ops.splat(col_scales[block_idx + 3]),
                        ];
                        for i in 0..8 {
                            let rhs_f32 = i32_ops.to_float(rhs_i32[i]);
                            let rhs_scaled = ops.mul(rhs_f32, scales[i / 2]);
                            let lhs = ops.load(&row_vblock[i * vlen..][..vlen]);
                            acc[i % acc.len()] = ops.mul_add(lhs, rhs_scaled, acc[i % acc.len()]);
                        }
                    }
                    8 => {
                        let block_idx = vblock_idx * 8;
                        for i in 0..8 {
                            let scale = ops.splat(col_scales[block_idx + i]);
                            let rhs_f32 = i32_ops.to_float(rhs_i32[i]);
                            let rhs_scaled = ops.mul(rhs_f32, scale);
                            let lhs = ops.load(&row_vblock[i * vlen..][..vlen]);
                            acc[i % acc.len()] = ops.mul_add(lhs, rhs_scaled, acc[i % acc.len()]);
                        }
                    }
                    _ => unreachable!(),
                }
            }

            // Sum accumulators
            let acc_01 = ops.add(acc[0], acc[1]);
            let acc_23 = ops.add(acc[2], acc[3]);
            let acc = ops.add(acc_01, acc_23);
            let mut acc = ops.sum(acc);

            // Scalar tail loop
            if !row_vblocks.remainder().is_empty() {
                let mut tail_acc = 0.;

                let lhs_tail_pairs = row_vblocks.remainder().as_chunks::<2>().0;
                let tail_scales = &col_scales[col_scales.len() - n_tail_scales..];
                let elements_per_scale = lhs_tail_pairs.len() / tail_scales.len();

                debug_assert_eq!(lhs_tail_pairs.len(), col_vblocks.remainder().len());
                for (i, (lhs, rhs)) in lhs_tail_pairs
                    .iter()
                    .zip(col_vblocks.remainder())
                    .enumerate()
                {
                    let zero_point = 8;
                    let rhs_lo = (rhs & 0x0F) as i32 - zero_point;
                    let rhs_hi = (rhs >> 4) as i32 - zero_point;

                    let scale = tail_scales[i / elements_per_scale];
                    let rhs_lo_scaled = (rhs_lo as f32) * scale;
                    let rhs_hi_scaled = (rhs_hi as f32) * scale;

                    tail_acc += lhs[0] * rhs_lo_scaled + lhs[1] * rhs_hi_scaled;
                }

                acc += tail_acc;
            }

            out.write(acc);
        }

        unsafe { out.assume_init() }
    }
}

/// Matrix which is quantized into blocks along the K dimension.
///
/// The data layout and supported bit/block sizes follow ONNX Runtime's MatMulNBits
/// operator. See <https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md>.
#[derive(Copy, Clone)]
pub struct BlockQuantizedMatrix<'a, T> {
    /// Quantized data of shape (N, k_blocks, block_size).
    quant: Contiguous<NdTensorView<'a, u8, 3>>,

    /// Scales of shape (N, k_blocks)
    scales: Contiguous<NdTensorView<'a, T, 2>>,

    /// Bits per quantized element.
    ///
    /// Must be a divisor of `block_size * 8`.
    bits: u8,
}

impl<'a, T: Copy> BlockQuantizedMatrix<'a, T> {
    /// Minimum supported number of elements per block.
    pub const MIN_BLOCK_SIZE: usize = 16;

    /// Create a block-quantized RHS matrix input.
    ///
    /// `quant` is the block-quantized input with shape (cols, k_blocks,
    /// block_size). `scales` are the per-block scales with shape (cols,
    /// k_blocks). `bits` specifies the number of bits per element in each
    /// block.
    ///
    /// The number of elements per quantized block must be a power of 2 of at
    /// least `MIN_BLOCK_SIZE`.
    pub fn new(
        quant: Contiguous<NdTensorView<'a, u8, 3>>,
        scales: Contiguous<NdTensorView<'a, T, 2>>,
        bits: u8,
    ) -> Result<Self, BlockQuantizedError> {
        // ONNX Runtime currently supports 2, 4 or 8 bits per element. These
        // values have the convenient property that a byte is a whole number
        // of elements. We only support 4 bits for the moment.
        if bits != 4 {
            return Err(BlockQuantizedError::UnsupportedElementSize);
        }
        let n_elem = 8 / bits;

        let [_batch, _k_blocks, block_bytes] = quant.shape();

        let block_size = block_bytes * n_elem as usize;
        if !block_size.is_power_of_two() || block_size < Self::MIN_BLOCK_SIZE {
            return Err(BlockQuantizedError::UnsupportedBlockSize);
        }

        Ok(Self {
            quant,
            scales,
            bits,
        })
    }

    /// Return the number of rows in the dequantized matrix.
    pub fn rows(&self) -> usize {
        self.blocks_per_column() * self.elements_per_block()
    }

    /// Return the number of columns in the dequantized matrix.
    pub fn cols(&self) -> usize {
        self.quant.size(0)
    }

    /// Extract a slice of columns from the matrix.
    pub(crate) fn slice(&self, col_range: Range<usize>) -> BlockQuantizedMatrix<'a, T> {
        BlockQuantizedMatrix {
            quant: Contiguous::new(self.quant.slice(col_range.clone())).unwrap(),
            scales: Contiguous::new(self.scales.slice(col_range)).unwrap(),
            bits: self.bits,
        }
    }

    /// Return the number of bits per element
    pub fn n_bits(&self) -> u8 {
        self.bits
    }

    /// Return the number of blocks in each column.
    pub(crate) fn blocks_per_column(&self) -> usize {
        self.quant.size(1)
    }

    pub(crate) fn elements_per_block(&self) -> usize {
        (self.bytes_per_block() * 8) / self.bits as usize
    }

    pub(crate) fn bytes_per_block(&self) -> usize {
        self.quant.size(2)
    }

    /// Return the packed data for a range of blocks in a column.
    pub(crate) fn column_data(
        &self,
        col: usize,
        start_block: usize,
        n_blocks: usize,
    ) -> Option<&[u8]> {
        if self.quant.size(1) < start_block + n_blocks {
            return None;
        }
        let offset = self.quant.offset([col, start_block, 0])?;
        let len = self.quant.size(2);

        Some(unsafe {
            std::slice::from_raw_parts(self.quant.data_ptr().add(offset), len * n_blocks)
        })
    }

    /// Return the scale factors for a range of blocks in a column.
    pub(crate) fn column_scales(
        &self,
        col: usize,
        start_block: usize,
        n_blocks: usize,
    ) -> Option<&[T]> {
        if self.scales.size(1) < start_block + n_blocks {
            return None;
        }
        let offset = self.scales.offset([col, start_block])?;

        Some(unsafe { std::slice::from_raw_parts(self.scales.data_ptr().add(offset), n_blocks) })
    }
}

/// Return the default zero point for n-bit quantization.
///
/// This is an i16 because the maximum value is 128 (when n_bits=8) and the
/// value is used in signed subtractions.
///
/// See docs for `zero_points` input to MatMulNBits in
/// <https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md>.
pub const fn nbit_zero_point(n_bits: u8) -> i16 {
    assert!(n_bits >= 2 && n_bits <= 8);
    1 << (n_bits - 1)
}

#[cfg(test)]
pub fn pack_4bit_elements(vals: &[i8], zero_point: i8) -> Vec<u8> {
    let (chunks, tail) = vals.as_chunks::<2>();
    assert!(tail.is_empty());
    chunks
        .iter()
        .copied()
        .map(|[even, odd]| {
            let lo = (even + zero_point) as u8;
            let hi = (odd + zero_point) as u8;
            (lo & 0x0F) | (hi << 4)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::test_util::expect_equal;
    use rten_tensor::{AsView, Contiguous, Layout, NdTensor, NdTensorView};
    use rten_testing::TestCases;

    use super::{BlockQuantizedGemm, BlockQuantizedMatrix, nbit_zero_point, pack_4bit_elements};

    fn reference_gemm_f32_with_block_quantized_rhs(
        lhs: NdTensorView<f32, 2>,
        rhs: NdTensorView<u8, 3>,
        rhs_scales: NdTensorView<f32, 2>,
    ) -> NdTensor<f32, 2> {
        let [m, k] = lhs.shape();
        let [n, _k_blocks, block_size] = rhs.shape();
        let elems_per_block = block_size * 2;
        let zero_point = 8;

        let mut out = NdTensor::zeros([m, n]);

        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.;
                for ki in 0..k {
                    let k_block = ki / elems_per_block;
                    let block_idx = ki % elems_per_block;
                    let scale = rhs_scales[[col, k_block]];

                    let byte = rhs[[col, k_block, block_idx / 2]];
                    let elem = if ki % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                    let dequantized_elem = (elem as i32 - zero_point) as f32 * scale;

                    acc += lhs[[row, ki]] * dequantized_elem;
                }
                out[[row, col]] = acc;
            }
        }

        out
    }

    #[test]
    fn test_block_quantized_matrix() {
        let zero_point = nbit_zero_point(4) as i8;
        let elems: Vec<i8> = (-8..8).cycle().take(256).collect();
        let packed = pack_4bit_elements(&elems, zero_point);

        let block_bytes = 16;
        let cols = 4;
        let k_blocks = 2;
        let n_bits = 4;

        let quants = NdTensor::from_data([cols, k_blocks, block_bytes], packed);
        let scales = NdTensorView::from_data([cols, k_blocks], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let mat = BlockQuantizedMatrix::new(
            Contiguous::new(quants.view()).unwrap(),
            Contiguous::new(scales.view()).unwrap(),
            n_bits,
        )
        .unwrap();

        assert_eq!(mat.rows(), 64);
        assert_eq!(mat.cols(), cols);
        assert_eq!(mat.elements_per_block(), 32);
        assert_eq!(mat.blocks_per_column(), k_blocks);

        assert_eq!(
            mat.column_data(0, 0, 1).unwrap(),
            pack_4bit_elements(&elems[..32], zero_point)
        );
        assert_eq!(mat.column_scales(0, 0, 2), Some([1.0, 2.0].as_slice()));
        assert_eq!(
            mat.column_data(3, 1, 1).unwrap(),
            pack_4bit_elements(&elems[256 - 32..], zero_point)
        );
        assert_eq!(mat.column_scales(3, 1, 1), Some([8.0].as_slice()));

        // Out of bounds column.
        assert_eq!(mat.column_data(4, 1, 1), None);
        // Out of bounds K block.
        assert_eq!(mat.column_data(3, 2, 1), None);
    }

    #[test]
    fn test_block_quantized_gemm() {
        #[derive(Clone, Debug)]
        struct Case {
            block_size: usize,
            n_cols: usize,
            n_blocks: usize,
        }

        // Max u4 elements in a SIMD vector.
        let max_vblock_size = 128;

        let mut cases = Vec::new();

        // The ONNX Runtime definition of MatMulNBits specifies that the block
        // size must be a power of 2 and >= 16. The ORT implementation supports
        // block sizes from 16 to 256. The implementation in this crate is more
        // general and supports larger block sizes. 256 is large enough to test
        // all code paths on all architectures.
        for block_size in [16, 32, 64, 128, 256] {
            cases.push(Case {
                n_cols: 3,
                n_blocks: (max_vblock_size / block_size).max(1),
                block_size,
            });
        }

        // Add a case that will exercise both the main and tail loops.
        cases.push(Case {
            n_cols: 1,
            // 16 x u4 = 64 bits, smaller than vector length.
            block_size: 16,
            // (max_vblock_size / 16) to use main loop once, plus one for a tail.
            n_blocks: (max_vblock_size / 16) + 1,
        });

        cases.test_each_clone(|case| {
            let Case {
                n_cols,
                n_blocks,
                block_size,
            } = case;

            let mut rng = XorShiftRng::new(1234);

            let gemm = BlockQuantizedGemm::new();
            let lhs = NdTensor::<f32, 2>::rand([1, n_blocks * block_size], &mut rng);
            let rhs_data = NdTensor::<u8, 3>::rand([n_cols, n_blocks, block_size / 2], &mut rng);
            let rhs_scales = NdTensor::<f32, 2>::rand([n_cols, n_blocks], &mut rng);
            let bqm = BlockQuantizedMatrix::new(
                Contiguous::new(rhs_data.view()).unwrap(),
                Contiguous::new(rhs_scales.view()).unwrap(),
                4,
            )
            .unwrap();

            let expected = reference_gemm_f32_with_block_quantized_rhs(
                lhs.view(),
                rhs_data.view(),
                rhs_scales.view(),
            );

            let mut out = Vec::with_capacity(n_cols);
            let result = gemm
                .batched_gemm_uninit(
                    out.spare_capacity_mut(),
                    lhs.reshaped([1, lhs.size(0), lhs.size(1)]).view(),
                    bqm,
                )
                .unwrap();
            let result_matrix = NdTensorView::from_data([1, result.len()], result.as_ref());
            expect_equal(&result_matrix, &expected.view()).unwrap();
        });
    }
}
