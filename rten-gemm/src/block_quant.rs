//! Matrix multiplication with block-quantized inputs.

use rten_tensor::{Contiguous, Layout, NdTensor, NdTensorView};

use std::mem::MaybeUninit;
use std::ops::Range;

use rayon::prelude::*;
use rten_base::iter::range_chunks;
use rten_simd::ops::{Extend, IntOps, Interleave, NumOps, ToFloat};
use rten_simd::{Isa, Simd, SimdOp};
use rten_tensor::{AsView, AssumeInit};

use crate::GemmResult;
use crate::errors::{BlockQuantizedError, GemmError};
use crate::i8dot::{Int8DotIsa, SimdInt8DotOp};

/// Specifies whether to quantize the LHS / "A" input to block-quantized matrix
/// multiplication.
///
/// Quantizing the LHS input can significantly improve performance but may
/// impact accuracy.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ComputeMode {
    /// Quantize LHS / "A" input to 8-bits.
    Int8,
    /// Do not quantize the LHS.
    Float,
}

/// Performs matrix-multiplication between an un-quantized LHS / "A" matrix
/// and a block-quantized RHS / "B" matrix.
pub struct BlockQuantizedGemm {
    mode: ComputeMode,
}

impl BlockQuantizedGemm {
    pub fn new() -> Self {
        BlockQuantizedGemm {
            mode: ComputeMode::Float,
        }
    }

    /// Set the compute mode controls the accuracy/performance balance.
    pub fn with_compute(mut self, mode: ComputeMode) -> Self {
        self.mode = mode;
        self
    }

    /// Return true if an optimized implementation for the given compute mode
    /// if available on the current platform.
    pub fn is_compute_optimized(mode: ComputeMode) -> bool {
        match mode {
            ComputeMode::Float => true,
            ComputeMode::Int8 => is_int8_compute_optimized(),
        }
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

        if rhs.bits != 4 {
            return Err(GemmError::QuantBitsNotSupported);
        }

        enum LhsRow<'a> {
            Float(&'a [f32]),
            Quant {
                data: Contiguous<NdTensorView<'a, i8, 2>>,
                scales: &'a [f32],
            },
        }

        let lhs = lhs.to_contiguous();

        let lhs_quant: Option<(NdTensor<i8, 4>, NdTensor<f32, 3>)> =
            if matches!(self.mode, ComputeMode::Int8) && m == 1 {
                Some(quantize(lhs.view(), rhs.elements_per_block()))
            } else {
                None
            };

        let col_block = 16;
        for (b, out_mat) in out.chunks_mut(n * m).enumerate() {
            // The handling of multiple rows here is inefficient. This is
            // because the initial focus is on efficient vector-matrix products.
            for (row, out_row) in out_mat.chunks_mut(n).enumerate() {
                let lhs_row = if let Some((lhs_data, lhs_scales)) = &lhs_quant {
                    LhsRow::Quant {
                        data: Contiguous::new(lhs_data.slice((b, row))).unwrap(),
                        scales: lhs_scales.slice((b, row)).data().unwrap(),
                    }
                } else {
                    LhsRow::Float(lhs.slice((b, row)).data().unwrap())
                };

                range_chunks(0..n, col_block)
                    .into_par_iter()
                    .zip(out_row.par_chunks_mut(col_block))
                    .for_each(|(col_range, out_row_chunk)| match lhs_row {
                        LhsRow::Quant { data, scales } => {
                            let op = VecDotMatrixQuant {
                                lhs_data: data,
                                lhs_scales: scales,
                                rhs: rhs.slice(col_range),
                                out: out_row_chunk,
                            };
                            op.dispatch();
                        }
                        LhsRow::Float(lhs) => {
                            let op = VecDotMatrix {
                                lhs,
                                rhs: rhs.slice(col_range),
                                out: out_row_chunk,
                            };
                            op.dispatch();
                        }
                    });
            }
        }

        Ok(unsafe { out.assume_init() })
    }
}

impl Default for BlockQuantizedGemm {
    fn default() -> Self {
        Self::new()
    }
}

/// Return true if a SIMD-optimized int8 dot product implementation is
/// available.
fn is_int8_compute_optimized() -> bool {
    struct HasInt8Simd;
    impl SimdInt8DotOp for HasInt8Simd {
        type Output = bool;

        fn eval<I: Int8DotIsa>(self, _isa: I) -> Self::Output {
            I::SIMD
        }
    }
    HasInt8Simd.dispatch()
}

/// SIMD operation which computes the product between an f32 vector and a 4-bit
/// quantized matrix.
struct VecDotMatrix<'a> {
    lhs: &'a [f32],
    rhs: BlockQuantizedMatrix<'a, f32>,
    out: &'a mut [MaybeUninit<f32>],
}

impl<'a> VecDotMatrix<'a> {
    #[inline(always)]
    fn eval_impl<I: Isa, const SCALES_PER_VBLOCK: usize>(self, isa: I) -> &'a mut [f32] {
        let ops = isa.f32();
        let i16_ops = isa.i16();
        let i32_ops = isa.i32();
        let i8_ops = isa.i8();
        let u8_ops = isa.u8();

        // Columns are processed in "vblocks" whose size is the number of
        // 4-bit elements that can be loaded into a SIMD vector. This can be
        // larger or smaller than the block size of the RHS.
        let elements_per_vec = u8_ops.len() * 2;

        let VecDotMatrix { lhs, rhs, out } = self;
        let vecs_per_block = rhs.elements_per_block() / elements_per_vec;

        // Convert division into shift.
        let vecs_per_block_log2 = if vecs_per_block != 0 {
            debug_assert!(vecs_per_block.is_power_of_two());
            vecs_per_block.ilog2()
        } else {
            0
        };

        // Max supported vector width is 512 bits. This is because the smallest
        // supported block size is 16 and we require f32 vector width >= block
        // size.
        assert!(ops.len() <= 16);

        let n_tail_scales = rhs.blocks_per_column() % SCALES_PER_VBLOCK;

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
                match SCALES_PER_VBLOCK {
                    1 => {
                        let scale = ops.splat(col_scales[vblock_idx >> vecs_per_block_log2]);
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

impl<'a> SimdOp for VecDotMatrix<'a> {
    type Output = &'a mut [f32];

    #[inline(always)]
    fn eval<I: Isa>(self, isa: I) -> Self::Output {
        let u8_ops = isa.u8();

        // Columns are processed in "vblocks" whose size is the number of
        // 4-bit elements that can be loaded into a SIMD vector. This can be
        // larger or smaller than the block size of the RHS.
        let elements_per_vec = u8_ops.len() * 2;

        // Number of scale values and zero points we will use for each vblock.
        // The maximum supported SIMD width is 512 bits (128 x u4) and the
        // minimum block size is 16, so the maximum value is 128/16 = 8.
        let scales_per_vblock = (elements_per_vec / self.rhs.elements_per_block()).max(1);

        match scales_per_vblock {
            1 => self.eval_impl::<I, 1>(isa),
            2 => self.eval_impl::<I, 2>(isa),
            4 => self.eval_impl::<I, 4>(isa),
            8 => self.eval_impl::<I, 8>(isa),
            _ => unreachable!("unsupported scales_per_vblock"),
        }
    }
}

/// Quantize blocks of `data` to i8 values.
///
/// `data` has shape (batch, row, col) and `block_size` is a power of 2 >= 16.
///
/// Returns a tuple of (quantized_data, scales) where `quantized_data` has shape
/// (batch, row, block, element) and `scales` has shape (batch, row, block).
/// Elements can be dequantized via `x as f32 * block_scale`.
fn quantize(
    data: Contiguous<NdTensorView<f32, 3>>,
    block_size: usize,
) -> (NdTensor<i8, 4>, NdTensor<f32, 3>) {
    let [batch, rows, k] = data.shape();

    assert!(block_size >= 16 && block_size.is_power_of_two());
    assert!(k.is_multiple_of(block_size));

    let n_blocks = k / block_size;
    let mut output = Vec::with_capacity(n_blocks * block_size);
    let mut scales = Vec::with_capacity(n_blocks);

    for block in data.data().chunks_exact(block_size) {
        let abs_max = block.iter().fold(0., |max, x| x.abs().max(max));
        let inv_scale = i8::MAX as f32 / abs_max;

        for &x in block {
            let qx = (x * inv_scale).round() as i8;
            output.push(qx);
        }

        scales.push(1. / inv_scale);
    }

    let quant_data = NdTensor::from_data([batch, rows, n_blocks, block_size], output);
    let scales = NdTensor::from_data([batch, rows, n_blocks], scales);
    (quant_data, scales)
}

/// Multiply an int8-quantized LHS by an int4-quantized RHS.
struct VecDotMatrixQuant<'a> {
    lhs_data: Contiguous<NdTensorView<'a, i8, 2>>,
    lhs_scales: &'a [f32],
    rhs: BlockQuantizedMatrix<'a, f32>,
    out: &'a mut [MaybeUninit<f32>],
}

impl<'a> VecDotMatrixQuant<'a> {
    #[inline(always)]
    fn eval_impl<I: Int8DotIsa, const SCALES_PER_VBLOCK: usize>(self, isa: I) -> &'a mut [f32] {
        let ops = isa.isa().f32();
        let i8_ops = isa.isa().i8();
        let u8_ops = isa.isa().u8();
        let i32_ops = isa.isa().i32();

        let VecDotMatrixQuant {
            lhs_data,
            lhs_scales,
            rhs,
            out,
        } = self;

        let rhs_data = rhs.quant.data();
        let rhs_cols = rhs_data.chunks_exact(rhs.blocks_per_column() * rhs.bytes_per_block());
        let scale_blocks = rhs.scales.data().chunks_exact(rhs.blocks_per_column());

        // Columns are processed in "vblocks" whose size is the number of
        // 4-bit elements that can be loaded into a SIMD vector. This can be
        // larger or smaller than the block size of the RHS.
        let elements_per_vec = u8_ops.len() * 2;
        let elements_per_block = rhs.elements_per_block();
        let vecs_per_block = elements_per_block / elements_per_vec;
        let blocks_per_vec = elements_per_vec.div_ceil(elements_per_block);
        let n_tail_blocks = rhs.blocks_per_column() % blocks_per_vec;

        // Convert division into shift.
        let vecs_per_block_log2 = if vecs_per_block != 0 {
            debug_assert!(vecs_per_block.is_power_of_two());
            vecs_per_block.ilog2()
        } else {
            0
        };

        // TESTING
        assert_eq!(rhs.bytes_per_block(), u8_ops.len());

        let col_bytes = rhs.blocks_per_column() * rhs.bytes_per_block();
        let n_vblocks = col_bytes / u8_ops.len();
        const K_TILE: usize = 4;
        let k_tiles = n_vblocks / K_TILE;

        // Masks used to select scales if we're using 4 or 8 scales per vblock.
        // let lo_half_mask = ops.first_n_mask(ops.len() / 2);
        // let lo_quad_mask = ops.first_n_mask(ops.len() / 4);
        // let lo_three_quads_mask = ops.first_n_mask(3 * ops.len() / 4);

        for ((col, col_scales), out) in rhs_cols.zip(scale_blocks).zip(out.iter_mut()) {
            let mut acc = [[ops.zero(); 2]; K_TILE];

            let zero_point = i8_ops.splat(8);
            let lo_mask = u8_ops.splat(0x0F);
            let vlen = u8_ops.len();
            let zero_i32 = i32_ops.zero();

            let row = lhs_data.data();

            for kt in 0..k_tiles {
                let col_scale = unsafe { ops.load_ptr(col_scales.as_ptr().add(kt * K_TILE)) };
                let row_scale = unsafe { ops.load_ptr(lhs_scales.as_ptr().add(kt * K_TILE)) };
                let scale = ops.mul(col_scale, row_scale);

                macro_rules! k_step {
                    ($k:literal) => {
                        let vblock_idx = kt * K_TILE + $k;

                        let col_vblock = unsafe { col.get_unchecked(vblock_idx * vlen..) };
                        let row_vblock = unsafe { row.get_unchecked(vblock_idx * vlen * 2..) };

                        // Load packed u4 values.
                        let rhs_vblock = unsafe { u8_ops.load_ptr(col_vblock.as_ptr()) };

                        // Unpack to u8.
                        let lo = u8_ops.and(rhs_vblock, lo_mask);
                        let hi = u8_ops.shift_right::<4>(rhs_vblock);
                        let (lo, hi) = (
                            u8_ops.interleave_low(lo, hi),
                            u8_ops.interleave_high(lo, hi),
                        );

                        // Re-interpret as i8
                        let mut lo = i8_ops.from_bits(lo.to_bits());
                        let mut hi = i8_ops.from_bits(hi.to_bits());

                        // Subtract zero point if using i8 x i8 dot product.
                        if !I::LHS_UNSIGNED {
                            lo = i8_ops.sub(lo, zero_point);
                            hi = i8_ops.sub(hi, zero_point);
                        }

                        // Load vblock elements from LHS
                        let lhs_lo = unsafe { i8_ops.load_ptr(row_vblock.as_ptr()) };
                        let lhs_hi = unsafe { i8_ops.load_ptr(row_vblock.as_ptr().add(vlen)) };

                        // Compute i8 x i8 -> i32 or u8 x i8 -> i32 dot product.
                        let mut dot_lo = isa.dot(lo, lhs_lo, zero_i32);
                        let mut dot_hi = isa.dot(hi, lhs_hi, zero_i32);

                        // If using u8 x i8 dot product, compensate for not subtracting
                        // the zero point before the dot product.
                        if I::LHS_UNSIGNED {
                            let zero_point_i32 = i32_ops.splat(8);
                            let lhs_lo_sum = isa.dot(i8_ops.splat(1), lhs_lo, zero_i32);
                            let lhs_hi_sum = isa.dot(i8_ops.splat(1), lhs_hi, zero_i32);
                            dot_lo = i32_ops.sub(dot_lo, i32_ops.mul(zero_point_i32, lhs_lo_sum));
                            dot_hi = i32_ops.sub(dot_hi, i32_ops.mul(zero_point_i32, lhs_hi_sum));
                        }

                        let float_lo = i32_ops.to_float(dot_lo);
                        let float_hi = i32_ops.to_float(dot_hi);

                        let scale = ops.broadcast_lane::<$k>(scale);
                        acc[$k][0] = ops.mul_add(float_lo, scale, acc[$k][0]);
                        acc[$k][1] = ops.mul_add(float_hi, scale, acc[$k][1]);
                    };
                }
                k_step!(0);
                k_step!(1);
                k_step!(2);
                k_step!(3);
            }

            let mut acc_sum = ops.zero();
            for k in 0..K_TILE {
                let k_sum = ops.add(acc[k][0], acc[k][1]);
                acc_sum = ops.add(acc_sum, k_sum);
            }

            let acc = ops.sum(acc_sum);

            // Handle tail blocks in column. This contains < 128 elements
            // (512 / 4).
            if n_tail_blocks > 0 {
                // let row_tail_blocks = row_vblocks.remainder().chunks_exact(elements_per_block);
                // let col_tail_blocks = col_vblocks.remainder().chunks_exact(elements_per_block / 2);
                // debug_assert_eq!(row_tail_blocks.len(), n_tail_blocks);

                // let mut tail_acc = 0.;
                // for (i, (lhs_block, rhs_block)) in row_tail_blocks.zip(col_tail_blocks).enumerate()
                // {
                //     let col_scale = col_scales[col_scales.len() - n_tail_blocks + i];
                //     let row_scale = lhs_scales[lhs_scales.len() - n_tail_blocks + i];
                //     let scale = col_scale * row_scale;

                //     let mut acc = 0.;
                //     for ([x_lo, x_hi], y) in lhs_block.as_chunks::<2>().0.iter().zip(rhs_block) {
                //         let y_lo = (y & 0x0F) as i32 - 8;
                //         let y_hi = (y >> 4) as i32 - 8;
                //         acc += (*x_lo as i32 * y_lo + *x_hi as i32 * y_hi) as f32 * scale;
                //     }
                //     tail_acc += acc;
                // }
                // acc += tail_acc;
            }

            out.write(acc);
        }

        unsafe { out.assume_init() }
    }
}

impl<'a> SimdInt8DotOp for VecDotMatrixQuant<'a> {
    type Output = &'a mut [f32];

    #[inline(always)]
    fn eval<I: Int8DotIsa>(self, isa: I) -> Self::Output {
        let u8_ops = isa.isa().u8();

        // Columns are processed in "vblocks" whose size is the number of
        // 4-bit elements that can be loaded into a SIMD vector. This can be
        // larger or smaller than the block size of the RHS.
        let elements_per_vec = u8_ops.len() * 2;

        // Number of scale values and zero points we will use for each vblock.
        // The maximum supported SIMD width is 512 bits (128 x u4) and the
        // minimum block size is 16, so the maximum value is 128/16 = 8.
        let scales_per_vblock = (elements_per_vec / self.rhs.elements_per_block()).max(1);

        match scales_per_vblock {
            1 => self.eval_impl::<I, 1>(isa),
            2 => self.eval_impl::<I, 2>(isa),
            4 => self.eval_impl::<I, 4>(isa),
            8 => self.eval_impl::<I, 8>(isa),
            _ => unreachable!("unsupported scales_per_vblock"),
        }
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
        if !matches!(bits, 4 | 8) {
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
        Some(&self.quant.data()[offset..offset + len * n_blocks])
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
        Some(&self.scales.data()[offset..offset + n_blocks])
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
    use rten_tensor::test_util::{expect_equal, expect_equal_with_tolerance};
    use rten_tensor::{AsView, Contiguous, Layout, NdTensor, NdTensorView};
    use rten_testing::TestCases;

    use super::{
        BlockQuantizedGemm, BlockQuantizedMatrix, ComputeMode, nbit_zero_point, pack_4bit_elements,
        quantize,
    };

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

    // The ONNX Runtime definition of MatMulNBits specifies that the block
    // size must be a power of 2 and >= 16. The ORT implementation supports
    // block sizes from 16 to 256. The implementation in this crate is more
    // general and supports larger block sizes. 256 is large enough to test
    // all code paths on all architectures.
    const BLOCK_SIZES: [usize; 5] = [16, 32, 64, 128, 256];

    #[test]
    fn test_quantize() {
        let mut rng = XorShiftRng::new(1234);
        for block_size in BLOCK_SIZES {
            let batch = 2;
            let rows = 3;
            let n_blocks = 2;
            let k = block_size * n_blocks;

            let mut data = NdTensor::rand([batch, rows, k], &mut rng);
            // Shift range from [0, 1] to [-1, -1]
            data.apply(|x| (x - 0.5) * 2.);

            let (quantized, scales) = quantize(Contiguous::new(data.view()).unwrap(), block_size);

            let dequantized: Vec<f32> = quantized
                .inner_iter::<1>()
                .zip(scales.iter())
                .flat_map(|(block, scale)| block.iter().map(move |x| *x as f32 * scale))
                .collect();

            let max_err = data
                .iter()
                .zip(dequantized)
                .map(|(x, y)| (x - y).abs())
                .fold(f32::MIN, |max, x| x.max(max));
            let threshold = 0.004;

            assert!(
                max_err <= threshold,
                "max_err {max_err} exceeds {threshold}"
            );
        }
    }

    #[test]
    fn test_block_quantized_gemm() {
        #[derive(Clone, Debug)]
        struct Case {
            block_size: usize,
            n_cols: usize,
            n_blocks: usize,
            compute: ComputeMode,
            tolerance: Option<f32>,
        }

        // Max u4 elements in a SIMD vector.
        let max_vblock_size = 128;

        let mut cases = Vec::new();

        for block_size in BLOCK_SIZES {
            cases.push(Case {
                n_cols: 3,
                n_blocks: (max_vblock_size / block_size).max(1),
                block_size,
                compute: ComputeMode::Float,
                tolerance: None,
            });
        }

        // Add a case that will exercise both the main and tail loops.
        cases.push(Case {
            n_cols: 1,
            // 16 x u4 = 64 bits, smaller than vector length.
            block_size: 16,
            // (max_vblock_size / 16) to use main loop once, plus one for a tail.
            n_blocks: (max_vblock_size / 16) + 1,
            compute: ComputeMode::Float,
            tolerance: None,
        });

        // Add cases that use int8 quantization of the LHS.
        for (block_size, atol) in [(16, 0.1), (32, 0.1), (64, 0.2), (128, 0.2), (256, 0.3)] {
            cases.push(Case {
                n_cols: 3,
                block_size,
                n_blocks: (max_vblock_size / block_size).max(1),
                compute: ComputeMode::Int8,
                tolerance: Some(atol),
            });
        }

        // Add a case that will exercise both the main and tail loops using int8.
        cases.push(Case {
            n_cols: 1,
            // 16 x u4 = 64 bits, smaller than vector length.
            block_size: 16,
            // (max_vblock_size / 16) to use main loop once, plus one for a tail.
            n_blocks: (max_vblock_size / 16) + 1,
            compute: ComputeMode::Int8,
            tolerance: Some(0.1),
        });

        cases.test_each_clone(|case| {
            let Case {
                n_cols,
                n_blocks,
                block_size,
                compute,
                tolerance,
            } = case;

            let mut rng = XorShiftRng::new(1234);

            let gemm = BlockQuantizedGemm::new().with_compute(compute);
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

            if let Some(atol) = tolerance {
                let rtol = 0.;
                expect_equal_with_tolerance(&result_matrix, &expected.view(), atol, rtol).unwrap();
            } else {
                expect_equal(&result_matrix, &expected.view()).unwrap();
            }
        });
    }
}
