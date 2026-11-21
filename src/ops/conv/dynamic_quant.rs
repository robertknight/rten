//! Convolution with int8 weights and dynamically quantized activations.

use std::borrow::Cow;
use std::mem::MaybeUninit;
use std::ops::Range;

use rayon::prelude::*;
use rten_gemm::{
    BMatrixSource, GemmExecutor, GemmInputA, GemmInputB, GemmUninitOptions, PackedAMatrix,
    QuantParams,
};
use rten_shape_inference::ops as shape_ops;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, Tensor, TensorView};
use smallvec::SmallVec;

use crate::buffer_pool::{AutoReturn, BufferPool};
use crate::infer_shapes::InferShapes;
use crate::operator::{
    IntoOpResult, OpError, OpRunContext, Operator, OutputList, OutputType, OutputTypeList,
    OutputTypesContext, PrepackedInput, check_eq, static_dims,
};
use crate::ops::Padding;
use crate::ops::binary_elementwise::add;
use crate::ops::matmul::shift_cast_gemm_lhs_to_u8;
use crate::ops::pooling::{RoundMode, calc_output_size_and_padding};
use crate::value::ValueView;

/// Geometry of a 2D convolution over one image.
#[derive(Clone, Copy)]
struct ConvGeometry {
    in_size: [usize; 2],
    kernel_size: [usize; 2],
    out_size: [usize; 2],
    /// Padding as `[top, left, bottom, right]`.
    padding: [usize; 4],
    strides: [usize; 2],
    dilations: [usize; 2],
}

impl ConvGeometry {
    /// Return the input coordinate along `axis` read by output position `out`
    /// and kernel position `k`, or `None` if it is in the padding region.
    #[inline]
    fn input_pos(&self, axis: usize, out: usize, k: usize) -> Option<usize> {
        let pos = (out * self.strides[axis] + k * self.dilations[axis]) as isize
            - self.padding[axis] as isize;
        (pos >= 0 && (pos as usize) < self.in_size[axis]).then_some(pos as usize)
    }

    /// Return the range of output X positions for which kernel X position
    /// `kx` reads from inside the image.
    fn valid_out_x(&self, kx: usize) -> Range<usize> {
        let [_, in_w] = self.in_size;
        let [_, out_w] = self.out_size;
        let [_, stride] = self.strides;
        let offset = (kx * self.dilations[1]) as isize - self.padding[1] as isize;

        // First output position with `out * stride + offset >= 0`.
        let start = if offset >= 0 {
            0
        } else {
            ((-offset) as usize).div_ceil(stride)
        };
        // First output position with `out * stride + offset >= in_w`.
        let end = if (in_w as isize) <= offset {
            0
        } else {
            (in_w as isize - offset) as usize
        }
        .div_ceil(stride);
        start.min(out_w)..end.min(out_w).max(start.min(out_w))
    }
}

/// Per-column activation quantization parameters for an im2col matrix.
struct ColumnQuant {
    /// Scale for each column.
    scale: Vec<f32>,
    /// Reciprocal of scale for each column.
    inv_scale: Vec<f32>,
    /// Zero point for each column.
    zero_point: Vec<i8>,
    /// Zero point for each column, as a float.
    zero_point_f32: Vec<f32>,
}

/// Compute per-column quantization parameters for the im2col matrix of
/// `image`, which is the contiguous data of a `[C, H, W]` tensor.
///
/// The range of each column covers all of the input values that the column
/// contains, plus zero (for padding and so that zero is exactly
/// representable).
fn column_quant_params(image: &[f32], geom: &ConvGeometry, levels: f32) -> ColumnQuant {
    let [in_h, in_w] = geom.in_size;
    let [out_h, out_w] = geom.out_size;

    // Min and max over channels at each input position. These include zero,
    // which also accounts for padding.
    let n_pos = in_h * in_w;
    let mut pos_min = vec![0f32; n_pos];
    let mut pos_max = vec![0f32; n_pos];
    const POS_CHUNK: usize = 1024;
    pos_min
        .par_chunks_mut(POS_CHUNK)
        .zip(pos_max.par_chunks_mut(POS_CHUNK))
        .enumerate()
        .for_each(|(chunk_idx, (pos_min, pos_max))| {
            let start = chunk_idx * POS_CHUNK;
            for chan in image.chunks(n_pos) {
                let chan = &chan[start..start + pos_min.len()];
                for ((lo, hi), &x) in pos_min.iter_mut().zip(pos_max.iter_mut()).zip(chan) {
                    *lo = lo.min(x);
                    *hi = hi.max(x);
                }
            }
        });

    let n_cols = out_h * out_w;
    let mut scale = vec![0f32; n_cols];
    let mut inv_scale = vec![0f32; n_cols];
    let mut zero_point = vec![0i8; n_cols];
    let mut zero_point_f32 = vec![0f32; n_cols];

    // Compute parameters for chunks of columns in parallel.
    const COL_CHUNK: usize = 1024;
    scale
        .par_chunks_mut(COL_CHUNK)
        .zip(inv_scale.par_chunks_mut(COL_CHUNK))
        .zip(zero_point.par_chunks_mut(COL_CHUNK))
        .zip(zero_point_f32.par_chunks_mut(COL_CHUNK))
        .enumerate()
        .for_each(
            |(chunk_idx, (((scale, inv_scale), zero_point), zero_point_f32))| {
                let chunk = chunk_idx * COL_CHUNK..chunk_idx * COL_CHUNK + scale.len();
                let mut lo = [0f32; COL_CHUNK];
                let mut hi = [0f32; COL_CHUNK];
                column_ranges(geom, &pos_min, &pos_max, chunk, &mut lo, &mut hi);

                for i in 0..scale.len() {
                    let (lo, hi) = (lo[i], hi[i]);
                    let s = if hi > lo { (hi - lo) / levels } else { 1. };
                    // Zero point such that `lo` maps to the minimum quantized value.
                    let zp = ((-lo / s).round_ties_even() - (levels + 1.) / 2.).clamp(-128., 127.);
                    scale[i] = s;
                    inv_scale[i] = 1. / s;
                    zero_point[i] = zp as i8;
                    zero_point_f32[i] = zp as i8 as f32;
                }
            },
        );

    ColumnQuant {
        scale,
        inv_scale,
        zero_point,
        zero_point_f32,
    }
}

/// Update `lo` and `hi` with the minimum and maximum values of each column in
/// `cols` of the im2col matrix, given the minimum and maximum values over
/// channels at each input position.
///
/// The range of each column is accumulated with one pass per kernel position
/// over the columns which read from inside the image.
fn column_ranges(
    geom: &ConvGeometry,
    pos_min: &[f32],
    pos_max: &[f32],
    cols: Range<usize>,
    lo: &mut [f32],
    hi: &mut [f32],
) {
    let [_, in_w] = geom.in_size;
    let [k_h, k_w] = geom.kernel_size;
    let [_, out_w] = geom.out_size;

    let mut col = cols.start;
    while col < cols.end {
        // Process the columns which share an output Y position.
        let out_y = col / out_w;
        let seg_end = ((out_y + 1) * out_w).min(cols.end);
        let x_range = (col % out_w)..(seg_end - out_y * out_w);
        let seg_start = col - cols.start;

        for ky in 0..k_h {
            let Some(in_y) = geom.input_pos(0, out_y, ky) else {
                continue;
            };
            for kx in 0..k_w {
                let valid_x = geom.valid_out_x(kx);
                let start = x_range.start.max(valid_x.start);
                let end = x_range.end.min(valid_x.end).max(start);
                let offset = seg_start + start - x_range.start;
                let lo = &mut lo[offset..offset + (end - start)];
                let hi = &mut hi[offset..offset + (end - start)];
                for (i, (lo, hi)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                    // `start..end` are positions where `in_x` is valid.
                    let in_x =
                        (start + i) * geom.strides[1] + kx * geom.dilations[1] - geom.padding[1];
                    let pos = in_y * in_w + in_x;
                    *lo = lo.min(pos_min[pos]);
                    *hi = hi.max(pos_max[pos]);
                }
            }
        }
        col = seg_end;
    }
}

/// Fill `out_row` with columns `cols` of row `row` of the quantized im2col
/// matrix for `image`, which is the contiguous data of a `[C, H, W]` tensor.
///
/// Rows of the im2col matrix are ordered by (channel, kernel_y, kernel_x).
fn pack_quantized_im2col_row(
    out_row: &mut [MaybeUninit<i8>],
    row: usize,
    image: &[f32],
    geom: &ConvGeometry,
    quant: &ColumnQuant,
    cols: Range<usize>,
) {
    let [in_h, in_w] = geom.in_size;
    let [k_h, k_w] = geom.kernel_size;
    let [_, out_w] = geom.out_size;
    let [_, stride_x] = geom.strides;

    let quantize = |x: f32, inv_scale: f32, zp: f32| -> i8 {
        // Clamp after converting to `i32`, as this compiles to fewer
        // instructions than clamping the float value.
        let q = (x * inv_scale + zp).round_ties_even() as i32;
        q.clamp(-128, 127) as i8
    };

    let chan = row / (k_h * k_w);
    let ky = (row / k_w) % k_h;
    let kx = row % k_w;
    let chan_data = &image[chan * in_h * in_w..(chan + 1) * in_h * in_w];
    let valid_x = geom.valid_out_x(kx);

    // Process the row in segments that share an output Y position.
    let mut col = cols.start;
    while col < cols.end {
        let out_y = col / out_w;
        let seg_end = ((out_y + 1) * out_w).min(cols.end);
        let x_range = (col % out_w)..(seg_end - out_y * out_w);
        let out_seg = &mut out_row[col - cols.start..seg_end - cols.start];
        let inv_scale_seg = &quant.inv_scale[col..seg_end];
        let zp_seg = &quant.zero_point[col..seg_end];
        let zp_f32_seg = &quant.zero_point_f32[col..seg_end];

        // Output X positions which read from inside the image. Others
        // read padding, which is zero and quantizes to the zero point.
        let inside_start = x_range.start.max(valid_x.start);
        let inside_end = x_range.end.min(valid_x.end).max(inside_start);
        let in_y = geom.input_pos(0, out_y, ky);
        let inside = match in_y {
            Some(_) => inside_start - x_range.start..inside_end - x_range.start,
            None => 0..0,
        };

        let (pad_start, rest) = out_seg.split_at_mut(inside.start);
        let pad_end = &mut rest[inside.len()..];
        for (q, &zp) in pad_start.iter_mut().zip(zp_seg) {
            q.write(zp);
        }
        for (q, &zp) in pad_end.iter_mut().zip(&zp_seg[inside.end..]) {
            q.write(zp);
        }

        if let Some(in_y) = in_y
            && !inside.is_empty()
        {
            let in_row = &chan_data[in_y * in_w..(in_y + 1) * in_w];
            let in_x_start = ((inside_start * stride_x + kx * geom.dilations[1]) as isize
                - geom.padding[1] as isize) as usize;
            let out_inside = &mut out_seg[inside.clone()];
            let inv_scale = &inv_scale_seg[inside.clone()];
            let zero_point = &zp_f32_seg[inside.clone()];

            if stride_x == 1 {
                let src = &in_row[in_x_start..in_x_start + out_inside.len()];
                quantize_slice(out_inside, src, inv_scale, zero_point);
            } else {
                for (i, ((q, &inv_s), &zp)) in out_inside
                    .iter_mut()
                    .zip(inv_scale)
                    .zip(zero_point)
                    .enumerate()
                {
                    q.write(quantize(in_row[in_x_start + i * stride_x], inv_s, zp));
                }
            }
        }

        col = seg_end;
    }
}

/// Quantize `src` into `out` using per-element inverse scales and zero
/// points.
///
/// This computes `clamp(round_ties_even(x * inv_scale + zero_point), -128,
/// 127)` for each element. Vectorized implementations use a fused
/// multiply-add, which is faster, so results may differ by one where
/// `x * inv_scale + zero_point` is within rounding error of a tie.
fn quantize_slice(out: &mut [MaybeUninit<i8>], src: &[f32], inv_scale: &[f32], zero_point: &[f32]) {
    let len = out.len();
    assert!(src.len() == len && inv_scale.len() == len && zero_point.len() == len);

    #[cfg(target_arch = "aarch64")]
    let n_done = quantize_slice_neon(out, src, inv_scale, zero_point);
    #[cfg(not(target_arch = "aarch64"))]
    let n_done = 0;

    for i in n_done..len {
        let q = (src[i] * inv_scale[i] + zero_point[i]).round_ties_even() as i32;
        out[i].write(q.clamp(-128, 127) as i8);
    }
}

/// Quantize a prefix of `src` into `out` using NEON instructions. See
/// [`quantize_slice`].
///
/// Returns the number of elements quantized, which is a multiple of 16.
#[cfg(target_arch = "aarch64")]
fn quantize_slice_neon(
    out: &mut [MaybeUninit<i8>],
    src: &[f32],
    inv_scale: &[f32],
    zero_point: &[f32],
) -> usize {
    use core::arch::aarch64::*;

    const BLOCK: usize = 16;
    let len = out.len();
    assert!(src.len() == len && inv_scale.len() == len && zero_point.len() == len);
    let n_blocks = len / BLOCK;

    // Safety: Loads read elements `i..i + 16` of each slice, where
    // `i + 16 <= len`, and stores write the same range of `out`. NEON is
    // always available on aarch64.
    unsafe {
        for i in (0..n_blocks * BLOCK).step_by(BLOCK) {
            let quantize = |offset: usize| {
                let x = vld1q_f32(src.as_ptr().add(i + offset));
                let inv_s = vld1q_f32(inv_scale.as_ptr().add(i + offset));
                let zp = vld1q_f32(zero_point.as_ptr().add(i + offset));
                vcvtnq_s32_f32(vfmaq_f32(zp, x, inv_s))
            };
            let lo = vcombine_s16(vqmovn_s32(quantize(0)), vqmovn_s32(quantize(4)));
            let hi = vcombine_s16(vqmovn_s32(quantize(8)), vqmovn_s32(quantize(12)));
            let q = vcombine_s8(vqmovn_s16(lo), vqmovn_s16(hi));
            vst1q_s8(out.as_mut_ptr().add(i) as *mut i8, q);
        }
    }

    n_blocks * BLOCK
}

/// The quantized im2col matrix of an image, which is computed on demand as
/// blocks of it are packed by the GEMM.
struct QuantizedIm2Col<'a> {
    /// Contiguous data of a `[C, H, W]` image.
    image: &'a [f32],
    geom: &'a ConvGeometry,
    quant: &'a ColumnQuant,
    rows: usize,
}

impl BMatrixSource<i8> for QuantizedIm2Col<'_> {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.quant.scale.len()
    }

    fn write_block(&self, out: &mut [MaybeUninit<i8>], rows: Range<usize>, cols: Range<usize>) {
        let geom = self.geom;
        let [in_h, in_w] = geom.in_size;
        let [k_h, k_w] = geom.kernel_size;
        let is_1d = in_h == 1 && k_h == 1 && geom.out_size[0] == 1;
        if !is_1d || geom.strides[1] != 1 {
            for (out_row, row) in out.chunks_mut(cols.len()).zip(rows) {
                pack_quantized_im2col_row(out_row, row, self.image, geom, self.quant, cols.clone());
            }
            return;
        }

        // For 1D convolutions with unit stride, process the rows for each
        // channel together. Rows for a channel read shifted copies of the
        // same input, so for columns where every row reads from inside the
        // input, the quantization parameters can be shared.
        let mut out_rows = out.chunks_mut(cols.len());
        let mut row = rows.start;
        while row < rows.end {
            let chan = row / k_w;
            let kx_start = row % k_w;
            let run_end = ((chan + 1) * k_w).min(rows.end);
            let kx_range = kx_start..kx_start + (run_end - row);

            // Columns which read from inside the input for every row in
            // the run.
            let inside_start = kx_range
                .clone()
                .map(|kx| geom.valid_out_x(kx).start)
                .max()
                .unwrap()
                .clamp(cols.start, cols.end);
            let inside_end = kx_range
                .clone()
                .map(|kx| geom.valid_out_x(kx).end)
                .min()
                .unwrap()
                .clamp(inside_start, cols.end);

            let chan_data = &self.image[chan * in_w..(chan + 1) * in_w];
            let mut run_rows: SmallVec<[&mut [MaybeUninit<i8>]; 16]> = SmallVec::new();
            let mut run_srcs: SmallVec<[&[f32]; 16]> = SmallVec::new();
            for kx in kx_range.clone() {
                let out_row = out_rows.next().unwrap();
                let row = chan * k_w + kx;

                // Columns outside the shared range are handled separately.
                let (before, rest) = out_row.split_at_mut(inside_start - cols.start);
                let (inside, after) = rest.split_at_mut(inside_end - inside_start);
                let quant = self.quant;
                pack_quantized_im2col_row(
                    before,
                    row,
                    self.image,
                    geom,
                    quant,
                    cols.start..inside_start,
                );
                pack_quantized_im2col_row(
                    after,
                    row,
                    self.image,
                    geom,
                    quant,
                    inside_end..cols.end,
                );

                let src = if inside.is_empty() {
                    &[]
                } else {
                    let src_start = inside_start + kx * geom.dilations[1] - geom.padding[1];
                    &chan_data[src_start..src_start + inside.len()]
                };
                run_rows.push(inside);
                run_srcs.push(src);
            }
            quantize_rows(
                &mut run_rows,
                &run_srcs,
                &self.quant.inv_scale[inside_start..inside_end],
                &self.quant.zero_point_f32[inside_start..inside_end],
            );
            row = run_end;
        }
    }
}

/// Quantize each slice in `srcs` into the corresponding slice in `outs`,
/// using the same per-element inverse scales and zero points for every slice.
///
/// See [`quantize_slice`].
fn quantize_rows(
    outs: &mut [&mut [MaybeUninit<i8>]],
    srcs: &[&[f32]],
    inv_scale: &[f32],
    zero_point: &[f32],
) {
    #[cfg(target_arch = "aarch64")]
    let n_done = quantize_rows_neon(outs, srcs, inv_scale, zero_point);
    #[cfg(not(target_arch = "aarch64"))]
    let n_done = 0;

    for (out, src) in outs.iter_mut().zip(srcs) {
        quantize_slice(
            &mut out[n_done..],
            &src[n_done..],
            &inv_scale[n_done..],
            &zero_point[n_done..],
        );
    }
}

/// Quantize a prefix of each slice in `srcs` into `outs` using NEON
/// instructions. See [`quantize_rows`].
///
/// Returns the number of elements quantized in each slice, which is a
/// multiple of 16.
#[cfg(target_arch = "aarch64")]
fn quantize_rows_neon(
    outs: &mut [&mut [MaybeUninit<i8>]],
    srcs: &[&[f32]],
    inv_scale: &[f32],
    zero_point: &[f32],
) -> usize {
    use core::arch::aarch64::*;

    const BLOCK: usize = 16;
    let len = inv_scale.len();
    assert_eq!(zero_point.len(), len);
    assert_eq!(outs.len(), srcs.len());
    for (out, src) in outs.iter().zip(srcs) {
        assert!(out.len() == len && src.len() == len);
    }
    let n_blocks = len / BLOCK;

    // Safety: Loads read elements `i..i + 16` of slices with length `len`,
    // where `i + 16 <= len`, and stores write the same range of slices in
    // `outs`, which also have length `len`. NEON is always available on
    // aarch64.
    unsafe {
        for i in (0..n_blocks * BLOCK).step_by(BLOCK) {
            let inv_s: [float32x4_t; 4] =
                std::array::from_fn(|j| vld1q_f32(inv_scale.as_ptr().add(i + j * 4)));
            let zp: [float32x4_t; 4] =
                std::array::from_fn(|j| vld1q_f32(zero_point.as_ptr().add(i + j * 4)));

            for (out, src) in outs.iter_mut().zip(srcs) {
                let quantize = |j: usize| {
                    let x = vld1q_f32(src.as_ptr().add(i + j * 4));
                    vcvtnq_s32_f32(vfmaq_f32(zp[j], x, inv_s[j]))
                };
                let lo = vcombine_s16(vqmovn_s32(quantize(0)), vqmovn_s32(quantize(1)));
                let hi = vcombine_s16(vqmovn_s32(quantize(2)), vqmovn_s32(quantize(3)));
                let q = vcombine_s8(vqmovn_s16(lo), vqmovn_s16(hi));
                vst1q_s8(out.as_mut_ptr().add(i) as *mut i8, q);
            }
        }
    }

    n_blocks * BLOCK
}

/// Convert int8 weights of shape `[O, K]` to `u8` and pack them for use as the
/// "A" input of the GEMM.
pub(crate) fn pack_weights(
    pool: &BufferPool,
    weight: NdTensorView<i8, 2>,
) -> Result<PackedAMatrix<u8>, OpError> {
    let out_c = weight.size(0);
    let (weight_u8, weight_zero) =
        shift_cast_gemm_lhs_to_u8(pool, weight.as_dyn(), vec![0i8; out_c]);
    let weight_u8 = weight_u8.auto_return(pool);
    GemmExecutor::<u8, i8, i32>::default()
        .prepack_a_quantized(
            weight_u8.nd_view::<2>(),
            QuantParams {
                zero_point: &weight_zero,
            },
        )
        .map_err(|_| OpError::invalid_value("failed to pack weights"))
}

/// Perform a 2D convolution of an `[N, C, H, W]` input with int8 weights of
/// shape `[O, C, kH, kW]` and per-output-channel scales `weight_scale`.
///
/// Activations are quantized dynamically to int8 with a separate scale and
/// zero point for each column of the im2col matrix (ie. each output
/// position). This tracks large local variations in activation magnitude much
/// better than a single per-tensor scale. The quantized im2col matrix is
/// computed by the GEMM as it packs blocks of it.
fn conv_dynamic_quant_2d(
    pool: &BufferPool,
    input: NdTensorView<f32, 4>,
    weight: NdTensorView<i8, 4>,
    weight_scale: &[f32],
    bias: Option<&[f32]>,
    packed_weight: Option<&PackedAMatrix<u8>>,
    residual: Option<&[f32]>,
    geom: ConvGeometry,
) -> Result<NdTensor<f32, 4>, OpError> {
    let [batch, in_c, _, _] = input.shape();
    let [out_c, k_in_c, k_h, k_w] = weight.shape();
    let [out_h, out_w] = geom.out_size;
    let n_cols = out_h * out_w;
    let k_depth = in_c * k_h * k_w;

    check_eq!(in_c, k_in_c, "input channels do not match weights")?;
    check_eq!(weight_scale.len(), out_c, "weight scale length")?;
    if let Some(bias) = bias {
        check_eq!(bias.len(), out_c, "bias length")?;
    }
    if let Some(residual) = residual {
        check_eq!(residual.len(), batch * out_c * n_cols, "residual length")?;
    }

    let gemm = GemmExecutor::<u8, i8, i32>::default();

    // Quantized activations use the full `i8` range unless the GEMM kernel
    // can saturate, in which case they are restricted to 7 bits.
    let levels = if gemm.may_saturate() { 127. } else { 255. };

    // Use the weights packed when the model was loaded, or pack them now.
    let packed_weight = match packed_weight {
        Some(packed) => Cow::Borrowed(packed),
        None => {
            let weight_mat = weight.reshaped_in(pool, [out_c, k_depth]).auto_return(pool);
            Cow::Owned(pack_weights(pool, weight_mat.view())?)
        }
    };

    let mut output = NdTensor::<f32, 3>::uninit_in(pool, [batch, out_c, n_cols]);

    for n in 0..batch {
        let image = input.slice(n).to_contiguous_in(pool).auto_return(pool);
        let image = image.data();
        let quant = column_quant_params(image, &geom, levels);
        let im2col = QuantizedIm2Col {
            image,
            geom: &geom,
            quant: &quant,
            rows: k_depth,
        };

        // Compute the int32 accumulators in the output buffer, then convert
        // them to floats in place.
        let mut out_item = output.slice_mut(n);
        let out_data = out_item.data_mut().unwrap();
        // Safety: `MaybeUninit<f32>` and `MaybeUninit<i32>` have the same size
        // and alignment, and any bit pattern is valid for both.
        let acc: &mut [MaybeUninit<i32>] = unsafe {
            std::slice::from_raw_parts_mut(out_data.as_mut_ptr() as *mut _, out_data.len())
        };
        gemm.gemm_uninit(
            acc,
            GemmInputA::Packed(&packed_weight),
            GemmInputB::Source(&im2col),
            GemmUninitOptions {
                b_quant: Some(QuantParams {
                    zero_point: &quant.zero_point,
                }),
                ..Default::default()
            },
        )
        .map_err(|_| OpError::invalid_value("GEMM failed"))?;

        // Rescale accumulators and add the bias and residual.
        let item_residual = residual.map(|r| &r[n * out_c * n_cols..][..out_c * n_cols]);
        out_data
            .par_chunks_mut(n_cols)
            .enumerate()
            .for_each(|(oc, out_row)| {
                let w_scale = weight_scale[oc];
                let b = bias.map(|b| b[oc]).unwrap_or(0.);
                let to_f32 = |y: &MaybeUninit<f32>, s: f32| {
                    // Safety: The GEMM initialized `y` with an `i32`.
                    let acc = i32::from_ne_bytes(unsafe { y.assume_init() }.to_ne_bytes());
                    acc as f32 * (w_scale * s) + b
                };
                if let Some(residual) = item_residual {
                    let residual = &residual[oc * n_cols..][..n_cols];
                    for ((y, &s), &r) in out_row.iter_mut().zip(&quant.scale).zip(residual) {
                        y.write(to_f32(y, s) + r);
                    }
                } else {
                    for (y, &s) in out_row.iter_mut().zip(&quant.scale) {
                        y.write(to_f32(y, s));
                    }
                }
            });
    }

    // Safety: Every output element was written by the epilogue.
    let output = unsafe { output.assume_init() };
    Ok(output.into_shape([batch, out_c, out_h, out_w]))
}

/// Multiply int8 weights of shape `[O, C]` with each position of a `[N, C, H,
/// W]` image, whose values are quantized dynamically with a separate scale for
/// each position.
///
/// This is a 1x1 convolution with dynamically quantized activations. Returns
/// a `[N, O, H, W]` tensor. `weight_scale` is the scale of each row of
/// `weight`. `packed_weight` is the result of [`pack_weights`] for `weight`, if
/// available.
pub(crate) fn conv_1x1_dynamic_quant(
    pool: &BufferPool,
    input: NdTensorView<f32, 4>,
    weight: NdTensorView<i8, 2>,
    weight_scale: &[f32],
    packed_weight: Option<&PackedAMatrix<u8>>,
) -> Result<NdTensor<f32, 4>, OpError> {
    let [_, _, in_h, in_w] = input.shape();
    let [out_c, in_c] = weight.shape();
    let geom = ConvGeometry {
        in_size: [in_h, in_w],
        kernel_size: [1, 1],
        out_size: [in_h, in_w],
        padding: [0; 4],
        strides: [1, 1],
        dilations: [1, 1],
    };
    let weight = weight
        .reshaped_in(pool, [out_c, in_c, 1, 1])
        .auto_return(pool);
    conv_dynamic_quant_2d(
        pool,
        input,
        weight.view(),
        weight_scale,
        None,
        packed_weight,
        None,
        geom,
    )
}

/// Convolution with int8 weights and dynamically quantized int8 activations.
///
/// Inputs are `X, W, W_scale, B, R`, where `W` is an int8 weight tensor with
/// shape `[O, C, ...kernel]`, `W_scale` is a per-output-channel scale with
/// shape `[O]`, `B` is an optional bias of shape `[O]` and `R` is an optional
/// residual which is added to the output. Only `groups = 1` is supported.
///
/// Activations are quantized with a separate scale and zero point for each
/// output position. This is not an ONNX operator. It is created from `Conv`
/// operators with `DequantizeLinear` weights when dynamic quantization of
/// convolutions is enabled.
#[derive(Clone, Debug)]
pub struct ConvDynamicQuant {
    pub padding: Padding,
    pub strides: Vec<usize>,
    pub dilations: Vec<usize>,
}

impl Operator for ConvDynamicQuant {
    fn name(&self) -> &str {
        "ConvDynamicQuant"
    }

    fn max_inputs(&self) -> Option<usize> {
        Some(5)
    }

    fn prepack_inputs(&self) -> SmallVec<[usize; 1]> {
        [1].into()
    }

    fn prepack(
        &self,
        index: usize,
        input: ValueView,
        _zero_point: Option<ValueView>,
    ) -> Option<PrepackedInput> {
        if index != 1 {
            return None;
        }
        let weight: TensorView<i8> = input.try_into().ok()?;
        let out_c = *weight.shape().first()?;
        let weight = weight.reshaped([out_c, weight.len() / out_c.max(1)]);
        let packed = pack_weights(&BufferPool::new(), weight.view()).ok()?;
        Some(PrepackedInput::UInt8AMatrix(packed))
    }

    fn run(&self, ctx: &OpRunContext) -> Result<OutputList, OpError> {
        let inputs = ctx.inputs();
        let input: TensorView<f32> = inputs.require_as(0)?;
        let weight: TensorView<i8> = inputs.require_as(1)?;
        let weight_scale: NdTensorView<f32, 1> = inputs.require_as(2)?;
        let bias: Option<NdTensorView<f32, 1>> = inputs.get_as(3)?;
        let residual: Option<TensorView<f32>> = inputs.get_as(4)?;
        let packed_weight = match inputs.get_prepacked(1) {
            Some(PrepackedInput::UInt8AMatrix(packed)) => Some(packed),
            _ => None,
        };

        let weight_scale = weight_scale.to_contiguous();
        let bias = bias.map(|b| b.to_contiguous());

        // Expand 1D convolutions to 2D.
        let is_1d = input.ndim() == 3;
        let (input, weight, padding, strides, dilations) = if is_1d {
            let mut input = input;
            input.insert_axis(2);
            let mut weight = weight;
            weight.insert_axis(2);
            let strides = match self.strides.as_slice() {
                &[s] => [1, s],
                _ => return Err(OpError::invalid_value("expected 1 stride value")),
            };
            let dilations = match self.dilations.as_slice() {
                &[d] => [1, d],
                _ => return Err(OpError::invalid_value("expected 1 dilation value")),
            };
            (
                input,
                weight,
                self.padding.expand_1d_to_2d()?,
                strides,
                dilations,
            )
        } else {
            let strides: [usize; 2] = self
                .strides
                .as_slice()
                .try_into()
                .map_err(|_| OpError::invalid_value("expected 2 stride values"))?;
            let dilations: [usize; 2] = self
                .dilations
                .as_slice()
                .try_into()
                .map_err(|_| OpError::invalid_value("expected 2 dilation values"))?;
            (input, weight, self.padding.clone(), strides, dilations)
        };

        let input = static_dims!(input, 4, "NCHW")?;
        let weight = static_dims!(weight, 4, "OCHW")?;
        let [_, _, in_h, in_w] = input.shape();
        let [_, _, k_h, k_w] = weight.shape();

        let (out_h, out_w, fixed_padding) = calc_output_size_and_padding(
            (in_h, in_w),
            (k_h, k_w),
            (strides[0], strides[1]),
            padding,
            Some((dilations[0], dilations[1])),
            RoundMode::default(),
        )?;
        let geom = ConvGeometry {
            in_size: [in_h, in_w],
            kernel_size: [k_h, k_w],
            out_size: [out_h, out_w],
            padding: fixed_padding,
            strides,
            dilations,
        };

        // Add the residual while converting accumulators to floats if it has
        // the same shape as the output, or separately otherwise.
        let [batch, _, _, _] = input.shape();
        let out_c = weight.size(0);
        let out_shape: SmallVec<[usize; 4]> = if is_1d {
            SmallVec::from_slice(&[batch, out_c, out_w])
        } else {
            SmallVec::from_slice(&[batch, out_c, out_h, out_w])
        };
        let pool = ctx.pool();
        let fused_residual = residual
            .as_ref()
            .filter(|r| r.shape() == out_shape.as_slice())
            .map(|r| r.to_contiguous_in(pool).auto_return(pool));

        let output = conv_dynamic_quant_2d(
            pool,
            input,
            weight,
            weight_scale.data(),
            bias.as_ref().map(|b| b.data()),
            packed_weight,
            fused_residual.as_ref().map(|r| r.data()),
            geom,
        )?;

        let mut output: Tensor<f32> = output.into_dyn();
        output.reshape(&out_shape);
        match residual {
            Some(residual) if fused_residual.is_none() => {
                add(pool, output.view(), residual)?.into_op_result()
            }
            _ => output.into_op_result(),
        }
    }

    fn output_types(&self, _ctx: &OutputTypesContext) -> Option<OutputTypeList> {
        Some([OutputType::CopyFromInput(0)].into())
    }

    fn as_infer_shapes(&self) -> Option<&dyn InferShapes> {
        Some(self)
    }
}

impl rten_shape_inference::InferShapes for ConvDynamicQuant {
    fn infer_shapes(
        &self,
        inputs: rten_shape_inference::InferShapesContext,
        sym_gen: &mut rten_shape_inference::SymbolGen,
    ) -> Result<Vec<rten_shape_inference::SymTensor>, rten_shape_inference::InferShapesError> {
        let conv = shape_ops::Conv {
            strides: &self.strides,
            dilations: &self.dilations,
            padding: self.padding.as_shape_inference_padding(),
        };
        let conv_inputs: Vec<_> = (0..4).map(|i| inputs.get(i).cloned()).collect();
        let conv_outputs = conv.infer_shapes(conv_inputs.into(), sym_gen)?;

        // The residual is broadcast with the convolution output.
        match (inputs.get(4), conv_outputs.as_slice()) {
            (Some(residual), [conv_out]) => {
                shape_ops::Add.infer_shapes([conv_out.clone(), residual.clone()].into(), sym_gen)
            }
            _ => Ok(conv_outputs),
        }
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::rng::XorShiftRng;
    use rten_tensor::{NdTensor, Tensor};
    use rten_testing::TestCases;

    use super::ConvDynamicQuant;
    use crate::buffer_pool::BufferPool;
    use crate::operator::{InputList, OpRunContext, Operator, OperatorExt, OutputMask};
    use crate::ops::binary_elementwise::add;
    use crate::ops::{Padding, conv};

    /// Reference implementation which quantizes each im2col column
    /// separately.
    fn reference_conv_dynamic_quant(
        input: &Tensor<f32>,
        weight: &Tensor<i8>,
        weight_scale: &[f32],
        bias: Option<&[f32]>,
        pads: [usize; 4],
        strides: [usize; 2],
        dilations: [usize; 2],
    ) -> Tensor<f32> {
        let [batch, in_c, in_h, in_w]: [usize; 4] = input.shape().try_into().unwrap();
        let [out_c, _, k_h, k_w]: [usize; 4] = weight.shape().try_into().unwrap();
        let [pad_top, pad_left, pad_bottom, pad_right] = pads;
        let out_h = (in_h + pad_top + pad_bottom - dilations[0] * (k_h - 1) - 1) / strides[0] + 1;
        let out_w = (in_w + pad_left + pad_right - dilations[1] * (k_w - 1) - 1) / strides[1] + 1;

        let mut output = Tensor::zeros(&[batch, out_c, out_h, out_w]);
        for n in 0..batch {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    // Gather the im2col column.
                    let mut column = Vec::new();
                    for c in 0..in_c {
                        for ky in 0..k_h {
                            for kx in 0..k_w {
                                let y = (oy * strides[0] + ky * dilations[0]) as isize
                                    - pad_top as isize;
                                let x = (ox * strides[1] + kx * dilations[1]) as isize
                                    - pad_left as isize;
                                let inside =
                                    y >= 0 && x >= 0 && (y as usize) < in_h && (x as usize) < in_w;
                                column.push(if inside {
                                    input[[n, c, y as usize, x as usize]]
                                } else {
                                    0.
                                });
                            }
                        }
                    }

                    // Quantize the column.
                    let lo = column.iter().copied().fold(0f32, f32::min);
                    let hi = column.iter().copied().fold(0f32, f32::max);
                    let scale = if hi > lo { (hi - lo) / 255. } else { 1. };
                    let zp = ((-lo / scale).round_ties_even() - 128.).clamp(-128., 127.) as i8;
                    let inv_scale = 1. / scale;
                    let quant: Vec<i32> = column
                        .iter()
                        .map(|&x| {
                            let q = (x * inv_scale + zp as f32)
                                .round_ties_even()
                                .clamp(-128., 127.) as i8;
                            q as i32 - zp as i32
                        })
                        .collect();

                    for oc in 0..out_c {
                        let w = weight.slice(oc);
                        let acc: i32 = w.iter().zip(&quant).map(|(&w, &q)| w as i32 * q).sum();
                        let b = bias.map(|b| b[oc]).unwrap_or(0.);
                        output[[n, oc, oy, ox]] = acc as f32 * (weight_scale[oc] * scale) + b;
                    }
                }
            }
        }
        output
    }

    #[test]
    fn test_conv_dynamic_quant_residual() {
        let mut rng = XorShiftRng::new(1234);
        let input = Tensor::<f32>::rand(&[1, 3, 8], &mut rng);
        let weight = Tensor::<i8>::rand(&[4, 3, 3], &mut rng);
        let weight_scale = Tensor::from([0.01f32, 0.02, 0.03, 0.04]);
        let bias = Tensor::from([0.5f32, -0.5, 1., 0.]);
        let op = ConvDynamicQuant {
            padding: Padding::Fixed([1, 1].as_slice().into()),
            strides: [1].into(),
            dilations: [1].into(),
        };
        let output: Tensor<f32> = op
            .run_simple((
                input.view(),
                weight.view(),
                weight_scale.view(),
                bias.view(),
            ))
            .unwrap();

        // Residuals with the same shape as the output, which are added while
        // computing the output, and which are broadcast.
        let residuals = [
            Tensor::<f32>::rand(&[1, 4, 8], &mut rng),
            Tensor::<f32>::rand(&[1, 4, 1], &mut rng),
            Tensor::<f32>::rand(&[2, 1, 8], &mut rng),
        ];
        for residual in residuals {
            let result: Tensor<f32> = op
                .run_simple((
                    input.view(),
                    weight.view(),
                    weight_scale.view(),
                    bias.view(),
                    residual.view(),
                ))
                .unwrap();
            let expected = add(&BufferPool::new(), output.view(), residual.view()).unwrap();
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_conv_dynamic_quant() {
        #[derive(Debug)]
        struct Case {
            input_shape: Vec<usize>,
            weight_shape: Vec<usize>,
            with_bias: bool,
            pads: Vec<usize>,
            strides: Vec<usize>,
            dilations: Vec<usize>,
        }

        let cases = [
            // 1D conv where the im2col matrix has multiple depth blocks,
            // which split the rows for a channel.
            Case {
                input_shape: [1, 100, 70].into(),
                weight_shape: [4, 100, 11].into(),
                with_bias: true,
                pads: [15, 15].into(),
                strides: [1].into(),
                dilations: [3].into(),
            },
            // 1D conv with padding and dilation
            Case {
                input_shape: [1, 5, 40].into(),
                weight_shape: [6, 5, 3].into(),
                with_bias: true,
                pads: [3, 3].into(),
                strides: [1].into(),
                dilations: [3].into(),
            },
            // 1D conv with even kernel, uneven padding and stride
            Case {
                input_shape: [2, 4, 33].into(),
                weight_shape: [3, 4, 4].into(),
                with_bias: false,
                pads: [1, 2].into(),
                strides: [2].into(),
                dilations: [1].into(),
            },
            // 2D conv with padding
            Case {
                input_shape: [1, 3, 9, 7].into(),
                weight_shape: [4, 3, 3, 3].into(),
                with_bias: true,
                pads: [1, 1, 1, 1].into(),
                strides: [1, 1].into(),
                dilations: [1, 1].into(),
            },
            // 2D conv with stride and asymmetric padding
            Case {
                input_shape: [1, 2, 8, 10].into(),
                weight_shape: [5, 2, 3, 2].into(),
                with_bias: true,
                pads: [0, 1, 2, 0].into(),
                strides: [2, 3].into(),
                dilations: [1, 2].into(),
            },
            // Pointwise conv
            Case {
                input_shape: [1, 8, 4, 4].into(),
                weight_shape: [3, 8, 1, 1].into(),
                with_bias: false,
                pads: [0, 0, 0, 0].into(),
                strides: [1, 1].into(),
                dilations: [1, 1].into(),
            },
        ];

        cases.test_each(|case| {
            let mut rng = XorShiftRng::new(1234);
            // Inputs with a mix of magnitudes, including large outliers.
            let input = Tensor::<f32>::rand(&case.input_shape, &mut rng)
                .map(|&x| if x > 0.95 { (x - 0.5) * 40. } else { x - 0.3 });
            let weight = Tensor::<f32>::rand(&case.weight_shape, &mut rng)
                .map(|&x| ((x - 0.5) * 255.).round() as i8);
            let out_c = case.weight_shape[0];
            let weight_scale: Vec<f32> = (0..out_c).map(|i| 0.01 * (i + 1) as f32).collect();
            let weight_scale = NdTensor::from_data([out_c], weight_scale);
            let bias = case.with_bias.then(|| {
                NdTensor::from_data([out_c], (0..out_c).map(|i| i as f32).collect::<Vec<_>>())
            });

            let op = ConvDynamicQuant {
                padding: Padding::Fixed(case.pads.as_slice().into()),
                strides: case.strides.clone(),
                dilations: case.dilations.clone(),
            };
            let result: Tensor<f32> = match &bias {
                Some(bias) => op
                    .run_simple((
                        input.view(),
                        weight.view(),
                        weight_scale.view(),
                        bias.view(),
                    ))
                    .unwrap(),
                None => op
                    .run_simple((input.view(), weight.view(), weight_scale.view()))
                    .unwrap(),
            };

            // Run again with prepacked weights. The unpacked weights are
            // replaced with zeros to check that the prepacked ones are used.
            let packed_weight = op.prepack(1, weight.view().into(), None).unwrap();
            let get_prepacked = |idx| (idx == 1).then_some(&packed_weight);
            let zero_weight = Tensor::<i8>::zeros(weight.shape());
            let input_values = [
                Some(input.view().into()),
                Some(zero_weight.view().into()),
                Some(weight_scale.view().into()),
                bias.as_ref().map(|b| b.view().into()),
            ];
            let inputs = InputList::from_optional(&input_values).with_prepacked(&get_prepacked);
            let pool = BufferPool::new();
            let ctx = OpRunContext::new(&pool, &inputs, OutputMask::all_used(1));
            let prepacked_result: Tensor<f32> = op.run(&ctx).unwrap().remove(0).try_into().unwrap();
            assert_eq!(prepacked_result, result);

            // Compare against the reference, using 2D shapes for 1D cases.
            let is_1d = input.ndim() == 3;
            let expand = |mut t: Tensor<f32>| {
                if is_1d {
                    t.insert_axis(2);
                }
                t
            };
            let mut weight_2d = weight.clone();
            let (pads, strides, dilations) = if is_1d {
                weight_2d.insert_axis(2);
                (
                    [0, case.pads[0], 0, case.pads[1]],
                    [1, case.strides[0]],
                    [1, case.dilations[0]],
                )
            } else {
                (
                    case.pads.as_slice().try_into().unwrap(),
                    case.strides.as_slice().try_into().unwrap(),
                    case.dilations.as_slice().try_into().unwrap(),
                )
            };
            let expected = reference_conv_dynamic_quant(
                &expand(input.clone()),
                &weight_2d,
                weight_scale.data().unwrap(),
                bias.as_ref().map(|b| b.data().unwrap()),
                pads,
                strides,
                dilations,
            );
            let result_2d = expand(result.clone());
            assert_eq!(result_2d.shape(), expected.shape());
            for (r, e) in result_2d.iter().zip(expected.iter()) {
                assert!((r - e).abs() <= 1e-5 * e.abs().max(1.), "{r} != {e}");
            }

            // Check the error vs an f32 convolution is small relative to the
            // output magnitude.
            let pool = BufferPool::new();
            let weight_f32 = Tensor::from_fn(weight.shape(), |idx| {
                weight[idx] as f32 * weight_scale[[idx[0]]]
            });
            let reference_f32 = conv(
                &pool,
                input.view(),
                weight_f32.view(),
                bias.as_ref().map(|b| b.view()),
                Padding::Fixed(case.pads.as_slice().into()),
                1,
                &case.strides,
                &case.dilations,
            )
            .unwrap();
            let max_abs = reference_f32.iter().fold(0f32, |m, x| m.max(x.abs()));
            let max_err = result
                .iter()
                .zip(reference_f32.iter())
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(
                max_err < 0.02 * max_abs,
                "max error {max_err} vs max abs {max_abs}"
            );
        })
    }
}
