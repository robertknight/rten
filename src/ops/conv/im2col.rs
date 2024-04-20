use std::ops::Range;

use rten_tensor::prelude::*;
use rten_tensor::NdTensorView;
use rten_vecmath::simd_vec::{SimdFloat, SimdInt};

#[cfg(feature = "avx512")]
use rten_vecmath::is_avx512_supported;

use crate::gemm::{round_up, KernelType, VirtualMatrix};
use crate::ops::pooling::calc_output_size_and_padding;
use crate::ops::Padding;

struct RowOffsets {
    /// Map of channel index to `channel * channel_stride`.
    chan: Vec<i32>,

    /// Map of row index to `row * row_stride`.
    y: Vec<i32>,

    /// Map of col index to `col * col_stride`.
    x: Vec<i32>,
}

struct ColOffsets {
    /// Map of column index to `row * row_stride`.
    y: Vec<i32>,

    /// Map of column index to `col * col_stride`.
    x: Vec<i32>,
}

/// Unrolls patches of an image as columns of a virtual matrix.
///
/// The input image has shape [C,H,W] and is transformed into a matrix with
/// shape [C * Kh * kW, Oh * Ow] where Kh/Kw are convolution kernel sizes and
/// Oh/Ow are the number of patches in the Y and X directions.
///
/// The transform is virtual because the matrix is not actually materialized
/// in memory. Instead blocks of it are produced on-demand during a matrix
/// multiplication operation.
pub struct VirtualIm2Col<'a> {
    image: NdTensorView<'a, f32, 3>,

    /// Map of im2col row index to input image coordinate, premultiplied with
    /// the corresponding stride.
    row_offsets: RowOffsets,

    /// Map of im2col column index to input image coordinate, premultiplied with
    /// the corresponding stride. The length of `col_offsets` is rounded up
    /// to the nearest multiple of the panel width. `n_cols` contains the
    /// number of columns in the virtual matrix.
    col_offsets: ColOffsets,

    /// Number of columns in the im2col matrix.
    n_cols: usize,

    /// Maximum valid sum of `row_offsets.y + col_offsets.y`. Values above this
    /// correspond to the padding region.
    max_y_offset: i32,

    /// Maximum valid sum of `row_offsets.x + col_offsets.x`. Values above this
    /// correspond to the padding region.
    max_x_offset: i32,

    /// Gemm kernel that is going to be used.
    gemm_kernel: KernelType,
}

impl<'a> VirtualIm2Col<'a> {
    /// Create a virtual im2col matrix from a [C, H, W] input tensor and
    /// convolution parameters.
    pub fn new(
        gemm_kernel: KernelType,
        image: NdTensorView<'a, f32, 3>,
        kernel: [usize; 2],
        padding: [usize; 4],
        strides: [usize; 2],
        dilations: [usize; 2],
        panel_width: usize,
    ) -> VirtualIm2Col {
        // Ensure image has at least one cell.
        assert!(image.len() > 0);

        let [chans, h, w] = image.shape();
        let [k_h, k_w] = kernel;
        let [stride_h, stride_w] = strides;
        let [dilation_y, dilation_x] = dilations;
        let [pad_top, pad_left, _pad_bottom, _pad_right] = padding;
        let (y_patches, x_patches, _) = calc_output_size_and_padding(
            (h, w),
            (k_h, k_w),
            (stride_h, stride_w),
            Padding::Fixed(padding.into()),
            Some((dilation_y, dilation_x)),
        )
        .expect("invalid im2col params");

        let [im_stride_c, im_stride_h, im_stride_w]: [i32; 3] =
            image.strides().map(|s| s.try_into().unwrap());

        // Build lookup table of row index in the virtual im2col matrix to
        // offsets in the image.
        let n_rows = chans * k_h * k_w;
        let row_offsets = (0..n_rows).map(|row| {
            let in_chan = row as i32 / (k_h * k_w) as i32;
            let kernel_element = row as i32 % (k_h * k_w) as i32;
            let k_y = kernel_element / k_w as i32;
            let k_x = kernel_element % k_w as i32;

            // Offset to image channel
            (
                in_chan * im_stride_c,
                // Offset from top-left corner of patch
                (
                    im_stride_h * k_y * dilation_y as i32,
                    im_stride_w * k_x * dilation_x as i32,
                ),
            )
        });
        let (row_chan_offsets, row_yx_offsets): (Vec<i32>, Vec<(i32, i32)>) = row_offsets.unzip();
        let (row_y_offsets, row_x_offsets) = row_yx_offsets.into_iter().unzip();

        // Build lookup table of column index in the virtual im2col matrix to
        // offsets in the image.
        let n_cols = x_patches * y_patches;
        let n_cols_padded = round_up(n_cols, panel_width);

        let col_offsets = (0..n_cols_padded).map(|col| {
            let patch_y = col as i32 / x_patches as i32;
            let patch_x = col as i32 % x_patches as i32;
            let img_x = (patch_x * stride_w as i32) - pad_left as i32;
            let img_y = (patch_y * stride_h as i32) - pad_top as i32;
            (img_y * im_stride_h, img_x * im_stride_w)
        });
        let (col_y_offsets, col_x_offsets): (Vec<i32>, Vec<i32>) = col_offsets.unzip();

        // Compute max valid X / Y offsets for testing whether an element is in
        // the padding region or not.
        let max_y_offset: i32 = ((image.size(1) - 1) * image.stride(1))
            .try_into()
            .expect("invalid im2col params");
        let max_x_offset: i32 = ((image.size(2) - 1) * image.stride(2))
            .try_into()
            .expect("invalid im2col params");

        VirtualIm2Col {
            gemm_kernel,
            image,
            row_offsets: RowOffsets {
                chan: row_chan_offsets,
                y: row_y_offsets,
                x: row_x_offsets,
            },
            col_offsets: ColOffsets {
                y: col_y_offsets,
                x: col_x_offsets,
            },
            n_cols,

            max_y_offset,
            max_x_offset,
        }
    }

    /// Pack part of an image according to the requirements of
    /// [VirtualMatrix::pack_b].
    ///
    /// `NR_REGS` specifies the width of each column panel in terms of the width
    /// of vector registers (`S::LEN`). ie. `panel_width` must exactly equal
    /// `NR_REGS * S::LEN`.
    #[inline(always)]
    unsafe fn pack_b_impl<S: SimdFloat, const NR_REGS: usize>(
        &self,
        out: &mut [f32],
        panel_width: usize,
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        assert_eq!(panel_width, S::LEN * NR_REGS);

        let col_range = cols.start..round_up(cols.end, panel_width);
        assert!(out.len() >= rows.len() * col_range.len());

        let col_y_offsets = &self.col_offsets.y[col_range.clone()];
        let col_x_offsets = &self.col_offsets.x[col_range.clone()];
        let row_chan_offsets = &self.row_offsets.chan[rows.clone()];
        let row_y_offsets = &self.row_offsets.y[rows.clone()];
        let row_x_offsets = &self.row_offsets.x[rows.clone()];

        let img_data = self.image.non_contiguous_data();

        // Loop over column panels, then rows, then `S::LEN`-wide column groups
        // within each panel.
        let out_ptr = out.as_mut_ptr();
        let mut out_offset = 0;

        for start_col in (0..col_y_offsets.len()).step_by(S::LEN * NR_REGS) {
            let col_y_offset: [S::Int; NR_REGS] = std::array::from_fn(|i| {
                S::Int::load(col_y_offsets.as_ptr().add(start_col + S::LEN * i))
            });
            let col_x_offset: [S::Int; NR_REGS] = std::array::from_fn(|i| {
                S::Int::load(col_x_offsets.as_ptr().add(start_col + S::LEN * i))
            });
            let max_x_offset = S::Int::splat(self.max_x_offset);
            let max_y_offset = S::Int::splat(self.max_y_offset);

            for ((&row_chan_offset, &row_y_offset), &row_x_offset) in row_chan_offsets
                .iter()
                .zip(row_y_offsets.iter())
                .zip(row_x_offsets.iter())
            {
                let row_chan_offset = S::Int::splat(row_chan_offset);
                let row_y_offset = S::Int::splat(row_y_offset);
                let row_x_offset = S::Int::splat(row_x_offset);

                for i in 0..NR_REGS {
                    let y_offset = col_y_offset[i].add(row_y_offset);
                    let x_offset = col_x_offset[i].add(row_x_offset);
                    let offsets = row_chan_offset.add(y_offset).add(x_offset);

                    // Create mask to specify offsets which are valid. Others
                    // correspond to the padding region.
                    let zero = S::Int::zero();
                    let pad_mask = S::Int::splat(-1)
                        .blend(zero, y_offset.lt(zero))
                        .blend(zero, y_offset.gt(max_y_offset))
                        .blend(zero, x_offset.lt(zero))
                        .blend(zero, x_offset.gt(max_x_offset));

                    let elts = S::gather_mask(img_data.as_ptr(), offsets, pad_mask.to_float_mask());

                    elts.store(out_ptr.add(out_offset));
                    out_offset += S::LEN;
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn pack_b_impl_avx(
        &self,
        out: &mut [f32],
        panel_width: usize,
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        use std::arch::x86_64::__m256;
        self.pack_b_impl::<__m256, 2>(out, panel_width, rows.clone(), cols.clone());
    }

    #[cfg(feature = "avx512")]
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    #[target_feature(enable = "avx512vl")]
    unsafe fn pack_b_impl_avx512(
        &self,
        out: &mut [f32],
        panel_width: usize,
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        use std::arch::x86_64::__m512;
        self.pack_b_impl::<__m512, 2>(out, panel_width, rows.clone(), cols.clone());
    }
}

impl<'a> VirtualMatrix for VirtualIm2Col<'a> {
    fn rows(&self) -> usize {
        self.row_offsets.chan.len()
    }

    fn cols(&self) -> usize {
        self.n_cols
    }

    fn pack_b(&self, out: &mut [f32], panel_width: usize, rows: Range<usize>, cols: Range<usize>) {
        match (self.gemm_kernel, panel_width) {
            #[cfg(feature = "avx512")]
            #[cfg(target_arch = "x86_64")]
            (KernelType::Avx512, 32) => unsafe {
                assert!(is_avx512_supported());
                self.pack_b_impl_avx512(out, panel_width, rows.clone(), cols.clone());
            },
            #[cfg(target_arch = "x86_64")]
            (KernelType::Fma, 16) => unsafe {
                assert!(is_x86_feature_detected!("avx2"));
                self.pack_b_impl_avx(out, panel_width, rows.clone(), cols.clone());
            },
            #[cfg(target_arch = "aarch64")]
            (KernelType::ArmNeon, 8) => unsafe {
                // Safety: Neon is always available.
                use std::arch::aarch64::float32x4_t;
                self.pack_b_impl::<float32x4_t, 2>(out, panel_width, rows, cols);
            },
            #[cfg(target_arch = "wasm32")]
            (KernelType::Wasm, 8) => unsafe {
                // Safety: SIMD support is checked when WASM binary is loaded.
                use rten_vecmath::simd_vec::wasm::v128f;
                self.pack_b_impl::<v128f, 2>(out, panel_width, rows, cols);
            },
            (KernelType::Base, 4) => unsafe {
                self.pack_b_impl::<f32, 4>(out, panel_width, rows, cols);
            },
            _ => {
                panic!(
                    "unsupported (kernel, panel_width) for im2col: ({:?}, {})",
                    self.gemm_kernel, panel_width
                );
            }
        }
    }
}
