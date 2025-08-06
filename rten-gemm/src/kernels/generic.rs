use std::mem::MaybeUninit;
use std::ops::Range;

use rten_base::byte_cast::{cast_pod_slice, cast_uninit_pod_mut_slice};
use rten_simd::{isa::GenericIsa, Isa};
use rten_tensor::{Matrix, MatrixLayout};

use super::simd_generic::{simd_gemv, GemmDispatch};
use super::{Kernel, Lhs, MatVecOutput, PackedLayout, QuantParams, TempTile};
use crate::packing;
use crate::packing::{pack_a_block, pack_b_block, packed_a_layout, packed_b_layout};
use crate::Im2Col;

/// This is the base kernel that does not use architecture-specific intrinsics
/// but is autovectorization-friendly. It is expected to perform the same as
/// a kernel using SSE intrinsics (or equivalent).
pub struct GenericKernel {
    isa: GenericIsa,
}

impl GenericKernel {
    const MR: usize = 8;

    // The base kernel will most likely be compiled to SSE or equivalent. SSE
    // registers are 128 bits wide = 4 x f32, so this should be a multiple of
    // that.
    const NR: usize = 4;
}

const X32_LANES: usize = size_of::<<GenericIsa as Isa>::I32>() / size_of::<i32>();

// Safety - Base kernel is always supported
unsafe impl Kernel<f32, f32, f32> for GenericKernel {
    fn new() -> Option<Self> {
        Some(GenericKernel {
            isa: GenericIsa::new(),
        })
    }

    fn mr(&self) -> usize {
        Self::MR
    }

    fn nr(&self) -> usize {
        Self::NR
    }

    fn name(&self) -> &'static str {
        "generic-f32"
    }

    fn packed_a_layout(
        &self,
        a: Matrix,
        rows: usize,
        cols: usize,
        _quant: Option<QuantParams<f32>>,
    ) -> PackedLayout {
        let mut info = packed_a_layout::<f32, { Self::MR }>(rows, cols);
        info.must_pack = a.col_stride() != 1;
        info
    }

    fn pack_a_block(
        &self,
        out: &mut [MaybeUninit<u8>],
        a: Matrix,
        rows: Range<usize>,
        cols: Range<usize>,
        _quant: Option<QuantParams<f32>>,
    ) {
        let out = cast_uninit_pod_mut_slice(out).unwrap();
        pack_a_block::<f32, { Self::MR }>(out, a, rows, cols);
    }

    fn packed_b_layout(
        &self,
        rows: usize,
        cols: usize,
        _quant: Option<QuantParams<f32>>,
    ) -> PackedLayout {
        packed_b_layout::<f32, { Self::NR }>(rows, cols)
    }

    fn pack_b_block(
        &self,
        out: &mut [MaybeUninit<u8>],
        b: Matrix,
        rows: Range<usize>,
        cols: Range<usize>,
        _quant: Option<QuantParams<f32>>,
    ) {
        let out = cast_uninit_pod_mut_slice(out).unwrap();
        pack_b_block::<f32, { Self::NR }>(out, b, rows, cols);
    }

    fn pack_im2col(
        &self,
        out: &mut [MaybeUninit<u8>],
        image: &Im2Col<f32>,
        rows: Range<usize>,
        cols: Range<usize>,
        _zero_point: Option<f32>,
    ) {
        const NR_REGS: usize = GenericKernel::NR / X32_LANES;

        // Safety: Scalar "SIMD" types are always supported
        let out = cast_uninit_pod_mut_slice(out).unwrap();
        image.pack_block::<_, NR_REGS>(self.isa, out, Self::NR, rows, cols);
    }

    unsafe fn kernel(
        &self,
        tile_ptr: *mut f32,
        tile_row_stride: usize,
        a: Lhs<f32>,
        b: &[u8],
        used_rows: usize,
        used_cols: usize,
        depth: usize,
        alpha: f32,
        beta: f32,
        _a_quant: Option<QuantParams<f32>>,
        _b_quant: Option<QuantParams<f32>>,
    ) {
        const MR: usize = GenericKernel::MR;
        const NR: usize = GenericKernel::NR;
        const NR_REGS: usize = NR / 4;

        let b = cast_pod_slice(b).unwrap();
        let mut tmp_tile = TempTile::<f32, MR, NR>::new();
        let (dest_ptr, dest_row_stride, dest_beta) = if used_cols == NR {
            (tile_ptr, tile_row_stride, beta)
        } else {
            (tmp_tile.as_mut_ptr() as *mut f32, NR, 0.)
        };

        let gemm = GemmDispatch::<_, MR, NR_REGS>::new(
            self.isa,
            dest_ptr,
            dest_row_stride,
            a,
            b,
            depth,
            alpha,
            dest_beta,
        );

        match used_rows {
            8 => gemm.dispatch::<8>(),
            7 => gemm.dispatch::<7>(),
            6 => gemm.dispatch::<6>(),
            5 => gemm.dispatch::<5>(),
            4 => gemm.dispatch::<4>(),
            3 => gemm.dispatch::<3>(),
            2 => gemm.dispatch::<2>(),
            1 => gemm.dispatch::<1>(),
            _ => panic!("unsupported `used_rows` {}", used_rows),
        }

        if used_cols != NR {
            tmp_tile.accumulate_into(
                tile_ptr as *mut MaybeUninit<f32>,
                used_rows,
                used_cols,
                tile_row_stride,
                beta,
            );
        }
    }

    fn gemv_kernel(
        &self,
        out: MatVecOutput<f32>,
        a: &[f32],
        b: Matrix,
        alpha: f32,
        _a_quant: Option<QuantParams<f32>>,
        _b_quant: Option<QuantParams<f32>>,
    ) {
        simd_gemv::<_, 1>(self.isa, out, a, b, alpha);
    }
}

pub struct GenericInt8Kernel {
    isa: GenericIsa,
}

impl GenericInt8Kernel {
    const MR: usize = 4;
    const NR: usize = 4;
    const K_TILE: usize = 1;
}

unsafe impl Kernel<u8, i8, i32> for GenericInt8Kernel {
    fn new() -> Option<Self> {
        Some(GenericInt8Kernel {
            isa: GenericIsa::new(),
        })
    }

    fn mr(&self) -> usize {
        Self::MR
    }

    fn nr(&self) -> usize {
        Self::NR
    }

    fn name(&self) -> &'static str {
        "generic-u8i8i32"
    }

    fn packed_a_layout(
        &self,
        _a: Matrix<u8>,
        rows: usize,
        cols: usize,
        _quant: Option<QuantParams<u8>>,
    ) -> PackedLayout {
        let mut layout =
            packing::int8::packed_a_layout::<{ Self::MR }, { Self::K_TILE }>(rows, cols);
        layout.must_pack = true;
        layout
    }

    fn pack_a_block(
        &self,
        out: &mut [MaybeUninit<u8>],
        a: Matrix<u8>,
        rows: Range<usize>,
        cols: Range<usize>,
        quant: Option<QuantParams<u8>>,
    ) {
        let out = cast_uninit_pod_mut_slice(out).unwrap();
        packing::int8::pack_a::<{ Self::MR }, { Self::K_TILE }>(
            out,
            a.slice((rows.clone(), cols)),
            quant.map(|q| &q.zero_point[rows]),
        )
    }

    fn packed_b_layout(
        &self,
        rows: usize,
        cols: usize,
        _quant: Option<QuantParams<i8>>,
    ) -> PackedLayout {
        packing::int8::packed_b_layout::<{ Self::NR }, { Self::K_TILE }>(rows, cols)
    }

    fn pack_b_block(
        &self,
        out: &mut [MaybeUninit<u8>],
        b: Matrix<i8>,
        rows: Range<usize>,
        cols: Range<usize>,
        quant: Option<QuantParams<i8>>,
    ) {
        packing::int8::pack_b_cast_i8_u8::<{ Self::NR }, { Self::K_TILE }>(
            out,
            b.slice((rows, cols.clone())),
            quant.map(|q| &q.zero_point[cols]),
        )
    }

    fn pack_im2col(
        &self,
        out: &mut [MaybeUninit<u8>],
        image: &Im2Col<i8>,
        rows: Range<usize>,
        cols: Range<usize>,
        _zero_point: Option<i8>,
    ) {
        const NR_REGS: usize = GenericInt8Kernel::NR / 4;

        // Safety: Scalar "SIMD" types are always supported
        let out = cast_uninit_pod_mut_slice(out).unwrap();
        image.pack_block::<_, NR_REGS>(self.isa, out, Self::NR, rows, cols);
    }

    unsafe fn kernel(
        &self,
        tile_ptr: *mut i32,
        tile_row_stride: usize,
        a: Lhs<u8>,
        b: &[u8],
        used_rows: usize,
        used_cols: usize,
        depth: usize,
        _alpha: f32,
        beta: i32,
        _a_quant: Option<QuantParams<u8>>,
        _b_quant: Option<QuantParams<i8>>,
    ) {
        let a_data = match a {
            Lhs::Packed(data) => data,
            Lhs::Unpacked { .. } => panic!("lhs must be packed"),
        };

        let (a, a_meta) = packing::int8::extract_packed_a::<{ Self::MR }>(a_data);
        let (b, b_meta) = packing::int8::extract_packed_b::<{ Self::NR }>(b);

        const MR: usize = GenericInt8Kernel::MR;
        const NR: usize = GenericInt8Kernel::NR;

        // Zero accumulators. We use unsigned here for consistency with the data
        // type of packed elements in A and B.
        //
        // We could also make both the packed elements and accumulators signed.
        // What matters for efficiency is that the accumulators and packed
        // elements have the same sign.
        let mut tmp = [[0u32; NR]; MR];

        // Loop over K and for each step, add outer product of row tile of A and
        // column tile of B to accumulators.
        for k in 0..depth {
            let col: [u16; NR] = std::array::from_fn(|c| *b.get_unchecked(k * NR + c) as u16);

            for r in 0..MR {
                let row_elt = *a.get_unchecked(k * MR + r) as u16;
                for c in 0..NR {
                    tmp[r][c] += row_elt as u32 * col[c] as u32;
                }
            }
        }

        // Convert from u32 -> i32.
        let mut tmp = tmp.map(|row| row.map(|x| x as i32));

        // Adjust accumulators to reflect zero point.
        for r in 0..MR {
            for c in 0..NR {
                tmp[r][c] = tmp[r][c]
                    + depth as i32 * a_meta.zero_points[r] * b_meta.zero_points[c]
                    - a_meta.row_sums[r] * b_meta.zero_points[c]
                    - b_meta.col_sums[c] * a_meta.zero_points[r];
            }
        }

        // Write to output
        let out_ptr = |row, col| tile_ptr.add(row * tile_row_stride + col);
        let accumulate = beta != 0;
        if used_rows == MR && used_cols == NR {
            if accumulate {
                for r in 0..MR {
                    for c in 0..NR {
                        *out_ptr(r, c) += tmp[r][c];
                    }
                }
            } else {
                for r in 0..MR {
                    for c in 0..NR {
                        *out_ptr(r, c) = tmp[r][c];
                    }
                }
            }
        } else {
            if accumulate {
                for r in 0..used_rows {
                    for c in 0..used_cols {
                        *out_ptr(r, c) += tmp[r][c];
                    }
                }
            } else {
                for r in 0..used_rows {
                    for c in 0..used_cols {
                        *out_ptr(r, c) = tmp[r][c];
                    }
                }
            }
        }
    }

    fn gemv_kernel(
        &self,
        out: MatVecOutput<i32>,
        a: &[u8],
        b: Matrix<i8>,
        alpha: f32,
        a_quant: Option<QuantParams<u8>>,
        b_quant: Option<QuantParams<i8>>,
    ) {
        int8_gemv(out, a, b, alpha, a_quant, b_quant)
    }
}

/// Generic implementation of [`Kernel::gemv`] for u8 x i8 -> i32 kernels.
pub fn int8_gemv(
    out: MatVecOutput<i32>,
    a: &[u8],
    b: Matrix<i8>,
    alpha: f32,
    a_quant: Option<QuantParams<u8>>,
    b_quant: Option<QuantParams<i8>>,
) {
    assert!(out.beta == 0 || out.beta == 1);
    assert_eq!(alpha, 1.);
    assert_eq!(b.rows(), a.len());
    assert_eq!(out.data.len(), b.cols());

    let a_zero = a_quant.map(|aq| aq.zero_point[0] as i32).unwrap_or(0);
    let depth = a.len();

    for (out_el, col) in out.data.iter_mut().zip(0..b.cols()) {
        let b_zero = b_quant.map(|bq| bq.zero_point[col] as i32).unwrap_or(0);
        let mut acc = 0;
        let mut row_sum = 0;
        let mut col_sum = 0;

        for k in 0..depth {
            let a_el = unsafe { *a.get_unchecked(k) } as i32;
            let b_el = unsafe { *b.get_unchecked([k, col]) } as i32;
            acc += a_el * b_el;
            row_sum += a_el;
            col_sum += b_el;
        }

        // Subtract zero points. This is equivalent to doing
        // `acc += (a - a_zero) * (b - b_zero)` in the loop over K, but more
        // efficient.
        acc = depth as i32 * a_zero * b_zero + acc - row_sum * b_zero - col_sum * a_zero;

        if out.beta == 0 {
            out_el.write(acc);
        } else {
            // Safety: Output is initialized when beta is non-zero
            unsafe {
                out_el.write(out_el.assume_init() + acc);
            }
        }
    }
}
