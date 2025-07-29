use std::arch::aarch64::{int32x4_t, int8x16_t};
use std::mem::MaybeUninit;
use std::ops::Range;

use rten_base::byte_cast::{cast_pod_slice, cast_uninit_pod_mut_slice};
use rten_simd::isa::ArmNeonIsa;
use rten_tensor::{Matrix, MatrixLayout};

use super::simd_generic::{simd_gemv, simd_int8_gemm, simd_int8_gemv, GemmDispatch};
use super::{Int8DotProduct, Kernel, Lhs, MatVecOutput, PackedLayout, QuantParams, TempTile};
use crate::packing::{pack_a_block, pack_b_block, packed_a_layout, packed_b_layout};
use crate::{packing, Im2Col};

pub struct ArmNeonKernel {
    isa: ArmNeonIsa,
}

impl ArmNeonKernel {
    const MR: usize = 4;
    const NR: usize = 16;
}

/// Number of 32-bit lanes in an Arm Neon SIMD vector.
const X32_LANES: usize = 4;

/// Size of K tiles in kernels using int 8 dot product instructions.
const I8DOT_K_TILE: usize = 4;

/// Size of K tiles in kernels using i8mm instructions.
const I8MM_K_TILE: usize = 8;

// Safety - We assume that Rust code on Arm is always compiled with Arm Neon
// available.
unsafe impl Kernel<f32, f32, f32> for ArmNeonKernel {
    fn new() -> Option<Self> {
        ArmNeonIsa::new().map(|isa| ArmNeonKernel { isa })
    }

    fn name(&self) -> &'static str {
        "aarch64-f32-neon"
    }

    fn mr(&self) -> usize {
        Self::MR
    }

    fn nr(&self) -> usize {
        Self::NR
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
        let out = cast_uninit_pod_mut_slice(out).expect("incorrect alignment for packing buffer");
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
        let out = cast_uninit_pod_mut_slice(out).expect("incorrect alignment for packing buffer");
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
        const NR_REGS: usize = ArmNeonKernel::NR / X32_LANES;

        // Safety: Arm Neon instructions are supported
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
        const MR: usize = ArmNeonKernel::MR;
        const NR: usize = ArmNeonKernel::NR;
        const NR_REGS: usize = NR / X32_LANES;

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

        debug_assert_eq!(MR, 4);
        match used_rows {
            4 => gemm.dispatch_broadcast_lane::<4>(),
            3 => gemm.dispatch_broadcast_lane::<3>(),
            2 => gemm.dispatch_broadcast_lane::<2>(),
            1 => gemm.dispatch_broadcast_lane::<1>(),
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
        simd_gemv::<_, 4>(self.isa, out, a, b, alpha);
    }
}

macro_rules! impl_arm_int8_common {
    ($self_type:ty) => {
        fn mr(&self) -> usize {
            Self::MR
        }

        fn nr(&self) -> usize {
            Self::NR
        }

        fn im2col_row_count_step(&self) -> usize {
            4
        }

        fn packed_a_layout(
            &self,
            _a: Matrix<u8>,
            rows: usize,
            cols: usize,
            _quant: Option<QuantParams<u8>>,
        ) -> PackedLayout {
            let mut layout =
                packing::int8::packed_a_layout::<{ Self::MR }, I8DOT_K_TILE>(rows, cols);
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
            packing::int8::pack_a::<{ Self::MR }, I8DOT_K_TILE>(
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
            packing::int8::packed_b_layout::<{ Self::NR }, I8DOT_K_TILE>(rows, cols)
        }

        fn pack_b_block(
            &self,
            out: &mut [MaybeUninit<u8>],
            b: Matrix<i8>,
            rows: Range<usize>,
            cols: Range<usize>,
            quant: Option<QuantParams<i8>>,
        ) {
            packing::int8::pack_b_cast_i8_u8::<{ Self::NR }, I8DOT_K_TILE>(
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
            zero_point: Option<i8>,
        ) {
            // Safety: Arm Neon is supported
            const NR: usize = <$self_type>::NR;
            const NR_REGS: usize = NR / X32_LANES;
            image.pack_block_i8_dot_cast_u8::<_, NR, NR_REGS, I8DOT_K_TILE>(
                self.isa,
                out,
                rows,
                cols,
                zero_point.unwrap_or_default(),
            )
        }
    };
}

/// 8-bit integer matrix multiplication kernel using Arm dot product
/// instructions.
pub struct ArmInt8DotKernel {
    isa: ArmNeonIsa,
    dot_isa: NeonNativeDotProd,
}

impl ArmInt8DotKernel {
    const MR: usize = 4;
    const NR: usize = 16;
}

unsafe impl Kernel<u8, i8, i32> for ArmInt8DotKernel {
    fn new() -> Option<Self> {
        let isa = ArmNeonIsa::new()?;
        let dot_isa = NeonNativeDotProd::new()?;
        Some(ArmInt8DotKernel { isa, dot_isa })
    }

    fn name(&self) -> &'static str {
        "aarch64-u8i8i32-dotprod"
    }

    impl_arm_int8_common!(ArmInt8DotKernel);

    #[target_feature(enable = "dotprod")]
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
        let (a_data, a_meta) = packing::int8::extract_packed_a::<{ Self::MR }>(a_data);
        let (b_data, b_meta) = packing::int8::extract_packed_b::<{ Self::NR }>(b);

        const NR_REGS: usize = ArmInt8DotKernel::NR / X32_LANES;
        simd_int8_gemm::<_, _, { Self::MR }, { Self::NR }, NR_REGS>(
            self.isa,
            tile_ptr,
            tile_row_stride,
            a_data,
            b_data,
            used_rows,
            used_cols,
            depth,
            beta != 0, // accumulate
            a_meta.zero_points,
            b_meta.zero_points,
            &a_meta.row_sums,
            &b_meta.col_sums,
            self.dot_isa,
        )
    }

    fn gemv_kernel(
        &self,
        mut out: MatVecOutput<i32>,
        a: &[u8],
        b: Matrix<i8>,
        _alpha: f32,
        a_quant: Option<QuantParams<u8>>,
        b_quant: Option<QuantParams<i8>>,
    ) {
        let a_zero = a_quant.map(|aq| aq.zero_point[0]).unwrap_or(0);
        let b_zero = b_quant.map(|bq| bq.zero_point);
        let out = out.as_bool_beta();
        simd_int8_gemv::<_, true /* CAST_B_U8 */>(self.isa, out, a, b, a_zero, b_zero, self.dot_isa)
    }
}

/// 8-bit integer matrix multiplication kernel for Arm CPUs using UMLAL/SMLAL
/// instructions.
pub struct ArmInt8MlalKernel {
    isa: ArmNeonIsa,
}

impl ArmInt8MlalKernel {
    const MR: usize = 8;
    const NR: usize = 8;

    // The int8 packing methods pack data with layout `[K / K_TILE, MR, K_TILE]`
    // for the LHS or `[K / K_TILE, NR, K_TILE]` for the RHS. For this kernel we
    // want the packed data to have layout `[K, MR]` and `[K, NR]` respectively.
    // Hence we set K_TILE=1.
    const K_TILE: usize = 1;
}

unsafe impl Kernel<u8, i8, i32> for ArmInt8MlalKernel {
    fn new() -> Option<Self> {
        ArmNeonIsa::new().map(|isa| Self { isa })
    }

    fn name(&self) -> &'static str {
        "aarch64-u8i8i32-mlal"
    }

    fn mr(&self) -> usize {
        Self::MR
    }

    fn nr(&self) -> usize {
        Self::NR
    }

    fn im2col_row_count_step(&self) -> usize {
        Self::K_TILE
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
        zero_point: Option<i8>,
    ) {
        const NR: usize = ArmInt8MMKernel::NR;
        const NR_REGS: usize = NR / X32_LANES;
        image.pack_block_i8_dot_cast_u8::<_, NR, NR_REGS, { Self::K_TILE }>(
            self.isa,
            out,
            rows,
            cols,
            zero_point.unwrap_or_default(),
        )
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
        use rten_simd::{
            ops::{Extend, NumOps},
            Isa, Simd,
        };
        use std::arch::aarch64::{vget_low_u16, vmlal_high_laneq_u16, vmlal_laneq_u16};

        let a_data = match a {
            Lhs::Packed(data) => data,
            Lhs::Unpacked { .. } => panic!("lhs must be packed"),
        };

        let (a_data, a_meta) = packing::int8::extract_packed_a::<{ Self::MR }>(a_data);
        let (b, b_meta) = packing::int8::extract_packed_b::<{ Self::NR }>(b);

        let accumulate = beta != 0;

        const MR: usize = ArmInt8MlalKernel::MR;
        const NR: usize = ArmInt8MlalKernel::NR;
        const NR_REGS: usize = ArmInt8MlalKernel::NR / X32_LANES;

        let ops = self.isa.i32();
        let u8_ops = self.isa.u8();

        type I32 = <ArmNeonIsa as Isa>::I32;
        type U32 = <ArmNeonIsa as Isa>::U32;

        let mut tmp: [[U32; NR_REGS]; MR] =
            std::array::from_fn(|_| std::array::from_fn(|_| ops.zero().reinterpret_cast()));

        // Compute `tmp[row][0..2] += a[row] * b` where `a` and `b` are `uint16x8_t`s.
        macro_rules! compute_row {
            ($row:literal, $a:ident, $b:ident) => {
                tmp[$row][0] = vmlal_laneq_u16::<$row>(tmp[$row][0], vget_low_u16($b), $a);
                tmp[$row][1] = vmlal_high_laneq_u16::<$row>(tmp[$row][1], $b, $a);
            };
        }

        // Loop over K in steps of 2. Each iteration loads one vector of 8-bit
        // elements from A and B and extends each to two vectors of 16-bit
        // elements. The outer product of each matching pair of A and B vectors
        // is computed and added to the accumulator registers.
        let depth_blocks = depth / 2;
        let depth_remainder = depth % 2;

        for k_block in 0..depth_blocks {
            let a_vec = u8_ops.load_ptr(a_data.as_ptr().add(k_block * u8_ops.len()));
            let (a_lo, a_hi) = u8_ops.extend(a_vec);

            let b_vec = u8_ops.load_ptr(b.as_ptr().add(k_block * u8_ops.len()));
            let (b_lo, b_hi) = u8_ops.extend(b_vec);

            // Compute first outer product update of this block.
            compute_row!(0, a_lo, b_lo);
            compute_row!(1, a_lo, b_lo);
            compute_row!(2, a_lo, b_lo);
            compute_row!(3, a_lo, b_lo);
            compute_row!(4, a_lo, b_lo);
            compute_row!(5, a_lo, b_lo);
            compute_row!(6, a_lo, b_lo);
            compute_row!(7, a_lo, b_lo);

            // Compute second outer product update of this block.
            compute_row!(0, a_hi, b_hi);
            compute_row!(1, a_hi, b_hi);
            compute_row!(2, a_hi, b_hi);
            compute_row!(3, a_hi, b_hi);
            compute_row!(4, a_hi, b_hi);
            compute_row!(5, a_hi, b_hi);
            compute_row!(6, a_hi, b_hi);
            compute_row!(7, a_hi, b_hi);
        }

        // Final update if K is odd.
        //
        // This loads vectors from a padded array and discards the high half.
        if depth_remainder != 0 {
            let a_buf: [u8; 16] = std::array::from_fn(|i| {
                if i < MR {
                    *a_data.as_ptr().add(depth_blocks * u8_ops.len() + i)
                } else {
                    0
                }
            });
            let b_buf: [u8; 16] = std::array::from_fn(|i| {
                if i < NR {
                    *b.as_ptr().add(depth_blocks * u8_ops.len() + i)
                } else {
                    0
                }
            });

            let a_vec = u8_ops.load_ptr(a_buf.as_ptr());
            let (a_lo, _a_hi) = u8_ops.extend(a_vec);

            let b_vec = u8_ops.load_ptr(b_buf.as_ptr());
            let (b_lo, _b_hi) = u8_ops.extend(b_vec);

            compute_row!(0, a_lo, b_lo);
            compute_row!(1, a_lo, b_lo);
            compute_row!(2, a_lo, b_lo);
            compute_row!(3, a_lo, b_lo);
            compute_row!(4, a_lo, b_lo);
            compute_row!(5, a_lo, b_lo);
            compute_row!(6, a_lo, b_lo);
            compute_row!(7, a_lo, b_lo);
        }

        // Cast accumulator from U32 to I32
        let mut tmp: [[I32; NR_REGS]; MR] =
            std::mem::transmute::<[[U32; NR_REGS]; MR], [[I32; NR_REGS]; MR]>(tmp);

        // Add `k * a_zero_point[row] * b_zero_point[col]`
        let b_zero = ops.load_many::<NR_REGS>(&b_meta.zero_points);
        let k_mul_b_zero: [I32; NR_REGS] =
            std::array::from_fn(|i| ops.mul(ops.splat(depth as i32), b_zero[i]));
        for row in 0..MR {
            let a_zero = ops.splat(a_meta.zero_points[row]);
            for i in 0..NR_REGS {
                tmp[row][i] = ops.mul_add(k_mul_b_zero[i], a_zero, tmp[row][i]);
            }
        }

        // Scale zero points by row and column sums and subtract from output tile.
        let b_col_sums: [I32; NR_REGS] =
            std::array::from_fn(|i| ops.load_ptr(b_meta.col_sums.as_ptr().add(i * ops.len())));
        for row in 0..MR {
            let a_zero = ops.splat(a_meta.zero_points[row]);
            let a_sum = ops.splat(a_meta.row_sums[row]);

            for i in 0..NR_REGS {
                let a_sum_mul_b_zero = ops.mul(a_sum, b_zero[i]);
                let b_sum_mul_a_zero = ops.mul(b_col_sums[i], a_zero);
                let sum = ops.add(a_sum_mul_b_zero, b_sum_mul_a_zero);
                tmp[row][i] = ops.sub(tmp[row][i], sum);
            }
        }

        // Write from accumulator in registers back to output.
        let output_tile_ptr =
            |row, col_block| tile_ptr.add(row * tile_row_stride + col_block * ops.len());

        if used_rows == MR && used_cols == NR {
            // Full output tile
            for row in 0..MR {
                for c_block in 0..NR_REGS {
                    let tile_ptr = output_tile_ptr(row, c_block);
                    if accumulate {
                        tmp[row][c_block] = ops.add(ops.load_ptr(tile_ptr), tmp[row][c_block]);
                    }
                    ops.store_ptr(tmp[row][c_block], tile_ptr);
                }
            }
        } else {
            // Partial output tile
            for r in 0..used_rows {
                for c_block in 0..NR_REGS {
                    let tile_ptr = output_tile_ptr(r, c_block);
                    let used_cols = used_cols.saturating_sub(c_block * ops.len()).min(ops.len());
                    let mut tmp = tmp[r][c_block].to_array();

                    for c in 0..used_cols {
                        if accumulate {
                            tmp[c] += *tile_ptr.add(c);
                        }
                        tile_ptr.add(c).write(tmp[c]);
                    }
                }
            }
        }
    }

    fn gemv_kernel(
        &self,
        mut out: MatVecOutput<i32>,
        a: &[u8],
        b: Matrix<i8>,
        _alpha: f32,
        a_quant: Option<QuantParams<u8>>,
        b_quant: Option<QuantParams<i8>>,
    ) {
        let a_zero = a_quant.map(|aq| aq.zero_point[0]).unwrap_or(0);
        let b_zero = b_quant.map(|bq| bq.zero_point);
        let out = out.as_bool_beta();
        simd_int8_gemv::<_, true /* CAST_B_U8 */>(
            self.isa,
            out,
            a,
            b,
            a_zero,
            b_zero,
            NeonDotProd {},
        )
    }
}

/// Implementation of int8 dot product using the native UDOT / SDOT instructions.
#[derive(Copy, Clone)]
struct NeonNativeDotProd {
    _private: (),
}

impl NeonNativeDotProd {
    fn new() -> Option<Self> {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return None;
        }
        Some(NeonNativeDotProd { _private: () })
    }
}

// Safety: Constructor checked for "dotprod" feature support.
unsafe impl Int8DotProduct for NeonNativeDotProd {
    type X8 = int8x16_t;
    type I32 = int32x4_t;

    #[inline]
    fn supports_indexed_dot_product() -> bool {
        true
    }

    #[inline]
    fn dot_product(self, a: Self::X8, b: Self::X8, c: Self::I32) -> Self::I32 {
        #[target_feature(enable = "dotprod")]
        #[inline]
        unsafe fn dot_product(a: int8x16_t, b: int8x16_t, c: int32x4_t) -> int32x4_t {
            use core::arch::aarch64::{
                vreinterpretq_s32_u32, vreinterpretq_u32_s32, vreinterpretq_u8_s8,
            };
            use core::arch::asm;

            let a = vreinterpretq_u8_s8(a);
            let b = vreinterpretq_u8_s8(b);
            let mut c = vreinterpretq_u32_s32(c);

            // Use inline asm here because the `vdotq_u32` intrinsic is not
            // stabilized yet.
            asm! {
                "udot {result:v}.4s, {a:v}.16b, {b:v}.16b",
                result = inout(vreg) c,
                a = in(vreg) a,
                b = in(vreg) b,
                options(nostack)
            }

            vreinterpretq_s32_u32(c)
        }
        unsafe { dot_product(a, b, c) }
    }

    #[inline]
    fn indexed_dot_product<const IDX: u32>(
        self,
        a: Self::X8,
        b: Self::X8,
        c: Self::I32,
    ) -> Self::I32 {
        #[target_feature(enable = "dotprod")]
        #[inline]
        unsafe fn dot_product<const IDX: u32>(
            a: int8x16_t,
            b: int8x16_t,
            c: int32x4_t,
        ) -> int32x4_t {
            use core::arch::aarch64::{
                vreinterpretq_s32_u32, vreinterpretq_u32_s32, vreinterpretq_u8_s8,
            };
            use core::arch::asm;

            let a = vreinterpretq_u8_s8(a);
            let b = vreinterpretq_u8_s8(b);
            let mut c = vreinterpretq_u32_s32(c);

            // Use inline asm here because the `vdotq_laneq_u32` intrinsic is not
            // stabilized yet.
            asm! {
                "udot {result:v}.4s, {a:v}.16b, {b:v}.4B[{idx}]",
                result = inout(vreg) c,
                a = in(vreg) a,
                b = in(vreg) b,
                idx = const IDX,
                options(nostack)
            }

            vreinterpretq_s32_u32(c)
        }
        unsafe { dot_product::<IDX>(a, b, c) }
    }
}

/// Fallback implementation of u8 dot product for older Arm CPUs which don't
/// support dot product instructions (UDOT, SDOT).
#[derive(Copy, Clone)]
struct NeonDotProd;

// Safety: Neon instructions used are always supported on aarch64.
unsafe impl Int8DotProduct for NeonDotProd {
    type X8 = int8x16_t;
    type I32 = int32x4_t;

    #[inline]
    fn dot_product(self, a: Self::X8, b: Self::X8, c: Self::I32) -> Self::I32 {
        unsafe {
            use core::arch::aarch64::{
                vaddq_u32, vget_low_u8, vmull_high_u8, vmull_u8, vpaddlq_u16, vpaddq_u32,
                vreinterpretq_s32_u32, vreinterpretq_u32_s32, vreinterpretq_u8_s8,
            };

            let a = vreinterpretq_u8_s8(a);
            let b = vreinterpretq_u8_s8(b);
            let c = vreinterpretq_u32_s32(c);

            let mul_lo = vmull_u8(vget_low_u8(a), vget_low_u8(b));
            let mul_hi = vmull_high_u8(a, b);
            let tmp_lo = vpaddlq_u16(mul_lo);
            let tmp_hi = vpaddlq_u16(mul_hi);
            let dot_prod = vpaddq_u32(tmp_lo, tmp_hi);
            vreinterpretq_s32_u32(vaddq_u32(dot_prod, c))
        }
    }
}

/// 8-bit integer matrix multiplication kernel using Arm i8mm instructions.
pub struct ArmInt8MMKernel {
    dot_kernel: ArmInt8DotKernel,
}

impl ArmInt8MMKernel {
    const MR: usize = 8;
    const NR: usize = 8;
}

// Safety: We check for i8mm support in constructors.
unsafe impl Kernel<u8, i8, i32> for ArmInt8MMKernel {
    fn new() -> Option<Self> {
        // If a CPU supports i8mm, it should also support dotprod. For GEMV
        // operations this kernel delegates to the dotprod kernel.
        let dot_kernel = ArmInt8DotKernel::new()?;

        if !std::arch::is_aarch64_feature_detected!("i8mm") {
            return None;
        }

        Some(ArmInt8MMKernel { dot_kernel })
    }

    fn name(&self) -> &'static str {
        "aarch64-u8i8i32-i8mm"
    }

    fn mr(&self) -> usize {
        Self::MR
    }

    fn nr(&self) -> usize {
        Self::NR
    }

    fn packed_a_layout(
        &self,
        _a: Matrix<u8>,
        rows: usize,
        cols: usize,
        _quant: Option<QuantParams<u8>>,
    ) -> PackedLayout {
        let mut layout = packing::int8::packed_a_layout::<{ Self::MR }, I8MM_K_TILE>(rows, cols);
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
        packing::int8::pack_a::<{ Self::MR }, I8MM_K_TILE>(
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
        packing::int8::packed_b_layout::<{ Self::NR }, I8MM_K_TILE>(rows, cols)
    }

    fn pack_b_block(
        &self,
        out: &mut [MaybeUninit<u8>],
        b: Matrix<i8>,
        rows: Range<usize>,
        cols: Range<usize>,
        quant: Option<QuantParams<i8>>,
    ) {
        packing::int8::pack_b_cast_i8_u8::<{ Self::NR }, I8MM_K_TILE>(
            out,
            b.slice((rows, cols.clone())),
            quant.map(|q| &q.zero_point[cols]),
        )
    }

    fn im2col_row_count_step(&self) -> usize {
        I8MM_K_TILE
    }

    fn pack_im2col(
        &self,
        out: &mut [MaybeUninit<u8>],
        image: &Im2Col<i8>,
        rows: Range<usize>,
        cols: Range<usize>,
        zero_point: Option<i8>,
    ) {
        const NR: usize = ArmInt8MMKernel::NR;
        const NR_REGS: usize = NR / X32_LANES;
        image.pack_block_i8_dot_cast_u8::<_, NR, NR_REGS, I8MM_K_TILE>(
            self.dot_kernel.isa,
            out,
            rows,
            cols,
            zero_point.unwrap_or_default(),
        )
    }

    #[target_feature(enable = "i8mm")]
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
        use rten_simd::{
            ops::{Concat, NumOps},
            Isa, Simd,
        };

        const MR: usize = ArmInt8MMKernel::MR;
        const NR: usize = ArmInt8MMKernel::NR;
        const NR_REGS: usize = ArmInt8MMKernel::NR / X32_LANES;

        // Each i8mm instruction accumulates into a 2x2 accumulator in a single
        // register. eg. For MR=8, NR=8 we have 4x4 tiles.
        const ROW_TILES: usize = MR / 2;
        const COL_TILES: usize = NR / 2;
        const K_TILE: usize = 8;

        let a = match a {
            Lhs::Packed(data) => data,
            Lhs::Unpacked { .. } => panic!("lhs must be packed"),
        };
        let accumulate = beta != 0;

        let (a, a_meta) = packing::int8::extract_packed_a::<MR>(a);
        let a_zero_points = a_meta.zero_points;
        let a_row_sums = a_meta.row_sums;

        let (b, b_meta) = packing::int8::extract_packed_b::<NR>(b);
        let b_zero_points = b_meta.zero_points;
        let b_col_sums = b_meta.col_sums;

        let ops = self.dot_kernel.isa.i32();
        let u8_ops = self.dot_kernel.isa.u8();

        // Packed buffers contain 2x8 microtiles of LHS and 8x2 microtiles of RHS.
        assert_eq!(a.len(), Self::MR * depth.next_multiple_of(K_TILE));
        assert_eq!(b.len(), Self::NR * depth.next_multiple_of(K_TILE));

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let n_depth_tiles = depth.div_ceil(K_TILE);
        let b_zero = ops.load_many::<NR_REGS>(&b_zero_points);

        type I32 = <ArmNeonIsa as Isa>::I32;
        type U8 = <ArmNeonIsa as Isa>::U8;

        let a_tile_size = ROW_TILES * u8_ops.len();
        let b_tile_size = COL_TILES * u8_ops.len();

        // The value for each element in the output tile is computed as:
        //
        // c = (a[0] - a_zero_point) * (b[0] - b_zero_point) + ...
        //
        // (or `c += ...` when beta=1)
        //
        // Where `a_zero_point` is the zero point for the row of A and
        // `b_zero_point` is the zero point for the column of B.
        //
        // This can be expanded and re-arranged into:
        //
        // c = a[0]b[0] - a[0] * b_zero_point - b[0] * a_zero_point + a_zero_point * b_zero_point + ...
        // c = dot(a, b) - sum(a) * b_zero_point - sum(b) * a_zero_point + k * a_zero_point * b_zero_point
        // c = dot(a, b) + k * a_zero_point * b_zero_point - sum(a) * b_zero_point - sum(b) * a_zero_point
        //
        // We compute `dot(a, b)` first, then add the zero point adjustment.
        let mut tmp: [[I32; COL_TILES]; ROW_TILES] =
            std::array::from_fn(|_| std::array::from_fn(|_| ops.zero()));

        // Loop over K dimension.
        //
        // Each iteration loads ROW_TILES 2x8 tiles of A, COL_TILES 8x2 tiles of
        // B and performs ROW_TILES * COL_TILES mini matmuls (2x8 @ 8x2 => 2x2),
        // collectively updating an MR x NR x i32 output tile in registers.
        for k_block in 0..n_depth_tiles {
            let col_tiles: [U8; COL_TILES] = std::array::from_fn(|c| {
                u8_ops.load_ptr(b_ptr.add(k_block * b_tile_size + c * u8_ops.len()))
            });
            let row_tiles: [U8; ROW_TILES] = std::array::from_fn(|r| {
                u8_ops.load_ptr(a_ptr.add(k_block * a_tile_size + r * u8_ops.len()))
            });

            for r in 0..ROW_TILES {
                for c in 0..COL_TILES {
                    // Use inline asm here because the `vmmlaq_u32` intrinsic is
                    // not stabilized yet.
                    core::arch::asm! {
                        "ummla {result:v}.4s, {a:v}.16b, {b:v}.16b",
                        result = inout(vreg) tmp[r][c],
                        a = in(vreg) row_tiles[r],
                        b = in(vreg) col_tiles[c],
                        options(nostack)
                    }
                }
            }
        }

        // The accumulator is arranged as a ROW_TILES x COL_TILES grid:
        //
        // A0 A1 B0 B1 C0 C1 D0 D1
        // A2 A3 B2 B3 C2 C3 D2 D3
        // ...
        //
        // Where each letter denotes a register. We want to re-arrange these to
        // a row-major layout that is more convenient for adding the zero point
        // adjustment and writing back out to memory:
        //
        // [A0 A1 B0 B1] [C0 C1 D0 D1]
        // [A2 A3 B2 B3] [C2 C3 D2 D3]
        //
        // To do this we alternate combining the low and high halves of
        // registers from each row tile.

        let mut tmp: [[I32; NR_REGS]; MR] = std::array::from_fn(|r| {
            std::array::from_fn(|c| {
                let src_a = tmp[r / 2][c * 2];
                let src_b = tmp[r / 2][c * 2 + 1];
                if r % 2 == 0 {
                    ops.concat_low(src_a, src_b)
                } else {
                    ops.concat_high(src_a, src_b)
                }
            })
        });

        // Add `k * a_zero_point[row] * b_zero_point[col]`
        let k_mul_b_zero: [I32; NR_REGS] =
            std::array::from_fn(|i| ops.mul(ops.splat(depth as i32), b_zero[i]));
        for row in 0..MR {
            let a_zero = ops.splat(a_zero_points[row]);
            for i in 0..NR_REGS {
                tmp[row][i] = ops.mul_add(k_mul_b_zero[i], a_zero, tmp[row][i]);
            }
        }

        // Scale zero points by row and column sums and subtract from output tile.
        let b_col_sums: [I32; NR_REGS] =
            std::array::from_fn(|i| ops.load_ptr(b_col_sums.as_ptr().add(i * ops.len())));
        for row in 0..MR {
            let a_zero = ops.splat(a_zero_points[row]);
            let a_sum = ops.splat(a_row_sums[row]);

            for i in 0..NR_REGS {
                let a_sum_mul_b_zero = ops.mul(a_sum, b_zero[i]);
                let b_sum_mul_a_zero = ops.mul(b_col_sums[i], a_zero);
                let sum = ops.add(a_sum_mul_b_zero, b_sum_mul_a_zero);
                tmp[row][i] = ops.sub(tmp[row][i], sum);
            }
        }

        // Write from accumulator in registers back to output.
        let output_tile_ptr =
            |row, col_block| tile_ptr.add(row * tile_row_stride + col_block * ops.len());

        if used_rows == MR && used_cols == NR {
            // Full output tile
            for row in 0..MR {
                for c_block in 0..NR_REGS {
                    let tile_ptr = output_tile_ptr(row, c_block);
                    if accumulate {
                        tmp[row][c_block] = ops.add(ops.load_ptr(tile_ptr), tmp[row][c_block]);
                    }
                    ops.store_ptr(tmp[row][c_block], tile_ptr);
                }
            }
        } else {
            // Partial output tile
            for r in 0..used_rows {
                for c_block in 0..NR_REGS {
                    let tile_ptr = output_tile_ptr(r, c_block);
                    let used_cols = used_cols.saturating_sub(c_block * ops.len()).min(ops.len());
                    let mut tmp = tmp[r][c_block].to_array();

                    for c in 0..used_cols {
                        if accumulate {
                            tmp[c] += *tile_ptr.add(c);
                        }
                        tile_ptr.add(c).write(tmp[c]);
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
        self.dot_kernel
            .gemv_kernel(out, a, b, alpha, a_quant, b_quant)
    }
}
