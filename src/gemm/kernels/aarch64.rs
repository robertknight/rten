use std::arch::aarch64::float32x4_t;

use rten_tensor::Matrix;

use super::{simd_gemv, Kernel};
use crate::iter_util::unroll_loop;

#[derive(Default)]
pub struct ArmNeonKernel {}

impl Kernel for ArmNeonKernel {
    const MR: usize = 8;
    const NR: usize = 8;

    fn name() -> &'static str {
        "arm-neon"
    }

    fn supported() -> bool {
        true
    }

    unsafe fn kernel(
        tile_ptr: *mut f32,
        tile_row_stride: usize,
        a: &[f32],
        b: &[f32],
        depth: usize,
        alpha: f32,
        beta: f32,
    ) {
        use std::arch::aarch64::{
            vaddq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vmulq_f32, vst1q_f32,
        };

        const MR: usize = ArmNeonKernel::MR;
        const NR: usize = ArmNeonKernel::NR;
        const REG_SIZE: usize = 4;
        const NR_REGS: usize = NR / REG_SIZE;

        assert!(a.len() >= depth * MR);
        assert!(b.len() >= depth * NR);

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        // Outer product accumulation tile that fits in registers.
        let mut tmp = [[vdupq_n_f32(0.); NR_REGS]; MR];
        let mut b_rows = [vdupq_n_f32(0.); NR_REGS];

        unroll_loop!(0..depth, k, 8, {
            let a_off = k * MR;
            let b_off = k * NR;

            for i in 0..NR_REGS {
                b_rows[i] = vld1q_f32(b_ptr.add(b_off + i * REG_SIZE));
            }

            for i in 0..MR {
                let a_val = *a_ptr.add(a_off + i);
                let a_broadcast = vdupq_n_f32(a_val);

                for j in 0..NR_REGS {
                    tmp[i][j] = vfmaq_f32(tmp[i][j], a_broadcast, b_rows[j]);
                }
            }
        });

        if beta == 0. && alpha == 1. {
            for i in 0..MR {
                for j in 0..NR_REGS {
                    let out_ptr = tile_ptr.add(tile_row_stride * i + j * REG_SIZE);
                    vst1q_f32(out_ptr, tmp[i][j]);
                }
            }
        } else if beta == 1. && alpha == 1. {
            for i in 0..MR {
                for j in 0..NR_REGS {
                    let out_ptr = tile_ptr.add(tile_row_stride * i + j * REG_SIZE);
                    let out_val = vaddq_f32(vld1q_f32(out_ptr), tmp[i][j]);
                    vst1q_f32(out_ptr, out_val);
                }
            }
        } else {
            let alpha_broadcast = vdupq_n_f32(alpha);
            let beta_broadcast = vdupq_n_f32(beta);
            for i in 0..MR {
                for j in 0..NR_REGS {
                    let out_ptr = tile_ptr.add(tile_row_stride * i + j * REG_SIZE);
                    let out_val = vmulq_f32(vld1q_f32(out_ptr), beta_broadcast);
                    let out_val = vfmaq_f32(out_val, tmp[i][j], alpha_broadcast);
                    vst1q_f32(out_ptr, out_val);
                }
            }
        }
    }

    unsafe fn gemv_kernel(out: &mut [f32], a: &[f32], b: Matrix, alpha: f32, beta: f32) {
        simd_gemv::<float32x4_t, 2>(out, a, b, alpha, beta);
    }
}

super::impl_gemmops!(ArmNeonKernel);
