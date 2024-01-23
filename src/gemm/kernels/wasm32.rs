use super::Kernel;

use crate::iter_util::unroll_loop;

#[derive(Default)]
pub struct WasmSimdKernel {}

impl Kernel for WasmSimdKernel {
    const MR: usize = 8;
    const NR: usize = 8;

    fn name() -> &'static str {
        "wasm-simd"
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
        use std::arch::wasm32::{
            f32x4, f32x4_add, f32x4_mul, v128, v128_load, v128_load32_splat,
            v128_store
        };

        const MR: usize = WasmSimdKernel::MR;
        const NR: usize = WasmSimdKernel::NR;
        const REG_SIZE: usize = 4;
        const NR_REGS: usize = NR / REG_SIZE;

        assert!(a.len() >= depth * MR);
        assert!(b.len() >= depth * NR);

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        // Outer product accumulation tile that fits in registers.
        let mut tmp = [[f32x4(0., 0., 0., 0.); NR_REGS]; MR];
        let mut b_rows = [f32x4(0., 0., 0., 0.); NR_REGS];

        unroll_loop!(depth, k, 8, {
            let a_off = k * MR;
            let b_off = k * NR;

            for i in 0..NR_REGS {
                b_rows[i] = v128_load(b_ptr.add(b_off + i * REG_SIZE) as *const v128);
            }

            for i in 0..MR {
                let a_broadcast = v128_load32_splat(a_ptr.add(a_off + i) as *const u32);

                for j in 0..NR_REGS {
                    tmp[i][j] = f32x4_add(tmp[i][j], f32x4_mul(a_broadcast, b_rows[j]));
                }
            }
        });

        if beta == 0. && alpha == 1. {
            for i in 0..MR {
                for j in 0..NR_REGS {
                    let out_ptr = tile_ptr.add(tile_row_stride * i + j * REG_SIZE);
                    v128_store(out_ptr as *mut v128, tmp[i][j]);
                }
            }
        } else if beta == 1. && alpha == 1. {
            for i in 0..MR {
                for j in 0..NR_REGS {
                    let out_ptr = tile_ptr.add(tile_row_stride * i + j * REG_SIZE);
                    let out_val = f32x4_add(v128_load(out_ptr as *const v128), tmp[i][j]);
                    v128_store(out_ptr as *mut v128, out_val);
                }
            }
        } else {
            let alpha_broadcast = f32x4(alpha, alpha, alpha, alpha);
            let beta_broadcast = f32x4(beta, beta, beta, beta);
            for i in 0..MR {
                for j in 0..NR_REGS {
                    let out_ptr = tile_ptr.add(tile_row_stride * i + j * REG_SIZE);
                    let out_val = f32x4_mul(v128_load(out_ptr as *const v128), beta_broadcast);
                    let out_val = f32x4_add(out_val, f32x4_mul(tmp[i][j], alpha_broadcast));
                    v128_store(out_ptr as *mut v128, out_val);
                }
            }
        }
    }
}

super::impl_gemmops!(WasmSimdKernel);
