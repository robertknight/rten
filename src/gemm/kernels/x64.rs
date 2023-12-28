use super::super::packing::{pack_a_block, pack_b_block};
use super::gemm_impl;
use super::{GemmOps, Kernel};

/// Optimized kernel for x64 CPUs that support AVX + FMA instructions.
pub struct FMAKernel {}

impl Kernel for FMAKernel {
    const MR: usize = 6;

    // Chosen to fit 2 AVX registers and take advantage of the two FMA
    // execution ports.
    const NR: usize = 16;

    fn supported() -> bool {
        is_x86_feature_detected!("fma")
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
        Self::kernel_fma(tile_ptr, tile_row_stride, a, b, depth, alpha, beta)
    }
}

impl FMAKernel {
    #[target_feature(enable = "fma")]
    unsafe fn kernel_fma(
        tile_ptr: *mut f32,
        tile_row_stride: usize,
        a: &[f32],
        b: &[f32],
        depth: usize,
        alpha: f32,
        beta: f32,
    ) {
        use core::arch::x86_64::{
            __m256, _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps,
            _mm256_setzero_ps, _mm256_storeu_ps, _mm_prefetch, _MM_HINT_ET0, _MM_HINT_T0,
        };
        use std::mem::size_of;

        const MR: usize = FMAKernel::MR;
        const NR: usize = FMAKernel::NR;

        const REG_SIZE: usize = size_of::<__m256>() / size_of::<f32>();
        const NR_REGS: usize = NR / REG_SIZE;
        assert!(NR % REG_SIZE == 0);

        // Check that buffer accesses below are going to be valid.
        assert!(a.len() >= depth * MR);
        assert!(b.len() >= depth * NR);
        assert!(depth > 0);

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut tmp = [[_mm256_setzero_ps(); NR_REGS]; MR];
        let mut b_rows = [_mm256_setzero_ps(); NR_REGS];

        // Perform first `depth - 1` outer product updates.
        for k in 0..depth - 1 {
            let a_off = k * MR;
            let b_off = k * NR;

            // Prefetch B for the next iteration.
            _mm_prefetch(b_ptr.add((k + 1) * NR) as *const i8, _MM_HINT_T0);

            for i in 0..NR_REGS {
                b_rows[i] = _mm256_loadu_ps(b_ptr.add(b_off + i * REG_SIZE));
            }

            for i in 0..MR {
                let a_val = *a_ptr.add(a_off + i);
                let a_broadcast = _mm256_set1_ps(a_val);

                for j in 0..NR_REGS {
                    tmp[i][j] = _mm256_fmadd_ps(a_broadcast, b_rows[j], tmp[i][j]);
                }
            }
        }

        // Prefetch output before the final computation loop.
        for i in 0..MR {
            _mm_prefetch(tile_ptr.add(tile_row_stride * i) as *const i8, _MM_HINT_ET0);
        }

        // Perform final outer product update.
        let k = depth - 1;
        let a_off = k * MR;
        let b_off = k * NR;

        for i in 0..NR_REGS {
            b_rows[i] = _mm256_loadu_ps(b_ptr.add(b_off + i * REG_SIZE));
        }

        for i in 0..MR {
            let a_val = *a_ptr.add(a_off + i);
            let a_broadcast = _mm256_set1_ps(a_val);

            for j in 0..NR_REGS {
                tmp[i][j] = _mm256_fmadd_ps(a_broadcast, b_rows[j], tmp[i][j]);
            }
        }

        // Write to output tile
        if beta == 0. && alpha == 1. {
            for i in 0..MR {
                for j in 0..NR_REGS {
                    let out_ptr = tile_ptr.add(tile_row_stride * i + j * REG_SIZE);
                    _mm256_storeu_ps(out_ptr, tmp[i][j]);
                }
            }
        } else if beta == 1. && alpha == 1. {
            for i in 0..MR {
                for j in 0..NR_REGS {
                    let out_ptr = tile_ptr.add(tile_row_stride * i + j * REG_SIZE);
                    let out_val = _mm256_add_ps(_mm256_loadu_ps(out_ptr), tmp[i][j]);
                    _mm256_storeu_ps(out_ptr, out_val);
                }
            }
        } else {
            let alpha_broadcast = _mm256_set1_ps(alpha);
            let beta_broadcast = _mm256_set1_ps(beta);
            for i in 0..MR {
                for j in 0..NR_REGS {
                    let out_ptr = tile_ptr.add(tile_row_stride * i + j * REG_SIZE);
                    let out_val = _mm256_mul_ps(_mm256_loadu_ps(out_ptr), beta_broadcast);
                    let out_val = _mm256_fmadd_ps(tmp[i][j], alpha_broadcast, out_val);
                    _mm256_storeu_ps(out_ptr, out_val);
                }
            }
        }
    }
}

super::impl_gemmops!(FMAKernel);
