use super::Kernel;

use crate::iter_util::unroll_loop;

/// This is not a fully optimized ARM NEON kernel, just an initial version
/// which is a copy of the base kernel that has been tweaked to:
///
///  - Use a larger tile size
///  - Use FMA instructions via `f32::mul_add`
///  - Unroll the inner loop
#[derive(Default)]
pub struct ArmNeonKernel {}

impl Kernel for ArmNeonKernel {
    // ARM NEON has 32 registers. Empirically 14x4 is the largest tile size
    // this naive auto-vectorized implementation can use before LLVM spills
    // registers and performance drops. Better kernels in eg. OpenBLAS have
    // 64-element tiles (8x8 or 16x4).

    const MR: usize = 14;
    const NR: usize = 4;

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
        const MR: usize = ArmNeonKernel::MR;
        const NR: usize = ArmNeonKernel::NR;

        assert!(a.len() >= depth * MR);
        assert!(b.len() >= depth * NR);

        // Accumulate into a fixed-sized array to allow the compiler to generate
        // more efficient code for the loop over `depth`.
        let mut tmp = [[0.0; NR]; MR];

        unroll_loop!(depth, k, 8, {
            let a_off = k * MR;
            let b_off = k * NR;

            for i in 0..MR {
                for j in 0..NR {
                    tmp[i][j] = a
                        .get_unchecked(a_off + i)
                        .mul_add(*b.get_unchecked(b_off + j), tmp[i][j]);
                }
            }
        });

        if beta == 0. && alpha == 1. {
            for i in 0..MR {
                for j in 0..NR {
                    let out_el = tile_ptr.add(tile_row_stride * i + j);
                    *out_el = tmp[i][j];
                }
            }
        } else if beta == 1. && alpha == 1. {
            for i in 0..MR {
                for j in 0..NR {
                    let out_el = tile_ptr.add(tile_row_stride * i + j);
                    *out_el += tmp[i][j];
                }
            }
        } else {
            for i in 0..MR {
                for j in 0..NR {
                    let out_el = tile_ptr.add(tile_row_stride * i + j);
                    *out_el = beta * *out_el + alpha * tmp[i][j];
                }
            }
        }
    }
}

super::impl_gemmops!(ArmNeonKernel);
