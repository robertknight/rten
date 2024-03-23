use rten_tensor::Matrix;

use super::{simd_gemv, Kernel};

/// Optimized kernel for x64 CPUs that support AVX + FMA instructions.
#[derive(Default)]
pub struct FmaKernel {}

impl Kernel for FmaKernel {
    const MR: usize = 6;

    // Chosen to fit 2 AVX registers and take advantage of the two FMA
    // execution ports.
    const NR: usize = 16;

    fn name() -> &'static str {
        "fma"
    }

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

    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn gemv_kernel(out: &mut [f32], a: &[f32], b: Matrix, alpha: f32, beta: f32) {
        use core::arch::x86_64::__m256;
        simd_gemv::<__m256, 2>(out, a, b, alpha, beta);
    }
}

impl FmaKernel {
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

        const MR: usize = FmaKernel::MR;
        const NR: usize = FmaKernel::NR;

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

super::impl_gemmops!(FmaKernel);

/// Detect availability of AVX-512 on macOS, where `is_x86_feature_detected`
/// can return false even if AVX-512 is available.
///
/// See https://github.com/golang/go/issues/43089. Go chose to use the
/// `commpage` to get the info. We use `sysctlbyname` instead since it is
/// a documented API.
#[cfg(feature = "avx512")]
#[cfg(target_os = "macos")]
fn test_for_avx512_on_macos() -> bool {
    use std::ffi::CStr;
    use std::os::raw::{c_char, c_int, c_void};
    use std::sync::OnceLock;

    #[link(name = "c")]
    extern "C" {
        /// See https://developer.apple.com/documentation/kernel/1387446-sysctlbyname.
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *const c_void,
            newlen: usize,
        ) -> c_int;
    }

    static AVX512_AVAILABLE: OnceLock<bool> = OnceLock::new();

    *AVX512_AVAILABLE.get_or_init(|| {
        unsafe {
            let mut ret = 0u64;
            let mut size = std::mem::size_of::<u64>();

            // We test only for avx512vl, as this implies avx512f.
            let sysctl_ret = sysctlbyname(
                CStr::from_bytes_with_nul(b"hw.optional.avx512vl\0")
                    .unwrap()
                    .as_ptr(),
                std::mem::transmute(&mut ret),
                &mut size,
                std::ptr::null(),
                0,
            );
            sysctl_ret == 0 && ret == 1
        }
    })
}

/// Optimized kernel for x64 CPUs that support AVX 512 instructions.
#[cfg(feature = "avx512")]
#[derive(Default)]
pub struct Avx512Kernel {}

#[cfg(feature = "avx512")]
impl Kernel for Avx512Kernel {
    // The optimal value of MR depends on how many AVX-512 FMA units the CPU has.
    // Client Intel CPUs have one, server CPUs have two. This smaller value is
    // tuned for single-FMA CPUs.
    //
    // See https://github.com/robertknight/rten/issues/17.
    const MR: usize = 6;

    // 2 x 16-f32-wide registers.
    const NR: usize = 32;

    fn name() -> &'static str {
        "avx512"
    }

    fn supported() -> bool {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
            return true;
        }

        #[cfg(target_os = "macos")]
        if test_for_avx512_on_macos() {
            return true;
        }

        false
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
        Self::kernel_avx_512(tile_ptr, tile_row_stride, a, b, depth, alpha, beta)
    }

    #[target_feature(enable = "avx512f")]
    #[target_feature(enable = "avx512vl")]
    unsafe fn gemv_kernel(out: &mut [f32], a: &[f32], b: Matrix, alpha: f32, beta: f32) {
        // Re-use the AVX2 / FMA kernel because rten_vecmath doesn't provide
        // AVX-512 implementations for `SimdFloat` yet.
        FmaKernel::gemv_kernel(out, a, b, alpha, beta);
    }
}

#[cfg(feature = "avx512")]
impl Avx512Kernel {
    #[target_feature(enable = "avx512f")]
    #[target_feature(enable = "avx512vl")]
    unsafe fn kernel_avx_512(
        tile_ptr: *mut f32,
        tile_row_stride: usize,
        a: &[f32],
        b: &[f32],
        depth: usize,
        alpha: f32,
        beta: f32,
    ) {
        use core::arch::x86_64::{
            __m512, _mm512_add_ps, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_mul_ps, _mm512_set1_ps,
            _mm512_setzero_ps, _mm512_storeu_ps, _mm_prefetch, _MM_HINT_ET0, _MM_HINT_T0,
        };
        use std::mem::size_of;

        const MR: usize = Avx512Kernel::MR;
        const NR: usize = Avx512Kernel::NR;

        const REG_SIZE: usize = size_of::<__m512>() / size_of::<f32>();
        const NR_REGS: usize = NR / REG_SIZE;
        assert!(NR % REG_SIZE == 0);

        // Check that buffer accesses below are going to be valid.
        assert!(a.len() >= depth * MR);
        assert!(b.len() >= depth * NR);

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut tmp = [[_mm512_setzero_ps(); NR_REGS]; MR];
        let mut b_rows = [_mm512_setzero_ps(); NR_REGS];

        // Perform first `depth - 1` outer product updates.
        for k in 0..depth - 1 {
            let a_off = k * MR;
            let b_off = k * NR;

            // Prefetch B for the next iteration.
            _mm_prefetch(b_ptr.add((k + 1) * NR) as *const i8, _MM_HINT_T0);

            for i in 0..NR_REGS {
                b_rows[i] = _mm512_loadu_ps(b_ptr.add(b_off + i * REG_SIZE));
            }

            for i in 0..MR {
                let a_val = *a_ptr.add(a_off + i);
                let a_broadcast = _mm512_set1_ps(a_val);

                for j in 0..NR_REGS {
                    tmp[i][j] = _mm512_fmadd_ps(a_broadcast, b_rows[j], tmp[i][j]);
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
            b_rows[i] = _mm512_loadu_ps(b_ptr.add(b_off + i * REG_SIZE));
        }

        for i in 0..MR {
            let a_val = *a_ptr.add(a_off + i);
            let a_broadcast = _mm512_set1_ps(a_val);

            for j in 0..NR_REGS {
                tmp[i][j] = _mm512_fmadd_ps(a_broadcast, b_rows[j], tmp[i][j]);
            }
        }

        // Write to output tile.
        if beta == 0. && alpha == 1. {
            for i in 0..MR {
                for j in 0..NR_REGS {
                    let out_ptr = tile_ptr.add(tile_row_stride * i + j * REG_SIZE);
                    _mm512_storeu_ps(out_ptr, tmp[i][j]);
                }
            }
        } else if beta == 1. && alpha == 1. {
            for i in 0..MR {
                for j in 0..NR_REGS {
                    let out_ptr = tile_ptr.add(tile_row_stride * i + j * REG_SIZE);
                    let out_val = _mm512_add_ps(_mm512_loadu_ps(out_ptr), tmp[i][j]);
                    _mm512_storeu_ps(out_ptr, out_val);
                }
            }
        } else {
            let alpha_broadcast = _mm512_set1_ps(alpha);
            let beta_broadcast = _mm512_set1_ps(beta);
            for i in 0..MR {
                for j in 0..NR_REGS {
                    let out_ptr = tile_ptr.add(tile_row_stride * i + j * REG_SIZE);
                    let out_val = _mm512_mul_ps(_mm512_loadu_ps(out_ptr), beta_broadcast);
                    let out_val = _mm512_fmadd_ps(tmp[i][j], alpha_broadcast, out_val);
                    _mm512_storeu_ps(out_ptr, out_val);
                }
            }
        }
    }
}

#[cfg(feature = "avx512")]
super::impl_gemmops!(Avx512Kernel);
