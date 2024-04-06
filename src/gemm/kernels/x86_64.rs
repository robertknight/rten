use std::arch::x86_64::__m256;

#[cfg(feature = "avx512")]
use std::arch::x86_64::__m512;

use rten_tensor::Matrix;
use rten_vecmath::simd_vec::SimdFloat;

use super::{simd_gemm, simd_gemv, Kernel};

/// Optimized kernel for x64 CPUs that support AVX + FMA instructions.
#[derive(Default)]
pub struct FmaKernel {
    _private: (),
}

// Safety - The `new` fn tests for AVX-2 / FMA support.
unsafe impl Kernel for FmaKernel {
    const MR: usize = 6;

    // Chosen to fit 2 AVX registers and take advantage of the two FMA
    // execution ports.
    const NR: usize = 16;

    fn new() -> Option<Self> {
        let supported = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        supported.then_some(FmaKernel { _private: () })
    }

    fn name(&self) -> &'static str {
        "fma"
    }

    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn kernel(
        &self,
        tile_ptr: *mut f32,
        tile_row_stride: usize,
        a: &[f32],
        b: &[f32],
        depth: usize,
        alpha: f32,
        beta: f32,
    ) {
        const MR: usize = FmaKernel::MR;
        const NR: usize = FmaKernel::NR;
        const NR_REGS: usize = NR / <__m256 as SimdFloat>::LEN;

        simd_gemm::<__m256, MR, NR_REGS>(tile_ptr, tile_row_stride, a, b, depth, alpha, beta);
    }

    fn gemv_kernel(&self, out: &mut [f32], a: &[f32], b: Matrix, alpha: f32, beta: f32) {
        #[target_feature(enable = "avx2")]
        #[target_feature(enable = "fma")]
        unsafe fn gemv_kernel_impl(out: &mut [f32], a: &[f32], b: Matrix, alpha: f32, beta: f32) {
            simd_gemv::<__m256, 2>(out, a, b, alpha, beta);
        }
        // Safety: Kernel can only be constructed if supported.
        unsafe {
            gemv_kernel_impl(out, a, b, alpha, beta);
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
pub struct Avx512Kernel {
    _private: (),
}

// Safety - The `new` fn checks for AVX-512 support.
#[cfg(feature = "avx512")]
unsafe impl Kernel for Avx512Kernel {
    // The optimal value of MR depends on how many AVX-512 FMA units the CPU has.
    // Client Intel CPUs have one, server CPUs have two. This smaller value is
    // tuned for single-FMA CPUs.
    //
    // See https://github.com/robertknight/rten/issues/17.
    const MR: usize = 6;

    // 2 x 16-f32-wide registers.
    const NR: usize = 32;

    fn new() -> Option<Self> {
        let supported = {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
                true
            } else {
                #[cfg(target_os = "macos")]
                {
                    test_for_avx512_on_macos()
                }
                #[cfg(not(target_os = "macos"))]
                {
                    false
                }
            }
        };
        supported.then_some(Avx512Kernel { _private: () })
    }

    fn name(&self) -> &'static str {
        "avx512"
    }

    #[target_feature(enable = "avx512f")]
    #[target_feature(enable = "avx512vl")]
    unsafe fn kernel(
        &self,
        tile_ptr: *mut f32,
        tile_row_stride: usize,
        a: &[f32],
        b: &[f32],
        depth: usize,
        alpha: f32,
        beta: f32,
    ) {
        const MR: usize = Avx512Kernel::MR;
        const NR: usize = Avx512Kernel::NR;
        const NR_REGS: usize = NR / <__m512 as SimdFloat>::LEN;

        simd_gemm::<__m512, MR, NR_REGS>(tile_ptr, tile_row_stride, a, b, depth, alpha, beta)
    }

    fn gemv_kernel(&self, out: &mut [f32], a: &[f32], b: Matrix, alpha: f32, beta: f32) {
        #[target_feature(enable = "avx512f")]
        #[target_feature(enable = "avx512vl")]
        unsafe fn gemv_kernel_impl(out: &mut [f32], a: &[f32], b: Matrix, alpha: f32, beta: f32) {
            simd_gemv::<__m512, 2>(out, a, b, alpha, beta);
        }
        // Safety: Kernel can only be constructed if supported.
        unsafe {
            gemv_kernel_impl(out, a, b, alpha, beta);
        }
    }
}

#[cfg(feature = "avx512")]
super::impl_gemmops!(Avx512Kernel);
