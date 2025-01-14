//! Functions for testing the availability of instruction sets at runtime.

/// Detect availability of AVX-512 on macOS, where `is_x86_feature_detected`
/// can return false even if AVX-512 is available.
///
/// See https://github.com/golang/go/issues/43089. Go chose to use the
/// `commpage` to get the info. We use `sysctlbyname` instead since it is
/// a documented API.
#[cfg(feature = "avx512")]
#[cfg(target_os = "macos")]
fn test_for_avx512_on_macos() -> bool {
    use std::sync::OnceLock;

    static AVX512_AVAILABLE: OnceLock<bool> = OnceLock::new();

    *AVX512_AVAILABLE.get_or_init(|| unsafe {
        // We test for the minimum AVX-512 extensions we require, but not
        // avx512f, as that is implied if the extensions are supported.
        get_sysctl_bool(b"hw.optional.avx512vl\0") && get_sysctl_bool(b"hw.optional.avx512bw\0")
    })
}

/// Get a sysctl int value by name and interpret it as a boolean.
///
/// `name` must be a nul-terminated.
#[cfg(feature = "avx512")]
#[cfg(target_os = "macos")]
unsafe fn get_sysctl_bool(name: &[u8]) -> bool {
    use std::os::raw::{c_char, c_int, c_void};

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

    use std::ffi::CStr;

    let mut ret = 0u64;
    let mut size = std::mem::size_of::<u64>();

    let sysctl_ret = sysctlbyname(
        CStr::from_bytes_with_nul(name).unwrap().as_ptr(),
        std::mem::transmute(&mut ret),
        &mut size,
        std::ptr::null(),
        0,
    );

    sysctl_ret == 0 && ret == 1
}

/// Test if the current system has basic AVX-512 support.
///
/// "Basic support" is defined as:
///  - AVX512F
///  - AVX512VL
///  - AVX512BW
///
/// These features are available on Skylake (2016) and later.
/// See https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#AVX-512_CPU_compatibility_table.
///
/// This is unfortunately not as simple as using `is_x86_feature_detected`
/// because that can return incorrect results on macOS.
#[cfg(feature = "avx512")]
#[cfg(target_arch = "x86_64")]
pub fn is_avx512_supported() -> bool {
    if is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512vl")
        && is_x86_feature_detected!("avx512bw")
    {
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
}
