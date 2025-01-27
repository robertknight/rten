//! Functions for testing the availability of instruction sets at runtime.

/// Functions for reading system info on macOS.
#[cfg(target_os = "macos")]
#[allow(unused)]
pub mod macos {
    /// Detect availability of AVX-512 on macOS, where `is_x86_feature_detected`
    /// can return false even if AVX-512 is available.
    ///
    /// See https://github.com/golang/go/issues/43089. Go chose to use the
    /// `commpage` to get the info. We use `sysctlbyname` instead since it is
    /// a documented API.
    fn test_for_avx512_on_macos() -> bool {
        use std::sync::OnceLock;

        static AVX512_AVAILABLE: OnceLock<bool> = OnceLock::new();

        *AVX512_AVAILABLE.get_or_init(|| {
            // We test for the minimum AVX-512 extensions we require, but not
            // avx512f, as that is implied if the extensions are supported.
            sysctl_bool(c"hw.optional.avx512vl").unwrap_or(false)
                && sysctl_bool(c"hw.optional.avx512bw").unwrap_or(false)
        })
    }

    /// Error code returned by `sysctlbyname`.
    #[derive(Copy, Clone, Debug, PartialEq)]
    pub struct SysctlError(i32);

    /// Read system info on macOS via the `sysctlbyname` API.
    ///
    /// See the output of `sysctl -a` on the command line for available settings.
    ///
    /// See also [Apple's documentation](https://developer.apple.com/documentation/kernel/1387446-sysctlbyname).
    pub fn sysctl_int(name: &std::ffi::CStr) -> Result<i64, SysctlError> {
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

        // Use i64 for the return type per example in Apple's docs.
        let mut result = 0i64;
        let mut size = std::mem::size_of::<i64>();

        let sysctl_ret = unsafe {
            sysctlbyname(
                name.as_ptr(),
                &mut result as *mut i64 as *mut c_void,
                &mut size,
                std::ptr::null(),
                0,
            )
        };

        if sysctl_ret != 0 {
            return Err(SysctlError(sysctl_ret));
        }
        Ok(result)
    }

    /// Read a system configuration integer value and interpret it as a boolean.
    pub fn sysctl_bool(name: &std::ffi::CStr) -> Result<bool, SysctlError> {
        sysctl_int(name).map(|val| val == 1)
    }
}

/// Test if the current system has basic AVX-512 support.
///
/// "Basic support" is defined as:
///  - AVX512F
///  - AVX512VL
///  - AVX512BW
///
/// These features are available on Skylake (2016) and later.
/// See <https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#AVX-512_CPU_compatibility_table>.
///
/// This is unfortunately not as simple as using `is_x86_feature_detected`
/// because that can return incorrect results on macOS.
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
            macos::test_for_avx512_on_macos()
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86_64")]
    use super::is_avx512_supported;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_is_avx512_supported() {
        // Just test that the function runs.
        is_avx512_supported();
    }
}
