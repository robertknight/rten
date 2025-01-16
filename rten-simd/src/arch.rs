//! Architecture-specific functionality.

/// Dummy arch which implements SIMD traits for Rust scalars (i32, f32 etc.)
mod scalar;

/// Dummy arch which implements SIMD traits for arrays
mod array;

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(target_arch = "wasm32")]
#[cfg(target_feature = "simd128")]
pub mod wasm;

use crate::{Simd, SimdMask};

/// Fallback implementation for [`SimdFloat::gather_mask`], for CPUs where
/// a native gather implementation is unavailable or unusable.
///
/// The caller must set `LEN` to `S::LEN`.
///
/// # Safety
///
/// See notes in [`SimdFloat::gather_mask`]. In particular, `src` must point
/// to a non-empty buffer, so that `src[0]` is valid.
#[inline]
unsafe fn simd_gather_mask<
    M: SimdMask,
    S: Simd<Mask = M>,
    SI: Simd<Elem = i32, Mask = M>,
    const LEN: usize,
>(
    src: *const S::Elem,
    offsets: SI,
    mask: M,
) -> S {
    // Set offset to zero where masked out. `src` is required to point to
    // a non-empty buffer, so index zero can be loaded as a dummy. This avoids
    // an unpredictable branch.
    let offsets = SI::zero().blend(offsets, mask);
    let mut offset_array = [0; LEN];
    offsets.store(offset_array.as_mut_ptr());

    let values: [S::Elem; LEN] = std::array::from_fn(|i| *src.add(offset_array[i] as usize));
    S::zero().blend(S::load(values.as_ptr()), mask)
}
