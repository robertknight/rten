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
