//! Architecture-specific functionality.

/// Dummy arch which implements SIMD vector types for Rust scalars (i32, f32 etc.)
mod scalar;

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(target_arch = "wasm32")]
pub mod wasm;
