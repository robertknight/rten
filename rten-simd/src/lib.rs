//! Portable SIMD types for implementing vectorized functions that work across
//! different architectures.
//!
//! Compared to [std::simd](https://doc.rust-lang.org/std/simd/index.html) it
//! offers the following benefits:
//!
//! - Works on stable Rust
//! - Includes infrastructure for dispatching vectorized operations using the
//!   optimal instruction set as determined at runtime.
//! - Includes higher order functions for vectorized maps, folds etc.
//!
//! ## Supported architectures
//!
//! SIMD wrappers are provided for the following architectures:
//!
//! - Arm Neon
//! - AVX 2 / FMA
//! - AVX-512 (requires `avx512` feature and nightly Rust)
//! - WebAssembly SIMD
//!
//! There is also a scalar fallback that works on all platforms, but provides no
//! performance benefit over non-SIMD code.

#![cfg_attr(
    feature = "avx512",
    feature(stdarch_x86_avx512),
    feature(avx512_target_feature)
)]

pub mod arch;

pub mod dispatch;
pub mod functional;
pub mod isa_detection;
pub mod span;
mod vec;

pub use vec::{vec_count, SimdFloat, SimdInt, SimdMask, SimdVal};

#[cfg(feature = "avx512")]
#[cfg(target_arch = "x86_64")]
pub use isa_detection::is_avx512_supported;
