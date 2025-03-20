mod avx2;
pub use avx2::Avx2Isa;

#[cfg(feature = "avx512")]
mod avx512;

#[cfg(feature = "avx512")]
pub use avx512::Avx512Isa;
