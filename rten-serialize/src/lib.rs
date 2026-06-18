

#[cfg(feature = "npy")]
mod npy;
#[cfg(feature = "npy")]
pub use npy::*;

#[cfg(feature = "npz")]
mod npz;
#[cfg(feature = "npz")]
pub use npz::*;
