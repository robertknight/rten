//! Parallel-processing utilities.
//!
//! This crate provides common utilities related to multi-threaded parallelism,
//! for use by the RTen crates.

pub mod par_init;
pub mod par_iter;

pub use par_init::par_init_chunks;
