//! Branch prediction hints for stable Rust.
//!
//! This module contains implementations of [`std::hint::likely`] and
//! [`std::hint::unlikely`] for stable Rust.
//!
//! The implementation is taken from the `hashbrown` crate via
//! <https://users.rust-lang.org/t/compiler-hint-for-unlikely-likely-for-if-branches/62102/4>.

#[inline]
#[cold]
fn cold() {}

/// Wrap the condition value of a branch to mark it as likely to be taken.
#[inline]
pub fn likely(b: bool) -> bool {
    if !b {
        cold()
    }
    b
}

/// Wrap the condition value of a branch to mark it as unlikely to be taken.
#[inline]
pub fn unlikely(b: bool) -> bool {
    if b {
        cold()
    }
    b
}
