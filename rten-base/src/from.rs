//! Utilities for defining `From` and `TryFrom` impls.

/// Define a `From` impl which creates an enum variant from a value.
#[macro_export]
macro_rules! enum_from {
    ($enum:ty, $variant:ident, $from:ty) => {
        impl From<$from> for $enum {
            fn from(val: $from) -> Self {
                Self::$variant(val)
            }
        }
    };
}

pub use enum_from;
