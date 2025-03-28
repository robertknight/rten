#[cfg(target_arch = "aarch64")]
pub mod aarch64;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "wasm32")]
#[cfg(target_feature = "simd128")]
pub mod wasm32;

pub mod generic;

use crate::simd::Simd;

/// Return the number of lanes in a SIMD vector with compile-time known size.
const fn lanes<S: Simd>() -> usize {
    size_of::<S>() / size_of::<S::Elem>()
}

/// Create a wrapper type for a platform-specific intrinsic type.
#[allow(unused_macros)] // Not used on some platforms
macro_rules! simd_type {
    ($type:ident, $inner:ty, $elem:ty, $mask:ty, $isa:ty) => {
        // The platform intrinsic is exposed as a public field so that
        // downstream crates can implement custom SIMD operations. It might be
        // better to support an `Into` conversion from the wrapper to the
        // platform type instead?

        #[derive(Copy, Clone, Debug)]
        #[repr(transparent)]
        pub struct $type(pub $inner);

        impl From<$inner> for $type {
            fn from(val: $inner) -> Self {
                Self(val)
            }
        }

        impl Simd for $type {
            type Elem = $elem;
            type Mask = $mask;
            type Array = [Self::Elem; size_of::<Self>() / size_of::<$elem>()];
            type Isa = $isa;

            #[inline]
            fn to_bits(self) -> <Self::Isa as Isa>::Bits {
                #[allow(clippy::useless_transmute)]
                unsafe {
                    transmute::<Self, <Self::Isa as Isa>::Bits>(self)
                }
            }

            #[inline]
            fn from_bits(bits: <Self::Isa as Isa>::Bits) -> Self {
                #[allow(clippy::useless_transmute)]
                unsafe {
                    transmute::<<Self::Isa as Isa>::Bits, Self>(bits)
                }
            }

            #[inline]
            fn to_array(self) -> Self::Array {
                unsafe { transmute::<Self, Self::Array>(self) }
            }
        }
    };
}

#[allow(unused_imports)] // Not used on some platforms
use simd_type;
