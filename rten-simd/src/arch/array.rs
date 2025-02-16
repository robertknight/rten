use crate::{Simd, SimdMask};

impl<const N: usize> SimdMask for [bool; N]
where
    [bool; N]: Default,
{
    type Array = Self;

    unsafe fn and(self, rhs: Self) -> Self {
        std::array::from_fn(|i| self[i] && rhs[i])
    }

    unsafe fn to_array(self) -> Self {
        self
    }

    unsafe fn from_array(mask: Self) -> Self {
        mask
    }
}

macro_rules! impl_simd_for_array {
    ($type:ty) => {
        impl<const N: usize> Simd for [$type; N]
        where
            [bool; N]: Default,
        {
            const LEN: Option<usize> = Some(N);

            type Array = Self;
            type Elem = $type;
            type Mask = [bool; N];

            #[inline]
            unsafe fn blend(self, rhs: Self, mask: Self::Mask) -> Self {
                std::array::from_fn(|i| if !mask[i] { self[i] } else { rhs[i] })
            }

            #[inline]
            unsafe fn load(ptr: *const $type) -> Self {
                std::array::from_fn(|i| *ptr.add(i))
            }

            #[inline]
            unsafe fn splat(val: $type) -> Self {
                [val; N]
            }

            #[inline]
            unsafe fn store(self, ptr: *mut $type) {
                for i in 0..N {
                    ptr.add(i).write(self[i]);
                }
            }

            #[inline]
            unsafe fn to_array(self) -> Self::Array {
                self
            }
        }
    };
}

impl_simd_for_array!(u8);
