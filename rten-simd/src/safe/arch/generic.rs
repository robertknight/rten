use std::array;
use std::mem::transmute;

use crate::safe::{Isa, Mask, MaskOps, Simd, SimdFloatOps, SimdIntOps, SimdOps};

const LEN: usize = 4;

#[repr(align(16))]
#[derive(Copy, Clone, Debug)]
pub struct I32x4([i32; LEN]);

#[repr(align(16))]
#[derive(Copy, Clone, Debug)]
pub struct F32x4([f32; LEN]);

#[derive(Copy, Clone)]
pub struct GenericIsa {
    _private: (),
}

impl GenericIsa {
    pub fn new() -> Self {
        GenericIsa { _private: () }
    }
}

impl Default for GenericIsa {
    fn default() -> Self {
        Self::new()
    }
}

// Safety: Instructions used by generic ISA are always supported.
unsafe impl Isa for GenericIsa {
    type F32 = F32x4;
    type I32 = I32x4;
    type Bits = I32x4;

    fn f32(self) -> impl SimdFloatOps<Self::F32, Int = Self::I32> {
        self
    }

    fn i32(self) -> impl SimdIntOps<Self::I32> {
        self
    }
}

macro_rules! simd_ops_x32_common {
    ($simd:ident, $elem:ty) => {
        #[inline]
        fn mask_ops(self) -> impl MaskOps<I32x4> {
            self
        }

        #[inline]
        fn len(self) -> usize {
            LEN
        }

        #[inline]
        fn first_n_mask(self, n: usize) -> <$simd as Simd>::Mask {
            let mask: [i32; LEN] = std::array::from_fn(|i| if i < n { -1 } else { 0 });
            I32x4(mask)
        }

        #[inline]
        unsafe fn load_ptr_mask(
            self,
            ptr: *const <$simd as Simd>::Elem,
            mask: <$simd as Simd>::Mask,
        ) -> $simd {
            let mask_array = mask.0;
            let mut vec = <Self as SimdOps<$simd>>::zero(self).0;
            for i in 0..mask_array.len() {
                if mask_array[i] != 0 {
                    vec[i] = *ptr.add(i);
                }
            }
            self.load_ptr(vec.as_ref().as_ptr())
        }

        #[inline]
        unsafe fn store_ptr_mask(
            self,
            x: $simd,
            ptr: *mut <$simd as Simd>::Elem,
            mask: <$simd as Simd>::Mask,
        ) {
            let mask_array = mask.0;
            let x_array = x.0;
            for i in 0..<Self as SimdOps<$simd>>::len(self) {
                if mask_array[i] != 0 {
                    *ptr.add(i) = x_array[i];
                }
            }
        }

        #[inline]
        fn add(self, x: $simd, y: $simd) -> $simd {
            let xs = array::from_fn(|i| x.0[i] + y.0[i]);
            $simd(xs)
        }

        #[inline]
        fn sub(self, x: $simd, y: $simd) -> $simd {
            let xs = array::from_fn(|i| x.0[i] - y.0[i]);
            $simd(xs)
        }

        #[inline]
        fn mul(self, x: $simd, y: $simd) -> $simd {
            let xs = array::from_fn(|i| x.0[i] * y.0[i]);
            $simd(xs)
        }

        #[inline]
        fn mul_add(self, a: $simd, b: $simd, c: $simd) -> $simd {
            let xs = array::from_fn(|i| a.0[i] * b.0[i] + c.0[i]);
            $simd(xs)
        }

        #[inline]
        fn lt(self, x: $simd, y: $simd) -> I32x4 {
            let xs = array::from_fn(|i| if x.0[i] < y.0[i] { -1 } else { 0 });
            I32x4(xs)
        }

        #[inline]
        fn le(self, x: $simd, y: $simd) -> I32x4 {
            let xs = array::from_fn(|i| if x.0[i] <= y.0[i] { -1 } else { 0 });
            I32x4(xs)
        }

        #[inline]
        fn eq(self, x: $simd, y: $simd) -> I32x4 {
            let xs = array::from_fn(|i| if x.0[i] == y.0[i] { -1 } else { 0 });
            I32x4(xs)
        }

        #[inline]
        fn ge(self, x: $simd, y: $simd) -> I32x4 {
            let xs = array::from_fn(|i| if x.0[i] >= y.0[i] { -1 } else { 0 });
            I32x4(xs)
        }

        #[inline]
        fn gt(self, x: $simd, y: $simd) -> I32x4 {
            let xs = array::from_fn(|i| if x.0[i] > y.0[i] { -1 } else { 0 });
            I32x4(xs)
        }

        #[inline]
        fn min(self, x: $simd, y: $simd) -> $simd {
            let xs = array::from_fn(|i| x.0[i].min(y.0[i]));
            $simd(xs)
        }

        #[inline]
        fn max(self, x: $simd, y: $simd) -> $simd {
            let xs = array::from_fn(|i| x.0[i].max(y.0[i]));
            $simd(xs)
        }

        #[inline]
        fn splat(self, x: $elem) -> $simd {
            $simd([x; LEN])
        }

        #[inline]
        unsafe fn load_ptr(self, ptr: *const $elem) -> $simd {
            let xs = array::from_fn(|i| *ptr.add(i));
            $simd(xs)
        }

        #[inline]
        fn select(self, x: $simd, y: $simd, mask: <$simd as Simd>::Mask) -> $simd {
            let xs = array::from_fn(|i| if mask.0[i] != 0 { x.0[i] } else { y.0[i] });
            $simd(xs)
        }

        #[inline]
        unsafe fn store_ptr(self, x: $simd, ptr: *mut $elem) {
            for i in 0..LEN {
                *ptr.add(i) = x.0[i];
            }
        }
    };
}

unsafe impl SimdOps<F32x4> for GenericIsa {
    simd_ops_x32_common!(F32x4, f32);
}

impl SimdFloatOps<F32x4> for GenericIsa {
    type Int = <Self as Isa>::I32;

    #[inline]
    fn div(self, x: F32x4, y: F32x4) -> F32x4 {
        let xs = array::from_fn(|i| x.0[i] / y.0[i]);
        F32x4(xs)
    }

    #[inline]
    fn neg(self, x: F32x4) -> F32x4 {
        let xs = array::from_fn(|i| -x.0[i]);
        F32x4(xs)
    }

    #[inline]
    fn abs(self, x: F32x4) -> F32x4 {
        let xs = array::from_fn(|i| x.0[i].abs());
        F32x4(xs)
    }

    #[inline]
    fn to_int_trunc(self, x: F32x4) -> Self::Int {
        let xs = array::from_fn(|i| x.0[i] as i32);
        I32x4(xs)
    }
}

unsafe impl SimdOps<I32x4> for GenericIsa {
    simd_ops_x32_common!(I32x4, i32);
}

impl SimdIntOps<I32x4> for GenericIsa {
    #[inline]
    fn neg(self, x: I32x4) -> I32x4 {
        let xs = array::from_fn(|i| -x.0[i]);
        I32x4(xs)
    }

    #[inline]
    fn shift_left<const SHIFT: i32>(self, x: I32x4) -> I32x4 {
        let xs = array::from_fn(|i| x.0[i] << SHIFT);
        I32x4(xs)
    }
}

impl Mask for I32x4 {
    type Array = [bool; LEN];

    #[inline]
    fn to_array(self) -> Self::Array {
        let array = self.0;
        std::array::from_fn(|i| array[i] != 0)
    }
}

unsafe impl MaskOps<I32x4> for GenericIsa {
    #[inline]
    fn and(self, x: I32x4, y: I32x4) -> I32x4 {
        I32x4(array::from_fn(|i| x.0[i] & y.0[i]))
    }
}

macro_rules! simd_x32_common {
    ($simd:ty, $elem:ty) => {
        type Array = [$elem; LEN];
        type Mask = I32x4;
        type Isa = GenericIsa;

        #[inline]
        fn to_bits(self) -> <Self::Isa as Isa>::Bits {
            #[allow(clippy::useless_transmute)]
            I32x4(unsafe { transmute::<[$elem; LEN], [i32; LEN]>(self.0) })
        }

        #[inline]
        fn from_bits(bits: <Self::Isa as Isa>::Bits) -> Self {
            #[allow(clippy::useless_transmute)]
            Self(unsafe { transmute::<[i32; LEN], [$elem; LEN]>(bits.0) })
        }
    };
}

impl Simd for F32x4 {
    type Elem = f32;

    simd_x32_common!(F32x4, f32);

    #[inline]
    fn to_array(self) -> Self::Array {
        self.0
    }
}

impl Simd for I32x4 {
    type Elem = i32;

    simd_x32_common!(I32x4, i32);

    #[inline]
    fn to_array(self) -> Self::Array {
        self.0
    }
}
