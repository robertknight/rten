use std::array;
use std::mem::transmute;

use crate::ops::{
    Concat, Extend, FloatOps, IntOps, Interleave, MaskOps, NarrowSaturate, NumOps, SignedIntOps,
};
use crate::{Isa, Mask, Simd};

// Size of SIMD vector in 32-bit lanes.
const LEN_X32: usize = 4;

macro_rules! simd_type {
    ($simd:ident, $elem:ty, $len:expr) => {
        #[repr(align(16))]
        #[derive(Copy, Clone, Debug)]
        pub struct $simd([$elem; $len]);

        impl From<[$elem; $len]> for $simd {
            fn from(val: [$elem; $len]) -> $simd {
                $simd(val)
            }
        }
    };
}

// Define SIMD vector types.
simd_type!(F32x4, f32, LEN_X32);
simd_type!(I32x4, i32, LEN_X32);
simd_type!(I16x8, i16, LEN_X32 * 2);
simd_type!(I8x16, i8, LEN_X32 * 4);
simd_type!(U8x16, u8, LEN_X32 * 4);
simd_type!(U16x8, u16, LEN_X32 * 2);
simd_type!(U32x4, u32, LEN_X32);

// Define mask vector types. `Mn` is a mask for a vector with n-bit lanes.
simd_type!(M32, i32, LEN_X32);
simd_type!(M16, i16, LEN_X32 * 2);
simd_type!(M8, i8, LEN_X32 * 4);

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
    type I16 = I16x8;
    type I8 = I8x16;
    type U8 = U8x16;
    type U16 = U16x8;
    type U32 = U32x4;
    type Bits = I32x4;

    fn f32(self) -> impl FloatOps<f32, Simd = Self::F32, Int = Self::I32> {
        self
    }

    fn i32(
        self,
    ) -> impl SignedIntOps<i32, Simd = Self::I32>
           + NarrowSaturate<i32, i16, Output = Self::I16>
           + Concat<i32> {
        self
    }

    fn i16(
        self,
    ) -> impl SignedIntOps<i16, Simd = Self::I16>
           + NarrowSaturate<i16, u8, Output = Self::U8>
           + Extend<i16, Output = Self::I32>
           + Interleave<i16> {
        self
    }

    fn i8(
        self,
    ) -> impl SignedIntOps<i8, Simd = Self::I8> + Extend<i8, Output = Self::I16> + Interleave<i8>
    {
        self
    }

    fn u8(self) -> impl Extend<u8, Output = Self::U16, Simd = Self::U8> {
        self
    }

    fn u16(self) -> impl IntOps<u16, Simd = Self::U16> {
        self
    }
}

macro_rules! simd_ops_common {
    ($simd:ident, $elem:ty, $len:expr, $mask:ident) => {
        #[inline]
        fn mask_ops(self) -> impl MaskOps<$mask> {
            self
        }

        #[inline]
        fn len(self) -> usize {
            $len
        }

        #[inline]
        fn first_n_mask(self, n: usize) -> $mask {
            let mask = std::array::from_fn(|i| if i < n { !0 } else { 0 });
            $mask(mask)
        }

        #[inline]
        unsafe fn load_ptr_mask(
            self,
            ptr: *const <$simd as Simd>::Elem,
            mask: <$simd as Simd>::Mask,
        ) -> $simd {
            let mask_array = mask.0;
            let mut vec = <Self as NumOps<$elem>>::zero(self).0;
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
            for i in 0..<Self as NumOps<$elem>>::len(self) {
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
        fn eq(self, x: $simd, y: $simd) -> $mask {
            let xs = array::from_fn(|i| if x.0[i] == y.0[i] { !0 } else { 0 });
            $mask(xs)
        }

        #[inline]
        fn ge(self, x: $simd, y: $simd) -> $mask {
            let xs = array::from_fn(|i| if x.0[i] >= y.0[i] { !0 } else { 0 });
            $mask(xs)
        }

        #[inline]
        fn gt(self, x: $simd, y: $simd) -> $mask {
            let xs = array::from_fn(|i| if x.0[i] > y.0[i] { !0 } else { 0 });
            $mask(xs)
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
            $simd([x; $len])
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
            for i in 0..$len {
                *ptr.add(i) = x.0[i];
            }
        }
    };
}

macro_rules! simd_int_ops_common {
    ($simd:ty) => {
        #[inline]
        fn and(self, x: $simd, y: $simd) -> $simd {
            array::from_fn(|i| x.0[i] & y.0[i]).into()
        }

        #[inline]
        fn or(self, x: $simd, y: $simd) -> $simd {
            array::from_fn(|i| x.0[i] | y.0[i]).into()
        }

        #[inline]
        fn not(self, x: $simd) -> $simd {
            array::from_fn(|i| !x.0[i]).into()
        }

        #[inline]
        fn xor(self, x: $simd, y: $simd) -> $simd {
            array::from_fn(|i| x.0[i] ^ y.0[i]).into()
        }
    };
}

unsafe impl NumOps<f32> for GenericIsa {
    type Simd = F32x4;

    simd_ops_common!(F32x4, f32, 4, M32);

    #[inline]
    fn and(self, x: F32x4, y: F32x4) -> F32x4 {
        array::from_fn(|i| f32::from_bits(x.0[i].to_bits() & y.0[i].to_bits())).into()
    }

    #[inline]
    fn not(self, x: F32x4) -> F32x4 {
        array::from_fn(|i| f32::from_bits(!x.0[i].to_bits())).into()
    }

    #[inline]
    fn or(self, x: F32x4, y: F32x4) -> F32x4 {
        array::from_fn(|i| f32::from_bits(x.0[i].to_bits() | y.0[i].to_bits())).into()
    }

    #[inline]
    fn xor(self, x: F32x4, y: F32x4) -> F32x4 {
        array::from_fn(|i| f32::from_bits(x.0[i].to_bits() ^ y.0[i].to_bits())).into()
    }
}

impl FloatOps<f32> for GenericIsa {
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

    #[inline]
    fn to_int_round(self, x: F32x4) -> Self::Int {
        let xs = array::from_fn(|i| x.0[i].round_ties_even() as i32);
        I32x4(xs)
    }
}

macro_rules! impl_simd_int_ops {
    ($simd:ident, $elem:ty, $len:expr, $mask:ident) => {
        unsafe impl NumOps<$elem> for GenericIsa {
            type Simd = $simd;

            simd_ops_common!($simd, $elem, $len, $mask);
            simd_int_ops_common!($simd);
        }

        impl IntOps<$elem> for GenericIsa {
            #[inline]
            fn shift_left<const SHIFT: i32>(self, x: $simd) -> $simd {
                let xs = array::from_fn(|i| x.0[i] << SHIFT);
                $simd(xs)
            }
        }
    };
}

macro_rules! impl_simd_signed_int_ops {
    ($simd:ident, $elem:ty, $len:expr, $mask:ident) => {
        impl_simd_int_ops!($simd, $elem, $len, $mask);

        impl SignedIntOps<$elem> for GenericIsa {
            #[inline]
            fn neg(self, x: $simd) -> $simd {
                let xs = array::from_fn(|i| -x.0[i]);
                $simd(xs)
            }
        }
    };
}

impl_simd_signed_int_ops!(I32x4, i32, 4, M32);
impl_simd_signed_int_ops!(I16x8, i16, 8, M16);
impl_simd_signed_int_ops!(I8x16, i8, 16, M8);

macro_rules! impl_extend {
    ($src:ty, $elem:ty, $dst:ty) => {
        impl Extend<$elem> for GenericIsa {
            type Output = $dst;

            fn extend(self, x: $src) -> ($dst, $dst) {
                let extended = x.0.map(|x| x as <$dst as Simd>::Elem);
                let low = array::from_fn(|i| extended[i]);
                let high = array::from_fn(|i| extended[i + extended.len() / 2]);
                (low.into(), high.into())
            }
        }
    };
}
impl_extend!(I8x16, i8, I16x8);
impl_extend!(I16x8, i16, I32x4);
impl_extend!(U8x16, u8, U16x8);

macro_rules! impl_concat {
    ($elem:ty, $simd:ty) => {
        impl Concat<$elem> for GenericIsa {
            fn concat_low(self, a: $simd, b: $simd) -> $simd {
                let half_len = a.0.len() / 2;
                array::from_fn(|i| {
                    if i < half_len {
                        a.0[i]
                    } else {
                        b.0[i - half_len]
                    }
                })
                .into()
            }

            fn concat_high(self, a: $simd, b: $simd) -> $simd {
                let half_len = a.0.len() / 2;
                array::from_fn(|i| {
                    if i < half_len {
                        a.0[half_len + i]
                    } else {
                        b.0[i]
                    }
                })
                .into()
            }
        }
    };
}

impl_concat!(i32, I32x4);

macro_rules! impl_interleave {
    ($elem:ty, $simd:ty) => {
        impl Interleave<$elem> for GenericIsa {
            fn interleave_low(self, a: $simd, b: $simd) -> $simd {
                array::from_fn(|i| if i % 2 == 0 { a.0[i / 2] } else { b.0[i / 2] }).into()
            }

            fn interleave_high(self, a: $simd, b: $simd) -> $simd {
                let start = a.0.len() / 2;
                array::from_fn(|i| {
                    if i % 2 == 0 {
                        a.0[start + i / 2]
                    } else {
                        b.0[start + i / 2]
                    }
                })
                .into()
            }
        }
    };
}
impl_interleave!(i8, I8x16);
impl_interleave!(i16, I16x8);

impl_simd_int_ops!(U8x16, u8, 16, M8);
impl_simd_int_ops!(U16x8, u16, 8, M16);

trait NarrowSaturateElem<T> {
    fn narrow_saturate(self) -> T;
}

impl NarrowSaturateElem<i16> for i32 {
    fn narrow_saturate(self) -> i16 {
        self.clamp(i16::MIN as i32, i16::MAX as i32) as i16
    }
}

impl NarrowSaturateElem<u8> for i16 {
    fn narrow_saturate(self) -> u8 {
        self.clamp(u8::MIN as i16, u8::MAX as i16) as u8
    }
}

macro_rules! impl_narrow {
    ($from:ident, $from_elem:ty, $to:ident, $to_elem:ty) => {
        impl NarrowSaturate<$from_elem, $to_elem> for GenericIsa {
            type Output = $to;

            fn narrow_saturate(self, lo: $from, hi: $from) -> $to {
                let mid = lo.0.len() / 2;
                let xs = array::from_fn(|i| {
                    let x = if i < mid { lo.0[i] } else { hi.0[i] };
                    x.narrow_saturate()
                });
                $to(xs)
            }
        }
    };
}
impl_narrow!(I32x4, i32, I16x8, i16);
impl_narrow!(I16x8, i16, U8x16, u8);

macro_rules! impl_mask {
    ($mask:ident, $len:expr) => {
        impl Mask for $mask {
            type Array = [bool; $len];

            #[inline]
            fn to_array(self) -> Self::Array {
                let array = self.0;
                array::from_fn(|i| array[i] != 0)
            }
        }

        unsafe impl MaskOps<$mask> for GenericIsa {
            #[inline]
            fn and(self, x: $mask, y: $mask) -> $mask {
                let xs = array::from_fn(|i| x.0[i] & y.0[i]);
                $mask(xs)
            }
        }
    };
}

impl_mask!(M32, LEN_X32);
impl_mask!(M16, LEN_X32 * 2);
impl_mask!(M8, LEN_X32 * 4);

macro_rules! impl_simd {
    ($simd:ty, $elem:ty, $mask:ty, $len:expr) => {
        impl Simd for $simd {
            type Mask = $mask;
            type Elem = $elem;
            type Array = [$elem; $len];
            type Isa = GenericIsa;

            #[inline]
            fn to_bits(self) -> <Self::Isa as Isa>::Bits {
                #[allow(clippy::useless_transmute)]
                I32x4(unsafe { transmute::<[$elem; $len], [i32; LEN_X32]>(self.0) })
            }

            #[inline]
            fn from_bits(bits: <Self::Isa as Isa>::Bits) -> Self {
                #[allow(clippy::useless_transmute)]
                Self(unsafe { transmute::<[i32; LEN_X32], [$elem; $len]>(bits.0) })
            }

            #[inline]
            fn to_array(self) -> Self::Array {
                self.0
            }
        }
    };
}

impl_simd!(F32x4, f32, M32, 4);
impl_simd!(I32x4, i32, M32, 4);
impl_simd!(I16x8, i16, M16, 8);
impl_simd!(I8x16, i8, M8, 16);
impl_simd!(U8x16, u8, M8, 16);
impl_simd!(U16x8, u16, M16, 8);
impl_simd!(U32x4, u32, M32, 4);
