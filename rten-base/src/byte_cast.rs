//! Cast values and slices to/from bytes.

use std::alloc::Layout;
use std::mem::{ManuallyDrop, MaybeUninit};

/// Types which can be transmuted to a byte array.
///
/// # Safety
///
/// This type must only be implemented for `Copy` types which contain no
/// padding.
pub unsafe trait ToByteArray: Copy {
    /// View of this type as an array of bytes.
    type Bytes: AsRef<[u8]>;

    /// Convert this type to an array of bytes in native order.
    ///
    /// This is the same as `to_ne_bytes` for primitive types.
    fn to_bytes(self) -> Self::Bytes;
}

macro_rules! impl_to_byte_array {
    ($type:ty) => {
        unsafe impl ToByteArray for $type {
            type Bytes = [u8; size_of::<$type>()];

            fn to_bytes(self) -> Self::Bytes {
                self.to_ne_bytes()
            }
        }
    };
}

impl_to_byte_array!(i8);
impl_to_byte_array!(u8);
impl_to_byte_array!(i16);
impl_to_byte_array!(u16);
impl_to_byte_array!(i32);
impl_to_byte_array!(u32);
impl_to_byte_array!(i64);
impl_to_byte_array!(u64);
impl_to_byte_array!(f32);
impl_to_byte_array!(f64);

unsafe impl ToByteArray for bool {
    type Bytes = [u8; 1];

    fn to_bytes(self) -> [u8; 1] {
        [self as u8]
    }
}

/// Types which can be transmuted from a byte array.
///
/// These are types for which any bit pattern is valid.
///
/// # Safety
///
/// This type must only be implemented for types for which any bit pattern is
/// valid.
pub unsafe trait FromByteArray: ToByteArray {
    /// Convert an array of bytes, in native order, into this type.
    ///
    /// This is the same as `from_ne_bytes` for primitive types.
    fn from_bytes(bytes: Self::Bytes) -> Self;

    /// Convert this type to another of the same size.
    ///
    /// This should compile to a zero-cost transmute.
    fn cast_bytes<Dst>(self) -> Dst
    where
        Dst: FromByteArray<Bytes = Self::Bytes>,
    {
        Dst::from_bytes(self.to_bytes())
    }
}

macro_rules! impl_from_byte_array {
    ($type:ty) => {
        unsafe impl FromByteArray for $type {
            fn from_bytes(val: Self::Bytes) -> Self {
                Self::from_ne_bytes(val)
            }
        }
    };
}

impl_from_byte_array!(i8);
impl_from_byte_array!(u8);
impl_from_byte_array!(i16);
impl_from_byte_array!(u16);
impl_from_byte_array!(i32);
impl_from_byte_array!(u32);
impl_from_byte_array!(i64);
impl_from_byte_array!(u64);
impl_from_byte_array!(f32);
impl_from_byte_array!(f64);

/// Return the length of a slice transmuted from `Src` to `Dst`, or `None` if
/// the transmute is not possible.
fn transmuted_slice_len<Src, Dst>(src: &[Src]) -> Option<usize> {
    if !(src.as_ptr() as usize).is_multiple_of(align_of::<Dst>()) {
        return None;
    }

    let src_byte_len = std::mem::size_of_val(src);
    if !src_byte_len.is_multiple_of(size_of::<Dst>()) {
        return None;
    }

    Some(src_byte_len / size_of::<Dst>())
}

/// Transmute a slice of elements from one type to another.
///
/// This cast is safe because all bytes of the source elements are initialized
/// and all bit patterns are valid for the destination elements.
///
/// Returns `None` if the source pointer is not correctly aligned for the
/// destination type.
pub fn cast_slice<Src: ToByteArray, Dst: FromByteArray>(src: &[Src]) -> Option<&[Dst]> {
    let new_len = transmuted_slice_len::<_, Dst>(src)?;

    // Safety:
    // - Pointer cast is safe since any bit pattern is valid for POD types
    // - Length has been adjusted for `Dst` type
    Some(unsafe { std::slice::from_raw_parts(src.as_ptr() as *const Dst, new_len) })
}

/// Transmute a mutable slice of elements from one type to another.
///
/// This cast is safe because all bytes of the source elements are initialized
/// and all bit patterns are valid for the destination elements.
///
/// Returns `None` if the source pointer is not correctly aligned for the
/// destination type.
#[allow(unused)]
pub fn cast_mut_slice<Src: ToByteArray, Dst: FromByteArray>(src: &mut [Src]) -> Option<&mut [Dst]> {
    let new_len = transmuted_slice_len::<_, Dst>(src)?;

    // Safety:
    // - Pointer cast is safe since any bit pattern is valid for POD types
    // - Length has been adjusted for `Dst` type
    Some(unsafe { std::slice::from_raw_parts_mut(src.as_mut_ptr() as *mut Dst, new_len) })
}

/// Transmute a mutable slice of elements from one uninitialized type to another.
///
/// Returns `None` if the source pointer is not correctly aligned for the
/// destination type.
pub fn cast_uninit_mut_slice<Src: ToByteArray, Dst: FromByteArray>(
    src: &mut [MaybeUninit<Src>],
) -> Option<&mut [MaybeUninit<Dst>]> {
    let new_len = transmuted_slice_len::<_, Dst>(src)?;

    // Safety:
    // - Pointer cast is safe since any bit pattern is valid for POD types
    // - Length has been adjusted for `Dst` type
    Some(unsafe {
        std::slice::from_raw_parts_mut(src.as_mut_ptr() as *mut MaybeUninit<Dst>, new_len)
    })
}

/// Transmute a vector of elements from one type to another.
///
/// The source and destination types must have the same size and alignment.
/// The length and capacity of the vector will be the same afterwards.
pub fn cast_vec<Src: ToByteArray, Dst: FromByteArray>(src: Vec<Src>) -> Option<Vec<Dst>> {
    // From `Vec::into_raw_parts`.
    let mut src = ManuallyDrop::new(src);
    let (src_ptr, src_len, src_cap) = (src.as_mut_ptr(), src.len(), src.capacity());

    if Layout::array::<Src>(src_cap) != Layout::array::<Dst>(src_cap) {
        return None;
    }

    // Safety: `Src` and `Dest` types have the same layout for an array of
    // `src_cap` elements, so the allocation is compatible, all bytes of `Src`
    // are initialized and any bit pattern is valid as a value of `Dst`.
    Some(unsafe { Vec::from_raw_parts(src_ptr as *mut Dst, src_len, src_cap) })
}

/// Types which can be viewed as a slice of bytes.
///
/// # Safety
///
/// To implement this trait, types must:
///
/// - Have a defined layout (eg. using `#[repr(C)]`)
/// - Contain no padding bytes or other uninitialized bytes
/// - Contain no interior mutability
pub unsafe trait AsBytes: Sized {
    /// Transmute a reference to a byte slice.
    fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self as *const Self as *const u8, size_of::<Self>()) }
    }

    /// Transmute a value to a signed byte slice.
    fn as_signed_bytes(&self) -> &[i8] {
        unsafe { std::slice::from_raw_parts(self as *const Self as *const i8, size_of::<Self>()) }
    }
}

/// Types which can be transmuted from a slice of bytes.
///
/// # Safety
///
/// To implement this trait, types must:
///
/// - Allow all bit patterns
/// - Contain no interior mutability
pub unsafe trait FromBytes: Sized {
    /// Transmute a reference to a byte slice into a reference to this type.
    ///
    /// Panics of the size or alignment of the slice does not match that required
    /// for `self`.
    fn from_bytes(bytes: &[u8]) -> &Self {
        assert!(
            bytes.len() == size_of::<Self>()
                && (bytes.as_ptr() as usize).is_multiple_of(align_of::<Self>())
        );
        unsafe { &*bytes.as_ptr().cast::<Self>() }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use super::{
        AsBytes, FromByteArray, FromBytes, cast_mut_slice, cast_slice, cast_uninit_mut_slice,
        cast_vec,
    };

    #[test]
    fn test_cast_bytes() {
        let float_val = 1.2f32;
        let int_val: i32 = float_val.cast_bytes();
        assert_eq!(int_val, i32::from_ne_bytes(float_val.to_ne_bytes()));
    }

    #[test]
    fn test_cast_slice() {
        // Convert to narrower type
        let i32s = [1, 2, 3];
        let i8s = cast_slice::<i32, i8>(&i32s).unwrap();
        assert_eq!(i8s.as_ptr(), i32s.as_ptr() as *const i8);
        assert_eq!(i8s.len(), i32s.len() * 4);

        // Convert back to wider type
        let i32s_v2 = cast_slice::<i8, i32>(&i8s).unwrap();
        assert_eq!(i32s_v2, i32s);
    }

    #[test]
    fn test_cast_vec() {
        // Compatible types
        let i32s = Vec::from([1, 2, 3]);
        let (ptr, len, cap) = (i32s.as_ptr(), i32s.len(), i32s.capacity());
        let f32s = cast_vec::<i32, f32>(i32s).unwrap();
        assert_eq!(f32s.as_ptr(), ptr as *const f32);
        assert_eq!(f32s.len(), len);
        assert_eq!(f32s.capacity(), cap);

        // Incompatible types
        let i32s = Vec::from([1, 2, 3]);
        let i8s = cast_vec::<i32, i8>(i32s);
        assert!(i8s.is_none());
    }

    #[test]
    fn test_cast_slice_fails_if_unaligned() {
        let i8s = [1, 2, 3, 4, 5];
        let i32s_a = cast_slice::<i8, i32>(&i8s);
        let i32s_b = cast_slice::<i8, i32>(&i8s[1..]);

        // At least one of `i32s_a` or `i32s_b`` will be incorrectly aligned for i32.
        assert!(i32s_a.is_none() || i32s_b.is_none());
    }

    #[test]
    fn test_cast_slice_fails_if_size_not_multiple_of_dst_size() {
        let i8s = [1, 2, 3, 4, 5];
        let i32s = cast_slice::<i8, i32>(&i8s);
        assert!(i32s.is_none());
    }

    #[test]
    fn test_cast_mut_slice() {
        let mut i32s = [1, 2, 3];
        let i32s_ptr = i32s.as_ptr();
        let i8s = cast_mut_slice::<i32, i8>(&mut i32s).unwrap();
        assert_eq!(i8s.as_ptr(), i32s_ptr as *const i8);
        assert_eq!(i8s.len(), i32s.len() * 4);
    }

    #[test]
    fn test_cast_uninit_mut_slice() {
        let mut i32s = [1, 2, 3].map(MaybeUninit::new);
        let i32s_ptr = i32s.as_ptr();
        let i8s = cast_uninit_mut_slice::<i32, i8>(&mut i32s).unwrap();
        assert_eq!(i8s.as_ptr(), i32s_ptr as *const MaybeUninit<i8>);
        assert_eq!(i8s.len(), i32s.len() * 4);
    }

    #[test]
    fn test_as_bytes_and_from_bytes() {
        #[derive(Debug, PartialEq)]
        struct SomeData {
            field_a: [i32; 2],
            field_b: [i32; 2],
        }
        // Safety: SomeData meets requirements for AsBytes, FromBytes.
        unsafe impl AsBytes for SomeData {}
        unsafe impl FromBytes for SomeData {}

        let data = SomeData {
            field_a: [1, 2],
            field_b: [3, 4],
        };
        let bytes = data.as_bytes();

        let recreated_data = SomeData::from_bytes(bytes);
        assert_eq!(recreated_data, &data);
    }
}
