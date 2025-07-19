use std::alloc::Layout;
use std::mem::{ManuallyDrop, MaybeUninit};

/// Marker trait for "plain old data".
///
/// POD types which are simple value types that impl `Copy`, have no padding,
/// and for which any bit pattern is valid.
///
/// This means an arbitrary byte sequence can be converted to this type, as
/// long as the byte sequence length is a multiple of the type's size.
///
/// # Safety
///
/// This type must only be implemented for types which are initialized and for
/// which any bit pattern is valid.
pub unsafe trait Pod: Copy {
    /// View of this type as an array of bytes.
    type Bytes: AsRef<[u8]>;

    /// Convert this type to an array of bytes in native order.
    ///
    /// This is the same as `to_ne_bytes` for primitive types.
    fn to_bytes(self) -> Self::Bytes;

    /// Convert an array of bytes, in native order, into this type.
    ///
    /// This is the same as `from_ne_bytes` for primitive types.
    fn from_bytes(bytes: Self::Bytes) -> Self;

    /// Convert this type to another of the same size.
    ///
    /// This should compile to a zero-cost transmute.
    fn cast_bytes<Dst>(self) -> Dst
    where
        Dst: Pod<Bytes = Self::Bytes>,
    {
        Dst::from_bytes(self.to_bytes())
    }
}

macro_rules! impl_pod {
    ($type:ty) => {
        unsafe impl Pod for $type {
            type Bytes = [u8; size_of::<$type>()];

            fn to_bytes(self) -> Self::Bytes {
                self.to_ne_bytes()
            }

            fn from_bytes(val: Self::Bytes) -> Self {
                Self::from_ne_bytes(val)
            }
        }
    };
}
impl_pod!(i8);
impl_pod!(u8);
impl_pod!(f32);
impl_pod!(i32);
impl_pod!(u32);
impl_pod!(u64);

/// Return the length of a slice transmuted from `Src` to `Dst`, or `None` if
/// the transmute is not possible.
fn transmuted_slice_len<Src, Dst>(src: &[Src]) -> Option<usize> {
    if (src.as_ptr() as usize) % align_of::<Dst>() != 0 {
        return None;
    }

    let src_byte_len = std::mem::size_of_val(src);
    if src_byte_len % size_of::<Dst>() != 0 {
        return None;
    }

    Some(src_byte_len / size_of::<Dst>())
}

/// Transmute a slice of elements from one [`Pod`] type to another.
///
/// This cast is safe because all bit patterns are valid for `Pod` elements.
///
/// Returns `None` if the source pointer is not correctly aligned for the
/// destination type.
pub fn cast_pod_slice<Src: Pod, Dst: Pod>(src: &[Src]) -> Option<&[Dst]> {
    let new_len = transmuted_slice_len::<_, Dst>(src)?;

    // Safety:
    // - Pointer cast is safe since any bit pattern is valid for POD types
    // - Length has been adjusted for `Dst` type
    Some(unsafe { std::slice::from_raw_parts(src.as_ptr() as *const Dst, new_len) })
}

/// Transmute a mutable slice of elements from one [`Pod`] type to another.
///
/// This cast is safe because all bit patterns are valid for `Pod` elements.
///
/// Returns `None` if the source pointer is not correctly aligned for the
/// destination type.
#[allow(unused)]
pub fn cast_pod_mut_slice<Src: Pod, Dst: Pod>(src: &mut [Src]) -> Option<&mut [Dst]> {
    let new_len = transmuted_slice_len::<_, Dst>(src)?;

    // Safety:
    // - Pointer cast is safe since any bit pattern is valid for POD types
    // - Length has been adjusted for `Dst` type
    Some(unsafe { std::slice::from_raw_parts_mut(src.as_mut_ptr() as *mut Dst, new_len) })
}

/// Transmute a mutable slice of elements from one uninitialized [`Pod`] type to another.
///
/// Returns `None` if the source pointer is not correctly aligned for the
/// destination type.
pub fn cast_uninit_pod_mut_slice<Src: Pod, Dst: Pod>(
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

/// Transmute a vector of elements from one [`Pod`] type to another.
///
/// The source and destination types must have the same size and alignment.
/// The length and capacity of the vector will be the same afterwards.
pub fn cast_pod_vec<Src: Pod, Dst: Pod>(src: Vec<Src>) -> Option<Vec<Dst>> {
    // From `Vec::into_raw_parts`.
    let mut src = ManuallyDrop::new(src);
    let (src_ptr, src_len, src_cap) = (src.as_mut_ptr(), src.len(), src.capacity());

    if Layout::array::<Src>(src_cap) != Layout::array::<Dst>(src_cap) {
        return None;
    }

    // Safety: `Src` and `Dest` types have the same layout for an array of
    // `src_cap` elements, so the allocation is compatible, and are `Pod` types
    // so we can transmute any value of `Src` to a value of `Dst`.
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
            bytes.len() == size_of::<Self>() && bytes.as_ptr() as usize % align_of::<Self>() == 0
        );
        unsafe { &*bytes.as_ptr().cast::<Self>() }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use super::{
        cast_pod_mut_slice, cast_pod_slice, cast_pod_vec, cast_uninit_pod_mut_slice, AsBytes,
        FromBytes, Pod,
    };

    #[test]
    fn test_cast_pod() {
        let float_val = 1.2f32;
        let int_val: i32 = float_val.cast_bytes();
        assert_eq!(int_val, i32::from_ne_bytes(float_val.to_ne_bytes()));
    }

    #[test]
    fn test_cast_pod_slice() {
        // Convert to narrower type
        let i32s = [1, 2, 3];
        let i8s = cast_pod_slice::<i32, i8>(&i32s).unwrap();
        assert_eq!(i8s.as_ptr(), i32s.as_ptr() as *const i8);
        assert_eq!(i8s.len(), i32s.len() * 4);

        // Convert back to wider type
        let i32s_v2 = cast_pod_slice::<i8, i32>(&i8s).unwrap();
        assert_eq!(i32s_v2, i32s);
    }

    #[test]
    fn test_cast_pod_vec() {
        // Compatible types
        let i32s = Vec::from([1, 2, 3]);
        let (ptr, len, cap) = (i32s.as_ptr(), i32s.len(), i32s.capacity());
        let f32s = cast_pod_vec::<i32, f32>(i32s).unwrap();
        assert_eq!(f32s.as_ptr(), ptr as *const f32);
        assert_eq!(f32s.len(), len);
        assert_eq!(f32s.capacity(), cap);

        // Incompatible types
        let i32s = Vec::from([1, 2, 3]);
        let i8s = cast_pod_vec::<i32, i8>(i32s);
        assert!(i8s.is_none());
    }

    #[test]
    fn test_cast_pod_slice_fails_if_unaligned() {
        let i8s = [1, 2, 3, 4, 5];
        let i32s_a = cast_pod_slice::<i8, i32>(&i8s);
        let i32s_b = cast_pod_slice::<i8, i32>(&i8s[1..]);

        // At least one of `i32s_a` or `i32s_b`` will be incorrectly aligned for i32.
        assert!(i32s_a.is_none() || i32s_b.is_none());
    }

    #[test]
    fn test_cast_pod_slice_fails_if_size_not_multiple_of_dst_size() {
        let i8s = [1, 2, 3, 4, 5];
        let i32s = cast_pod_slice::<i8, i32>(&i8s);
        assert!(i32s.is_none());
    }

    #[test]
    fn test_cast_pod_mut_slice() {
        let mut i32s = [1, 2, 3];
        let i32s_ptr = i32s.as_ptr();
        let i8s = cast_pod_mut_slice::<i32, i8>(&mut i32s).unwrap();
        assert_eq!(i8s.as_ptr(), i32s_ptr as *const i8);
        assert_eq!(i8s.len(), i32s.len() * 4);
    }

    #[test]
    fn test_cast_uninit_pod_mut_slice() {
        let mut i32s = [1, 2, 3].map(MaybeUninit::new);
        let i32s_ptr = i32s.as_ptr();
        let i8s = cast_uninit_pod_mut_slice::<i32, i8>(&mut i32s).unwrap();
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
