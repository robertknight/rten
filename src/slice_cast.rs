use std::mem::MaybeUninit;

/// Marker trait for "plain old data".
///
/// POD types which are simple value types that impl `Copy`, have no padding,
/// and for which any bit pattern is valid.
///
/// This means an arbitrary byte sequence can be converted to this type, as
/// long as the byte sequence length is a multiple of the type's size.
pub trait Pod: Copy {}
impl Pod for i8 {}
impl Pod for u8 {}
impl Pod for f32 {}
impl Pod for i32 {}
impl Pod for u32 {}
impl Pod for u64 {}
impl<T: Pod> Pod for MaybeUninit<T> {}

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
pub fn cast_pod_mut_slice<Src: Pod, Dst: Pod>(src: &mut [Src]) -> Option<&mut [Dst]> {
    let new_len = transmuted_slice_len::<_, Dst>(src)?;

    // Safety:
    // - Pointer cast is safe since any bit pattern is valid for POD types
    // - Length has been adjusted for `Dst` type
    Some(unsafe { std::slice::from_raw_parts_mut(src.as_mut_ptr() as *mut Dst, new_len) })
}

#[cfg(test)]
mod tests {
    use super::{cast_pod_mut_slice, cast_pod_slice};

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
}
