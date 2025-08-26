//! Slice-like types used as inputs and outputs for vectorized operations.

use std::mem::{MaybeUninit, transmute};

enum SrcDestInner<'src, 'dst, T> {
    InOut(&'src [T], &'dst mut [MaybeUninit<T>]),
    InMut(&'dst mut [T]),
}

/// Input-output buffer for vectorized operations.
///
/// This can either be a single mutable buffer for operations that execute
/// in-place (`&mut [T]`) or a pair of input and output buffers where the
/// output is uninitialized (`([T], &mut [MaybeUninit<T>])`) and both buffers
/// must have the same length.
pub struct SrcDest<'src, 'dst, T: Copy> {
    inner: SrcDestInner<'src, 'dst, T>,
}

impl<'dst, T: Copy> SrcDest<'_, 'dst, T> {
    /// Return the source slice.
    pub fn src(&self) -> &[T] {
        match &self.inner {
            SrcDestInner::InOut(src, _dest) => src,
            SrcDestInner::InMut(src_mut) => src_mut,
        }
    }

    /// Return the length of the input and output slices.
    pub fn len(&self) -> usize {
        self.src().len()
    }

    /// Return true if the input and output slices are empty.
    pub fn is_empty(&self) -> bool {
        self.src().is_empty()
    }

    /// Return source and destination slice pointers and the length.
    ///
    /// The source and destination will either alias, or the destination will
    /// be a non-aliasing, uninitialized slice.
    pub fn src_dest_ptr(&mut self) -> (*const T, *mut MaybeUninit<T>, usize) {
        match &mut self.inner {
            SrcDestInner::InOut(src, dest) => (src.as_ptr(), dest.as_mut_ptr(), src.len()),
            SrcDestInner::InMut(src) => (
                src.as_ptr(),
                src.as_mut_ptr() as *mut MaybeUninit<T>,
                src.len(),
            ),
        }
    }

    /// Return the initialized destination slice.
    ///
    /// # Safety
    ///
    /// If this instance was constructed with an uninitialized destination
    /// buffer, all elements must have been initialized before this is called.
    pub unsafe fn dest_assume_init(self) -> &'dst mut [T] {
        match self.inner {
            SrcDestInner::InOut(_src, dest) => unsafe {
                transmute::<&mut [MaybeUninit<T>], &mut [T]>(dest)
            },
            SrcDestInner::InMut(src) => src,
        }
    }
}

impl<'src, 'dst, T: Copy> From<(&'src [T], &'dst mut [MaybeUninit<T>])> for SrcDest<'src, 'dst, T> {
    fn from(val: (&'src [T], &'dst mut [MaybeUninit<T>])) -> Self {
        let (src, dest) = val;
        assert_eq!(
            src.len(),
            dest.len(),
            "src len {} != dest len {}",
            src.len(),
            dest.len(),
        );
        SrcDest {
            inner: SrcDestInner::InOut(src, dest),
        }
    }
}

impl<'dst, T: Copy> From<&'dst mut [T]> for SrcDest<'dst, 'dst, T> {
    fn from(val: &'dst mut [T]) -> Self {
        SrcDest {
            inner: SrcDestInner::InMut(val),
        }
    }
}
