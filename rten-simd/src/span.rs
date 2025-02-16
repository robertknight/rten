//! Slice-like types used as inputs and outputs for vectorized operations.

use std::mem::{transmute, MaybeUninit};

enum SrcDestInner<'a, T> {
    InOut(&'a [T], &'a mut [MaybeUninit<T>]),
    InMut(&'a mut [T]),
}

/// Input-output buffer for vectorized operations.
///
/// This can either be a single mutable buffer for operations that execute
/// in-place (`&mut [T]`) or a pair of input and output buffers where the
/// output is uninitialized (`([T], &mut [MaybeUninit<T>])`) and both buffers
/// must have the same length.
pub struct SrcDest<'a, T: Copy> {
    inner: SrcDestInner<'a, T>,
}

impl<'a, T: Copy> SrcDest<'a, T> {
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
    pub unsafe fn dest_assume_init(self) -> &'a mut [T] {
        match self.inner {
            SrcDestInner::InOut(_src, dest) => transmute::<&mut [MaybeUninit<T>], &mut [T]>(dest),
            SrcDestInner::InMut(src) => src,
        }
    }
}

impl<'a, T: Copy> From<(&'a [T], &'a mut [MaybeUninit<T>])> for SrcDest<'a, T> {
    fn from(val: (&'a [T], &'a mut [MaybeUninit<T>])) -> Self {
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

impl<'a, T: Copy> From<&'a mut [T]> for SrcDest<'a, T> {
    fn from(val: &'a mut [T]) -> Self {
        SrcDest {
            inner: SrcDestInner::InMut(val),
        }
    }
}
