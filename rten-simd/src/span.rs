//! Slice-like types without the restrictions on aliasing.

use std::mem::MaybeUninit;

/// Const pointer to a range of `T`s.
///
/// This is like an `&[T]`, but without the guarantee that no mutable aliases
/// exist. This is useful as it enables re-using the same unsafe code for
/// mutating and non-mutating variants of a function.
#[derive(Copy, Clone)]
pub struct PtrLen<T> {
    ptr: *const T,
    len: usize,
}

impl<T> PtrLen<T> {
    pub fn ptr(&self) -> *const T {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<'a, T> From<&'a [T]> for PtrLen<T> {
    fn from(val: &'a [T]) -> PtrLen<T> {
        PtrLen {
            ptr: val.as_ptr(),
            len: val.len(),
        }
    }
}

impl<'a, T> From<&'a mut [T]> for PtrLen<T> {
    fn from(val: &'a mut [T]) -> PtrLen<T> {
        PtrLen {
            ptr: val.as_ptr(),
            len: val.len(),
        }
    }
}

impl<T> From<MutPtrLen<T>> for PtrLen<T> {
    fn from(val: MutPtrLen<T>) -> PtrLen<T> {
        PtrLen {
            ptr: val.ptr,
            len: val.len,
        }
    }
}

/// Mutable pointer to a range of `T`s.
///
/// This is like an `&mut [T]`, but without the guarantee that no aliases exist.
#[derive(Copy, Clone)]
pub struct MutPtrLen<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> MutPtrLen<T> {
    pub fn ptr(&self) -> *mut T {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> MutPtrLen<MaybeUninit<T>> {
    /// Promise that the span of `T`s that are pointed to have been initialized.
    ///
    /// # Safety
    ///
    /// The caller must ensure that all elements referenced by this range have
    /// been initialized.
    pub unsafe fn assume_init(self) -> MutPtrLen<T> {
        MutPtrLen {
            ptr: unsafe { std::mem::transmute(self.ptr) },
            len: self.len,
        }
    }
}

impl<T> MutPtrLen<T> {
    /// Transmute a span of initialized `T`s to uninitialized `T`s.
    pub fn as_uninit(self) -> MutPtrLen<MaybeUninit<T>>
    where
        T: Copy,
    {
        MutPtrLen {
            ptr: unsafe { std::mem::transmute(self.ptr) },
            len: self.len,
        }
    }
}

impl<'a, T> From<&'a mut [T]> for MutPtrLen<T> {
    fn from(val: &'a mut [T]) -> MutPtrLen<T> {
        MutPtrLen {
            ptr: val.as_mut_ptr(),
            len: val.len(),
        }
    }
}
