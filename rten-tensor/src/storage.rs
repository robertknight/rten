use std::borrow::Cow;
use std::marker::PhantomData;
use std::ops::Range;

/// Trait for backing storage used by tensors and views.
///
/// This provides common operations needed by the tensor implementations.
/// Owned tensors and views additionally implement [StorageMut].
pub trait Storage {
    /// The element type.
    type Elem;

    /// Return the number of elements in the storage.
    fn len(&self) -> usize;

    /// Return true if `self.len() == 0`.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return a pointer to the first element in the storage.
    fn as_ptr(&self) -> *const Self::Elem;

    /// Return the element at a given offset, or None if `offset >= self.len()`.
    fn get(&self, offset: usize) -> Option<&Self::Elem>;

    /// Return a reference to the element at `offset`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `offset < len`.
    unsafe fn get_unchecked(&self, offset: usize) -> &Self::Elem;

    /// Return a view of a sub-region of the storage.
    fn slice(&self, range: Range<usize>) -> ViewData<Self::Elem> {
        assert!(range.end <= self.len());
        ViewData {
            // Safety: `range.start < range.end` and `range.end <= self.len())`,
            // so this is in-bounds.
            ptr: unsafe { self.as_ptr().add(range.start) },
            len: range.len(),
            _marker: PhantomData,
        }
    }

    /// Shorthand for `self.slice(0..self.len())`.
    fn view(&self) -> ViewData<Self::Elem> {
        self.slice(0..self.len())
    }

    /// Return the contents of the storage as a slice.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no mutable references exist to any element
    /// in the storage.
    unsafe fn as_slice(&self) -> &[Self::Elem] {
        std::slice::from_raw_parts(self.as_ptr(), self.len())
    }
}

/// Trait for converting owned and borrowed element containers (`Vec<T>`, slices)
/// into their corresponding `Storage` type.
///
/// This is used by [`Tensor::from_data`](crate::TensorBase::from_data).
pub trait IntoStorage {
    type Output: Storage;

    fn into_storage(self) -> Self::Output;
}

impl<T> IntoStorage for Vec<T> {
    type Output = Self;

    fn into_storage(self) -> Self {
        self
    }
}

impl<'a, T> IntoStorage for &'a [T] {
    type Output = ViewData<'a, T>;

    fn into_storage(self) -> ViewData<'a, T> {
        ViewData {
            ptr: self.as_ptr(),
            len: self.len(),
            _marker: PhantomData,
        }
    }
}

impl<'a, T, const N: usize> IntoStorage for &'a [T; N] {
    type Output = ViewData<'a, T>;

    fn into_storage(self) -> ViewData<'a, T> {
        self.as_slice().into_storage()
    }
}

impl<'a, T> IntoStorage for &'a mut [T] {
    type Output = ViewMutData<'a, T>;

    fn into_storage(self) -> ViewMutData<'a, T> {
        ViewMutData {
            ptr: self.as_mut_ptr(),
            len: self.len(),
            _marker: PhantomData,
        }
    }
}

/// Trait for backing storage used by mutable tensors and views.
///
/// This provides common operations needed by the tensor implementation.
pub trait StorageMut: Storage {
    /// Return a mutable pointer to the first element in storage.
    fn as_mut_ptr(&mut self) -> *mut Self::Elem;

    /// Mutable version of [Storage::get].
    fn get_mut(&mut self, offset: usize) -> Option<&mut Self::Elem>;

    /// Mutable version of [Storage::get_unchecked].
    ///
    /// # Safety
    ///
    /// The caller must ensure that `offset < self.len()`.
    unsafe fn get_unchecked_mut(&mut self, offset: usize) -> &mut Self::Elem;

    /// Return a slice of this storage.
    fn slice_mut(&mut self, range: Range<usize>) -> ViewMutData<Self::Elem> {
        assert!(range.end <= self.len());
        ViewMutData {
            // Safety: `range.start <= self.len()`
            ptr: unsafe { self.as_mut_ptr().add(range.start) },
            len: range.len(),
            _marker: PhantomData,
        }
    }

    /// Shorthand for `self.slice_mut(0..self.len())`.
    fn view_mut(&mut self) -> ViewMutData<Self::Elem> {
        self.slice_mut(0..self.len())
    }

    /// Return the stored elements as a mutable slice.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the storage is contiguous (ie. no used
    /// elements) and that there are no references to any elements in the
    /// storage.
    unsafe fn as_slice_mut(&mut self) -> &mut [Self::Elem] {
        std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len())
    }
}

impl<T> Storage for Vec<T> {
    type Elem = T;

    fn len(&self) -> usize {
        self.len()
    }

    fn get(&self, offset: usize) -> Option<&T> {
        self.as_slice().get(offset)
    }

    fn as_ptr(&self) -> *const T {
        self.as_ptr()
    }

    unsafe fn get_unchecked(&self, offset: usize) -> &T {
        self.as_slice().get_unchecked(offset)
    }
}

impl<T> StorageMut for Vec<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.as_mut_ptr()
    }

    fn get_mut(&mut self, offset: usize) -> Option<&mut T> {
        self.as_mut_slice().get_mut(offset)
    }

    unsafe fn get_unchecked_mut(&mut self, offset: usize) -> &mut T {
        self.as_mut_slice().get_unchecked_mut(offset)
    }
}

/// Storage for an immutable tensor view.
///
/// This has the same representation in memory as a slice: a pointer and a
/// length. Unlike a slice it allows for other storage objects to reference
/// memory ranges that overlap with this one. It is up to APIs built on top of
/// this (ie. [Tensor](crate::Tensor)) to ensure that consumers cannot obtain
/// multiple mutable references to the same element.
#[derive(Debug)]
pub struct ViewData<'a, T> {
    ptr: *const T,
    len: usize,
    _marker: PhantomData<&'a T>,
}

unsafe impl<'a, T> Send for ViewData<'a, T> {}
unsafe impl<'a, T> Sync for ViewData<'a, T> {}

impl<'a, T> Clone for ViewData<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, T> Copy for ViewData<'a, T> {}

impl<'a, T> ViewData<'a, T> {
    /// Variant of [Storage::get] which preserves lifetimes.
    pub fn get(&self, offset: usize) -> Option<&'a T> {
        if offset < self.len {
            Some(unsafe { &*self.ptr.add(offset) })
        } else {
            None
        }
    }

    /// Variant of [Storage::get_unchecked] which preserves lifetimes.
    ///
    /// # Safety
    ///
    /// See [Storage::get_unchecked].
    pub unsafe fn get_unchecked(&self, offset: usize) -> &'a T {
        debug_assert!(offset < self.len);
        &*self.ptr.add(offset)
    }

    /// Variant of [Storage::slice] which preserves lifetimes.
    pub fn slice(&self, range: Range<usize>) -> ViewData<'a, T> {
        assert!(range.end <= self.len());
        ViewData {
            // Safety: `range.start < range.end` and `range.end <= self.len())`,
            // so this is in-bounds.
            ptr: unsafe { self.as_ptr().add(range.start) },
            len: range.len(),
            _marker: PhantomData,
        }
    }

    /// Variant of [Storage::view] which preserves lifetimes.
    pub fn view(&self) -> ViewData<'a, T> {
        self.slice(0..self.len())
    }

    /// Return the contents of the storage as a slice.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no mutable references exist to any element
    /// in the storage.
    pub unsafe fn as_slice(&self) -> &'a [T] {
        std::slice::from_raw_parts(self.ptr, self.len)
    }
}

impl<'a, T> Storage for ViewData<'a, T> {
    type Elem = T;

    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, offset: usize) -> Option<&T> {
        if offset < self.len {
            Some(unsafe { &*self.ptr.add(offset) })
        } else {
            None
        }
    }

    fn as_ptr(&self) -> *const T {
        self.ptr
    }

    unsafe fn get_unchecked(&self, offset: usize) -> &T {
        debug_assert!(offset < self.len);
        &*self.ptr.add(offset)
    }
}

/// Storage for a mutable tensor view.
///
/// This has the same representation in memory as a mutable slice: a pointer
/// and a length. Unlike a slice it allows for other storage objects to
/// reference memory ranges that overlap with this one. It is up to
/// APIs built on top of this (ie. [Tensor](crate::Tensor)) to ensure that
/// consumers cannot obtain multiple mutable references to the same element.
#[derive(Debug)]
pub struct ViewMutData<'a, T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<&'a mut T>,
}
unsafe impl<'a, T> Send for ViewMutData<'a, T> {}

impl<'a, T> ViewMutData<'a, T> {
    /// Variant of [StorageMut::as_slice_mut] which preserves the underlying
    /// lifetime in the result.
    ///
    /// # Safety
    ///
    /// See [StorageMut::as_slice_mut].
    pub unsafe fn to_slice_mut(mut self) -> &'a mut [T] {
        std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len())
    }
}

impl<'a, T> Storage for ViewMutData<'a, T> {
    type Elem = T;

    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, offset: usize) -> Option<&T> {
        if offset < self.len {
            Some(unsafe { &*self.ptr.add(offset) })
        } else {
            None
        }
    }

    fn as_ptr(&self) -> *const T {
        self.ptr
    }

    unsafe fn get_unchecked(&self, offset: usize) -> &T {
        debug_assert!(offset < self.len);
        &*self.ptr.add(offset)
    }
}

impl<'a, T> StorageMut for ViewMutData<'a, T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    fn get_mut(&mut self, offset: usize) -> Option<&mut T> {
        if offset < self.len {
            Some(unsafe { &mut *self.ptr.add(offset) })
        } else {
            None
        }
    }

    unsafe fn get_unchecked_mut(&mut self, offset: usize) -> &mut T {
        &mut *self.ptr.add(offset)
    }
}

impl<'a, T> Storage for Cow<'a, [T]>
where
    [T]: ToOwned,
{
    type Elem = T;

    fn len(&self) -> usize {
        self.as_ref().len()
    }

    fn get(&self, offset: usize) -> Option<&T> {
        self.as_ref().get(offset)
    }

    fn as_ptr(&self) -> *const T {
        self.as_ref().as_ptr()
    }

    unsafe fn get_unchecked(&self, offset: usize) -> &T {
        self.as_slice().get_unchecked(offset)
    }
}
