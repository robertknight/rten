use std::borrow::Cow;
use std::marker::PhantomData;
use std::ops::Range;

/// Trait for backing storage used by tensors and views.
///
/// Mutable tensors have storage which also implement [StorageMut].
///
/// This specifies a contiguous array of elements in memory, as a pointer and a
/// length. The storage may be owned or borrowed. For borrowed storage, there
/// may be other storage whose ranges overlap. This is necessary to support
/// mutable views of non-contiguous tensors (eg. independent columns of a
/// matrix, whose data is stored in row-major order).
///
/// # Safety
///
/// Since different storage objects can have memory ranges that overlap, it is
/// up to the caller to ensure that mutable tensors cannot logically overlap any
/// other tensors. In other words, whenever a mutable tensor is split or sliced
/// or iterated, it should not be possible to get duplicate mutable references
/// to the same elements from those views.
///
/// Implementations of this trait must ensure that the
/// [`as_ptr`](Storage::as_ptr) and [`len`](Storage::len) methods define a valid
/// range of memory within the same allocated object, which is correctly aligned
/// for the `Elem` type. For the case where the storage is contiguous, these
/// requirements are the same as
/// [`slice::from_raw_parts`](std::slice::from_raw_parts).
pub unsafe trait Storage {
    /// The element type.
    type Elem;

    /// Return the number of elements in the storage.
    fn len(&self) -> usize;

    /// Return true if the storage contains no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return a pointer to the first element in the storage.
    fn as_ptr(&self) -> *const Self::Elem;

    /// Return the element at a given offset, or None if `offset >= self.len()`.
    ///
    /// # Safety
    ///
    /// - The caller must ensure that no mutable references to the same element
    ///   can be created.
    unsafe fn get(&self, offset: usize) -> Option<&Self::Elem> {
        if offset < self.len() {
            Some(&*self.as_ptr().add(offset))
        } else {
            None
        }
    }

    /// Return a reference to the element at `offset`.
    ///
    /// # Safety
    ///
    /// This has the same safety requirements as [`get`](Storage::get) plus
    /// the caller must ensure that `offset < len`.
    unsafe fn get_unchecked(&self, offset: usize) -> &Self::Elem {
        debug_assert!(offset < self.len());
        &*self.as_ptr().add(offset)
    }

    /// Return a view of a sub-region of the storage.
    ///
    /// Panics if the range is out of bounds.
    fn slice(&self, range: Range<usize>) -> ViewData<Self::Elem> {
        assert!(
            range.start <= self.len() && range.end <= self.len(),
            "invalid slice range {:?} for storage length {}",
            range,
            self.len()
        );
        ViewData {
            // Safety: We verified that `range` is in bounds.
            ptr: unsafe { self.as_ptr().add(range.start) },
            len: range.len(),
            _marker: PhantomData,
        }
    }

    /// Return an immutable view of this storage.
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

impl<T: Storage> IntoStorage for T {
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

fn assert_storage_range_valid<S: Storage + ?Sized>(storage: &S, range: Range<usize>) {
    assert!(
        range.start <= storage.len() && range.end <= storage.len(),
        "invalid slice range {:?} for storage length {}",
        range,
        storage.len()
    );
}

/// Trait for backing storage used by mutable tensors and views.
///
/// This extends [Storage] with methods to get mutable pointers and references
/// to elements in the storage.
///
/// # Safety
///
/// The [`as_mut_ptr`](StorageMut::as_mut_ptr) method has the same safety
/// requirements as [`Storage::as_ptr`]. The result of `as_mut_ptr` must also
/// be equal to `as_ptr`.
pub unsafe trait StorageMut: Storage {
    /// Return a mutable pointer to the first element in storage.
    fn as_mut_ptr(&mut self) -> *mut Self::Elem;

    /// Mutable version of [Storage::get].
    ///
    /// # Safety
    ///
    /// This has the same safety requirements as [`get`](Storage::get).
    unsafe fn get_mut(&mut self, offset: usize) -> Option<&mut Self::Elem> {
        if offset < self.len() {
            Some(&mut *self.as_mut_ptr().add(offset))
        } else {
            None
        }
    }

    /// Mutable version of [Storage::get_unchecked].
    ///
    /// # Safety
    ///
    /// This has the same requirement as [`get_mut`](StorageMut::get_mut) plus
    /// the caller must ensure that `offset < self.len()`.
    unsafe fn get_unchecked_mut(&mut self, offset: usize) -> &mut Self::Elem {
        debug_assert!(offset < self.len());
        &mut *self.as_mut_ptr().add(offset)
    }

    /// Return a slice of this storage.
    fn slice_mut(&mut self, range: Range<usize>) -> ViewMutData<Self::Elem> {
        assert_storage_range_valid(self, range.clone());
        ViewMutData {
            // Safety: We verified that `range` is in bounds.
            ptr: unsafe { self.as_mut_ptr().add(range.start) },
            len: range.len(),
            _marker: PhantomData,
        }
    }

    /// Return two sub-views of the storage.
    ///
    /// Unlike splitting a slice, this does *not* ensure that the two halves
    /// do not overlap, only that the "left" and "right" ranges are valid.
    fn split_mut(
        &mut self,
        left: Range<usize>,
        right: Range<usize>,
    ) -> (ViewMutData<Self::Elem>, ViewMutData<Self::Elem>) {
        assert_storage_range_valid(self, left.clone());
        assert_storage_range_valid(self, right.clone());

        let ptr = self.as_mut_ptr();
        let left = ViewMutData {
            ptr: unsafe { ptr.add(left.start) },
            len: left.len(),
            _marker: PhantomData,
        };
        let right = ViewMutData {
            ptr: unsafe { ptr.add(right.start) },
            len: right.len(),
            _marker: PhantomData,
        };
        (left, right)
    }

    /// Return a mutable view of this storage.
    fn view_mut(&mut self) -> ViewMutData<Self::Elem> {
        self.slice_mut(0..self.len())
    }

    /// Return the stored elements as a mutable slice.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the storage is contiguous (ie. no unused
    /// elements) and that there are no other references to any elements in the
    /// storage.
    unsafe fn as_slice_mut(&mut self) -> &mut [Self::Elem] {
        std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len())
    }
}

unsafe impl<T> Storage for Vec<T> {
    type Elem = T;

    fn len(&self) -> usize {
        self.len()
    }

    fn as_ptr(&self) -> *const T {
        self.as_ptr()
    }
}

unsafe impl<T> StorageMut for Vec<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
}

/// Storage for an immutable tensor view.
///
/// This has the same representation in memory as a slice: a pointer and a
/// length. Unlike a slice it allows for other mutable storage to reference
/// memory ranges that overlap with this one. It is up to APIs built on top of
/// this to ensure uniqueness of mutable element references.
#[derive(Debug)]
pub struct ViewData<'a, T> {
    ptr: *const T,
    len: usize,
    _marker: PhantomData<&'a T>,
}

// Safety: `ViewData` does not provide mutable access to its elements, so it
// is `Send` and `Sync`.
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
    ///
    /// # Safety
    ///
    /// See [Storage::get].
    pub unsafe fn get(&self, offset: usize) -> Option<&'a T> {
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

unsafe impl<'a, T> Storage for ViewData<'a, T> {
    type Elem = T;

    fn len(&self) -> usize {
        self.len
    }

    fn as_ptr(&self) -> *const T {
        self.ptr
    }
}

/// Storage for a mutable tensor view.
///
/// This has the same representation in memory as a mutable slice: a pointer
/// and a length. Unlike a slice it allows for other storage objects to
/// reference memory ranges that overlap with this one. It is up to
/// APIs built on top of this to ensure uniqueness of mutable references.
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

unsafe impl<'a, T> Storage for ViewMutData<'a, T> {
    type Elem = T;

    fn len(&self) -> usize {
        self.len
    }

    fn as_ptr(&self) -> *const T {
        self.ptr
    }
}

unsafe impl<'a, T> StorageMut for ViewMutData<'a, T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

/// Tensor storage which may be either owned or borrowed.
///
/// The name is taken from [std::borrow::Cow] in the standard library,
/// which is conceptually similar.
pub enum CowData<'a, T> {
    /// A [CowData] that owns its data.
    Owned(Vec<T>),
    /// A [CowData] that borrows data.
    Borrowed(ViewData<'a, T>),
}

unsafe impl<'a, T> Storage for CowData<'a, T> {
    type Elem = T;

    fn len(&self) -> usize {
        match self {
            CowData::Owned(vec) => vec.len(),
            CowData::Borrowed(view) => view.len(),
        }
    }

    fn as_ptr(&self) -> *const T {
        match self {
            CowData::Owned(vec) => vec.as_ptr(),
            CowData::Borrowed(view) => view.as_ptr(),
        }
    }
}

impl<'a, T> IntoStorage for Cow<'a, [T]>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    type Output = CowData<'a, T>;

    fn into_storage(self) -> Self::Output {
        match self {
            Cow::Owned(vec) => CowData::Owned(vec),
            Cow::Borrowed(slice) => CowData::Borrowed(slice.into_storage()),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::{IntoStorage, Storage, StorageMut, ViewData, ViewMutData};

    fn test_storage_impl<S: Storage<Elem = i32>>(s: S, expected: &[i32]) {
        // Test `len`, `get`.
        assert_eq!(s.len(), expected.len());
        for i in 0..s.len() {
            assert_eq!(unsafe { s.get(i) }, expected.get(i));
        }
        assert_eq!(unsafe { s.get(s.len()) }, None);

        // Test slicing storage.
        let range = 1..s.len() - 1;
        let slice = s.slice(range.clone());
        assert_eq!(slice.len(), range.len());
        for i in 0..slice.len() {
            assert_eq!(unsafe { slice.get(i) }, expected[range.clone()].get(i));
        }

        // Test restoring a slice.
        assert_eq!(unsafe { s.as_slice() }, expected);
    }

    #[test]
    fn test_storage() {
        let data = &mut [1, 2, 3, 4];

        let owned = data.to_vec();
        test_storage_impl(owned, data);

        let view: ViewData<i32> = data.as_slice().into_storage();
        test_storage_impl(view, data);

        let cow_view = Cow::Borrowed(data.as_slice()).into_storage();
        test_storage_impl(cow_view, data);

        let mut_view: ViewMutData<i32> = data.as_mut_slice().into_storage();
        test_storage_impl(mut_view, &[1, 2, 3, 4]);
    }

    #[test]
    #[should_panic(expected = "invalid slice range 5..2 for storage length 4")]
    fn test_storage_slice_invalid_start() {
        let data = vec![1, 2, 3, 4];
        Storage::slice(&data, 5..2);
    }

    #[test]
    #[should_panic(expected = "invalid slice range 2..5 for storage length 4")]
    fn test_storage_slice_invalid_end() {
        let data = vec![1, 2, 3, 4];
        Storage::slice(&data, 2..5);
    }

    #[test]
    #[should_panic(expected = "invalid slice range 5..2 for storage length 4")]
    fn test_storage_slice_mut_invalid_start() {
        let mut data = vec![1, 2, 3, 4];
        StorageMut::slice_mut(&mut data, 5..2);
    }

    #[test]
    #[should_panic(expected = "invalid slice range 2..5 for storage length 4")]
    fn test_storage_slice_mut_invalid_end() {
        let mut data = vec![1, 2, 3, 4];
        StorageMut::slice_mut(&mut data, 2..5);
    }
}
