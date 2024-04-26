use std::cell::RefCell;
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};

use rten_tensor::prelude::*;
use rten_tensor::{Alloc, IntoLayout, MutLayout, TensorBase};

/// A memory buffer that can be used to satisfy a future allocation from
/// a [TensorPool].
struct Buffer {
    /// Pointer and capacity extracted from the `Vec`. The length is always
    /// zero.
    ptr: *mut u8,
    capacity: usize,

    /// The original layout, based on `Layout::array`.
    layout: std::alloc::Layout,

    /// Pointer to a function that frees the buffer by reconsituting a `Vec<T>`
    /// and dropping it.
    drop: fn(&mut Buffer),
}

impl Buffer {
    /// Clear `vec` using [Vec::clear] and convert it into a buffer.
    fn from_vec<T>(mut vec: Vec<T>) -> Buffer {
        let layout = std::alloc::Layout::array::<T>(vec.capacity()).unwrap();

        vec.clear();

        let mut vec_md = std::mem::ManuallyDrop::new(vec);
        Buffer {
            ptr: vec_md.as_mut_ptr() as *mut u8,
            capacity: vec_md.capacity(),
            layout,
            drop: Buffer::release::<T>,
        }
    }

    /// Return true if this buffer could be used to satisfy an allocation
    /// request for an array of `capacity` values of type `T`.
    fn can_fit<T>(&self, capacity: usize) -> bool {
        self.layout_match::<T>() && self.capacity >= capacity
    }

    /// Convert this buffer into a zero-length vector.
    fn into_vec<T>(self) -> Vec<T> {
        // This code assumes that it is safe to transmute the buffer provided
        // that the original and new array layouts are the same.
        assert!(self.layout_match::<T>());

        let vec = unsafe { Vec::from_raw_parts(self.ptr as *mut T, 0, self.capacity) };

        // Don't drop self, as that would deallocate the buffer.
        std::mem::forget(self);

        vec
    }

    /// Test if this buffer has the same layout as one with the same capacity
    /// allocated for type `T`.
    fn layout_match<T>(&self) -> bool {
        std::alloc::Layout::array::<T>(self.capacity)
            .map(|layout| layout == self.layout)
            .unwrap_or(false)
    }

    /// Drop the buffer by reconstructing a `Vec<T>`.
    fn release<T>(this: &mut Buffer) {
        let vec = unsafe { Vec::<T>::from_raw_parts(this.ptr as *mut T, 0, this.capacity) };
        std::mem::drop(vec);
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        (self.drop)(self);
    }
}

/// A pool which enables reuse of tensor data buffers.
///
/// Reusing buffers for operator outputs, as opposed to allocating a fresh
/// buffer from the global allocator and freeing it when no longer needed,
/// can provide a significant performance improvement.
///
/// Tensors can be allocated from the pool either using the `alloc_*` methods of
/// the pool, or by passing the pool to a tensor method with an `_in` suffix as
/// the allocator (eg. [`map_in`](TensorPool::map_in). The pool will reuse an
/// existing buffer from the pool if available or allocate a new one using the
/// global allocator otherwise.
///
/// When a tensor is no longer needed, its buffer can be added to the pool using
/// [`add`](TensorPool::add), making it available for future allocations.
/// Tensors can also be wrapped in a [PoolRef] smart pointer to automatically
/// return them to the pool when the `PoolRef` is dropped.
pub struct TensorPool {
    /// List of buffers currently in the pool.
    buffers: RefCell<Vec<Buffer>>,

    /// Number of allocation requests received.
    alloc_count: RefCell<usize>,

    /// Number of allocation requests fulfilled from the pool.
    hit_count: RefCell<usize>,
}

impl TensorPool {
    /// Return a new, empty pool.
    ///
    /// This is a cheap operation that does not allocate, so it can be used
    /// to create a temporary pool to pass to a function that requires one,
    /// if the caller does not have a pool otherwise available.
    pub fn new() -> TensorPool {
        TensorPool {
            buffers: RefCell::new(Vec::new()),
            alloc_count: RefCell::new(0),
            hit_count: RefCell::new(0),
        }
    }

    /// Allocate a tensor from the pool if possible, or fall back to the
    /// global allocator otherwise.
    ///
    /// The contents of the returned tensor are uninitialized. After
    /// initializing its contents, [`assume_init`](Tensor::assume_init) can
    /// be used to mark it as initialized.
    ///
    /// When it is no longer needed, the tensor can be returned to the pool
    /// using [`add`](TensorPool::add) to make it available for subsequent
    /// allocations.
    pub fn alloc<T, S: IntoLayout>(
        &self,
        shape: S,
    ) -> TensorBase<MaybeUninit<T>, Vec<MaybeUninit<T>>, S::Layout> {
        let layout = shape.into_layout();
        let len = layout.len();

        let mut buf = self.alloc_vec(len);
        // Safety: Since the data is `MaybeUninit<T>` it is already "initialized".
        unsafe {
            buf.set_len(len);
        }

        TensorBase::from_data(layout.shape(), buf)
    }

    /// Allocate a tensor using [`alloc`](TensorPool::alloc) and fill all
    /// entries with zero.
    pub fn alloc_zeroed<T: Copy + Default, S: IntoLayout>(
        &self,
        shape: S,
    ) -> TensorBase<T, Vec<T>, S::Layout> {
        let mut tensor = self.alloc(shape);
        tensor.fill(MaybeUninit::new(T::default()));

        // Safety: We just populated all the elements.
        unsafe { tensor.assume_init() }
    }

    /// Allocate an empty vec with a given capacity from the pool.
    ///
    /// This is useful for scenarios where the data buffer for a tensor is
    /// built up and then passed to `Tensor::from_data`.
    pub fn alloc_vec<T>(&self, required_len: usize) -> Vec<T> {
        *self.alloc_count.borrow_mut() += 1;

        // Find best fit item that matches the requested type and size with
        // the least excess capacity.
        let best_fit =
            self.buffers
                .borrow()
                .iter()
                .enumerate()
                .fold(None, |best_fit, (idx, buffer)| {
                    if !buffer.can_fit::<T>(required_len) {
                        return best_fit;
                    };

                    if let Some((best_fit_idx, best_fit_size)) = best_fit {
                        if buffer.capacity >= best_fit_size {
                            return Some((best_fit_idx, best_fit_size));
                        }
                    }
                    Some((idx, buffer.capacity))
                });

        let data = if let Some((best_fit, _overhead)) = best_fit {
            *self.hit_count.borrow_mut() += 1;

            let item = self.buffers.borrow_mut().remove(best_fit);
            item.into_vec::<T>()
        } else {
            // No match :( - Fall back to the global allocator.
            Vec::with_capacity(required_len)
        };

        data
    }

    /// Add the data buffer from a tensor into the pool, so it can be used
    /// to satisfy future calls to [`alloc`](TensorPool::alloc).
    pub fn add<T, L: MutLayout>(&self, tensor: TensorBase<T, Vec<T>, L>) {
        self.add_vec(tensor.into_non_contiguous_data());
    }

    /// Add a data buffer to the pool.
    ///
    /// This is like [`add`](TensorPool::add) but takes a buffer directly,
    /// instead of a tensor from which a buffer can be extracted.
    pub fn add_vec<T>(&self, vec: Vec<T>) {
        self.buffers.borrow_mut().push(Buffer::from_vec(vec));
    }

    /// Return the total number of allocation requests.
    pub fn alloc_count(&self) -> usize {
        *self.alloc_count.borrow()
    }

    /// Return the number of allocation requests that were fulfilled using
    /// items in the pool.
    pub fn hit_count(&self) -> usize {
        *self.hit_count.borrow()
    }

    /// Return the number of buffers currently in the pool.
    pub fn len(&self) -> usize {
        self.buffers.borrow().len()
    }

    /// Return true if the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.buffers.borrow().is_empty()
    }
}

impl Alloc for TensorPool {
    fn alloc<T>(&self, capacity: usize) -> Vec<T> {
        self.alloc_vec(capacity)
    }
}

impl Default for TensorPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for wrapping a tensor in a [PoolRef] which automatically returns the
/// tensor to a pool when it goes out of scope.
pub trait AutoReturn {
    type Elem;
    type Layout: MutLayout;

    /// Wrap `self` in a [PoolRef]. When the returned [PoolRef] is dropped,
    /// `self` will be returned to `pool`.
    fn auto_return(self, pool: &TensorPool) -> PoolRef<Self::Elem, Self::Layout>;
}

impl<T, L: MutLayout> AutoReturn for TensorBase<T, Vec<T>, L> {
    type Elem = T;
    type Layout = L;

    fn auto_return(self, pool: &TensorPool) -> PoolRef<T, L> {
        PoolRef::new(pool, self)
    }
}

/// A smart pointer which wraps a tensor and adds it to a pool when dropped.
pub struct PoolRef<'a, T, L: MutLayout> {
    pool: &'a TensorPool,

    /// Wrapped tensor, set to `None` after the PoolRef is dropped.
    tensor: Option<TensorBase<T, Vec<T>, L>>,
}

impl<'a, T, L: MutLayout> PoolRef<'a, T, L> {
    /// Create a `PoolRef` which will wrap `tensor` and return it to `pool`
    /// when dropped.
    pub fn new(pool: &'a TensorPool, tensor: TensorBase<T, Vec<T>, L>) -> Self {
        PoolRef {
            pool,
            tensor: Some(tensor),
        }
    }

    /// Extract the wrapped tensor. After this is used, the tensor will no
    /// longer be added to the pool when `self` is dropped.
    pub fn take(mut self) -> TensorBase<T, Vec<T>, L> {
        self.tensor.take().unwrap()
    }
}

impl<'a, T, L: MutLayout> Deref for PoolRef<'a, T, L> {
    type Target = TensorBase<T, Vec<T>, L>;

    fn deref(&self) -> &Self::Target {
        self.tensor.as_ref().unwrap()
    }
}

impl<'a, T, L: MutLayout> DerefMut for PoolRef<'a, T, L> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.tensor.as_mut().unwrap()
    }
}

impl<'a, T, L: MutLayout> Drop for PoolRef<'a, T, L> {
    fn drop(&mut self) {
        if let Some(tensor) = self.tensor.take() {
            self.pool.add(tensor)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AutoReturn, TensorPool};
    use rten_tensor::prelude::*;

    #[test]
    fn test_pool_alloc() {
        let pool = TensorPool::new();

        // nb. These tests use `alloc_zeroed` because `TensorPool::add` expects
        // an initialized tensor.

        // Initial alloc. There is nothing in the pool so this will use the
        // system allocator.
        let tensor = pool.alloc_zeroed::<f32, _>([2, 2]);
        assert_eq!(tensor.shape(), [2, 2]);
        assert_eq!(pool.alloc_count(), 1);
        assert_eq!(pool.hit_count(), 0);
        let ptr = tensor.data().unwrap().as_ptr();

        pool.add(tensor);

        // Alloc with an exact size match. This will be fulfilled from the pool.
        let tensor = pool.alloc_zeroed::<f32, _>([2, 2]);
        assert_eq!(tensor.shape(), [2, 2]);
        assert_eq!(pool.alloc_count(), 2);
        assert_eq!(pool.hit_count(), 1);

        // Check we really did get the same data back.
        assert_eq!(tensor.data().unwrap().as_ptr(), ptr);

        pool.add(tensor);

        // Alloc with a smaller size. This will be fulfilled from the pool.
        let tensor = pool.alloc_zeroed::<f32, _>([2, 1]);
        assert_eq!(tensor.shape(), [2, 1]);
        assert_eq!(pool.alloc_count(), 3);
        assert_eq!(pool.hit_count(), 2);

        pool.add(tensor);

        // Alloc with a larger size. This will return a new tensor.
        let tensor = pool.alloc_zeroed::<f32, _>([2, 3]);
        assert_eq!(tensor.shape(), [2, 3]);
        assert_eq!(pool.alloc_count(), 4);
        assert_eq!(pool.hit_count(), 2);

        // Alloc with a size that matches the item in the pool, but a different
        // type. This will return a new tensor.
        let int_tensor = pool.alloc_zeroed::<i8, _>([2, 2]);
        assert_eq!(int_tensor.shape(), [2, 2]);
        assert_eq!(pool.alloc_count(), 5);
        assert_eq!(pool.hit_count(), 2);

        pool.add(int_tensor);

        // Alloc that matches the tensor we just freed. This will return the
        // item just added to the pool.
        let int_tensor = pool.alloc_zeroed::<i8, _>([2, 2]);
        assert_eq!(int_tensor.shape(), [2, 2]);
        assert_eq!(pool.alloc_count(), 6);
        assert_eq!(pool.hit_count(), 3);
    }

    #[test]
    fn test_pool_alloc_vec() {
        let pool = TensorPool::new();

        let vec = pool.alloc_vec::<f32>(128);
        assert_eq!(vec.capacity(), 128);
        assert_eq!(vec.len(), 0);
        assert_eq!(pool.alloc_count(), 1);
        assert_eq!(pool.hit_count(), 0);

        pool.add_vec(vec);

        let vec = pool.alloc_vec::<f32>(64);
        assert_eq!(vec.capacity(), 128);
        assert_eq!(vec.len(), 0);
        assert_eq!(pool.alloc_count(), 2);
        assert_eq!(pool.hit_count(), 1);
    }

    #[test]
    fn test_pool_alloc_zst() {
        let pool = TensorPool::new();

        let vec = pool.alloc_vec::<()>(128);
        assert_eq!(vec.capacity(), usize::MAX);
        pool.add_vec(vec);

        let vec = pool.alloc_vec::<()>(512);
        assert_eq!(vec.capacity(), usize::MAX);
        pool.add_vec(vec);

        assert_eq!(pool.alloc_count(), 2);
        assert_eq!(pool.hit_count(), 1);
    }

    #[test]
    fn test_pool_alloc_non_copy_type() {
        let pool = TensorPool::new();

        let mut vec = pool.alloc_vec::<String>(5);
        vec.push("hello".into());
        vec.push("world".into());
        let ptr = vec.as_ptr();

        pool.add_vec(vec);

        let vec = pool.alloc_vec::<String>(3);
        assert_eq!(vec.as_ptr(), ptr);
        assert_eq!(vec.capacity(), 5);
        assert_eq!(vec.len(), 0);
    }

    #[test]
    fn test_pool_ref() {
        let pool = TensorPool::new();
        assert_eq!(pool.len(), 0);

        {
            let tensor = pool.alloc_zeroed::<f32, _>([2, 2]).auto_return(&pool);
            assert_eq!(tensor.shape(), [2, 2]);
        }

        assert_eq!(pool.alloc_count(), 1);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn test_pool_auto_return() {
        let pool = TensorPool::new();
        assert_eq!(pool.len(), 0);
        {
            let tensor = pool.alloc_zeroed::<f32, _>([2, 2]).auto_return(&pool);
            assert_eq!(tensor.shape(), [2, 2]);
            tensor.take(); // Take tensor out of the `PoolRef`
        }
        assert_eq!(pool.len(), 0);
    }
}
