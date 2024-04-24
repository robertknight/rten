use std::any::Any;
use std::cell::RefCell;
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};

use rten_tensor::prelude::*;
use rten_tensor::{IntoLayout, MutLayout, TensorBase};

/// A pool which enables reuse of tensor data buffers.
///
/// Reusing buffers for operator outputs, as opposed to allocating a fresh
/// buffer from the global allocator and freeing it when no longer needed,
/// can provide a significant performance improvement.
///
/// Tensors are allocated from the pool using [`alloc`](TensorPool::alloc)
/// and returned to the pool after use using [`add`](TensorPool::add). If
/// an allocation request cannot be satisfied by the pool, it will fall back
/// to the global allocator.
///
/// To simplify the implementation, the pool is limited to working with
/// buffers of `Copy + 'static` types.
pub struct TensorPool {
    /// List of buffers currently in the pool. Each is a
    /// `Vec<MaybeUninit<T>>` where `T` is a `Copy` type.
    items: RefCell<Vec<Box<dyn Any>>>,

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
            items: RefCell::new(Vec::new()),
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
    pub fn alloc<T: Copy + Any, S: IntoLayout>(
        &self,
        shape: S,
    ) -> TensorBase<MaybeUninit<T>, Vec<MaybeUninit<T>>, S::Layout> {
        let layout = shape.into_layout();
        TensorBase::from_data(layout.shape(), self.alloc_buf(layout.len()))
    }

    /// Allocate a tensor using [`alloc`](TensorPool::alloc) and fill all
    /// entries with zero.
    pub fn alloc_zeroed<T: Copy + Any + Default, S: IntoLayout>(
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
    pub fn alloc_vec<T: Copy + Any>(&self, capacity: usize) -> Vec<T> {
        let mut buf = self.alloc_buf::<T>(capacity);
        buf.clear();

        // Safety: Since the vec is empty, it is fully initialized and we
        // can convert `Vec<MaybeUninit<T>>` -> `Vec<T>`.
        unsafe { std::mem::transmute(buf) }
    }

    fn alloc_buf<T: Copy + Any>(&self, required_len: usize) -> Vec<MaybeUninit<T>> {
        *self.alloc_count.borrow_mut() += 1;

        // Find best fit item that matches the requested type and size with
        // the least excess capacity.
        let best_fit =
            self.items
                .borrow()
                .iter()
                .enumerate()
                .fold(None, |best_fit, (idx, tensor)| {
                    let Some(tensor) = tensor.downcast_ref::<Vec<MaybeUninit<T>>>() else {
                        return best_fit;
                    };

                    let len = tensor.capacity();
                    if len < required_len {
                        return best_fit;
                    }
                    let overhead = len - required_len;

                    if let Some((best_fit_idx, best_fit_overhead)) = best_fit {
                        if overhead >= best_fit_overhead {
                            return Some((best_fit_idx, best_fit_overhead));
                        }
                    }

                    Some((idx, overhead))
                });

        let mut data = if let Some((best_fit, _overhead)) = best_fit {
            *self.hit_count.borrow_mut() += 1;

            let item = self.items.borrow_mut().remove(best_fit);
            *item.downcast().expect("buffer type mismatch")
        } else {
            // No match :( - Fall back to the global allocator.
            Vec::with_capacity(required_len)
        };

        // Safety: Changing the length of a `MaybeUninit<T>` is safe since items
        // are de-facto "initialized".
        unsafe {
            assert!(required_len <= data.capacity());
            data.set_len(required_len);
        }

        data
    }

    /// Add the data buffer from a tensor into the pool, so it can be used
    /// to satisfy future calls to [`alloc`](TensorPool::alloc).
    ///
    /// This method expects `T` to be an initialized type (ie. not an
    /// uninitialized tensor as returned by `Tensor::uninit`).
    pub fn add<T: Any + Copy, L: MutLayout>(&self, tensor: TensorBase<T, Vec<T>, L>) {
        self.add_vec(tensor.into_non_contiguous_data());
    }

    /// Add a data buffer to the pool.
    ///
    /// This is like [`add`](TensorPool::add) but takes a buffer directly,
    /// instead of a tensor from which a buffer can be extracted.
    pub fn add_vec<T: Any + Copy>(&self, mut vec: Vec<T>) {
        vec.clear();

        // The buffer is now empty, so we can mark it uninitialized.
        let data: Vec<MaybeUninit<T>> = unsafe { std::mem::transmute(vec) };

        self.items.borrow_mut().insert(0, Box::new(data));
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
        self.items.borrow().len()
    }

    /// Return true if the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.items.borrow().is_empty()
    }
}

impl Default for TensorPool {
    fn default() -> Self {
        Self::new()
    }
}

/// A smart pointer which wraps a tensor and adds it to a pool when dropped.
pub struct PoolRef<'a, T: Any + Copy, L: MutLayout> {
    pool: &'a TensorPool,

    /// Wrapped tensor, set to `None` after the PoolRef is dropped.
    tensor: Option<TensorBase<T, Vec<T>, L>>,
}

impl<'a, T: Any + Copy, L: MutLayout> PoolRef<'a, T, L> {
    /// Create a `PoolRef` which will wrap `tensor` and return it to `pool`
    /// when dropped.
    pub fn new(pool: &'a TensorPool, tensor: TensorBase<T, Vec<T>, L>) -> Self {
        PoolRef {
            pool,
            tensor: Some(tensor),
        }
    }
}

impl<'a, T: Any + Copy, L: MutLayout> Deref for PoolRef<'a, T, L> {
    type Target = TensorBase<T, Vec<T>, L>;

    fn deref(&self) -> &Self::Target {
        self.tensor.as_ref().unwrap()
    }
}

impl<'a, T: Any + Copy, L: MutLayout> DerefMut for PoolRef<'a, T, L> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.tensor.as_mut().unwrap()
    }
}

impl<'a, T: Any + Copy, L: MutLayout> Drop for PoolRef<'a, T, L> {
    fn drop(&mut self) {
        self.pool.add(self.tensor.take().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;

    use super::{PoolRef, TensorPool};

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
        let int_tensor = pool.alloc_zeroed::<i32, _>([2, 2]);
        assert_eq!(int_tensor.shape(), [2, 2]);
        assert_eq!(pool.alloc_count(), 5);
        assert_eq!(pool.hit_count(), 2);

        pool.add(int_tensor);

        // Alloc that matches the tensor we just freed. This will return the
        // item just added to the pool.
        let int_tensor = pool.alloc_zeroed::<i32, _>([2, 2]);
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
    fn test_pool_ref() {
        let pool = TensorPool::new();
        assert_eq!(pool.len(), 0);

        {
            let tensor = PoolRef::new(&pool, pool.alloc_zeroed::<f32, _>([2, 2]));
            assert_eq!(tensor.shape(), [2, 2]);
        }

        assert_eq!(pool.alloc_count(), 1);
        assert_eq!(pool.len(), 1);
    }
}
