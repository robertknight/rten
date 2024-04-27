use std::cell::RefCell;
use std::ops::{Deref, DerefMut};

use rten_tensor::{Alloc, MutLayout, TensorBase};

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

        // Safety: We are reconstructing the vec with the same raw parts into
        // which it was decomposed in `from_vec`. The type may be different,
        // but it has the same alignment.
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
        // Safety: We are reconstructing the vec with the same raw parts into
        // which it was decomposed in `from_vec`.
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
/// Tensors can be allocated from the pool by passing a reference to pool to the
/// various `Tensor::*_in` methods, eg. [Tensor::zeros_in]. Allocation requests
/// will be satisfied from the pool if there is a suitable buffer available, or
/// it will fall back to the global allocator otherwise.
///
/// When a tensor is no longer needed, it's buffer can be added to the pool
/// using [`add_tensor`](TensorPool::add_tensor), making it available for future
/// allocations. Alternately
/// [`tensor.auto_return(&pool)`](TensorBase::auto_return) can be used to wrap a
/// tensor in a smart pointer that returns it to the pool when no longer needed.
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

    /// Allocate an empty vec with a given capacity from the pool.
    pub fn alloc<T>(&self, capacity: usize) -> Vec<T> {
        *self.alloc_count.borrow_mut() += 1;

        // Find best fit item that matches the requested type and size with
        // the least excess capacity.
        let best_fit =
            self.buffers
                .borrow()
                .iter()
                .enumerate()
                .fold(None, |best_fit, (idx, buffer)| {
                    if !buffer.can_fit::<T>(capacity) {
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
            Vec::with_capacity(capacity)
        };

        data
    }

    /// Add a data buffer to the pool.
    ///
    /// The buffer will be cleared using [Vec::clear] and then made available
    /// to fulfill future allocation requests.
    pub fn add<T>(&self, vec: Vec<T>) {
        self.buffers.borrow_mut().push(Buffer::from_vec(vec));
    }

    /// Extract the data buffer from a tensor and add it to the pool.
    pub fn add_tensor<T, L: MutLayout>(&self, tensor: TensorBase<T, Vec<T>, L>) {
        self.add(tensor.into_non_contiguous_data());
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
        self.alloc(capacity)
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
            self.pool.add_tensor(tensor)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AutoReturn, TensorPool};
    use rten_tensor::prelude::*;
    use rten_tensor::NdTensor;

    #[test]
    fn test_pool_alloc_tensor() {
        let pool = TensorPool::new();

        // Initial alloc. There is nothing in the pool so this will use the
        // system allocator.
        let tensor = NdTensor::<f32, 2>::zeros_in(&pool, [2, 2]);
        assert_eq!(tensor.shape(), [2, 2]);
        assert_eq!(pool.alloc_count(), 1);
        assert_eq!(pool.hit_count(), 0);
        let ptr = tensor.data().unwrap().as_ptr();

        pool.add_tensor(tensor);

        // Alloc with an exact size match. This will be fulfilled from the pool.
        let tensor = NdTensor::<f32, 2>::zeros_in(&pool, [2, 2]);
        assert_eq!(tensor.shape(), [2, 2]);
        assert_eq!(pool.alloc_count(), 2);
        assert_eq!(pool.hit_count(), 1);

        // Check we really did get the same data back.
        assert_eq!(tensor.data().unwrap().as_ptr(), ptr);

        pool.add_tensor(tensor);

        // Alloc with a smaller size. This will be fulfilled from the pool.
        let tensor = NdTensor::<f32, 2>::zeros_in(&pool, [2, 1]);
        assert_eq!(tensor.shape(), [2, 1]);
        assert_eq!(pool.alloc_count(), 3);
        assert_eq!(pool.hit_count(), 2);

        pool.add_tensor(tensor);

        // Alloc with a larger size. This will return a new tensor.
        let tensor = NdTensor::<f32, 2>::zeros_in(&pool, [2, 3]);
        assert_eq!(tensor.shape(), [2, 3]);
        assert_eq!(pool.alloc_count(), 4);
        assert_eq!(pool.hit_count(), 2);

        // Alloc with a size that matches the item in the pool, but a different
        // type. This will return a new tensor.
        let int_tensor = NdTensor::<i8, 2>::zeros_in(&pool, [2, 2]);
        assert_eq!(int_tensor.shape(), [2, 2]);
        assert_eq!(pool.alloc_count(), 5);
        assert_eq!(pool.hit_count(), 2);

        pool.add_tensor(int_tensor);

        // Alloc that matches the tensor we just freed. This will return the
        // item just added to the pool.
        let int_tensor = NdTensor::<i8, 2>::zeros_in(&pool, [2, 2]);
        assert_eq!(int_tensor.shape(), [2, 2]);
        assert_eq!(pool.alloc_count(), 6);
        assert_eq!(pool.hit_count(), 3);
    }

    #[test]
    fn test_pool_alloc() {
        let pool = TensorPool::new();

        let vec = pool.alloc::<f32>(128);
        assert_eq!(vec.capacity(), 128);
        assert_eq!(vec.len(), 0);
        assert_eq!(pool.alloc_count(), 1);
        assert_eq!(pool.hit_count(), 0);

        pool.add(vec);

        let vec = pool.alloc::<f32>(64);
        assert_eq!(vec.capacity(), 128);
        assert_eq!(vec.len(), 0);
        assert_eq!(pool.alloc_count(), 2);
        assert_eq!(pool.hit_count(), 1);
    }

    #[test]
    fn test_pool_alloc_zst() {
        let pool = TensorPool::new();

        let vec = pool.alloc::<()>(128);
        assert_eq!(vec.capacity(), usize::MAX);
        pool.add(vec);

        let vec = pool.alloc::<()>(512);
        assert_eq!(vec.capacity(), usize::MAX);
        pool.add(vec);

        assert_eq!(pool.alloc_count(), 2);
        assert_eq!(pool.hit_count(), 1);
    }

    #[test]
    fn test_pool_alloc_non_copy_type() {
        let pool = TensorPool::new();

        let mut vec = pool.alloc::<String>(5);
        vec.push("hello".into());
        vec.push("world".into());
        let ptr = vec.as_ptr();

        pool.add(vec);

        let vec = pool.alloc::<String>(3);
        assert_eq!(vec.as_ptr(), ptr);
        assert_eq!(vec.capacity(), 5);
        assert_eq!(vec.len(), 0);
    }

    #[test]
    fn test_pool_ref_auto_return() {
        let pool = TensorPool::new();
        assert_eq!(pool.len(), 0);

        {
            let tensor = NdTensor::<f32, 2>::zeros_in(&pool, [2, 2]).auto_return(&pool);
            assert_eq!(tensor.shape(), [2, 2]);
        }

        assert_eq!(pool.alloc_count(), 1);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn test_pool_ref_take() {
        let pool = TensorPool::new();
        assert_eq!(pool.len(), 0);
        {
            let tensor = NdTensor::<f32, 2>::zeros_in(&pool, [2, 2]).auto_return(&pool);
            assert_eq!(tensor.shape(), [2, 2]);
            tensor.take(); // Take tensor out of the `PoolRef`
        }
        assert_eq!(pool.len(), 0);
    }
}
