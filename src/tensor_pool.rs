use std::cell::RefCell;
use std::ops::{Deref, DerefMut};

use rten_tensor::{Alloc, CowData, MutLayout, TensorBase};

/// A memory buffer that can be used to satisfy a future allocation from
/// a [`TensorPool`].
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
    /// Clear `vec` using [`Vec::clear`] and convert it into a buffer.
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

/// A pool which enables reuse of data buffers from tensors and other containers.
///
/// Reusing buffers for operator outputs, as opposed to allocating a fresh
/// buffer from the global allocator and freeing it when no longer needed,
/// can provide a significant performance improvement.
///
/// [`TensorPool`] implements the [`Alloc`] trait, enabling tensors to be allocated
/// from the pool using the various `Tensor::*_in` methods, eg.
/// [`Tensor::zeros_in`](rten_tensor::Tensor::zeros_in). Allocation requests
/// will be satisfied from the pool if there is a suitable buffer available, or
/// it will fall back to the global allocator otherwise.
///
/// When a tensor is no longer needed, it's buffer can be added to the pool
/// using `pool.add(tensor.extract_buffer())`, making it available for future
/// allocations. A more convenient method is to wrap the tensor in a [`PoolRef`]
/// smart pointer which will auto-return the tensor to the pool when dropped. A
/// tensor can be wrapped using `tensor.auto_return(pool)`. The [`PoolRef`] smart
/// pointer can also be used with other container types, by implementing the
/// [`ExtractBuffer`] trait for them.
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
    /// The buffer will be cleared using [`Vec::clear`] and then made available
    /// to fulfill future allocation requests.
    pub fn add<T>(&self, vec: Vec<T>) {
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
        self.alloc(capacity)
    }
}

impl Default for TensorPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for extracting the data buffer from a tensor or other container.
///
/// This is used to extract the buffer from a container that is no longer
/// needed, in order to return it to a [`TensorPool`].
pub trait ExtractBuffer {
    type Elem;

    /// Consume `self` and return it's data buffer if it was uniquely owned, or
    /// `None` otherwise.
    fn extract_buffer(self) -> Option<Vec<Self::Elem>>;
}

impl<T, L: MutLayout> ExtractBuffer for TensorBase<Vec<T>, L> {
    type Elem = T;

    fn extract_buffer(self) -> Option<Vec<Self::Elem>> {
        Some(self.into_non_contiguous_data())
    }
}

impl<T, L: MutLayout> ExtractBuffer for TensorBase<CowData<'_, T>, L> {
    type Elem = T;

    fn extract_buffer(self) -> Option<Vec<Self::Elem>> {
        self.into_non_contiguous_data()
    }
}

/// Trait for wrapping a container in a [`PoolRef`] which automatically returns
/// the container's data buffer to a pool when it goes out of scope.
pub trait AutoReturn {
    /// Wrap `self` in a [`PoolRef`].
    ///
    /// When the returned ref is dropped, `self` will be returned to the pool.
    fn auto_return(self, pool: &TensorPool) -> PoolRef<Self>
    where
        Self: Sized + ExtractBuffer;
}

impl<EB: ExtractBuffer> AutoReturn for EB {
    fn auto_return(self, pool: &TensorPool) -> PoolRef<EB> {
        PoolRef::new(pool, self)
    }
}

/// A smart pointer which wraps a tensor or other container and returns it to
/// a pool when dropped.
///
/// [`PoolRef`] is not currently [`Sync`], so if you want to wrap a container and
/// then reference it inside a parallel block, you will need to deref the
/// [`PoolRef`] outside the parallel block.
pub struct PoolRef<'a, T: ExtractBuffer> {
    pool: &'a TensorPool,
    container: Option<T>,
}

impl<'a, T: ExtractBuffer> PoolRef<'a, T> {
    /// Create a `PoolRef` which will wrap `tensor` and return it to `pool`
    /// when dropped.
    pub fn new(pool: &'a TensorPool, tensor: T) -> Self {
        PoolRef {
            pool,
            container: Some(tensor),
        }
    }

    /// Extract the wrapped tensor. After this is used, the tensor will no
    /// longer be added to the pool when `self` is dropped.
    pub fn take(mut self) -> T {
        self.container.take().unwrap()
    }
}

impl<T: ExtractBuffer> Deref for PoolRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.container.as_ref().unwrap()
    }
}

impl<T: ExtractBuffer> DerefMut for PoolRef<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.container.as_mut().unwrap()
    }
}

impl<T: ExtractBuffer> Drop for PoolRef<'_, T> {
    fn drop(&mut self) {
        if let Some(container) = self.container.take() {
            if let Some(buffer) = container.extract_buffer() {
                self.pool.add(buffer)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AutoReturn, ExtractBuffer, TensorPool};
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

        pool.add(tensor.extract_buffer().unwrap());

        // Alloc with an exact size match. This will be fulfilled from the pool.
        let tensor = NdTensor::<f32, 2>::zeros_in(&pool, [2, 2]);
        assert_eq!(tensor.shape(), [2, 2]);
        assert_eq!(pool.alloc_count(), 2);
        assert_eq!(pool.hit_count(), 1);

        // Check we really did get the same data back.
        assert_eq!(tensor.data().unwrap().as_ptr(), ptr);

        pool.add(tensor.extract_buffer().unwrap());

        // Alloc with a smaller size. This will be fulfilled from the pool.
        let tensor = NdTensor::<f32, 2>::zeros_in(&pool, [2, 1]);
        assert_eq!(tensor.shape(), [2, 1]);
        assert_eq!(pool.alloc_count(), 3);
        assert_eq!(pool.hit_count(), 2);

        pool.add(tensor.extract_buffer().unwrap());

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

        pool.add(int_tensor.extract_buffer().unwrap());

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
            // Owned tensor. This will auto-return to the pool.
            let tensor = NdTensor::<f32, 2>::zeros_in(&pool, [2, 2]).auto_return(&pool);
            assert_eq!(tensor.shape(), [2, 2]);

            // Conditional copy which doesn't copy. This will not return to the pool.
            tensor.to_contiguous_in(&pool).auto_return(&pool);

            // Conditional copy which does copy. This will return to the pool.
            tensor
                .transposed()
                .to_contiguous_in(&pool)
                .auto_return(&pool);
        }

        assert_eq!(pool.alloc_count(), 2);
        assert_eq!(pool.len(), 2);
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
