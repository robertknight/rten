use std::ops::{Deref, DerefMut};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use rten_gemm::{PackedAMatrix, PackedBMatrix};
use rten_tensor::{Alloc, CowData, MutLayout, TensorBase};

/// A memory buffer that can be used to satisfy a future allocation from
/// a [`BufferPool`].
///
/// This is conceptually like a `Vec` which has zero length, possibly non-zero
/// capacity and has been type-erased. A buffer can be cheaply converted into an
/// empty `Vec` of any type which has the same size and alignment as the type
/// of Vec from which the buffer was constructed. Converting a `Vec` to a
/// `Buffer` and back preserves the capacity of the vec but not the length.
pub struct Buffer {
    /// Pointer to the allocation. The data may be uninitialized.
    ptr: *mut u8,

    /// Capacity of the Vec from which the buffer was constructed. Note this
    /// is a count of elements rather than bytes.
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
    ///
    /// This will return `None` if the size or alignment of `T` does not match
    /// the size and alignment of the `Vec` that the buffer was originally
    /// created from.
    fn into_vec<T>(self) -> Option<Vec<T>> {
        // This code assumes that it is safe to transmute the buffer provided
        // that the original and new array layouts are the same.
        if !self.layout_match::<T>() {
            return None;
        }

        // Safety: We are reconstructing the vec with the same raw parts into
        // which it was decomposed in `from_vec`. The type may be different,
        // but it has the same size and alignment.
        let vec = unsafe { Vec::from_raw_parts(self.ptr as *mut T, 0, self.capacity) };

        // Don't drop self, as that would deallocate the buffer.
        std::mem::forget(self);

        Some(vec)
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

// Safety: Buffer does not have any interior mutability (ie. we don't allow
// modifying it via non-mut references). It isn't auto Send/Sync due to pointer
// fields.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Drop for Buffer {
    fn drop(&mut self) {
        (self.drop)(self);
    }
}

impl<T> From<Vec<T>> for Buffer {
    fn from(val: Vec<T>) -> Buffer {
        Self::from_vec(val)
    }
}

/// A pool which enables reuse of data buffers from tensors and other containers.
///
/// The purpose of this pool is to minimize the overhead from allocating and
/// de-allocating large buffers repeatedly during model inference.
///
/// This pool is optimized to store a small number of large buffers. For
/// allocations below a configurable threshold, the global allocator is used.
///
/// # Usage
///
/// [`BufferPool`] implements the [`Alloc`] trait, enabling tensors to be
/// allocated from the pool using the various `Tensor::*_in` methods, eg.
/// [`Tensor::zeros_in`](rten_tensor::Tensor::zeros_in). Allocation requests
/// will be satisfied from the pool if there is a suitable buffer available and
/// the requested capacity exceeds a threshold, otherwise the global allocator
/// is used.
///
/// When a tensor is no longer needed, its buffer can be added to the pool
/// using `pool.add(tensor.extract_buffer())`, making it available for future
/// allocations. A more convenient method is to wrap the tensor in a [`PoolRef`]
/// smart pointer which will auto-return the tensor to the pool when dropped. A
/// tensor can be wrapped using `tensor.auto_return(pool)`. The [`PoolRef`] smart
/// pointer can also be used with other container types, by implementing the
/// [`ExtractBuffer`] trait for them.
///
/// # Performance
///
/// Reusing buffers for operator outputs, as opposed to allocating a fresh
/// buffer from the global allocator and freeing it when no longer needed, can
/// provide a significant performance improvement for medium and especially
/// large allocations. This is because allocation or de-allocation of memory
/// blocks above a certain size can incur overhead from zeroing memory
/// ([^1][^2]) and memory mapping management in the OS. For small allocations
/// (<1KB) there is typically no benefit compared to using the system allocator.
///
/// The pool assumes that it will be managing a relatively small number of
/// buffers at any given time, and isn't optimized for managing a large number
/// of buffers.
///
/// [^1]: <https://mjtsai.com/blog/2022/09/20/zeroing-freed-memory/>
/// [^2]: <https://randomascii.wordpress.com/2014/12/10/hidden-costs-of-memory-allocation/>
pub struct BufferPool {
    /// List of buffers currently in the pool.
    buffers: Mutex<Vec<Buffer>>,

    /// Number of allocation requests received.
    alloc_count: AtomicUsize,

    /// Number of allocation requests fulfilled from the pool.
    hit_count: AtomicUsize,

    /// Minimum size, in bytes, of buffers to store in the pool.
    ///
    /// For small buffers it is more efficient to use the system allocator.
    /// The purpose of this pool is to avoid the overhead of allocating and
    /// freeing large buffers.
    min_size: usize,
}

impl BufferPool {
    /// Return a new, empty pool.
    ///
    /// This is a cheap operation that does not allocate, so it can be used
    /// to create a temporary pool to pass to a function that requires one,
    /// if the caller does not have a pool otherwise available.
    pub fn new() -> BufferPool {
        BufferPool {
            buffers: Mutex::new(Vec::new()),
            alloc_count: AtomicUsize::new(0),
            hit_count: AtomicUsize::new(0),
            min_size: 128,
        }
    }

    /// Configure the minimum size for allocations from the pool.
    ///
    /// Allocations below this size will fall back to the global allocator.
    pub fn with_min_size(mut self, n_bytes: usize) -> Self {
        self.min_size = n_bytes;
        self
    }

    /// Allocate an empty vec with a given capacity from the pool.
    ///
    /// The returned buffer will have a [`capacity`](Vec::capacity) of at least
    /// the requested size, but _may have more_.
    pub fn alloc<T>(&self, capacity: usize) -> Vec<T> {
        // Skip the pool for small buffers.
        if capacity * size_of::<T>() < self.min_size {
            return Vec::with_capacity(capacity);
        }

        self.alloc_count.fetch_add(1, Ordering::AcqRel);

        let mut buffers = self.buffers.lock().unwrap();

        // Find best fit item that matches the requested type and size with
        // the least excess capacity.
        let best_fit = buffers
            .iter()
            .enumerate()
            .fold(None, |best_fit, (idx, buffer)| {
                if !buffer.can_fit::<T>(capacity) {
                    return best_fit;
                };

                if let Some((best_fit_idx, best_fit_size)) = best_fit
                    && buffer.capacity >= best_fit_size
                {
                    return Some((best_fit_idx, best_fit_size));
                }
                Some((idx, buffer.capacity))
            });

        if let Some((best_fit, _overhead)) = best_fit {
            self.hit_count.fetch_add(1, Ordering::AcqRel);

            let item = buffers.remove(best_fit);
            return item.into_vec::<T>().expect("alignment should match");
        }

        // No suitable buffer was found. Fall back to the global allocator, but
        // release the mutex before we do.
        std::mem::drop(buffers);

        Vec::with_capacity(capacity)
    }

    /// Add a data buffer to the pool.
    ///
    /// The buffer will be cleared using [`Vec::clear`] and then made available
    /// to fulfill future allocation requests.
    pub fn add<B: Into<Buffer>>(&self, buf: B) {
        let buf: Buffer = buf.into();
        if buf.layout.size() >= self.min_size {
            self.buffers.lock().unwrap().push(buf);
        }
    }

    /// Return the total number of allocation requests.
    ///
    /// This excludes allocations below the minimum size threshold.
    pub fn alloc_count(&self) -> usize {
        self.alloc_count.load(Ordering::Acquire)
    }

    /// Return the number of allocation requests that were fulfilled using
    /// items in the pool.
    pub fn hit_count(&self) -> usize {
        self.hit_count.load(Ordering::Acquire)
    }

    /// Return the number of buffers currently in the pool.
    pub fn len(&self) -> usize {
        self.buffers.lock().unwrap().len()
    }

    /// Return true if the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.buffers.lock().unwrap().is_empty()
    }
}

impl Alloc for BufferPool {
    fn alloc<T>(&self, capacity: usize) -> Vec<T> {
        self.alloc(capacity)
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for extracting the data buffer from a tensor or other container.
///
/// This is used to extract the buffer from a container that is no longer
/// needed, in order to return it to a [`BufferPool`].
pub trait ExtractBuffer {
    /// Consume `self` and return its data buffer if it was uniquely owned, or
    /// `None` otherwise.
    fn extract_buffer(self) -> Option<Buffer>;
}

impl<T> ExtractBuffer for Vec<T> {
    fn extract_buffer(self) -> Option<Buffer> {
        if self.capacity() > 0 {
            Some(self.into())
        } else {
            None
        }
    }
}

impl<T, L: MutLayout> ExtractBuffer for TensorBase<Vec<T>, L> {
    fn extract_buffer(self) -> Option<Buffer> {
        Some(self.into_non_contiguous_data().into())
    }
}

impl<T, L: MutLayout> ExtractBuffer for TensorBase<CowData<'_, T>, L> {
    fn extract_buffer(self) -> Option<Buffer> {
        self.into_non_contiguous_data().map(|data| data.into())
    }
}

impl<T> ExtractBuffer for PackedAMatrix<T> {
    fn extract_buffer(self) -> Option<Buffer> {
        Some(self.into_vec().into())
    }
}

impl<T> ExtractBuffer for PackedBMatrix<T> {
    fn extract_buffer(self) -> Option<Buffer> {
        Some(self.into_vec().into())
    }
}

/// Trait for wrapping a container in a [`PoolRef`] which automatically returns
/// the container's data buffer to a pool when it goes out of scope.
pub trait AutoReturn {
    /// Wrap `self` in a [`PoolRef`].
    ///
    /// When the returned ref is dropped, `self` will be returned to the pool.
    fn auto_return(self, pool: &BufferPool) -> PoolRef<'_, Self>
    where
        Self: Sized + ExtractBuffer;
}

impl<EB: ExtractBuffer> AutoReturn for EB {
    fn auto_return(self, pool: &BufferPool) -> PoolRef<'_, EB> {
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
    pool: &'a BufferPool,
    container: Option<T>,
}

impl<'a, T: ExtractBuffer> PoolRef<'a, T> {
    /// Create a `PoolRef` which will wrap `tensor` and return it to `pool`
    /// when dropped.
    pub fn new(pool: &'a BufferPool, tensor: T) -> Self {
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
        if let Some(container) = self.container.take()
            && let Some(buffer) = container.extract_buffer()
        {
            self.pool.add(buffer)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AutoReturn, Buffer, BufferPool, ExtractBuffer};
    use rten_tensor::NdTensor;
    use rten_tensor::prelude::*;

    #[test]
    fn test_buffer() {
        let vec = vec![1i32, 2, 3];
        let cap = vec.capacity();
        let buf = Buffer::from_vec(vec);

        // Convert into Vec with a different type that has the same layout.
        let new_vec: Vec<f32> = buf.into_vec().unwrap();
        assert_eq!(new_vec.capacity(), cap); // Capacity is preserved
        assert_eq!(new_vec.len(), 0); // ...but the length is not.

        // Attempt to convert into Vec with larger alignment.
        let buf = Buffer::from_vec(new_vec);
        let new_vec: Option<Vec<i64>> = buf.into_vec();
        assert!(new_vec.is_none());

        // Attempt to convert into Vec with smaller alignment.
        let vec = vec![1i32, 2, 3];
        let buf = Buffer::from_vec(vec);
        let new_vec: Option<Vec<u8>> = buf.into_vec();
        assert!(new_vec.is_none());
    }

    #[test]
    fn test_empty_buffer() {
        // Empty Vec is special as it doesn't allocate.
        let buf = Buffer::from_vec(Vec::<i32>::new());
        std::mem::drop(buf);
    }

    #[test]
    fn test_zst_buffer() {
        // Make sure Buffer behaves correctly with zero-sized types.
        let vec = vec![(), ()];
        let cap = vec.capacity();
        let buf = Buffer::from_vec(vec);
        let new_vec: Vec<()> = buf.into_vec().unwrap();
        assert_eq!(new_vec.len(), 0);
        assert_eq!(new_vec.capacity(), cap);

        let buf = Buffer::from_vec(new_vec);
        std::mem::drop(buf);
    }

    #[test]
    fn test_pool_alloc_tensor() {
        let pool = BufferPool::new().with_min_size(0);

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
        let pool = BufferPool::new().with_min_size(0);

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
    fn test_pool_alloc_small() {
        let pool = BufferPool::new().with_min_size(18);

        // 16 byte allocation, just below min size.
        let vec = pool.alloc::<f32>(4);
        assert_eq!(vec.capacity(), 4);
        assert_eq!(vec.len(), 0);
        assert_eq!(pool.alloc_count(), 0);
        assert_eq!(pool.hit_count(), 0);

        // 20 byte allocation, just above min size
        let vec = pool.alloc::<f32>(5);
        assert_eq!(vec.capacity(), 5);
        assert_eq!(vec.len(), 0);
        assert_eq!(pool.alloc_count(), 1);
        assert_eq!(pool.hit_count(), 0);
    }

    #[test]
    fn test_pool_add_small() {
        let pool = BufferPool::new().with_min_size(18);

        // `add` discards buffers < min size.
        pool.add(vec![0.0f32; 4]);
        assert_eq!(pool.len(), 0);

        // `add` saves buffers >= min size.
        pool.add(vec![0.0f32; 5]);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn test_pool_alloc_zst() {
        let pool = BufferPool::new().with_min_size(0);

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
        let pool = BufferPool::new().with_min_size(0);

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
        let pool = BufferPool::new().with_min_size(0);
        assert_eq!(pool.len(), 0);

        // Owned tensor. This will auto-return to the pool.
        let tensor = NdTensor::<f32, 2>::zeros_in(&pool, [2, 2]).auto_return(&pool);
        assert_eq!(tensor.shape(), [2, 2]);
        assert_eq!(pool.alloc_count(), 1);
        assert_eq!(pool.len(), 0);

        // Conditional copy which doesn't copy. This will not return to the pool.
        let copy = tensor.to_contiguous_in(&pool).auto_return(&pool);
        std::mem::drop(copy);
        assert_eq!(pool.alloc_count(), 1);
        assert_eq!(pool.len(), 0);

        // Conditional copy which does copy. This will return to the pool.
        let copy = tensor
            .transposed()
            .to_contiguous_in(&pool)
            .auto_return(&pool);
        std::mem::drop(copy);
        assert_eq!(pool.alloc_count(), 2);
        assert_eq!(pool.len(), 1);

        std::mem::drop(tensor);
        assert_eq!(pool.len(), 2);

        // Non-empty vector. This will return to the pool.
        let non_empty = Vec::<f32>::with_capacity(16).auto_return(&pool);
        std::mem::drop(non_empty);
        assert_eq!(pool.len(), 3);

        // Empty vector. This will not return to the pool.
        let empty = Vec::<f32>::new().auto_return(&pool);
        std::mem::drop(empty);
        assert_eq!(pool.len(), 3);
    }

    #[test]
    fn test_pool_ref_take() {
        let pool = BufferPool::new().with_min_size(0);
        assert_eq!(pool.len(), 0);
        {
            let tensor = NdTensor::<f32, 2>::zeros_in(&pool, [2, 2]).auto_return(&pool);
            assert_eq!(tensor.shape(), [2, 2]);
            tensor.take(); // Take tensor out of the `PoolRef`
        }
        assert_eq!(pool.len(), 0);
    }
}
