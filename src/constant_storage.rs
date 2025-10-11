//! Storage for constants (ie. weights) in a graph.

use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;

use rten_base::byte_cast::Pod;
use rten_tensor::{DynLayout, Storage, TensorBase};

#[cfg(feature = "mmap")]
use memmap2::Mmap;

/// Return the range of pointer addresses of a slice.
fn slice_address_range<T>(slice: &[T]) -> Range<usize> {
    let addr = slice.as_ptr() as usize;
    addr..(addr + std::mem::size_of_val(slice))
}

/// A buffer containing aligned data for heterogenous tensor types. This will
/// usually be a model data/weights file (eg. in FlatBuffers format) read in
/// or memory-mapped from disk.
///
/// This can be used as the storage for tensor views by creating [`ArcSlice`]
/// instances which reference a region of this buffer, and then using that
/// slice as the storage for an [`ArcTensorView`].
#[derive(Debug)]
pub enum ConstantStorage {
    /// Storage that references a memory-mapped file.
    #[cfg(feature = "mmap")]
    Mmap(Mmap),

    /// An in-memory buffer, such as a FlatBuffers file that has been read
    /// into memory using functions from `std::fs`.
    Buffer(Vec<u8>),

    /// A static buffer.
    ///
    /// This can be used to load a model baked into the binary using
    /// [`include_bytes`].
    StaticSlice(&'static [u8]),
}

impl ConstantStorage {
    /// Return the data in this storage as a slice of bytes.
    pub fn data(&self) -> &[u8] {
        match &self {
            ConstantStorage::Buffer(data) => data,
            ConstantStorage::StaticSlice(data) => data,
            #[cfg(feature = "mmap")]
            ConstantStorage::Mmap(mmap) => mmap,
        }
    }

    /// Return the byte offsets of a sub-slice of this storage as a range, or
    /// `None` if any part of `data` lies outside storage.
    ///
    /// Note this always returns `None` if `T` is a zero-sized type.
    fn byte_range_of<T>(&self, data: &[T]) -> Option<Range<usize>> {
        // See https://internals.rust-lang.org/t/proposal-get-range-of-sub-slice/16556
        if std::mem::size_of::<T>() == 0 {
            return None;
        }

        let self_range = slice_address_range(self.data());
        let data_range = slice_address_range(data);

        if !self_range.contains(&data_range.start) || self_range.end < data_range.end {
            return None;
        }

        let start = data_range.start - self_range.start;
        let end = data_range.end - self_range.start;
        Some(start..end)
    }
}

/// Tensor storage which references data owned by an `Arc<ConstantStorage>`.
#[derive(Debug)]
pub struct ArcSlice<T> {
    /// Shared byte buffer. The alignment must be a multiple of `T`s alignment.
    storage: Arc<ConstantStorage>,

    /// Offset of first element as a count of bytes. This must be a multiple
    /// of the size and alignment of `T`.
    byte_offset: usize,

    /// Number of elements in this storage.
    len: usize,

    phantom: PhantomData<T>,
}

// Manual implementation of `ArcSlice<T>` avoids adding a `T: Clone` bound.
impl<T> Clone for ArcSlice<T> {
    fn clone(&self) -> ArcSlice<T> {
        ArcSlice {
            storage: self.storage.clone(),
            byte_offset: self.byte_offset,
            len: self.len,
            phantom: PhantomData,
        }
    }
}

impl<T> ArcSlice<T> {
    /// Return an ArcSlice which references the subslice of `storage` specified
    /// by `data`.
    ///
    /// Returns `None` if the data slice is not contained within `storage` or
    /// is incorrectly aligned.
    pub fn new(storage: Arc<ConstantStorage>, data: &[T]) -> Option<ArcSlice<T>> {
        let byte_range = storage.byte_range_of(data)?;
        Some(ArcSlice::<T> {
            storage,
            byte_offset: byte_range.start,
            len: data.len(),
            phantom: PhantomData,
        })
    }

    /// Create an ArcSlice from a buffer of bytes.
    ///
    /// This places `buf` in a reference-counted [`ConstantStorage`] container
    /// and returns a view of the data as a slice of `T`s. The size and
    /// alignment of the buffer must be a multiple of the size and alignment of
    /// `T` respectively.
    pub fn from_bytes(buf: Vec<u8>) -> Option<ArcSlice<T>>
    where
        T: Pod,
    {
        // Vecs with zero capacity have a non-null dangling pointer which can
        // have a smaller alignment that the minimum used by the global
        // allocator. Create a new vec with zero length but non-zero capacity to
        // ensure the data pointer has the required alignment. This assumes that
        // `T` has an alignment <= the minimum alignment of the allocator. If
        // not, this operation will still fail.
        let buf = if buf.capacity() == 0 {
            Vec::with_capacity(1)
        } else {
            buf
        };

        if !(buf.as_ptr() as usize).is_multiple_of(align_of::<T>())
            || !buf.len().is_multiple_of(size_of::<T>())
        {
            return None;
        }
        let len = buf.len().checked_div(size_of::<T>())?;
        let storage = Arc::new(ConstantStorage::Buffer(buf));
        Some(ArcSlice::<T> {
            storage,
            byte_offset: 0,
            len,
            phantom: PhantomData,
        })
    }
}

// Safety: ArcSlice constructors ensure that the byte range defined by
// `self.byte_offset..self.byte_offset + self.len` is in-bounds for the storage
// length and that the storage's data pointer is correctly aligned for the type
// T.
unsafe impl<T> Storage for ArcSlice<T> {
    type Elem = T;

    const MUTABLE: bool = false;

    fn len(&self) -> usize {
        self.len
    }

    fn as_ptr(&self) -> *const Self::Elem {
        // Safety: We checked the data range was in-bounds when the ArcSlice
        // was constructed.
        unsafe {
            let ptr = self.storage.data().as_ptr().add(self.byte_offset);
            std::mem::transmute(ptr)
        }
    }
}

/// Tensor view whose data is a slice of a buffer owned by a [`ConstantStorage`].
pub type ArcTensorView<T> = TensorBase<ArcSlice<T>, DynLayout>;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rten_base::byte_cast::cast_pod_slice;
    use rten_tensor::prelude::*;

    use super::{ArcSlice, ArcTensorView, ConstantStorage};

    /// Convert a `Vec<i32>` to native-endian bytes.
    fn vec_to_ne_bytes(vec: Vec<i32>) -> Vec<u8> {
        vec.into_iter()
            .flat_map(|x| x.to_ne_bytes().into_iter())
            .collect()
    }

    #[test]
    fn test_constant_storage() {
        let data: Vec<i32> = (0..16).collect();
        let bytes = vec_to_ne_bytes(data);
        let storage = Arc::new(ConstantStorage::Buffer(bytes));

        // Create two slices referencing memory from the storage.
        let slice_one = cast_pod_slice::<u8, i32>(&storage.data()[0..32]).unwrap();
        assert_eq!(slice_one, [0, 1, 2, 3, 4, 5, 6, 7]);

        let slice_two = cast_pod_slice::<u8, i32>(&storage.data()[32..64]).unwrap();
        assert_eq!(slice_two, [8, 9, 10, 11, 12, 13, 14, 15]);

        let arc_slice_one = ArcSlice::new(storage.clone(), slice_one).unwrap();
        let arc_slice_two = ArcSlice::new(storage.clone(), slice_two).unwrap();

        let view_one = ArcTensorView::from_data(&[2, 4], arc_slice_one);
        let view_two = ArcTensorView::from_data(&[4, 2], arc_slice_two);

        assert_eq!(view_one.shape(), &[2, 4]);
        assert_eq!(view_one.data().unwrap(), slice_one);

        assert_eq!(view_two.shape(), &[4, 2]);
        assert_eq!(view_two.data().unwrap(), slice_two);

        // Create a slice referencing data outside the storage.
        let slice_outside = &[1, 2, 3];
        assert!(ArcSlice::new(storage.clone(), slice_outside).is_none());

        // Try with a zero-sized type.
        let zst_slice = &[(), ()];
        assert!(ArcSlice::new(storage.clone(), zst_slice).is_none());
    }

    #[test]
    fn test_arc_slice_from_bytes() {
        let data: Vec<i32> = (0..16).collect();
        let bytes = vec_to_ne_bytes(data.clone());
        let slice = ArcSlice::<i32>::from_bytes(bytes).unwrap();
        let tensor = ArcTensorView::from_data(&[16], slice);
        assert_eq!(tensor.data().unwrap(), data);
    }

    #[test]
    fn test_arc_slice_from_empty_bytes() {
        let data: Vec<i32> = Vec::new();
        let bytes = vec_to_ne_bytes(data.clone());
        let slice = ArcSlice::<i32>::from_bytes(bytes).unwrap();
        let tensor = ArcTensorView::from_data(&[0], slice);
        assert_eq!(tensor.data().unwrap(), data);
    }

    #[test]
    fn test_arc_slice_from_bytes_invalid() {
        let data: Vec<i32> = (0..16).collect();
        let mut bytes = vec_to_ne_bytes(data.clone());
        bytes.push(0u8); // Make length a non-multiple of i32 size.
        assert!(ArcSlice::<i32>::from_bytes(bytes).is_none());
    }
}
