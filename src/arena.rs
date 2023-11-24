use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::cell::RefCell;
use std::mem::{align_of, size_of};
use std::ops::{Deref, DerefMut};

#[cfg(test)]
use std::ops::Range;

use wasnn_tensor::TensorViewMut;

struct Alloc {
    /// ID of the allocation, used when releasing it.
    id: usize,

    /// Offset of this allocation's elements within [Arena::buf].
    offset: usize,
    /// Size of this allocation.
    len: usize,
}

fn arena_buf_layout(capacity: usize) -> Layout {
    // This is the maximum alignment among the datatypes we expect to store in
    // the arena.
    let align = 8;
    Layout::from_size_align(capacity, align).unwrap()
}

fn round_up(val: usize, factor: usize) -> usize {
    let rem = val % factor;
    if rem == 0 {
        val
    } else {
        (val + factor) - rem
    }
}

/// An arena for allocating tensors during a graph run.
///
/// The arena makes several assumptions about its usage:
///
/// - At any given time, there will be a small number of large live allocations.
/// - The types stored in the arena are all copyable primitives (eg. f32, i32,
///   bool) that don't have a `Drop` impl.
pub struct Arena {
    buf: *mut u8,
    len: usize,

    allocs: RefCell<Vec<Alloc>>,
    next_id: RefCell<usize>,
}

impl Arena {
    /// Create a new arena with capacity for a given number of elements.
    ///
    /// The elements in the arena are initialized to zero.
    pub fn new(capacity: usize) -> Arena {
        assert!(capacity > 0, "arena capacity must be non-zero");
        let buf = unsafe { alloc_zeroed(arena_buf_layout(capacity)) as *mut u8 };
        assert!(!buf.is_null(), "failed to allocate");

        Arena {
            buf,
            len: capacity,

            allocs: RefCell::new(Vec::new()),
            next_id: RefCell::new(0),
        }
    }

    /// Allocate a new tensor in this arena with a given shape.
    ///
    /// The contents of the returned tensor will be either zeros or values
    /// written to a tensor that was previously allocated in this arena.
    ///
    /// Returns None if there is not enough space in the arena.
    pub fn alloc_uninit<T: Copy>(&self, shape: &[usize]) -> Option<ArenaRef<T>> {
        let align = align_of::<T>();
        assert!(self.buf as usize % align == 0, "buffer is not aligned to T");

        let next_offset = self
            .allocs
            .borrow()
            .iter()
            .map(|al| al.offset + al.len)
            .max()
            .unwrap_or(0);
        let next_offset = round_up(next_offset, align);

        let n_elts = shape.iter().product::<usize>();
        let len = n_elts * size_of::<T>();
        if next_offset + len > self.len {
            return None;
        }

        let mut next_id = self.next_id.borrow_mut();
        if *next_id == usize::MAX {
            // Prevent ID wraparound.
            return None;
        }

        let id = *next_id;
        *next_id += 1;

        let alloc = Alloc {
            id,
            offset: next_offset,
            len,
        };

        // Safety - `self.buf.add(alloc.offset)` is correctly aligned for T
        // and the allocation size is `len / size_of(T)`.
        let slice = unsafe {
            std::slice::from_raw_parts_mut(
                self.buf.add(alloc.offset) as *mut T,
                len / size_of::<T>(),
            )
        };
        self.allocs.borrow_mut().push(alloc);

        let view = TensorViewMut::<T>::from_data(shape, slice);
        let aref = ArenaRef {
            id,
            arena: self,
            view,
        };
        Some(aref)
    }

    /// Allocate a new tensor with all elements initialized to zero.
    ///
    /// Returns `None` if there is not enough space in the arena.
    pub fn alloc_zeros<T: Copy>(&self, shape: &[usize]) -> Option<ArenaRef<T>>
    where
        T: Clone + Default,
    {
        let mut tensor = self.alloc_uninit(shape)?;
        tensor.data_mut().unwrap().fill(T::default());
        Some(tensor)
    }

    /// Release an allocation in this arena.
    fn release(&self, alloc_id: usize) {
        if let Some(pos) = self.find_by_id(alloc_id) {
            self.allocs.borrow_mut().remove(pos);
        }
    }

    fn find_by_id(&self, id: usize) -> Option<usize> {
        self.allocs.borrow().iter().position(|al| al.id == id)
    }

    /// Return the range of elements within the arena's buffer that are
    /// associated with a particular allocation.
    #[cfg(test)]
    fn alloc_range(&self, alloc_id: usize) -> Range<usize> {
        let pos = self.find_by_id(alloc_id).unwrap();
        let alloc = &self.allocs.borrow()[pos];
        alloc.offset..alloc.offset + alloc.len
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        // The borrow checker will enforce that ArenaRefs don't outlive their
        // Arena. This is an internal sanity check.
        assert!(
            self.allocs.borrow().len() == 0,
            "dropped arena that is still in use"
        );

        // Safety: This is the same layout as when `self.buf` was allocated.
        let layout = arena_buf_layout(self.len);
        unsafe { dealloc(self.buf as *mut u8, layout) };
    }
}

/// A mutable view of a tensor whose elements are owned by an [Arena].
pub struct ArenaRef<'a, T> {
    id: usize,
    view: TensorViewMut<'a, T>,
    arena: &'a Arena,
}

impl<'a, T> ArenaRef<'a, T> {
    /// Return the offsets within the arena's buffer that are associated with
    /// this allocation.
    #[cfg(test)]
    fn alloc_range(&self) -> Range<usize> {
        self.arena.alloc_range(self.id)
    }
}

impl<'a, T> Drop for ArenaRef<'a, T> {
    fn drop(&mut self) {
        self.arena.release(self.id);
    }
}

impl<'a, T> Deref for ArenaRef<'a, T> {
    type Target = TensorViewMut<'a, T>;

    fn deref(&self) -> &Self::Target {
        &self.view
    }
}

impl<'a, T> DerefMut for ArenaRef<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.view
    }
}

#[cfg(test)]
mod tests {
    use wasnn_tensor::prelude::*;

    use super::Arena;
    use std::mem::size_of;

    #[test]
    fn test_arena_alloc() {
        let arena = Arena::new(512 * size_of::<f32>());

        // Two initial allocations. These should succeed since there is enough
        // room.
        let view_a = arena.alloc_uninit::<f32>(&[256]);
        assert!(view_a.is_some());
        let view_a = view_a.unwrap();
        assert_eq!(view_a.shape(), &[256]);
        assert_eq!(view_a.alloc_range(), 0..1024);

        let view_b = arena.alloc_uninit::<f32>(&[256]);
        assert!(view_b.is_some());
        let mut view_b = view_b.unwrap();
        assert_eq!(view_b.shape(), &[256]);
        assert_eq!(view_b.alloc_range(), 1024..2048);
        view_b.apply(|_| 42.);

        // Allocation into a full arena. This should fail.
        let view_c = arena.alloc_uninit::<f32>(&[256]);
        assert!(view_c.is_none());

        // Release one of the allocations and try again.
        std::mem::drop(view_b);

        let view_d = arena.alloc_uninit::<f32>(&[256]);
        assert!(view_d.is_some());
        let view_d = view_d.unwrap();
        assert_eq!(view_d.shape(), &[256]);
        assert_eq!(view_d.alloc_range(), 1024..2048);

        // Since `view_d` was allocated with `alloc_uninit`, it will initially
        // have the values from the earlier allocation.
        assert!(view_d.iter().all(|x| *x == 42.));
    }

    #[test]
    fn test_arena_alloc_zeros() {
        let arena = Arena::new(256 * size_of::<f32>());
        let view_a = arena.alloc_uninit::<f32>(&[256]);
        view_a.unwrap().apply(|_| 42.);

        let view_b = arena.alloc_zeros::<f32>(&[256]);
        assert!(view_b.unwrap().iter().all(|x| *x == 0.));
    }

    #[test]
    #[should_panic(expected = "arena capacity must be non-zero")]
    fn test_arena_panics_if_zero_capacity() {
        Arena::new(0);
    }
}
