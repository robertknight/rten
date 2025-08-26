use std::mem::{MaybeUninit, transmute};

use crate::Elem;
use crate::ops::NumOps;

/// Utility for incrementally filling an uninitialized slice, one SIMD vector
/// at a time.
pub struct SliceWriter<'a, T> {
    buf: &'a mut [MaybeUninit<T>],
    n_init: usize,
}

impl<'a, T: Elem> SliceWriter<'a, T> {
    /// Create a writer which initializes elements of `buf`.
    pub fn new(buf: &'a mut [MaybeUninit<T>]) -> Self {
        SliceWriter { buf, n_init: 0 }
    }

    /// Initialize the next `ops.len()` elements of the slice from the contents
    /// of SIMD vector `xs`.
    ///
    /// Panics if the slice does not have space for `ops.len()` elements.
    pub fn write_vec<O: NumOps<T>>(&mut self, ops: O, xs: O::Simd) {
        let written = ops.store_uninit(xs, &mut self.buf[self.n_init..]);
        self.n_init += written.len();
    }

    /// Initialize the next element of the slice from `x`.
    ///
    /// Panics if the slice does not have space for writing any more elements.
    pub fn write_scalar(&mut self, x: T) {
        self.buf[self.n_init].write(x);
        self.n_init += 1;
    }

    /// Finish writing the slice and return the initialized portion.
    pub fn into_mut_slice(self) -> &'a mut [T] {
        let init = &mut self.buf[0..self.n_init];

        // Safety: All elements in `init` have been initialized.
        unsafe { transmute::<&mut [MaybeUninit<T>], &mut [T]>(init) }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use crate::ops::NumOps;
    use crate::{Isa, SimdOp, SliceWriter};

    #[test]
    fn test_slice_writer() {
        struct MemCopy<'src, 'dest> {
            src: &'src [f32],
            dest: &'dest mut [MaybeUninit<f32>],
        }

        impl<'src, 'dest> SimdOp for MemCopy<'src, 'dest> {
            type Output = &'dest mut [f32];

            fn eval<I: Isa>(self, isa: I) -> &'dest mut [f32] {
                let ops = isa.f32();

                let mut src_chunks = self.src.chunks_exact(ops.len());
                let mut dest_writer = SliceWriter::new(self.dest);

                for chunk in src_chunks.by_ref() {
                    let xs = ops.load(chunk);
                    dest_writer.write_vec(ops, xs);
                }

                for x in src_chunks.remainder() {
                    dest_writer.write_scalar(*x);
                }

                dest_writer.into_mut_slice()
            }
        }

        // Length which should cover the vectorized body and tail cases for
        // every ISA.
        let len = 17;
        let src: Vec<_> = (0..len).map(|x| x as f32).collect();
        let mut dest = Vec::with_capacity(src.len());

        let copied = MemCopy {
            src: &src,
            dest: dest.spare_capacity_mut(),
        }
        .dispatch();
        assert_eq!(copied, src);
    }
}
