use std::mem::MaybeUninit;

use rayon::prelude::*;

/// Initialize chunks of a buffer in parallel.
///
/// The buffer is divided into chunks of size `chunk_size` and each is
/// initialized in parallel via calls to `init_chunk`. Each `init_chunk` call
/// receives the index and uninitialized slice for a single chunk. `init_chunk`
/// must return the initialized chunk. `chunk_size` must be non-zero. If `len`
/// is not a multiple of `chunk_size`, the final chunk passed to `init_chunk`
/// will be shorter.
///
/// For use cases where the chunk size is 1, you can use Rayon's
/// [`ParallelExtend`] instead.
///
/// Returns the initialized vec with length `len`.
pub fn par_init_chunks<T: Send>(
    mut vec: Vec<T>,
    len: usize,
    chunk_size: usize,
    init_chunk: impl Fn(usize, &mut [MaybeUninit<T>]) -> &mut [T] + Sync,
) -> Vec<T> {
    vec.clear();
    if len == 0 {
        return vec;
    }

    vec.reserve(len);
    assert_ne!(chunk_size, 0, "chunk_size must be > 0");

    let buf = &mut vec.spare_capacity_mut()[..len];
    buf.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(idx, chunk)| {
            let ptr = chunk.as_ptr();
            let len = chunk.len();

            let initialized = init_chunk(idx, chunk);

            assert!(
                initialized.as_ptr() as *const MaybeUninit<T> == ptr,
                "initialized chunk has incorrect ptr"
            );
            assert!(
                initialized.len() == len,
                "initialized chunk has incorrect length"
            );
        });

    // Safety: All chunks were fully initialized by `init_chunk`.
    unsafe { vec.set_len(len) };

    vec
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::par_init_chunks;

    #[test]
    fn test_par_init_chunks() {
        let chunk_size = 1024;
        let len = 4 * chunk_size + 1;
        let buf = par_init_chunks(Vec::new(), len, chunk_size, |idx, buf| {
            let start = idx * chunk_size;
            let end = (start + chunk_size).min(len);
            for i in 0..end - start {
                buf[i].write(start + i);
            }
            // Safety: The buffer was fully initialized.
            unsafe { std::mem::transmute(buf) }
        });
        let expected: Vec<_> = (0..len).collect();
        assert_eq!(buf, expected);
    }
}
