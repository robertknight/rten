use std::mem::MaybeUninit;

/// Extend a buffer by incrementally initializing spare capacity.
///
/// This is implemented for [`Vec<T>`], where it provides a safe API to
/// initialize the spare capacity returned by
/// [`spare_capacity_mut`](Vec::spare_capacity_mut).
pub trait ExtendInit {
    /// Element type in the buffer.
    type Elem;

    /// Extend the buffer by initializing a portion of the buffer's spare
    /// capacity.
    ///
    /// The function `f` is passed the uninitialized portion of the buffer and
    /// should return the portion that it has initialized. `extend_init` can
    /// be called many times, until the entire buffer has been initialized.
    ///
    /// # Panics
    ///
    /// Panics if `f` returns a slice that is not a prefix of the slice that
    /// was passed to it.
    fn extend_init<F: Fn(&mut [MaybeUninit<Self::Elem>]) -> &[Self::Elem]>(&mut self, f: F);
}

impl<T> ExtendInit for Vec<T> {
    type Elem = T;

    fn extend_init<F: Fn(&mut [MaybeUninit<Self::Elem>]) -> &[Self::Elem]>(&mut self, f: F) {
        let cap = self.spare_capacity_mut();
        let cap_ptr = cap.as_ptr();
        let cap_len = cap.len();

        let initialized = f(cap);
        assert_eq!(
            initialized.as_ptr(),
            cap_ptr as *const T,
            "returned slice must be a prefix of the input"
        );
        assert!(
            initialized.len() <= cap_len,
            "initialized slice length {} is longer than input {}",
            initialized.len(),
            cap_len
        );
        let n_init = initialized.len();

        // Safety: `n_init` elements from the spare capacity have been initialized.
        unsafe { self.set_len(self.len() + n_init) }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use super::ExtendInit;

    // Implementation of `MaybeUninit::fill` from nightly Rust.
    fn fill<T: Copy>(xs: &mut [MaybeUninit<T>], value: T) -> &mut [T] {
        for x in xs.iter_mut() {
            x.write(value);
        }
        unsafe { std::mem::transmute::<&mut [MaybeUninit<T>], &mut [T]>(xs) }
    }

    #[test]
    fn test_extend_init() {
        let mut vec = Vec::with_capacity(7);

        vec.extend_init(|uninit| {
            assert_eq!(uninit.len(), 7);
            fill(&mut uninit[..3], 1.)
        });
        assert_eq!(vec.len(), 3);
        assert_eq!(vec, &[1., 1., 1.]);

        vec.extend_init(|uninit| {
            assert_eq!(uninit.len(), 4);
            fill(uninit, 2.)
        });
        assert_eq!(vec.len(), 7);
        assert_eq!(vec, &[1., 1., 1., 2., 2., 2., 2.]);
    }
}
