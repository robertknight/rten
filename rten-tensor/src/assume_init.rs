use std::mem::MaybeUninit;

/// Trait for converting collections of uninitialized (`MaybeUninit<T>`) values
/// to collections of corresponding initializes values (`T`).
///
/// ## Example
///
/// ```
/// use std::mem::MaybeUninit;
/// use rten_tensor::AssumeInit;
///
/// fn scale_values<'a>(dst: &'a mut [MaybeUninit<f32>], src: &[f32], scale: f32) -> &'a mut [f32] {
///   for (y, x) in dst.into_iter().zip(src) {
///     y.write(x * scale);
///   }
///   // Safety: All elements have been initialized.
///   unsafe { dst.assume_init() }
/// }
///
/// let src = [1., 2., 3.];
/// let mut dst = [MaybeUninit::uninit(); 3];
/// let scaled = scale_values(&mut dst, &src, 2.);
/// assert_eq!(scaled, [2., 4., 6.]);
/// ```
pub trait AssumeInit {
    /// The type of the initialized storage.
    type Output;

    /// Cast `self` to a collection of initialized values.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that all elements have been initialized.
    unsafe fn assume_init(self) -> Self::Output;
}

impl<T> AssumeInit for Vec<MaybeUninit<T>> {
    type Output = Vec<T>;

    unsafe fn assume_init(mut self) -> Self::Output {
        let (ptr, len, capacity) = (self.as_mut_ptr(), self.len(), self.capacity());

        // Don't drop self, as that would deallocate.
        std::mem::forget(self);

        // Safety: We're re-constructing a `Vec` with the same length and
        // capacity and an element type that has the same size and alignment,
        // just cast from uninitialized to initialized.
        unsafe { Vec::from_raw_parts(ptr as *mut T, len, capacity) }
    }
}

impl<'a, T> AssumeInit for &'a [MaybeUninit<T>] {
    type Output = &'a [T];

    unsafe fn assume_init(self) -> Self::Output {
        std::mem::transmute(self)
    }
}

impl<'a, T> AssumeInit for &'a mut [MaybeUninit<T>] {
    type Output = &'a mut [T];

    unsafe fn assume_init(self) -> Self::Output {
        std::mem::transmute(self)
    }
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use super::AssumeInit;

    #[test]
    fn test_assume_init_vec() {
        let mut vec = vec![MaybeUninit::uninit(); 3];
        vec.reserve(4);

        for x in &mut vec {
            x.write(2.);
        }

        let vec = unsafe { vec.assume_init() };
        assert_eq!(vec.len(), 3);
        assert_eq!(vec.capacity(), 7);
        assert_eq!(vec, &[2., 2., 2.]);
    }
}
