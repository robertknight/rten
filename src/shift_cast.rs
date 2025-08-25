use rten_tensor::prelude::*;
use rten_tensor::{Alloc, CowTensor, TensorView};

/// Conversion from one numeric type to another that preserves the value's
/// offset from the minimum value.
///
/// This trait is also implemented for some collection types, which can
/// avoid allocating a new collection if the source and target types are the
/// same.
///
/// Converting `0i8` to `u8` via a normal cast returns 0, but a shift cast
/// returns `128u8`, since `0i8 - i8::MIN = 128` and `128u8 - u8::MIN = 128`.
pub trait ShiftCast<T> {
    /// Return a value of type T that has the same difference from `T::MIN`
    /// as `self` has from `Self::MIN`.
    fn shift_cast(self) -> T;

    /// Variant of [`shift_cast`](ShiftCast::shift_cast) that takes an allocator.
    fn shift_cast_in(self, _alloc: impl Alloc) -> T
    where
        Self: Sized,
    {
        self.shift_cast()
    }
}

macro_rules! impl_noop_cast {
    ($type:ty) => {
        impl ShiftCast<$type> for $type {
            fn shift_cast(self) -> Self {
                self
            }
        }
    };
}

impl_noop_cast!(i8);
impl ShiftCast<u8> for i8 {
    fn shift_cast(self) -> u8 {
        (self as u8) ^ 0x80
    }
}

impl_noop_cast!(u8);
impl ShiftCast<i8> for u8 {
    fn shift_cast(self) -> i8 {
        (self ^ 0x80) as i8
    }
}

impl<'a, T> ShiftCast<CowTensor<'a, T>> for TensorView<'a, T> {
    fn shift_cast(self) -> CowTensor<'a, T> {
        self.as_cow()
    }
}

impl<'a> ShiftCast<CowTensor<'a, u8>> for TensorView<'a, i8> {
    fn shift_cast(self) -> CowTensor<'a, u8> {
        self.map(|&x| x.shift_cast()).into_cow()
    }

    fn shift_cast_in(self, alloc: impl Alloc) -> CowTensor<'a, u8> {
        self.map_in(alloc, |&x| x.shift_cast()).into_cow()
    }
}

impl<'a> ShiftCast<CowTensor<'a, i8>> for TensorView<'a, u8> {
    fn shift_cast(self) -> CowTensor<'a, i8> {
        self.map(|&x| x.shift_cast()).into_cow()
    }

    fn shift_cast_in(self, alloc: impl Alloc) -> CowTensor<'a, i8> {
        self.map_in(alloc, |&x| x.shift_cast()).into_cow()
    }
}

impl<T, U> ShiftCast<Vec<U>> for Vec<T>
where
    T: ShiftCast<U>,
{
    fn shift_cast(self) -> Vec<U> {
        self.into_iter().map(|x| x.shift_cast()).collect()
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::Tensor;
    use rten_tensor::prelude::*;

    use super::{CowTensor, ShiftCast};

    #[test]
    fn test_shift_cast_scalar() {
        const LEN: usize = 5;

        let input = [-128i8, -1, 0, 1, 127];
        let expected = [0u8, 127, 128, 129, 255];

        let actual: [u8; LEN] = input.map(|x| x.shift_cast());
        assert_eq!(actual, expected);

        let actual_noop: [u8; LEN] = actual.map(|x| x.shift_cast());
        assert_eq!(actual_noop, expected);

        let actual_inverse: [i8; LEN] = expected.map(|x| x.shift_cast());
        assert_eq!(actual_inverse, input);

        let actual_inverse_noop: [i8; LEN] = input.map(|x| x.shift_cast());
        assert_eq!(actual_inverse_noop, input);
    }

    #[test]
    fn test_shift_cast_tensor() {
        let input = Tensor::from([-128i8, -1, 0, 1, 127]);
        let expected = Tensor::from([0u8, 127, 128, 129, 255]);

        let actual: CowTensor<u8> = input.view().shift_cast();
        assert_eq!(actual, expected);

        let noop_cast: CowTensor<u8> = actual.view().shift_cast();
        assert_eq!(noop_cast, actual);

        let actual_inverse: CowTensor<i8> = expected.view().shift_cast();
        assert_eq!(actual_inverse, input);
    }

    #[test]
    fn test_shift_cast_vec() {
        let input: Vec<_> = [-128i8, -1, 0, 1, 127].into();
        let expected = [0u8, 127, 128, 129, 255];
        let actual: Vec<u8> = input.shift_cast();
        assert_eq!(actual, expected);
    }
}
